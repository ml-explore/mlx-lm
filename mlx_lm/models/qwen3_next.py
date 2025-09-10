# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Tuple

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, MambaCache
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    linear_num_value_heads: int
    linear_num_key_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    num_experts: int
    num_experts_per_tok: int
    decoder_sparse_step: int
    shared_expert_intermediate_size: int
    mlp_only_layers: List[int]
    moe_intermediate_size: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    rope_theta: float
    tie_word_embeddings: bool
    max_position_embeddings: int
    norm_topk_prob: bool
    attention_bias: bool
    layer_types: Optional[List[str]] = None
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None


@mx.compile
def recurrent_gated_delta_rule(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    g: mx.array,
    beta: mx.array,
    initial_state: Optional[mx.array] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[mx.array, Optional[mx.array]]:
    """Minimal recurrent gated delta rule in MLX matching the Torch reference.
    Expects query/key/value shapes (B, S, H, D*) and g/beta shapes (B, S, H) or (B, S, H, 1).
    """
    # Optional L2 normalization on last dim for query/key
    if use_qk_l2norm_in_kernel:
        # Normalize along the feature dimension
        query = query / mx.maximum(mx.linalg.norm(query, axis=-1, keepdims=True), 1e-12)
        key = key / mx.maximum(mx.linalg.norm(key, axis=-1, keepdims=True), 1e-12)

    # Cast to float32 for numerical stability (like Torch .to(torch.float32))
    query = query.astype(mx.float32)
    key = key.astype(mx.float32)
    value = value.astype(mx.float32)
    beta = beta.astype(mx.float32)
    g = g.astype(mx.float32)

    # Allow beta and g to come with an extra trailing singleton dim: (B,S,H,1)
    if beta.ndim == 4 and beta.shape[-1] == 1:
        beta = beta.squeeze(-1)
    if g.ndim == 4 and g.shape[-1] == 1:
        g = g.squeeze(-1)

    B, S, H, Dk = key.shape
    Dv = value.shape[-1]

    # Scale queries by 1/sqrt(Dq) (Dq == last dim of query)
    scale = 1.0 / mx.sqrt(mx.array(query.shape[-1], dtype=mx.float32))
    query = query * scale

    # Precompute value*beta and key*beta to match the Torch reference
    v_beta = value * beta[..., None]
    k_beta = key * beta[..., None]

    # Initialize state: (B, H, Dk, Dv)
    if initial_state is None:
        state = mx.zeros((B, H, Dk, Dv), dtype=value.dtype)
    else:
        state = initial_state.astype(value.dtype)
        if state.shape != (B, H, Dk, Dv):
            state = state.reshape(B, H, Dk, Dv)

    # Output buffer: (B, S, H, Dv)
    out = mx.zeros((B, S, H, Dv), dtype=value.dtype)

    for t in range(S):
        q_t = query[:, t]       # (B, H, Dk)
        k_t = k_beta[:, t]      # (B, H, Dk)
        v_t = v_beta[:, t]      # (B, H, Dv)
        g_t = g[:, t]           # (B, H)

        # decay = exp(g_t)
        decay = mx.exp(g_t)[..., None]  # (B, H, 1)

        # state = state * decay.unsqueeze(-1) + k_t.unsqueeze(-1) @ v_t.unsqueeze(-2)
        state = state * decay[..., None] + mx.matmul(
            k_t[..., None],            # (B, H, Dk, 1)
            v_t[..., None, :],         # (B, H, 1,  Dv)
        )

        # out[:, t] = einsum("bhd,bhdv->bhv", q_t, state)
        out[:, t] = mx.einsum("bhd,bhdv->bhv", q_t, state)

    # Return (B, H, S, Dv) like Torch's out.transpose(1, 2)
    out = mx.transpose(out, (0, 2, 1, 3)).astype(query.dtype)

    if not output_final_state:
        state = None
    return out, state


class Qwen3NextRMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(hidden_size)

    def __call__(self, hidden_states: mx.array, gate: mx.array = None) -> mx.array:
        if gate is not None:
            hidden_states = hidden_states * nn.silu(gate)
        return mx.fast.rms_norm(hidden_states, self.weight, self.eps)


class Qwen3NextAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_key_value_heads = args.num_key_value_heads
        self.num_attention_heads = args.num_attention_heads
        self.head_dim = args.hidden_size // self.num_attention_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(args.hidden_size, self.num_attention_heads * self.head_dim * 2, bias=args.attention_bias)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, args.hidden_size, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        self.rope = nn.RoPE(
            self.head_dim,
            traditional=False,
            base=args.rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape
        
        q_proj_output = self.q_proj(x)
        queries, gate = mx.split(q_proj_output.reshape(B, L, self.num_attention_heads, -1, 2), 2, axis=-1)
        queries = queries.squeeze(-1)
        gate = gate.squeeze(-1).reshape(B, L, -1)
        
        keys, values = self.k_proj(x), self.v_proj(x)
        
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.num_key_value_heads, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        
        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)
        
        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        
        output = output * mx.sigmoid(gate)
        
        return self.o_proj(output)


class Qwen3NextMLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3NextGatedDeltaNet(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_norm_epsilon = config.rms_norm_eps

        self.conv_dim = self.key_dim * 2 + self.value_dim

        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
        )

        projection_size_qkvz = self.key_dim * 2 + self.value_dim * 2
        projection_size_ba = self.num_v_heads * 2
        self.in_proj_qkvz = nn.Linear(self.hidden_size, projection_size_qkvz, bias=False)
        self.in_proj_ba = nn.Linear(self.hidden_size, projection_size_ba, bias=False)

        self.dt_bias = mx.ones(self.num_v_heads)

        A = mx.random.uniform(low=0, high=16, shape=(self.num_v_heads,))
        self.A_log = mx.log(A)

        self.norm = Qwen3NextRMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)

        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)
    
    def fix_query_key_value_ordering(self, mixed_qkvz, mixed_ba):
        nq, nk, nv, dv = self.num_k_heads, self.head_k_dim, self.num_v_heads, self.head_v_dim
        mixed_qkvz = mixed_qkvz.reshape(*mixed_qkvz.shape[:-1], nq, 2*nk + 2*nv*dv//nq)
        mixed_ba = mixed_ba.reshape(*mixed_ba.shape[:-1], nq, 2*nv//nq)
        q,k,v,z = mx.split(mixed_qkvz,[nk,2*nk,2*nk+nv//nq*dv],axis=-1)
        b,a = mx.split(mixed_ba,[nv//nq],axis=-1)
        return (
            q,
            k,
            v.reshape(v.shape[0], v.shape[1], -1, dv),
            z.reshape(z.shape[0], z.shape[1], -1, dv),
            b.reshape(b.shape[0], b.shape[1], nv),
            a.reshape(a.shape[0], a.shape[1], nv),
        )
    
    
    def __call__(self, inputs: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None):
        B,L,_ = inputs.shape
        qkvz, ba = self.in_proj_qkvz(inputs), self.in_proj_ba(inputs)
        q,k,v,z,b,a = self.fix_query_key_value_ordering(qkvz, ba)
        q,k,v = (x.reshape(B,L,-1) for x in (q,k,v))
        mixed_qkv = mx.concatenate((q,k,v),-1)

        if cache is not None:
            conv_state, rec_state = cache
            if conv_state is None:
                conv_state = mx.zeros((B,self.conv_kernel_size-1,self.conv_dim),dtype=inputs.dtype)
            padded = mx.concatenate([conv_state,mixed_qkv],1)
            cache[0] = padded[:,-(self.conv_kernel_size-1):]
            mixed_qkv = nn.silu(self.conv1d(padded)[:,:L])
        else:
            padded = mx.pad(mixed_qkv,[(0,0),(self.conv_kernel_size-1,0),(0,0)])
            mixed_qkv = nn.silu(self.conv1d(padded)[:,:L]); rec_state=None

        q,k,v = mx.split(mixed_qkv,[self.key_dim,2*self.key_dim],-1)
        q = q.reshape(B,L,-1,self.head_k_dim); k = k.reshape(B,L,-1,self.head_k_dim); v = v.reshape(B,L,-1,self.head_v_dim)

        beta = mx.sigmoid(b).reshape(B,L,-1,1)
        g = (-mx.exp(self.A_log.astype(mx.float32))*nn.softplus(a.astype(mx.float32)+self.dt_bias)).reshape(B,L,-1,1)
        if self.num_v_heads//self.num_k_heads>1:
            q = mx.repeat(q,self.num_v_heads//self.num_k_heads,axis=2)
            k = mx.repeat(k,self.num_v_heads//self.num_k_heads,axis=2)

        if rec_state is None:
            rec_state = mx.zeros((B,self.num_v_heads,self.head_k_dim,self.head_v_dim),dtype=inputs.dtype)
        out,new_state = recurrent_gated_delta_rule(q,k,v,g,beta,rec_state,cache is not None,True)

        if cache is not None: cache[1]=new_state
        else: new_state=None

        out = self.norm(out.reshape(-1,out.shape[-1]),z.reshape(-1,z.shape[-1])).reshape(z.shape[0],z.shape[1],-1)
        return self.out_proj(out)


class Qwen3NextSparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        intermediate_size = args.moe_intermediate_size
        shared_expert_intermediate_size = args.shared_expert_intermediate_size

        self.num_experts = num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok

        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.switch_mlp = SwitchGLU(dim, intermediate_size, num_experts)

        self.shared_expert = Qwen3NextMLP(dim, shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(dim, 1, bias=False)

    def __call__(
        self,
        x: mx.array,
    ):
        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)

        k = self.top_k
        inds = mx.stop_gradient(mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k])
        scores = mx.take_along_axis(gates, inds, axis=-1)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        shared_expert_output = self.shared_expert(x)
        shared_expert_output = (
            mx.sigmoid(self.shared_expert_gate(x)) * shared_expert_output
        )

        return y + shared_expert_output


class Qwen3NextDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_type = args.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3NextGatedDeltaNet(args)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(args)
        
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.args = args
        
        if (layer_idx not in args.mlp_only_layers) and (
            args.num_experts > 0 and (layer_idx + 1) % args.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3NextSparseMoeBlock(args)
        else:
            self.mlp = Qwen3NextMLP(args.hidden_size, args.intermediate_size)
    
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if self.layer_type == "linear_attention":
            r = self.linear_attn(self.input_layernorm(x), mask, cache)
        elif self.layer_type == "full_attention":
            r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


class Qwen3NextModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Qwen3NextDecoderLayer(args=args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ):
        hidden_states = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        cache_counter = 0
        for layer in self.layers:
            if layer.layer_type == "linear_attention":
                c = cache[cache_counter]
                cache_counter += 1
            elif layer.layer_type == "full_attention":
                c = cache[cache_counter]
                cache_counter += 1
            else:
                c = None

            # Compute attention mask per layer as needed
            if layer.layer_type == "full_attention":
                mask_to_use = create_attention_mask(hidden_states, c)
            elif layer.layer_type == "linear_attention":
                mask_to_use = None
            else:
                mask_to_use = None
            hidden_states = layer(hidden_states, mask=mask_to_use, cache=c)

        return self.norm(hidden_states)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen3NextModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ):
        out = self.model(inputs, cache)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for l in self.layers:
            if l.layer_type == "linear_attention":
                caches.append(MambaCache())
            elif l.layer_type == "full_attention":
                caches.append(KVCache())
        return caches
    
    def sanitize(self, weights):
        if "model.layers.0.mlp.experts.0.up_proj.weight" not in weights:
            return weights
        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"
            for n in ["up_proj", "down_proj", "gate_proj"]:
                if f"{prefix}.mlp.experts.0.{n}.weight" in weights:
                    to_join = [
                        weights.pop(f"{prefix}.mlp.experts.{e}.{n}.weight")
                        for e in range(self.args.num_experts)
                    ]
                    weights[f"{prefix}.mlp.switch_mlp.{n}.weight"] = mx.stack(to_join)
        for k, v in weights.items():
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)
        return weights