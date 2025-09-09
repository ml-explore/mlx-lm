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
    query: mx.array, key: mx.array, value: mx.array, g: mx.array, beta: mx.array,
    initial_state: Optional[mx.array] = None, output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False
) -> Tuple[mx.array, Optional[mx.array]]:
    
    initial_dtype = query.dtype
    
    if use_qk_l2norm_in_kernel:
        query = query / mx.linalg.norm(query, axis=-1, keepdims=True)
        key = key / mx.linalg.norm(key, axis=-1, keepdims=True)
    
    # Transpose to match PyTorch: (B, H, T, D)
    query, key, value, beta, g = [mx.transpose(x, (0, 2, 1, 3)).astype(mx.float32) 
                                  for x in (query, key, value, beta, g)]
    
    B, H, T, Dk = key.shape
    Dv = value.shape[-1]
    scale = 1.0 / (query.shape[-1] ** 0.5)
    query = query * scale
    
    if initial_state is None:
        state = mx.zeros((B, H, Dk, Dv), dtype=mx.float32)
    else:
        state = initial_state.astype(mx.float32)
        if len(state.shape) == 4 and state.shape[1] == T:
            state = state[:, -1, :, :].reshape(B, H, Dk, Dv)
        else:
            state = state.reshape(B, H, Dk, Dv)
    
    outputs = []
    for t in range(T):
        g_t = mx.exp(g[:, :, t, :])  # exp of g, not -exp
        g_t = mx.expand_dims(g_t, -1)  # (B, H, Dv, 1)
        
        state = state * g_t
        mem = mx.einsum("bhkv,bhk->bhv", state, key[:, :, t])
        delta = (value[:, :, t] - mem) * beta[:, :, t]
        state = state + mx.einsum("bhk,bhv->bhkv", key[:, :, t], delta)
        outputs.append(mx.einsum("bhkv,bhk->bhv", state, query[:, :, t]))
    
    out = mx.transpose(mx.stack(outputs, axis=2), (0, 2, 1, 3))
    return out.astype(initial_dtype), state if output_final_state else None


@mx.compile
def apply_mask_to_padding_states(hidden_states: mx.array, attention_mask: mx.array) -> mx.array:
    if (
        attention_mask is not None
        and attention_mask.shape[0] > 1
        and attention_mask.shape[1] > 1
    ):
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).astype(dtype)

    return hidden_states


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
        mixed_qkvz = mixed_qkvz.reshape(mixed_qkvz.shape[:-1] + (nq, 2*nk + 2*nv*dv//nq))
        mixed_ba = mixed_ba.reshape(mixed_ba.shape[:-1] + (nq, 2*nv//nq))
        
        # Split indices are cumulative positions
        q, k, v, z = mx.split(mixed_qkvz, [nk, 2*nk, 2*nk + nv//nq*dv], axis=-1)
        b, a = mx.split(mixed_ba, [nv//nq], axis=-1)
        
        v = v.reshape(v.shape[0], v.shape[1], -1, dv)
        z = z.reshape(z.shape[0], z.shape[1], -1, dv)
        b = b.reshape(b.shape[0], b.shape[1], nv)
        a = a.reshape(a.shape[0], a.shape[1], nv)
        return q, k, v, z, b, a
    
    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ):
        if mask is not None:
            hidden_states = apply_mask_to_padding_states(inputs, mask)
        else:
            hidden_states = inputs

        batch_size, seq_len, _ = hidden_states.shape

        if cache is not None and cache[1] is not None:
            recurrent_state = cache[1]
        else:
            recurrent_state = mx.zeros(
                (batch_size, self.num_v_heads, self.head_k_dim, self.head_v_dim),
                dtype=hidden_states.dtype,
            )

        projected_states_qkvz = self.in_proj_qkvz(hidden_states)
        projected_states_ba = self.in_proj_ba(hidden_states)
        query, key, value, z, b, a = self.fix_query_key_value_ordering(projected_states_qkvz, projected_states_ba)
        query, key, value = (x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value))

        mixed_qkv = mx.concatenate((query, key, value), axis=-1)

        if cache is not None:
            if cache[0] is None:
                conv_state = mx.zeros(
                    (batch_size, self.conv_kernel_size - 1, self.conv_dim),
                    dtype=hidden_states.dtype,
                )
            else:
                conv_state = cache[0]
            padded_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
            cache[0] = padded_input[:, -(self.conv_kernel_size - 1):, :]
            mixed_qkv = nn.silu(self.conv1d(padded_input))
        else:
            padded_input = mx.pad(mixed_qkv, [(0, 0), (self.conv_kernel_size - 1, 0), (0, 0)])
            conv_output = self.conv1d(padded_input)
            mixed_qkv = nn.silu(conv_output[:, :seq_len, :])

        query, key, value = mx.split(
            mixed_qkv,
            [self.key_dim, self.key_dim + self.key_dim],
            axis=-1,
        )
        query = query.reshape(query.shape[0], query.shape[1], -1, self.head_k_dim)
        key = key.reshape(key.shape[0], key.shape[1], -1, self.head_k_dim)
        value = value.reshape(value.shape[0], value.shape[1], -1, self.head_v_dim)

        beta = mx.sigmoid(b).reshape(batch_size, seq_len, -1, 1)
        g = (
            -mx.exp(self.A_log.astype(mx.float32))
            * nn.softplus(a.astype(mx.float32) + self.dt_bias)
        ).reshape(batch_size, seq_len, -1, 1)
        if self.num_v_heads // self.num_k_heads > 1:
            query = mx.repeat(query, self.num_v_heads // self.num_k_heads, axis=2)
            key = mx.repeat(key, self.num_v_heads // self.num_k_heads, axis=2)

        core_attn_out, new_recurrent_state = recurrent_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=True if cache is not None else False,
            use_qk_l2norm_in_kernel=True,
        )
        # Updated storage of new_recurrent_state
        if cache is not None:
            cache[1] = new_recurrent_state
        else:
            new_recurrent_state = None

        z_shape_og = z.shape

        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1)

        return self.out_proj(core_attn_out)


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
        if isinstance(r, tuple):
            r, _ = r
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

        self.fa_idx = 0
        for b in args.layer_types:
            if b == "linear_attention":
                break
            elif b == "full_attention":
                self.fa_idx += 1

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ):
        hidden_states = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        attn_mask = None
        if mask is None:
            kv_caches = [c for c in cache if isinstance(c, KVCache)]
            if kv_caches:
                attn_mask = create_attention_mask(hidden_states, kv_caches)

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

            if layer.layer_type == "full_attention":
                mask_to_use = attn_mask
            elif layer.layer_type == "linear_attention":
                mask_to_use = mask
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
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ):
        out = self.model(inputs, mask, cache)
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