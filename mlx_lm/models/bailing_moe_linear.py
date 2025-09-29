# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union
import math

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, create_ssm_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    intermediate_size: int
    max_position_embeddings: int
    moe_intermediate_size: int
    num_experts: int
    num_shared_experts: int
    norm_topk_prob: bool
    num_attention_heads: int
    num_experts_per_tok: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    vocab_size: int
    first_k_dense_replace: int
    layer_group_size: int
    group_norm_size: int
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    use_bias: bool = False
    use_qkv_bias: bool = False
    norm_head: bool = False
    norm_softmax: bool = False
    use_qk_norm: bool = False
    tie_word_embeddings: bool = False
    partial_rotary_factor: float = 1.0
    moe_router_enable_expert_bias: bool = False
    moe_router_enable_routed_scaling: bool = True
    routed_scaling_factor: float = 1.0
    score_function: str = "softmax"
    n_group: int = 1
    topk_group: int = 4
    use_rmsnorm: bool = True
    moe_shared_expert_intermediate_size: Optional[int] = None
    moe_router_enable_shared_expert: bool = True
    head_dim: Optional[int] = None


def chunk_simple_gla(q: mx.array, k: mx.array, v: mx.array, g: mx.array) -> mx.array:
    B, H, L, K = q.shape
    V = v.shape[-1]
    chunk_size = 64
    h = mx.zeros((B, H, K, V), dtype=q.dtype)
    g = g.transpose(0, 2, 1)
    outputs = []
    num_chunks = (L + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, L)
        chunk_len = end_idx - start_idx
        q_chunk = q[:, :, start_idx:end_idx]
        k_chunk = k[:, :, start_idx:end_idx]
        v_chunk = v[:, :, start_idx:end_idx]
        g_chunk = g[:, :, start_idx:end_idx]
        last_idx = end_idx - 1
        g_last = g[:, :, last_idx:last_idx+1]
        h = h * mx.exp(g_last[:, :, :, None])
        
        chunk_output = []
        for t in range(chunk_len):
            q_t = q_chunk[:, :, t:t+1]
            o_h = mx.matmul(q_t, h)
            o_local = mx.zeros((B, H, 1, V), dtype=q.dtype)
            for s in range(t + 1):
                k_s = k_chunk[:, :, s:s+1]
                v_s = v_chunk[:, :, s:s+1]
                g_decay = mx.exp(g_chunk[:, :, t:t+1] - g_chunk[:, :, s:s+1])
                score = mx.matmul(q_t, k_s.transpose(0, 1, 3, 2))
                o_local = o_local + score * v_s * g_decay[:, :, :, None]
            
            chunk_output.append(o_h + o_local)
            k_t = k_chunk[:, :, t:t+1]
            v_t = v_chunk[:, :, t:t+1]
            g_decay = mx.exp(g_last - g_chunk[:, :, t:t+1])
            h = h + mx.matmul(k_t.transpose(0, 1, 3, 2), v_t * g_decay[:, :, :, None])
        
        outputs.append(mx.concatenate(chunk_output, axis=2))
    
    return mx.concatenate(outputs, axis=2)


def fused_recurrent_simple_gla(q: mx.array, k: mx.array, v: mx.array, g: mx.array) -> mx.array:  
    B, Hq, L, K = q.shape  
    Hv = k.shape[1]  
    V = v.shape[-1]

    if Hq != Hv:
        assert Hq % Hv == 0, f"Query heads {Hq} must be divisible by key/value heads {Hv}"
        repeat_factor = Hq // Hv
        k = mx.repeat(k, repeat_factor, axis=1)
        v = mx.repeat(v, repeat_factor, axis=1)
        Hv = Hq
    h = mx.zeros((B, Hv, K, V), dtype=q.dtype)  
    g = g.transpose(0, 2, 1)  
      
    outputs = []  
    for t in range(L):  
        q_t = q[:, :, t:t+1]
        k_t = k[:, :, t:t+1]
        v_t = v[:, :, t:t+1]
        g_t = g[:, :, t:t+1]
        o_t = mx.matmul(q_t, h)
        outputs.append(o_t)  
        g_decay = mx.exp(g_t.mean(axis=1, keepdims=True))
        h = h * g_decay[:, :, :, None]
        h = h + mx.matmul(k_t.transpose(0, 1, 3, 2), v_t)  
      
    return mx.concatenate(outputs, axis=2)  


class BailingMoeLinearV2GroupRMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5, group: int = 1):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.group = group
        self.eps = eps
    
    def __call__(self, x: mx.array) -> mx.array:
        shape = x.shape
        x = x.reshape(shape[:-1] + (self.group, -1))
        x = mx.fast.rms_norm(x, weight=None, eps=self.eps)
        return self.weight * x.reshape(shape)


class BailingMoeLinearMLP(nn.Module):
    def __init__(self, args: ModelArgs, intermediate_size: Optional[int] = None):
        super().__init__()
        self.intermediate_size = (
            intermediate_size
            if intermediate_size is not None
            else args.intermediate_size
        )

        self.gate_proj = nn.Linear(
            args.hidden_size, self.intermediate_size, bias=args.use_bias
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, args.hidden_size, bias=args.use_bias
        )
        self.up_proj = nn.Linear(
            args.hidden_size, self.intermediate_size, bias=args.use_bias
        )

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class BailingMoeLinearAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.use_qk_norm = args.use_qk_norm
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.hidden_size // self.num_attention_heads
        self.scale = self.head_dim**-0.5

        self.query_key_value = nn.Linear(
            args.hidden_size,
            (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=args.use_qkv_bias,
        )
        self.dense = nn.Linear(
            self.num_attention_heads * self.head_dim,
            args.hidden_size,
            bias=args.use_bias,
        )

        if args.use_qk_norm:
            self.key_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
            self.query_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        self.rope = initialize_rope(
            int(self.head_dim * args.partial_rotary_factor),
            args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        qkv = self.query_key_value(x)

        q_size = self.num_attention_heads * self.head_dim
        kv_size = self.num_key_value_heads * self.head_dim
        q, k, v = mx.split(qkv, [q_size, q_size + kv_size], axis=-1)

        queries = q.reshape(B, L, self.num_attention_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        keys = k.reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        values = v.reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        if self.use_qk_norm:
            queries = self.query_layernorm(queries)
            keys = self.key_layernorm(keys)

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
        return self.dense(output)


class BailingMoeLinearLinearAttention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_qk_norm = args.use_qk_norm
        self.num_hidden_layers = args.num_hidden_layers
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.hidden_size // self.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.mode = 'chunk'

        self.query_key_value = nn.Linear(
            args.hidden_size,
            (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim * 2,
            bias=args.use_qkv_bias,
        )

        self.dense = nn.Linear(
            self.num_attention_heads * self.head_dim,
            args.hidden_size,
            bias=args.use_bias,
        )

        self.g_proj = self.g_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.g_norm = BailingMoeLinearV2GroupRMSNorm(
            args.num_attention_heads * self.head_dim, 
            eps=args.rms_norm_eps, 
            group=args.group_norm_size
        )
        if args.use_qk_norm:
            self.key_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
            self.query_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        
        self.rope = initialize_rope(
            int(self.head_dim * args.partial_rotary_factor),
            args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )
    
    @property
    def slope(self):
        base_slope = - self.make_slope(self.num_attention_heads)
        layer_factor = (1 - (self.layer_idx - 1) / (self.num_hidden_layers - 1) + 1e-5)
        return base_slope * layer_factor
      
    def make_slope(self, n: int) -> mx.array:
        def slopes(n):
            if math.log2(n).is_integer():
                s = 2 ** (-(2 ** -(math.log2(n) - 3)))
                return [s * s ** i for i in range(n)]
            p = 2 ** math.floor(math.log2(n))
            return slopes(p) + slopes(2 * p)[0::2][:n - p]
        return mx.array(slopes(n))

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        mode = "fused_recurrent" if x.shape[1] <= 64 else self.mode
        qkv = self.query_key_value(x)

        # The QKV output is doubled in size, so we need to handle this correctly
        total_heads = self.num_attention_heads + 2 * self.num_key_value_heads
        qkv = qkv.reshape(B, L, total_heads, self.head_dim * 2)  # Note: head_dim * 2
        
        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        
        q = qkv[:, :, :self.num_attention_heads]  # First num_attention_heads
        k = qkv[:, :, self.num_attention_heads:self.num_attention_heads + self.num_key_value_heads]  # Next num_key_value_heads
        v = qkv[:, :, self.num_attention_heads + self.num_key_value_heads:]  # Remaining num_key_value_heads
        
        # Now we need to split each of Q, K, V to get the actual values
        # Assuming the doubling means we take the first half of each
        q = q[..., :self.head_dim]  # Take first half
        k = k[..., :self.head_dim]  # Take first half  
        v = v[..., :self.head_dim]  # Take first half
        
        queries = q.reshape(B, L, self.num_attention_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        keys = k.reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        values = v.reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        if self.use_qk_norm:
            queries = self.query_layernorm(queries)
            keys = self.key_layernorm(keys)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        slope_expanded = mx.tile(self.slope[None, None, :], (B, L, 1))

        if self.num_key_value_groups > 1:
            keys = mx.repeat(keys, self.num_key_value_groups, axis=1)
            values = mx.repeat(values, self.num_key_value_groups, axis=1)

        if mode == "fused_recurrent":
            output = fused_recurrent_simple_gla(
                q=queries,
                k=keys,
                v=values,
                g=slope_expanded,
            )
        else:
            output = chunk_simple_gla(
                q=queries,
                k=keys,
                v=values,
                g=slope_expanded,
            )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        output = self.g_norm(output) * mx.sigmoid(self.g_proj(x))
        return self.dense(output)


def group_expert_select(
    gates,
    e_score_correction_bias,
    top_k,
    n_group,
    topk_group,
    routed_scaling_factor,
    norm_topk_prob,
    score_function,
):

    in_type = gates.dtype
    if score_function == "sigmoid":
        scores = mx.sigmoid(gates.astype(mx.float32))
    else:
        scores = mx.softmax(gates.astype(mx.float32), axis=-1)
    orig_scores = scores
    if e_score_correction_bias is not None:
        scores = scores + e_score_correction_bias
    if n_group > 1:
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        scores = mx.put_along_axis(
            scores, mx.stop_gradient(group_idx), mx.array(0.0), axis=-2
        )
        scores = mx.flatten(scores, -2, -1)

    k = top_k
    inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)
    if top_k > 1 and norm_topk_prob:
        denominator = scores.sum(axis=-1, keepdims=True)
        scores = scores / denominator
    scores = scores * routed_scaling_factor

    return inds, scores.astype(in_type)


class BailingMoeLinearGate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.norm_topk_prob = args.norm_topk_prob

        self.top_k = args.num_experts_per_tok
        self.n_group = args.n_group
        self.topk_group = args.topk_group
        self.routed_scaling_factor = args.routed_scaling_factor
        self.enable_routed_scaling = args.moe_router_enable_routed_scaling

        self.gate_proj = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.expert_bias = (
            mx.zeros((args.num_experts,))
            if args.moe_router_enable_expert_bias
            else None
        )
        self.score_function = args.score_function

    def __call__(self, x: mx.array) -> mx.array:
        return group_expert_select(
            self.gate_proj(x),
            self.expert_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
            self.score_function,
        )


class BailingMoeLinearSparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.num_experts_per_tok = args.num_experts_per_tok
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.num_experts,
            bias=args.use_bias,
        )
        self.gate = BailingMoeLinearGate(args)
        shared_dim = (
            args.moe_shared_expert_intermediate_size or args.moe_intermediate_size
        )
        self.shared_experts = (
            BailingMoeLinearMLP(
                args=args,
                intermediate_size=shared_dim * args.num_shared_experts,
            )
            if args.num_shared_experts > 0 and args.moe_router_enable_shared_expert
            else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        topk_idx, topk_weight = self.gate(x)
        out = self.switch_mlp(x, topk_idx)
        out = (out * topk_weight[..., None]).sum(axis=-2)
        if self.shared_experts is not None:
            out = out + self.shared_experts(x)
        return out


class BailingMoeLinearDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.attention_layer_type = "attention" if (layer_idx + 1) % args.layer_group_size == 0 or \
            layer_idx >= args.num_hidden_layers // args.layer_group_size * args.layer_group_size else "linear_attention"

        if self.attention_layer_type == "attention":
            self.attention = BailingMoeLinearAttention(args)
        else:
            self.attention = BailingMoeLinearLinearAttention(args, layer_idx=layer_idx)

        self.mlp = (
            BailingMoeLinearSparseMoeBlock(args)
            if (
                args.num_experts is not None and layer_idx >= args.first_k_dense_replace
            )
            else BailingMoeLinearMLP(args)
        )
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.attention(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class BailingMoeLinearModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            BailingMoeLinearDecoderLayer(args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        self.fa_idx = []
        self.la_idx = []

        for i, layer in enumerate(self.layers):
            if layer.attention_layer_type == "attention":
                self.fa_idx.append(i)
            else:
                self.la_idx.append(i)
                
        self.fa_idx = self.fa_idx[0] if self.fa_idx else 0
        self.la_idx = self.la_idx[0] if self.la_idx else 0

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.word_embeddings(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        fa_mask = create_attention_mask(h, cache[self.fa_idx])
        # ssm_mask = create_ssm_mask(h, cache[self.la_idx])

        for layer, c in zip(self.layers, cache):
            mask = fa_mask if layer.attention_layer_type == "attention" else None
            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.norm_head = args.norm_head
        self.model_type = args.model_type
        self.model = BailingMoeLinearModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            out = self.model.word_embeddings.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        if self.norm_head:
            w = weights["lm_head.weight"]
            dtype = w.dtype
            weight_norm = (
                mx.linalg.norm(w.astype(mx.float32), axis=0, keepdims=True) + 1e-7
            )
            weights["lm_head.weight"] = (w / weight_norm).astype(dtype)

        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"
            
            # Fix linear attention QKV weights that were incorrectly doubled
            layer = self.model.layers[l]
            if hasattr(layer, 'attention') and hasattr(layer.attention, 'attention_layer_type'):
                if layer.attention_layer_type == "linear_attention":
                    qkv_key = f"{prefix}.attention.query_key_value.weight"
                    if qkv_key in weights:
                        qkv_weight = weights[qkv_key]
                        expected_size = (self.args.num_attention_heads + 2 * self.args.num_key_value_heads) * (self.args.hidden_size // self.args.num_attention_heads)
                        
                        # If the weight is doubled, take only the first half
                        if qkv_weight.shape[0] == expected_size * 2:
                            weights[qkv_key] = qkv_weight[:expected_size, :]
                        
                    # Same for bias if it exists
                    qkv_bias_key = f"{prefix}.attention.query_key_value.bias"
                    if qkv_bias_key in weights:
                        qkv_bias = weights[qkv_bias_key]
                        expected_size = (self.args.num_attention_heads + 2 * self.args.num_key_value_heads) * (self.args.hidden_size // self.args.num_attention_heads)
                        
                        if qkv_bias.shape[0] == expected_size * 2:
                            weights[qkv_bias_key] = qkv_bias[:expected_size]

            # Handle MoE layers
            if l >= self.args.first_k_dense_replace:
                for m in ["gate_proj", "down_proj", "up_proj"]:
                    for k in ["weight", "scales", "biases"]:
                        if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                            to_join = [
                                weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                                for e in range(self.args.num_experts)
                            ]
                            weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(
                                to_join
                            )

                if f"{prefix}.mlp.gate.weight" in weights:
                    gate_weight = weights.pop(f"{prefix}.mlp.gate.weight")
                    weights[f"{prefix}.mlp.gate.gate_proj.weight"] = gate_weight

                if f"{prefix}.mlp.gate.bias" in weights:
                    gate_bias = weights.pop(f"{prefix}.mlp.gate.bias")
                    weights[f"{prefix}.mlp.gate.gate_proj.bias"] = gate_bias

        return weights

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("mlp.gate.gate_proj"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def cast_predicate(self):
        def predicate(k):
            return "expert_bias" not in k

        return predicate

    @property
    def layers(self):
        return self.model.layers