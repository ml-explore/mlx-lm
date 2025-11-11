# Copyright © 2025 Apple Inc.

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import ArraysCache, KVCache
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
    rope_traditional: bool = False
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


def recurrent_gla(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    scale: float,
    h: Optional[mx.array] = None,
) -> mx.array:
    """
    Recurrence per (b, h):
        h_t = h_{t-1} * exp(g_t)
        h_t = h_t + k_t^T @ v_t
        y_t = (q_t @ h_t) * scale
    Returns y with shape [B, H, T, Dv].
    """
    B, Hq, L, K = q.shape
    Hv = k.shape[1]
    V = v.shape[-1]

    outputs = []
    exp_g = mx.exp(g)[:, None, None].astype(q.dtype)
    q = q * scale
    for t in range(L):
        q_t = q[:, :, t : t + 1]
        k_t = k[:, :, t : t + 1]
        v_t = v[:, :, t : t + 1]
        h_up = k_t.transpose(0, 1, 3, 2) @ v_t
        if h is not None:
            h = h * exp_g + h_up
        else:
            h = h_up
        o_t = q_t @ h
        outputs.append(o_t)

    return mx.concatenate(outputs, axis=2), h


class GroupRMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5, groups: int = 1):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.groups = groups
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        shape = x.shape
        x = mx.unflatten(x, axis=-1, shape=(self.groups, -1))
        x = mx.fast.rms_norm(x, weight=None, eps=self.eps)
        return self.weight * mx.flatten(x, -2)


class MLP(nn.Module):
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


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.use_qk_norm = args.use_qk_norm
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.head_dim or args.hidden_size // self.num_attention_heads
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
            traditional=args.rope_traditional,
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


class LinearAttention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_qk_norm = args.use_qk_norm
        self.num_hidden_layers = args.num_hidden_layers
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_attention_heads
        self.head_dim = args.hidden_size // self.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        assert self.num_key_value_groups == 1, "Grouped linear not yet supported."

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

        self.g_proj = nn.Linear(
            args.hidden_size, args.num_attention_heads * self.head_dim, bias=False
        )
        self.g_norm = GroupRMSNorm(
            args.num_attention_heads * self.head_dim,
            eps=args.rms_norm_eps,
            groups=args.group_norm_size,
        )

        if args.use_qk_norm:
            self.key_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
            self.query_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        self.rope = initialize_rope(
            int(self.head_dim * args.partial_rotary_factor),
            args.rope_theta,
            traditional=args.rope_traditional,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )
        self._slope = self._get_slopes()

    def _get_slopes(self) -> mx.array:
        n = self.num_attention_heads

        def power_of_2_slopes(n):
            return [2 ** (-(2 ** -(math.log2(n) - 3)) * (i + 1)) for i in range(n)]

        if math.log2(n).is_integer():
            slopes = power_of_2_slopes(n)
        else:
            p = 2 ** math.floor(math.log2(n))
            slopes = power_of_2_slopes(p) + power_of_2_slopes(2 * p)[::2][: n - p]

        slopes = mx.array(slopes, dtype=mx.float32)
        denom = max(1, self.num_hidden_layers - 1)
        layer_pos = max(0, self.layer_idx - 1)
        layer_factor = 1 - (layer_pos / denom) + 1e-5
        return -slopes * layer_factor

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        offset: int = 0,
    ) -> mx.array:
        B, L, D = x.shape

        qkv = self.query_key_value(x)
        qkv_mix = qkv.reshape(
            B,
            L,
            (self.num_attention_heads + 2 * self.num_key_value_heads),
            self.head_dim,
        )
        q, k, v = mx.split(
            qkv_mix,
            [
                self.num_attention_heads,
                self.num_attention_heads + self.num_key_value_heads,
            ],
            axis=2,
        )

        queries = q.transpose(0, 2, 1, 3)
        keys = k.transpose(0, 2, 1, 3)
        values = v.transpose(0, 2, 1, 3)

        if self.use_qk_norm:
            queries = self.query_layernorm(queries)
            keys = self.key_layernorm(keys)

        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)

        if cache is None:
            cache = [None]
        output, cache[0] = recurrent_gla(
            q=queries,
            k=keys,
            v=values,
            g=self._slope,
            scale=self.scale,
            h=cache[0],
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        output = self.g_norm(output) * mx.sigmoid(self.g_proj(x))
        return self.dense(output)


def group_expert_select(
    gates: mx.array,
    e_score_correction_bias: mx.array,
    top_k: int,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
    score_function: str,
) -> Tuple[mx.array, mx.array]:
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


class Gate(nn.Module):
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


class SparseMoeBlock(nn.Module):
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
        self.gate = Gate(args)
        shared_dim = (
            args.moe_shared_expert_intermediate_size or args.moe_intermediate_size
        )
        self.shared_experts = (
            MLP(
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


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.is_global = (
            (layer_idx + 1) % args.layer_group_size == 0
            or layer_idx
            >= args.num_hidden_layers // args.layer_group_size * args.layer_group_size
        )

        if self.is_global:
            self.attention = Attention(args)
        else:
            self.attention = LinearAttention(args, layer_idx=layer_idx)

        self.mlp = (
            SparseMoeBlock(args)
            if (
                args.num_experts is not None and layer_idx >= args.first_k_dense_replace
            )
            else MLP(args)
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
        offset: int = 0,
    ) -> mx.array:
        if self.is_global:
            r = self.attention(self.input_layernorm(x), mask, cache)
        else:
            r = self.attention(self.input_layernorm(x), mask, cache, offset=offset)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class LanguageModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.gla_idx = 0
        self.attn_idx = args.layer_group_size - 1

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.word_embeddings(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        offset = 0
        attn_mask = create_attention_mask(h, cache[self.attn_idx])
        gla_mask = create_ssm_mask(h, cache[self.gla_idx])
        if cache[self.attn_idx] is not None:
            offset = cache[self.attn_idx].offset

        for layer, c in zip(self.layers, cache):
            mask = attn_mask if layer.is_global else gla_mask
            h = layer(h, mask, c, offset=offset)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.norm_head = args.norm_head
        self.model_type = args.model_type
        self.model = LanguageModel(args)
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

    def make_cache(self):
        caches = []
        for l in self.layers:
            if l.is_global:
                caches.append(KVCache())
            else:
                caches.append(ArraysCache(size=1))
        return caches
