# Copyright © 2026 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import sum_gradients

from mlx_lm.layers.attention import scaled_dot_product_attention
from mlx_lm.layers.common import swiglu
from mlx_lm.layers.moe import SwitchGLU
from mlx_lm.layers.rope import initialize_rope
from mlx_lm.models.exaone_moe.config import ModelArgs

__all__ = [
    "Attention",
    "DecoderLayer",
    "MLP",
    "MoE",
    "MoEGate",
    "group_expert_select",
]


class MLP(nn.Module):
    def __init__(self, args: ModelArgs, intermediate_size: Optional[int] = None):
        super().__init__()
        hidden_size = args.hidden_size
        intermediate_size = intermediate_size or args.intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()

        self.hidden_size = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            self.hidden_size, self.n_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, self.hidden_size, bias=False
        )

        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        self.is_sliding_window = args.layer_types[layer_idx] == "sliding_attention"
        self.apply_rope_all_layers = "sliding_attention" not in args.layer_types
        self.use_rope = self.is_sliding_window or self.apply_rope_all_layers

        if self.use_rope:
            self.rope = initialize_rope(
                self.head_dim,
                base=args.rope_theta,
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

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(
            0, 2, 1, 3
        )
        keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            if self.use_rope:
                queries = self.rope(queries, offset=cache.offset)
                keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        elif self.use_rope:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


@mx.compile
def group_expert_select(
    gates,
    e_score_correction_bias,
    top_k,
    n_group,
    topk_group,
    routed_scaling_factor,
    norm_topk_prob,
):
    scores = mx.sigmoid(gates.astype(mx.float32))
    orig_scores = scores
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
    inds = mx.stop_gradient(inds)
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)

    if top_k > 1 and norm_topk_prob:
        denominator = scores.sum(axis=-1, keepdims=True)
        scores = scores / (denominator + 1e-20)

    scores = scores * routed_scaling_factor
    return inds, scores


class MoEGate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.n_routed_experts = args.num_experts
        self.routed_scaling_factor = args.routed_scaling_factor
        self.n_group = args.n_group
        self.topk_group = args.topk_group
        self.weight = mx.zeros((self.n_routed_experts, args.hidden_size))
        self.e_score_correction_bias = mx.zeros((self.n_routed_experts,))
        assert args.topk_method == "noaux_tc", "Unsupported topk method."

    def __call__(self, x):
        return group_expert_select(
            x @ self.weight.T,
            self.e_score_correction_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )


class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.num_experts,
        )

        self.gate = MoEGate(args)

        self.shared_experts = (
            MLP(
                args,
                intermediate_size=args.moe_intermediate_size * args.num_shared_experts,
            )
            if args.num_shared_experts is not None and args.num_shared_experts > 0
            else None
        )

        self.sharding_group = None

    def __call__(self, x):
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)

        inds, scores = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        if self.shared_experts is not None:
            y = y + self.shared_experts(x)

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)

        return y


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()

        self.self_attn = Attention(args, layer_idx)
        self.mlp = MoE(args) if args.is_moe_layer[layer_idx] else MLP(args)
        self.is_sliding_window = self.self_attn.is_sliding_window

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
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r
