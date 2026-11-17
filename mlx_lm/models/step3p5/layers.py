# Copyright © 2026 Apple Inc.

from functools import partial
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import sum_gradients

from mlx_lm.layers.activations import swiglu
from mlx_lm.layers.attention import scaled_dot_product_attention
from mlx_lm.layers.norms import ZeroCenteredRMSNorm
from mlx_lm.layers.rope import initialize_rope
from mlx_lm.layers.switch import SwiGLU, SwitchGLU
from mlx_lm.models.step3p5.config import ModelArgs

__all__ = [
    "ClampedSwiGLU",
    "Step3p5Attention",
    "Step3p5DecoderLayer",
    "Step3p5MLP",
    "Step3p5MoE",
    "clamped_swiglu",
    "moe_gate_select",
]


@partial(mx.compile, shapeless=True)
def clamped_swiglu(x, gate, limit):
    gate = mx.clip(nn.silu(gate), a_min=None, a_max=limit)
    x = mx.clip(x, a_min=-limit, a_max=limit)
    return gate * x


class ClampedSwiGLU(nn.Module):
    def __init__(self, limit: float):
        super().__init__()
        self.limit = limit

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        return clamped_swiglu(x, gate, self.limit)


class Step3p5MLP(nn.Module):
    def __init__(
        self, args: ModelArgs, intermediate_size: int, swiglu_limit: float = 0
    ):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        self.limit = swiglu_limit if swiglu_limit and swiglu_limit > 0 else None

    def __call__(self, x: mx.array) -> mx.array:
        if self.limit is not None:
            return self.down_proj(
                clamped_swiglu(self.up_proj(x), self.gate_proj(x), self.limit)
            )
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


@mx.compile
def moe_gate_select(gates, router_bias, top_k, routed_scaling_factor, norm_topk_prob):
    scores = mx.sigmoid(gates.astype(mx.float32))
    corrected_scores = scores + router_bias

    topk_indices = mx.argpartition(-corrected_scores, kth=top_k - 1, axis=-1)[
        ..., :top_k
    ]
    topk_indices = mx.stop_gradient(topk_indices)
    topk_weights = mx.take_along_axis(scores, topk_indices, axis=-1)

    if norm_topk_prob:
        topk_weights = topk_weights / (
            mx.sum(topk_weights, axis=-1, keepdims=True) + 1e-20
        )

    return topk_indices, topk_weights * routed_scaling_factor


class Step3p5MoE(SwitchGLU):
    """The routed experts and their router, moe in the checkpoint."""

    def __init__(self, args: ModelArgs, swiglu_limit: float):
        activation = ClampedSwiGLU(swiglu_limit) if swiglu_limit > 0 else SwiGLU()
        super().__init__(
            args.hidden_size,
            args.moe_intermediate_size,
            args.moe_num_experts,
            activation=activation,
        )
        self.top_k = args.moe_top_k
        self.routed_scaling_factor = args.moe_router_scaling_factor
        self.norm_topk_prob = args.norm_expert_weight
        self.gate = nn.Linear(args.hidden_size, args.moe_num_experts, bias=False)
        self.router_bias = mx.zeros((args.moe_num_experts,))

    def __call__(self, x: mx.array) -> mx.array:
        topk_indices, topk_weights = moe_gate_select(
            self.gate(x),
            self.router_bias,
            self.top_k,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )
        routed_output = super().__call__(x, topk_indices)
        return (
            (routed_output * topk_weights[..., None])
            .sum(axis=-2)
            .astype(routed_output.dtype)
        )


class Step3p5Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        dim = args.hidden_size

        layer_types = args.layer_types or []
        if layer_types:
            self.is_sliding = layer_types[layer_idx] == "sliding_attention"
        else:
            self.is_sliding = layer_idx % 2 == 0

        if self.is_sliding and args.attention_other_setting:
            self.num_heads = args.attention_other_setting["num_attention_heads"]
            self.num_kv_heads = args.attention_other_setting["num_attention_groups"]
        else:
            self.num_heads = args.num_attention_heads
            self.num_kv_heads = args.num_attention_groups

        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, dim, bias=False)

        self.q_norm = ZeroCenteredRMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = ZeroCenteredRMSNorm(self.head_dim, eps=args.rms_norm_eps)

        self.use_head_wise_attn_gate = args.use_head_wise_attn_gate
        if self.use_head_wise_attn_gate:
            self.g_proj = nn.Linear(dim, self.num_heads, bias=False)

        rope_theta = args.rope_theta
        if isinstance(rope_theta, list):
            rope_theta = rope_theta[layer_idx]

        partial_rotary_factor = 1.0
        if args.partial_rotary_factors and layer_idx < len(args.partial_rotary_factors):
            partial_rotary_factor = args.partial_rotary_factors[layer_idx]

        rope_dims = int(self.head_dim * partial_rotary_factor)

        yarn_only_types = args.yarn_only_types or []
        layer_type = layer_types[layer_idx] if layer_types else "full_attention"
        if yarn_only_types and layer_type not in yarn_only_types:
            rope_scaling = None
        else:
            rope_scaling = args.rope_scaling

        self.rope = initialize_rope(
            dims=rope_dims,
            base=rope_theta,
            traditional=False,
            scaling_config=rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = self.q_norm(queries.reshape(B, L, self.num_heads, -1)).transpose(
            0, 2, 1, 3
        )
        keys = self.k_norm(keys.reshape(B, L, self.num_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.num_kv_heads, -1).transpose(0, 2, 1, 3)

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
        output = output.transpose(0, 2, 1, 3)

        if self.use_head_wise_attn_gate:
            output = output * mx.sigmoid(self.g_proj(x))[..., None]

        return self.o_proj(output.reshape(B, L, -1))


class Step3p5DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()

        self.self_attn = Step3p5Attention(args, layer_idx)
        self.is_sliding = self.self_attn.is_sliding

        moe_layers_idx = set()
        if args.moe_layers_enum:
            moe_layers_idx = {int(i) for i in args.moe_layers_enum.strip().split(",")}
        else:
            moe_layers_idx = set(range(1, args.num_hidden_layers))

        self.is_moe_layer = layer_idx in moe_layers_idx

        swiglu_limit_shared = 0
        if args.swiglu_limits_shared and layer_idx < len(args.swiglu_limits_shared):
            swiglu_limit_shared = args.swiglu_limits_shared[layer_idx] or 0

        if self.is_moe_layer:
            swiglu_limit = 0
            if args.swiglu_limits and layer_idx < len(args.swiglu_limits):
                swiglu_limit = args.swiglu_limits[layer_idx] or 0
            self.moe = Step3p5MoE(args, swiglu_limit)
            self.share_expert = Step3p5MLP(
                args,
                intermediate_size=args.share_expert_dim,
                swiglu_limit=swiglu_limit_shared,
            )
            self.sharding_group = None
        else:
            self.mlp = Step3p5MLP(
                args,
                intermediate_size=args.intermediate_size,
                swiglu_limit=swiglu_limit_shared,
            )

        self.input_layernorm = ZeroCenteredRMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.post_attention_layernorm = ZeroCenteredRMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        h = x + r
        r = self._feed_forward(self.post_attention_layernorm(h))
        return h + r

    def _feed_forward(self, x: mx.array) -> mx.array:
        if not self.is_moe_layer:
            return self.mlp(x)
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)
        y = self.moe(x) + self.share_expert(x)
        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)
        return y
