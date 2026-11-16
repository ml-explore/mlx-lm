# Copyright © 2026 Apple Inc.

from typing import Any

import mlx.core as mx
from mlx import nn
from mlx.nn.layers.distributed import sum_gradients

from mlx_lm.kernels import (
    expert_routing,
    norm_add_norm,
    post_expert_norm_combine,
    qk_norm,
    qk_norm_rope,
    rms_norm_sigmoid_gate,
)
from mlx_lm.layers.activations import precise_swiglu
from mlx_lm.layers.attention import scaled_dot_product_attention
from mlx_lm.layers.distributed import gather_last_axis
from mlx_lm.layers.switch import SwitchGLU
from mlx_lm.models.muse_spark.config import TextArgs

__all__ = [
    "Attention",
    "InputScaleRMSNorm",
    "MoE",
    "MoEGate",
    "POOLED_TP_MAX_TOKENS",
    "PreciseLayerNorm",
    "PreciseSwiGLU",
    "TransformerBlock",
    "WeightlessRMSNorm",
    "residual_gate_coefficients",
    "row_parallel_linear",
]


class InputScaleRMSNorm(nn.Module):

    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(self.weight * x, None, self.eps)


def residual_gate_coefficients(
    gate: mx.array, *, temperature: float = 0.3, eps: float = 1e-3
) -> tuple[mx.array, mx.array]:
    """Return the weights of the stream and of the branch for a gate."""
    w = gate.astype(mx.float32) / temperature
    beta = mx.sigmoid(w)
    alpha = mx.sqrt(mx.clip(mx.sigmoid(-w) * (1.0 + beta), eps, None))
    return alpha, beta


POOLED_TP_MAX_TOKENS = 8


# Added to the sum of the top-k router scores.
_ROUTER_SCORE_EPS = 1e-15


def row_parallel_linear(
    linear: nn.Linear, x: mx.array, group: mx.distributed.Group
) -> mx.array:
    assert "bias" not in linear, "Linear layers with a bias are not supported."
    rows = linear.weight.shape[0] // group.size()
    r = group.rank()
    part = x @ linear.weight[r * rows : (r + 1) * rows].T
    return gather_last_axis(part, group)


class Attention(nn.Module):
    def __init__(self, args: TextArgs, *, use_rope: bool):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim

        self.scale = head_dim**-0.5
        # Queries carry a tuned constant scale of their own, on top of the
        # usual 1/sqrt(head_dim) applied inside the attention call
        self.query_scale = args.qk_scale_factor
        self.eps = args.rms_norm_eps

        self.qkvg_proj = nn.Linear(
            dim, 2 * (n_heads + n_kv_heads) * head_dim, bias=False
        )
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.use_rope = use_rope
        if use_rope:
            self.rope = nn.RoPE(head_dim, traditional=True, base=args.rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        B, L, _ = x.shape

        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim
        qkvg = self.qkvg_proj(x)
        norm_args = {
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "eps": self.eps,
            "query_scale": self.query_scale,
        }
        if self.use_rope:
            # Also rotates q and k, at the positions after the cache.
            offset = cache.offset if cache is not None else 0
            queries, keys = qk_norm_rope(
                qkvg, **norm_args, rope_base=self.rope.base, offset=offset
            )
        else:
            queries, keys = qk_norm(qkvg, **norm_args)
        values = qkvg[..., q_dim + kv_dim : q_dim + 2 * kv_dim]
        gate = qkvg[..., q_dim + 2 * kv_dim :]

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        return self.o_proj(rms_norm_sigmoid_gate(output, gate, eps=self.eps))


class PreciseSwiGLU(nn.Module):
    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        return precise_swiglu(x, gate, x)


class MoEGate(nn.Module):
    """The router, (num_experts, dim) for a gemv, and its score bias."""

    def __init__(self, dim: int, num_experts: int):
        super().__init__()
        self.weight = mx.zeros((num_experts, dim), dtype=mx.float32)
        self.e_score_correction_bias = mx.zeros((num_experts,), dtype=mx.float32)


class MoE(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()

        dim = args.hidden_size
        compressed = args.moe_hidden_size
        num_experts = args.num_local_experts

        self.top_k = args.num_experts_per_tok
        self.route_scale = args.routed_scaling_factor
        self.score_eps = _ROUTER_SCORE_EPS
        self.router_multiplier = args.output_multiplier

        self.gate = MoEGate(dim, num_experts)

        # The experts run at a compressed width, between the two projections.
        self.pre_expert_proj = nn.Linear(dim, compressed, bias=False)
        self.pre_expert_norm = nn.RMSNorm(compressed, eps=args.rms_norm_eps)
        self.experts = SwitchGLU(
            compressed,
            args.moe_intermediate_size,
            num_experts,
            activation=PreciseSwiGLU(),
        )
        # The checkpoint stores this norm under the experts.
        self.experts.post_expert_norm = InputScaleRMSNorm(
            compressed, eps=args.post_norm_eps
        )
        self.post_expert_proj = nn.Linear(compressed, dim, bias=False)

        self.sharding_group = None

    def __call__(self, x: mx.array) -> mx.array:
        group = self.sharding_group
        if group is not None:
            x = sum_gradients(group)(x)
        # Quantized expert projections stay whole.
        split = (
            group is not None
            and group.size() > 1
            and x.size // x.shape[-1] <= POOLED_TP_MAX_TOKENS
            and isinstance(self.pre_expert_proj, nn.Linear)
            and isinstance(self.post_expert_proj, nn.Linear)
        )

        logits = x.astype(mx.float32) @ self.gate.weight.T
        inds, weights = expert_routing(
            logits,
            self.gate.e_score_correction_bias,
            top_k=self.top_k,
            multiplier=self.router_multiplier,
            score_eps=self.score_eps,
            route_scale=self.route_scale,
        )

        h = (
            row_parallel_linear(self.pre_expert_proj, x, group)
            if split
            else self.pre_expert_proj(x)
        )
        h = self.experts(self.pre_expert_norm(h), inds)
        if group is not None:
            # Sum the expert outputs before the post-expert norm.
            h = mx.distributed.all_sum(h, group=group)
        norm = self.experts.post_expert_norm
        h = post_expert_norm_combine(h, weights, norm_weight=norm.weight, eps=norm.eps)
        return (
            row_parallel_linear(self.post_expert_proj, h, group)
            if split
            else self.post_expert_proj(h)
        )


class TransformerBlock(nn.Module):
    def __init__(self, args: TextArgs, *, use_rope: bool, use_sliding: bool):
        super().__init__()

        dim = args.hidden_size
        self.use_sliding = use_sliding
        self.post_norm_eps = args.post_norm_eps

        self.input_layernorm = nn.RMSNorm(dim, eps=args.rms_norm_eps)
        self.self_attn = Attention(args, use_rope=use_rope)
        self.pre_feedforward_layernorm = nn.RMSNorm(dim, eps=args.rms_norm_eps)
        self.mlp = MoE(args)
        self.post_feedforward_layernorm = InputScaleRMSNorm(dim, eps=args.post_norm_eps)
        # Sanitize computes these from the fixed gates. The defaults are the
        # values for a zero gate.
        zero_gate = mx.zeros((dim,))
        self.add_attn_alpha, self.add_attn_beta = residual_gate_coefficients(zero_gate)
        self.add_ffn_alpha, self.add_ffn_beta = residual_gate_coefficients(zero_gate)

    def __call__(
        self,
        x: mx.array,
        *,
        mask: mx.array | None = None,
        cache: Any | None = None,
        attn_input: mx.array | None = None,
        next_norm: nn.Module | None = None,
    ) -> tuple[mx.array, mx.array | None]:
        if attn_input is None:
            attn_input = self.input_layernorm(x).astype(mx.bfloat16)
        r = self.self_attn(attn_input, mask, cache)
        h, ffn_input = norm_add_norm(
            r,
            x,
            alpha=self.add_attn_alpha,
            beta=self.add_attn_beta,
            post_weight=None,
            post_eps=self.post_norm_eps,
            next_norm=self.pre_feedforward_layernorm,
        )
        r = self.mlp(ffn_input)
        return norm_add_norm(
            r,
            h,
            alpha=self.add_ffn_alpha,
            beta=self.add_ffn_beta,
            post_weight=self.post_feedforward_layernorm.weight,
            post_eps=self.post_feedforward_layernorm.eps,
            next_norm=next_norm,
        )


class WeightlessRMSNorm(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, None, self.eps)


class PreciseLayerNorm(nn.LayerNorm):
    """As PyTorch: LayerNorm in fp32, rounded once."""

    def __call__(self, x: mx.array) -> mx.array:
        f = mx.float32
        w, b = self.weight.astype(f), self.bias.astype(f)
        return mx.fast.layer_norm(x.astype(f), w, b, self.eps).astype(x.dtype)
