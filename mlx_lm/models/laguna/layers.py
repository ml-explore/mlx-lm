# Copyright © 2026 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import scaled_dot_product_attention
from mlx_lm.layers.common import swiglu
from mlx_lm.layers.moe import SwitchGLU
from mlx_lm.layers.rope import initialize_rope
from mlx_lm.models.laguna.config import ModelArgs

__all__ = [
    "Attention",
    "MLP",
    "MoE",
    "MoEGate",
    "TransformerBlock",
]


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


def _layer_rope(args: ModelArgs, layer_type: str):
    """Build the RoPE module for a given layer type.

    ``rope_parameters`` is a nested ``{layer_type: rope_dict}`` mapping (the
    v5 Laguna-XS schema). Each sub-dict carries its own ``partial_rotary_factor``
    so that full-attention layers (YaRN, partial rotary) and sliding-attention
    layers (default RoPE, full rotary) can differ.
    """
    rope_params = (args.rope_parameters or {}).get(layer_type) or {}
    # ``rope_theta`` is the base; the rest configure the scaling.
    base = rope_params.get("rope_theta", 10000.0)
    partial = rope_params.get(
        "partial_rotary_factor",
        args.partial_rotary_factor if args.partial_rotary_factor is not None else 1.0,
    )
    dims = int(args.head_dim * partial)
    scaling_config = {k: v for k, v in rope_params.items() if k != "rope_theta"}
    if "rope_type" not in scaling_config and "type" not in scaling_config:
        scaling_config["rope_type"] = "default"
    return initialize_rope(
        dims,
        base=base,
        traditional=False,
        scaling_config=scaling_config,
        max_position_embeddings=args.max_position_embeddings,
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads_per_layer[layer_idx]
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim
        self.gating = args.gating

        layer_type = args.layer_types[layer_idx]
        self.use_sliding = layer_type == "sliding_attention"

        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=args.qkv_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.qkv_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.qkv_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=args.attention_bias)

        # QK normalization (like Qwen3), applied per head over ``head_dim``.
        self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)

        # Optional per-head attention output gating (softplus).
        if self.gating:
            self.g_proj = nn.Linear(dim, n_heads, bias=False)

        self.rope = _layer_rope(args, layer_type)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1)
        keys = keys.reshape(B, L, self.n_kv_heads, -1)
        values = values.reshape(B, L, self.n_kv_heads, -1)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys).transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

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

        if self.gating:
            # gate: [B, L, n_heads]; broadcast across head_dim.
            gate = nn.softplus(self.g_proj(x).astype(mx.float32)).astype(output.dtype)
            output = output.reshape(B, L, self.n_heads, self.head_dim)
            output = (output * gate[..., None]).reshape(B, L, -1)

        return self.o_proj(output)


class MoEGate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.num_experts = args.num_experts
        self.softcap = args.moe_router_logit_softcapping
        self.score_func = args.moe_router_score_func
        # A Linear, not a bare matrix, so a quantized router can load.
        self.proj = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.e_score_correction_bias = mx.zeros((args.num_experts,))

    def __call__(self, x):
        logits = self.proj(x).astype(mx.float32)
        if self.softcap > 0.0:
            logits = mx.tanh(logits / self.softcap) * self.softcap

        scores = self._routing_scores(logits)
        scores_for_selection = scores + self.e_score_correction_bias
        inds = mx.argpartition(-scores_for_selection, kth=self.top_k - 1, axis=-1)[
            ..., : self.top_k
        ]
        inds = mx.stop_gradient(inds)
        weights = mx.take_along_axis(scores, inds, axis=-1)
        weights = weights / weights.sum(axis=-1, keepdims=True)
        return inds, weights

    def _routing_scores(self, logits: mx.array) -> mx.array:
        if self.score_func == "sigmoid":
            return mx.sigmoid(logits)
        if self.score_func == "sqrtsoftplus":
            return mx.sqrt(nn.softplus(logits))
        raise ValueError(f"Unknown moe_router_score_func: {self.score_func!r}")


class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.routed_scaling_factor = args.moe_routed_scaling_factor
        self.gate = MoEGate(args)
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.num_experts,
        )
        self.shared_expert = MLP(args.hidden_size, args.shared_expert_intermediate_size)

    def __call__(self, x):
        shared_out = self.shared_expert(x)

        inds, weights = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = (y * weights[..., None]).sum(axis=-2).astype(x.dtype)

        return y * self.routed_scaling_factor + shared_out


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args, layer_idx)
        self.use_sliding = self.self_attn.use_sliding
        if args.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = MoE(args)
        else:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)
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
