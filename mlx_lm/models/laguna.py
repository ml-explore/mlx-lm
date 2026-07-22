# Copyright © 2025 Apple Inc.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "laguna"
    vocab_size: int = 100352
    hidden_size: int = 3072
    intermediate_size: int = 12288
    num_hidden_layers: int = 48
    num_attention_heads: int = 48
    num_key_value_heads: int = 8
    head_dim: int = 128
    attention_bias: bool = False
    gating: Any = True
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 1048576
    rope_parameters: Dict[str, Any] = field(
        default_factory=lambda: {"rope_type": "default", "rope_theta": 500000.0}
    )
    sliding_window: Optional[int] = None
    layer_types: Optional[List[str]] = None
    num_attention_heads_per_layer: Optional[List[int]] = None
    swa_attention_sink_enabled: bool = False
    num_experts: int = 256
    num_experts_per_tok: int = 10
    moe_intermediate_size: int = 1024
    shared_expert_intermediate_size: int = 1024
    norm_topk_prob: bool = True
    decoder_sparse_step: int = 1
    mlp_only_layers: Optional[List[int]] = None
    moe_routed_scaling_factor: float = 1.0
    moe_apply_router_weight_on_input: bool = False
    moe_router_logit_softcapping: float = 0.0
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.mlp_only_layers is None:
            self.mlp_only_layers = [0]
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        if self.moe_apply_router_weight_on_input:
            # Matches the transformers reference, which also raises for this
            # flag: routing the weight into the expert *input* (rather than
            # scaling the output) needs different math in SwitchGLU than what
            # this port implements. No known Laguna checkpoint sets this.
            raise NotImplementedError(
                "moe_apply_router_weight_on_input=True is not supported."
            )


def _resolve_rope_config(rope_parameters: Dict[str, Any], attention_type: str) -> Dict[str, Any]:
    """Laguna nests RoPE config by layer type when the model mixes full and
    sliding attention: ``{"full_attention": {...}, "sliding_attention": {...}}``.
    A flat (non-nested) config applies to every layer as-is."""
    sub = rope_parameters.get(attention_type)
    if isinstance(sub, dict):
        return sub
    return rope_parameters


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        head_dim = args.head_dim
        self.head_dim = head_dim

        per_layer_heads = args.num_attention_heads_per_layer
        self.n_heads = (
            per_layer_heads[layer_idx]
            if per_layer_heads is not None
            else args.num_attention_heads
        )
        self.n_kv_heads = args.num_key_value_heads
        self.scale = head_dim**-0.5

        attention_type = args.layer_types[layer_idx]
        self.is_sliding = attention_type == "sliding_attention"
        self.sliding_window = args.sliding_window if self.is_sliding else None

        self.q_proj = nn.Linear(
            args.hidden_size, self.n_heads * head_dim, bias=args.attention_bias
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.n_kv_heads * head_dim, bias=args.attention_bias
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.n_kv_heads * head_dim, bias=args.attention_bias
        )
        self.o_proj = nn.Linear(self.n_heads * head_dim, args.hidden_size, bias=False)

        self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)

        gating = args.gating
        self.gating = bool(gating)
        self.gate_per_head = gating == "per-head"
        if self.gating:
            g_out = self.n_heads if self.gate_per_head else self.n_heads * head_dim
            self.g_proj = nn.Linear(args.hidden_size, g_out, bias=False)

        self.sinks = (
            mx.zeros((self.n_heads,))
            if (self.is_sliding and args.swa_attention_sink_enabled)
            else None
        )

        rope_cfg = _resolve_rope_config(args.rope_parameters, attention_type)
        rope_type = rope_cfg.get("rope_type", "default")
        partial = rope_cfg.get("partial_rotary_factor", 1.0)
        self.rope = initialize_rope(
            dims=int(head_dim * partial),
            base=rope_cfg.get("rope_theta", 10000.0),
            traditional=False,
            scaling_config=rope_cfg if rope_type != "default" else None,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        B, L, _ = x.shape

        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)

        # RMSNorm always normalizes the last axis (head_dim), so applying it
        # here (before the head/seq transpose) or after is numerically
        # identical — this order matches the reference's own comments.
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(k).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        out = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=mask, sinks=self.sinks
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.n_heads * self.head_dim)

        if self.gating:
            gate = nn.softplus(self.g_proj(x))
            if self.gate_per_head:
                out = (
                    out.reshape(B, L, self.n_heads, self.head_dim) * gate[..., None]
                ).reshape(B, L, -1)
            else:
                out = out * gate

        return self.o_proj(out)
