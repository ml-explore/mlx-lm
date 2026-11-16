# Copyright © 2026 Apple Inc.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "bailing_hybrid"
    vocab_size: int = 157184
    hidden_size: int = 1536
    intermediate_size: int = 4608
    moe_intermediate_size: int = 512
    moe_shared_expert_intermediate_size: int = 512
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_experts: int = 128
    num_experts_per_tok: int = 8
    num_shared_experts: int = 1
    first_k_dense_replace: int = 1
    n_group: int = 8
    topk_group: int = 4
    routed_scaling_factor: float = 2.5
    layer_group_size: int = 4
    head_dim: int = 128
    q_lora_rank: int | None = 256
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    short_conv_kernel_size: int = 4
    kda_lower_bound: float = -5.0
    kda_safe_gate: bool = True
    no_kda_lora: bool = True
    rms_norm_eps: float = 1e-6
    rope_theta: float = 6000000.0
    rope_scaling: dict[str, Any] | None = None
    max_position_embeddings: int = 131072
    rope_interleave: bool = True
    tie_word_embeddings: bool = False
