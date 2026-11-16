# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Dict, Optional

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    max_position_embeddings: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    num_attention_heads: int
    n_group: int
    head_dim: int
    topk_group: int
    n_shared_experts: int
    n_routed_experts: int
    routed_scaling_factor: float
    num_experts_per_tok: int
    first_k_dense_replace: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: Optional[Dict]
    use_qk_norm: bool
    tie_word_embeddings: bool
    attention_bias: bool
    partial_rotary_factor: float
    scoring_func: str = "sigmoid"
    topk_method: str = "noaux_tc"
