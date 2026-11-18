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
    head_dim: int
    n_shared_experts: int
    n_routed_experts: int
    routed_scaling_factor: float
    num_experts_per_tok: int
    first_k_dense_replace: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    partial_rotary_factor: float
    # Solar Open's configs leave these out.
    rope_scaling: Optional[Dict] = None
    use_qk_norm: bool = False
    attention_bias: bool = False
    n_group: int = 1
    topk_group: int = 1
    scoring_func: str = "sigmoid"
    topk_method: str = "noaux_tc"
