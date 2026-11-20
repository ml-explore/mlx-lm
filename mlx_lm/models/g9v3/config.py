# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from typing import Dict, Optional

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "g9v3"
    vocab_size: int = 130560
    hidden_size: int = 2048
    num_attention_heads: int = 32
    head_dim: int = 128
    num_key_value_heads: int = 2
    use_gated_attention: Optional[bool] = False
    rope_theta: int = 5000000
    rope_scaling: Optional[Dict] = None
    first_k_dense_replace: int = 1
    intermediate_size: int = 8192
    n_routed_experts: int = 320
    num_experts_per_tok: int = 32
    routed_scaling_factor: float = 3.66
    n_group: Optional[int] = 1
    topk_group: Optional[int] = 1
    norm_topk_prob: Optional[bool] = True
    moe_intermediate_size: int = 512
    n_shared_experts: int = 1
    rms_norm_eps: float = 1e-06
    num_hidden_layers: int = 38
    max_position_embeddings: int = 131072
