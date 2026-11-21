# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from typing import Dict, List, Optional

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    vocab_size: int
    num_attention_heads: int
    num_attention_groups: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict] = None
    max_position_embeddings: int = 262144
    sliding_window: int = 512
    layer_types: Optional[List[str]] = None
    yarn_only_types: Optional[List[str]] = None
    partial_rotary_factors: Optional[List[float]] = None
    attention_other_setting: Optional[Dict] = None
    use_head_wise_attn_gate: bool = True
    moe_num_experts: int = 288
    moe_top_k: int = 8
    moe_intermediate_size: int = 1280
    share_expert_dim: int = 1280
    moe_layers_enum: Optional[str] = None
    moe_router_scaling_factor: float = 3.0
    norm_expert_weight: bool = True
    swiglu_limits: Optional[List[float]] = None
    swiglu_limits_shared: Optional[List[float]] = None
    tie_word_embeddings: bool = False
