# Copyright © 2024 Apple Inc.

from dataclasses import dataclass
from typing import List, Optional

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    num_experts_per_tok: int
    hybrid_layer_pattern: List[int]
    moe_layer_freq: List[int]
    add_swa_attention_sink_bias: bool
    add_full_attention_sink_bias: bool
    sliding_window_size: int
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    n_shared_experts: Optional[int]
    n_routed_experts: Optional[int]
    routed_scaling_factor: Optional[float]
    topk_method: str
    scoring_func: str
    norm_topk_prob: bool
    n_group: int
    topk_group: int
    max_position_embeddings: int
    layernorm_epsilon: float
    rope_theta: float
    swa_rope_theta: float
    swa_num_attention_heads: int
    swa_num_key_value_heads: int
    head_dim: int
    v_head_dim: int
    swa_head_dim: int
    swa_v_head_dim: int
    partial_rotary_factor: int
    attention_value_scale: Optional[float] = None
