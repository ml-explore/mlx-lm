# Copyright © 2023 Apple Inc.

from dataclasses import dataclass, field
from typing import Optional

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    hidden_size: int
    intermediate_size: int
    model_type: str
    max_position_embeddings: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    rms_norm_eps: float
    vocab_size: int
    rope_theta: float
    use_bias: bool
    tie_word_embeddings: bool
    moe_num_experts: int
    moe_layer_start_index: int = 0
    moe_intermediate_size: int = 0
    moe_capacity: list[int] = field(default_factory=list)
    moe_k: int = 1
    moe_layer_interval: int = 1
    moe_use_aux_free: bool = False
    moe_num_shared_experts: int = 0
    moe_layer_end_index: Optional[int] = None
    head_dim: Optional[int] = None
    moe_gate_act: str = "softmax"
