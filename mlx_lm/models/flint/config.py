# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    mlp_only_layers: List[int]
    layer_types: List[str]
    sliding_window: int
    rms_norm_eps: float
    vocab_size: int
    rope_theta: float = 10000.0
    rope_parameters: Optional[Dict[str, Any]] = None
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.rope_parameters and self.rope_parameters.get("rope_theta") is not None:
            self.rope_theta = self.rope_parameters["rope_theta"]
