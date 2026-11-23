# Copyright © 2024 Apple Inc.

from dataclasses import dataclass
from typing import Dict, Optional, Union

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    hidden_act: str
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    head_dim: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    attention_bias: bool = False
    mlp_bias: bool = False
    partial_rotary_factor: float = 0.5
    rope_theta: float = 10000.0
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.rope_scaling:
            if not "factor" in self.rope_scaling:
                raise ValueError("rope_scaling must contain 'factor'")
            rope_type = self.rope_scaling.get("type") or self.rope_scaling.get(
                "rope_type"
            )
            if rope_type is None:
                raise ValueError(
                    "rope_scaling must contain either 'type' or 'rope_type'"
                )
            if rope_type not in ["linear"]:
                raise ValueError("rope_scaling 'type' currently only supports 'linear'")
