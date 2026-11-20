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
    num_layers: int
    intermediate_size: int
    num_attention_heads: int
    vocab_size: int
    rope_theta: float
    layer_norm_epsilon: float
    num_key_value_heads: int
    head_dim: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    mlp_bias: bool = False
