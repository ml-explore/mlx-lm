# Copyright © 2023 Apple Inc.

from dataclasses import dataclass
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
    head_dim: Optional[int]
    num_hidden_layers: int
    rms_norm_eps: float
    vocab_size: int
    rope_theta: float
    use_bias: bool
    tie_word_embeddings: bool
