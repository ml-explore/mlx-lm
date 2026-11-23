# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Optional

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
    rms_norm_eps: float
    vocab_size: int
    head_dim: int
    num_key_value_heads: int
    max_position_embeddings: Optional[int] = None
    attention_bias: bool = False
    rope_theta: float = 10000
    tie_word_embeddings: bool = True
