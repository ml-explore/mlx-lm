# Copyright © 2023 Apple Inc.

from dataclasses import dataclass

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
    norm_epsilon: float = 1e-5
    vocab_size: int = 49152
    rope_theta: float = 100000
    tie_word_embeddings: bool = True
