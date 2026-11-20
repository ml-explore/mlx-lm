# Copyright © 2025 Apple Inc.

from dataclasses import dataclass

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    norm_eps: float
    head_dim: int
    num_hidden_layers: int
    a_low_rank_dim: int
    v_low_rank_dim: int
    gate_low_rank_dim: int
    decay_low_rank_dim: int
    tie_word_embeddings: bool = False
