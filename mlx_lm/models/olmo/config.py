# Copyright © 2023 Apple Inc.

from dataclasses import dataclass

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    d_model: int
    n_layers: int
    mlp_hidden_size: int
    n_heads: int
    vocab_size: int
    embedding_size: int
    rope_theta: float = 10000
    rope_traditional: bool = False
    mlp_ratio: int = 4
    weight_tying: bool = False

    def __post_init__(self):
        self.mlp_hidden_size = (
            self.mlp_hidden_size
            if self.mlp_hidden_size is not None
            else self.mlp_ratio * self.d_model
        )
