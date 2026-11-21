# Copyright © 2023 Apple Inc.

from dataclasses import dataclass

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "phi"
    max_position_embeddings: int = 2048
    vocab_size: int = 51200
    hidden_size: int = 2560
    num_attention_heads: int = 32
    num_hidden_layers: int = 32
    num_key_value_heads: int = 32
    partial_rotary_factor: float = 0.4
    intermediate_size: int = 10240
    layer_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
