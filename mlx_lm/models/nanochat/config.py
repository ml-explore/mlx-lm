# Copyright © 2025 Apple Inc.

from dataclasses import dataclass

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "nanochat"
    hidden_size: int = 1280
    num_hidden_layers: int = 20
    num_attention_heads: int = 10
    num_key_value_heads: int = 10
    vocab_size: int = 65536
    max_position_embeddings: int = 2048
    intermediate_size: int = 5120  # 4 * hidden_size
    rope_theta: float = 10000.0
