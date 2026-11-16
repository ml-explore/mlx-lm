# Copyright © 2023 Apple Inc.

from dataclasses import dataclass

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int = 2048
    num_attention_heads: int = 16
    num_hidden_layers: int = 24
    kv_channels: int = 128
    max_position_embeddings: int = 8192
    layer_norm_epsilon: float = 1e-6
    intermediate_size: int = 11008
    no_bias: bool = True
    vocab_size: int = 151936
    num_key_value_heads = None

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
