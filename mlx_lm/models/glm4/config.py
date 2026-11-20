# Copyright © 2025 Apple Inc.

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
    attention_bias: bool
    head_dim: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    partial_rotary_factor: float
    rope_theta: float
    rope_traditional: bool = True
    max_position_embeddings: int = 32768
