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
    num_attention_heads: int
    num_hidden_layers: int
    vocab_size: int
    intermediate_size: int
    intermediate_size_mlp: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    head_dim: int
    tie_word_embeddings: bool
    no_rope_layers: list
    use_qk_norm: bool
