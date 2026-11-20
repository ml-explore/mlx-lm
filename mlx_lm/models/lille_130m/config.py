# Copyright © 2025 Apple Inc.

from dataclasses import dataclass

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    block_size: int
    layer_norm_eps: float
    n_embd: int
    n_head: int
    n_kv_heads: int
    n_layer: int
    rope_theta: float
    vocab_size: int
    tie_word_embeddings: bool = True
