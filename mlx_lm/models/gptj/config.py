# Copyright © 2026 Apple Inc.

from dataclasses import dataclass

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "gptj"
    n_embd: int = 4096
    vocab_size: int = 50400
    layer_norm_epsilon: float = 1e-5
    n_positions: int = 2048
    n_head: int = 16
    rotary_dim: int = 64
    n_layer: int = 28
