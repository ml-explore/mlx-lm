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
    hidden_dim: int
    num_layers: int
    num_kv_reuse_layers: int
    num_heads: int
    num_kv_heads: int
    hidden_dim_scale_factor: float = 3.25
    rope_theta: float = 50000
    rms_norm_eps: float = 1e-5
