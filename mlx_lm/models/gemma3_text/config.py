# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Dict

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int = 1152
    num_hidden_layers: int = 26
    intermediate_size: int = 6912
    num_attention_heads: int = 4
    head_dim: int = 256
    rms_norm_eps: float = 1.0e-6
    vocab_size: int = 262144
    num_key_value_heads: int = 1
    rope_theta: float = 1_000_000.0
    rope_local_base_freq: float = 10_000.0
    query_pre_attn_scalar: float = 256
    sliding_window: int = 512
    sliding_window_pattern: int = 6
    max_position_embeddings: int = 32768
    rope_scaling: Dict = None
