# Copyright © 2023 Apple Inc.

from dataclasses import dataclass
from typing import List

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    head_dim: int
    num_transformer_layers: int
    model_dim: int
    vocab_size: int
    ffn_dim_divisor: int
    num_query_heads: List
    num_kv_heads: List
    ffn_multipliers: List
    ffn_with_glu: bool = True
    normalize_qk_projections: bool = True
    share_input_output_layers: bool = True
    rms_norm_eps: float = 1e-6
    rope_freq_constant: float = 10000
