# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Optional

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "plamo2"
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    hidden_size_per_head: int = 128
    max_position_embeddings: int = 2048
    attention_window_size: int = 2048
    full_attention_idx: Optional[list[int]] = None
    mamba_d_state: int = 64
    mamba_d_conv: int = 4
    mamba_num_heads: int = 64
    mamba_step: int = 2
    mamba_chunk_size: int = 256
    mamba_enabled: bool = True
    intermediate_size: int = 13312
    vocab_size: int = 32000
