# Copyright © 2025 Apple Inc.

from dataclasses import dataclass, field
from typing import List, Optional

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    attention_bias: bool = False
    attention_in_multiplier: float = 1.0
    attention_out_multiplier: float = 0.9375
    embedding_multiplier: float = 5.656854249492381
    head_dim: int = 64
    hidden_size: int = 1024
    initializer_range: float = 0.02
    intermediate_size: int = 2048
    key_multiplier: float = 0.390625
    lm_head_multiplier: float = 0.0390625
    mamba_chunk_size: int = 128
    mamba_conv_bias: bool = True
    mamba_d_conv: int = 4
    mamba_d_head: int = 64
    mamba_d_ssm: int = 1536
    mamba_d_state: int = 128
    mamba_expand: int = 2
    mamba_n_groups: int = 1
    mamba_n_heads: int = 24
    mamba_norm_before_gate: bool = False
    mamba_proj_bias: bool = False
    mamba_rms_norm: bool = False
    mamba_use_mlp: bool = True
    max_position_embeddings: int = 131072
    mlp_bias: bool = False
    mlp_expansion_factor: int = 8
    mlp_multipliers: List[float] = field(
        default_factory=lambda: [0.8838834764831844, 0.5859375]
    )
    model_type: str = "falcon_h1"
    num_attention_heads: int = 8
    num_hidden_layers: int = 36
    num_key_value_heads: int = 2
    projectors_bias: bool = False
    rms_norm_eps: float = 1e-05
    rope_traditional: bool = False
    rope_scaling: Optional[float] = None
    rope_theta: float = 100000000000.0
    ssm_in_multiplier: float = 1.25
    ssm_multipliers: List[float] = field(
        default_factory=lambda: [
            0.3535533905932738,
            0.25,
            0.3535533905932738,
            0.5,
            0.3535533905932738,
        ]
    )
    ssm_out_multiplier: float = 0.23570226039551587
    vocab_size: int = 32784
    tie_word_embeddings: bool = True
