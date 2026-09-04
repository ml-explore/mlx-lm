# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    vocab_size: int
    hidden_size: int
    head_dim: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    rope_theta: float = 10_000
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    layer_norm: str = "pre"  # "pre" or "post"
    init_std: float = 0.02
    rope_scaling_factor: float = 1.0
    original_max_position_embeddings: int = 8192
    # Quadratic attention
    partial_rotary_factor: float = 1.0
    attn_output_gate: bool = False

    quadratic_attn_interval: int = 1
    linear_attn_type: str = "gated_delta"
    mlp_type: str = "mlp"  # "mlp" or "sparse_moe"
    # Gated delta net
    linear_num_key_heads: Optional[int] = None
    linear_num_value_heads: Optional[int] = None
    linear_key_head_dim: Optional[int] = None
    linear_value_head_dim: Optional[int] = None
    linear_conv_kernel_dim: int = 4

    def __post_init__(self):
        if self.quadratic_attn_interval < 1:
            raise ValueError(
                "quadratic_attn_interval counts the layers per full attention "
                f"layer, so it must be at least 1, got {self.quadratic_attn_interval}"
            )
        if not 0 < self.partial_rotary_factor <= 1:
            raise ValueError(
                "partial_rotary_factor is the fraction of each head that RoPE "
                f"rotates, so it must be in (0, 1], got {self.partial_rotary_factor}"
            )
        if self.linear_num_key_heads is None:
            self.linear_num_key_heads = self.num_key_value_heads
        if self.linear_num_value_heads is None:
            self.linear_num_value_heads = self.num_attention_heads
        if self.linear_key_head_dim is None:
            self.linear_key_head_dim = self.head_dim
        if self.linear_value_head_dim is None:
            self.linear_value_head_dim = self.head_dim
