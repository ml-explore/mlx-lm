# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Optional, List, Any

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, scaled_dot_product_attention, create_attention_mask


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    intermediate_size: float
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    attn_layer_offset: int
    attn_layer_period: int
    expert_layer_offset: int
    expert_layer_period: int
    mamba_conv_bias: bool
    mamba_d_conv: int
    mamba_d_state: int
    mamba_dt_rank: int
    mamba_expand: int
    mamba_proj_bias: bool
    num_experts: int
    num_experts_per_tok: int
    rms_norm_eps: float
    max_position_embeddings: int
    vocab_size: int
    tie_word_embeddings: bool = True