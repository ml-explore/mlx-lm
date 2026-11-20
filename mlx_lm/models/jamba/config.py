# Copyright © 2025 Apple Inc.

import math
from dataclasses import dataclass
from typing import List, Optional, Union

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    attn_layer_offset: int
    attn_layer_period: int
    expert_layer_offset: int
    expert_layer_period: int
    mamba_d_conv: int
    mamba_d_state: int
    mamba_expand: int
    num_experts: int
    num_experts_per_tok: int
    rms_norm_eps: float
    max_position_embeddings: int
    vocab_size: int
    mamba_dt_rank: Union[str, int] = "auto"
    mamba_proj_bias: bool = False
    mamba_conv_bias: bool = True
    layers_block_type: Optional[List[str]] = None
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.mamba_dt_rank == "auto":
            self.mamba_dt_rank = math.ceil(self.hidden_size / 16)
        if self.layers_block_type is None:
            self.layers_block_type = [
                (
                    "attention"
                    if i % self.attn_layer_period == self.attn_layer_offset
                    else "mamba"
                )
                for i in range(self.num_hidden_layers)
            ]
