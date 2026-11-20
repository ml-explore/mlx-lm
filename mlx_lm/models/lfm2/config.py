# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import List, Optional

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    norm_eps: float
    conv_bias: bool
    conv_L_cache: int
    block_dim: int
    block_multiple_of: int
    block_ffn_dim_multiplier: float
    block_auto_adjust_ff_dim: bool
    block_ff_dim: Optional[int] = None
    intermediate_size: Optional[int] = None
    rope_theta: float = 1000000.0
    rope_parameters: Optional[dict] = None
    full_attn_idxs: Optional[List[int]] = None
    layer_types: Optional[List[str]] = None

    def __post_init__(self):
        if self.block_ff_dim is None:
            if self.intermediate_size is None:
                raise ValueError(
                    "Config must specify either 'block_ff_dim' or 'intermediate_size'"
                )
            self.block_ff_dim = self.intermediate_size
        if self.rope_parameters is not None and "rope_theta" in self.rope_parameters:
            self.rope_theta = self.rope_parameters["rope_theta"]
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.full_attn_idxs is None:
            self.full_attn_idxs = [
                i
                for i, layer_type in enumerate(self.layer_types)
                if layer_type == "full_attention"
            ]
