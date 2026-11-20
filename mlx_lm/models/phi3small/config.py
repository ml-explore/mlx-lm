# Copyright © 2023 Apple Inc.

from dataclasses import dataclass

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    dense_attention_every_n_layers: int
    ff_intermediate_size: int
    gegelu_limit: float
    num_hidden_layers: int
    num_attention_heads: int
    layer_norm_epsilon: float
    vocab_size: int
    num_key_value_heads: int
    mup_attn_multiplier: float = 1.0
    mup_use_scaling: bool = True
    mup_embedding_multiplier: float = 10.0
    mup_width_multiplier: float = 8.0
    rope_embedding_base: float = 1000000
    rope_position_scale: float = 1.0
    blocksparse_block_size: int = 64
    blocksparse_num_local_blocks: int = 16
    blocksparse_vert_stride: int = 8
