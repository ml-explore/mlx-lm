# Copyright © 2023 Apple Inc.

from dataclasses import dataclass
from typing import Any

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
    "TextArgs",
]


@dataclass
class TextArgs(BaseModelArgs):
    attention_bias: bool
    attention_chunk_size: int
    head_dim: int
    hidden_size: int
    interleave_moe_layer_step: int
    intermediate_size: int
    intermediate_size_mlp: int
    max_position_embeddings: int
    model_type: str
    num_attention_heads: int
    num_experts_per_tok: int
    num_hidden_layers: int
    num_key_value_heads: int
    num_local_experts: int
    rms_norm_eps: float
    rope_scaling: Any
    rope_theta: float
    use_qk_norm: bool
    vocab_size: int
    attn_temperature_tuning: int = 4
    floor_scale: int = 8192
    attn_scale: float = 0.1


@dataclass
class ModelArgs(BaseModelArgs):
    text_config: dict
    model_type: str

    def __post_init__(self):
        self.text = TextArgs.from_dict(self.text_config)
