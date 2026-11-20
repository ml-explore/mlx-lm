# Copyright © 2023 Apple Inc.

from dataclasses import dataclass

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    intermediate_size: int
    rope_theta: float
    use_qkv_bias: bool
    partial_rotary_factor: float
    layer_norm_eps: float
    use_parallel_residual: bool = False
    qk_layernorm: bool = False
