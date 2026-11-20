# Copyright © 2023 Apple Inc.

from dataclasses import dataclass

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int = 8192
    num_hidden_layers: int = 40
    intermediate_size: int = 22528
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    rope_theta: float = 8000000.0
    vocab_size: int = 256000
    layer_norm_eps: float = 1e-05
    logit_scale: float = 0.0625
    attention_bias: bool = False
    layer_norm_bias: bool = False
    use_qk_norm: bool = False
