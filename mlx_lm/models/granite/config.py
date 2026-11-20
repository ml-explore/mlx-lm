# Copyright © 2023 Apple Inc.

from dataclasses import dataclass
from typing import Dict, Optional, Union

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    logits_scaling: float
    attention_multiplier: float
    embedding_multiplier: float
    residual_multiplier: float
    max_position_embeddings: int
    num_key_value_heads: int
    attention_bias: bool
    mlp_bias: bool
    rope_theta: float
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = True
