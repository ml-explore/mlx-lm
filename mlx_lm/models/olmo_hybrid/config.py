# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from typing import Dict, List, Optional

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
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    linear_allow_neg_eigval: bool
    tie_word_embeddings: bool
    attention_bias: bool
    layer_norm_epsilon: float = 1e-5
    head_dim: Optional[int] = None
    layer_types: Optional[List[str]] = None
    rope_parameters: Optional[Dict] = None
    rope_theta: Optional[float] = 10000.0

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.rope_parameters is not None:
            self.rope_theta = self.rope_parameters.get("rope_theta", 10000.0)
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if (i % 4 == 3) else "linear_attention"
                for i in range(self.num_hidden_layers)
            ]
