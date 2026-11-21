# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

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
    max_position_embeddings: int
    sliding_window: int
    rope_theta: float
    attention_bias: bool = False
    layer_types: Optional[List[str]] = None
    num_key_value_heads: Optional[int] = None
    head_dim: Optional[int] = None
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if (i + 1) % 4 == 0 else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]
