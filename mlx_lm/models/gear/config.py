# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    layer_types: List[str]
    max_position_embeddings: Optional[int] = None
    hidden_activation: str = "gelu_pytorch_tanh"
    attention_bias: bool = False
    attention_dropout: float = 0.0
    query_pre_attn_scalar: Optional[float] = None
    sliding_window: int = 512
    conv_L_cache: int = 3
    rope_theta: float = 1000000.0
    rope_local_base_freq: float = 10000.0
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = True
    final_logit_softcapping: Optional[float] = None
    attn_logit_softcapping: Optional[float] = None
    use_bidirectional_attention: bool = False

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.query_pre_attn_scalar is None:
            self.query_pre_attn_scalar = self.head_dim
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"Expected {self.num_hidden_layers} layer types, "
                f"got {len(self.layer_types)}."
            )
        if self.use_bidirectional_attention:
            raise ValueError("Gear bidirectional attention is not supported.")
