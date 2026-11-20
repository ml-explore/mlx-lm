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
    dim_model_base: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    q_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    kv_lora_rank: int
    scale_depth: float
    scale_emb: float
    max_position_embeddings: int
    attention_bias: bool = False
    rope_theta: float = 1000000.0
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[str, float]]] = None
    tie_word_embeddings: bool = False
