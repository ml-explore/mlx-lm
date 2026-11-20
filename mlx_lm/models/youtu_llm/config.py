# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Dict, Optional

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "youtu_llm"
    vocab_size: int = 128256
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    qk_nope_head_dim: int = 128
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1600000
    rope_traditional: bool = True
    rope_scaling: Optional[Dict] = None
    attention_bias: bool = False
    mlp_bias: bool = False
    tie_word_embeddings: bool = True
