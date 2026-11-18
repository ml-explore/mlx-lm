# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import List

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
    attention_bias: bool
    mlp_only_layers: List[int]
    num_experts: int
    num_experts_per_tok: int
    decoder_sparse_step: int
    n_shared_experts: int
    moe_intermediate_size: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    rope_theta: float
    max_position_embeddings: int
    norm_topk_prob: bool
