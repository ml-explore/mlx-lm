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
    max_position_embeddings: Optional[int]
    num_key_value_heads: int
    first_k_dense_replace: int
    moe_intermediate_size: int
    n_routed_experts: int
    n_shared_experts: int
    norm_topk_prob: bool
    num_experts_per_tok: int
    rope_theta: float
    routed_scaling_factor: float
    head_dim: Optional[int] = None
    scoring_func: str = ("noaux_tc",)
    n_group: Optional[int] = 1
    topk_group: Optional[int] = 1
    attention_bias: bool = False
    mlp_bias: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = False
