# Copyright © 2025 Apple Inc.

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
    intermediate_size: int
    max_position_embeddings: int
    moe_intermediate_size: int
    num_experts: int
    num_shared_experts: int
    norm_topk_prob: bool
    num_attention_heads: int
    num_experts_per_tok: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    vocab_size: int
    first_k_dense_replace: int
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    use_bias: bool = False
    use_qkv_bias: bool = False
    norm_head: bool = False
    norm_softmax: bool = False
    use_qk_norm: bool = False
    tie_word_embeddings: bool = False
    partial_rotary_factor: float = 1.0
    rotary_dim: Optional[int] = None
    moe_router_enable_expert_bias: bool = False
    moe_router_enable_routed_scaling: bool = True
    routed_scaling_factor: float = 1.0
    score_function: str = "softmax"
    n_group: int = 1
    topk_group: int = 4
    moe_shared_expert_intermediate_size: Optional[int] = None
    moe_router_enable_shared_expert: bool = True
