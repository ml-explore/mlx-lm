# Copyright © 2024 Apple Inc.

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    layer_types: List[str]
    vocab_size: int = 200192
    hidden_size: int = 2048
    intermediate_size: int = 6144
    moe_intermediate_size: int = 1024
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int = 64
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = False
    # MoE config
    num_experts: int = 128
    num_experts_per_tok: int = 8
    num_shared_experts: int = 1
    num_dense_layers: int = 2
    route_norm: bool = True
    route_scale: float = 2.826
    score_func: str = "sigmoid"
    n_group: int = 1
    topk_group: int = 1
    sliding_window: int = 2048
    mup_enabled: bool = True
