# Copyright © 2024 Apple Inc.

from dataclasses import dataclass
from typing import Dict, List, Union

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "phimoe"
    vocab_size: int = 32064
    hidden_size: int = 4096
    intermediate_size: int = 6400
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 131072
    original_max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    rope_scaling: Dict[str, Union[float, List[float]]] = None
    num_local_experts: int = 16
    num_experts_per_tok: int = 2
    rope_theta: float = 10000.0
    router_jitter_noise: float = 0.01
