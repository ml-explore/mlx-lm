# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import List, Optional

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
    moe_intermediate_size: int
    num_hidden_layers: int
    num_experts: int
    num_experts_per_tok: int
    norm_topk_prob: bool
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    use_expert_bias: bool
    num_dense_layers: int
    norm_eps: float
    conv_bias: bool
    conv_L_cache: int
    rope_theta: float = 1000000.0
    routed_scaling_factor: float = 1.0
    rope_parameters: Optional[dict] = None
    full_attn_idxs: Optional[List[int]] = None
    layer_types: Optional[List[str]] = None

    def __post_init__(self):
        if self.rope_parameters is not None and "rope_theta" in self.rope_parameters:
            self.rope_theta = self.rope_parameters["rope_theta"]
        if self.full_attn_idxs is None:
            self.full_attn_idxs = [
                i
                for i, layer_type in enumerate(self.layer_types)
                if layer_type == "full_attention"
            ]
