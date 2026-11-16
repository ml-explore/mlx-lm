# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "laguna"
    vocab_size: int = 100352
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 40
    num_attention_heads: int = 48
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    qkv_bias: bool = False
    gating: bool = True
    tie_word_embeddings: bool = False
    sliding_window: Optional[int] = 512
    partial_rotary_factor: Optional[float] = None
    rope_parameters: Optional[Dict[str, Any]] = None
    layer_types: Optional[List[str]] = None
    num_attention_heads_per_layer: Optional[List[int]] = None
    mlp_layer_types: Optional[List[str]] = None
    # MoE
    num_experts: int = 256
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512
    moe_routed_scaling_factor: float = 1.0
    moe_router_logit_softcapping: float = 0.0
    moe_router_score_func: str = "sigmoid"

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] + ["sparse"] * (self.num_hidden_layers - 1)
        if self.num_attention_heads_per_layer is None:
            self.num_attention_heads_per_layer = [
                self.num_attention_heads
            ] * self.num_hidden_layers
