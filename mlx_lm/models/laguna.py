# Copyright © 2025 Apple Inc.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "laguna"
    vocab_size: int = 100352
    hidden_size: int = 3072
    intermediate_size: int = 12288
    num_hidden_layers: int = 48
    num_attention_heads: int = 48
    num_key_value_heads: int = 8
    head_dim: int = 128
    attention_bias: bool = False
    gating: Any = True
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 1048576
    rope_parameters: Dict[str, Any] = field(
        default_factory=lambda: {"rope_type": "default", "rope_theta": 500000.0}
    )
    sliding_window: Optional[int] = None
    layer_types: Optional[List[str]] = None
    num_attention_heads_per_layer: Optional[List[int]] = None
    swa_attention_sink_enabled: bool = False
    num_experts: int = 256
    num_experts_per_tok: int = 10
    moe_intermediate_size: int = 1024
    shared_expert_intermediate_size: int = 1024
    norm_topk_prob: bool = True
    decoder_sparse_step: int = 1
    mlp_only_layers: Optional[List[int]] = None
    moe_routed_scaling_factor: float = 1.0
    moe_apply_router_weight_on_input: bool = False
    moe_router_logit_softcapping: float = 0.0
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.mlp_only_layers is None:
            self.mlp_only_layers = [0]
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        if self.moe_apply_router_weight_on_input:
            # Matches the transformers reference, which also raises for this
            # flag: routing the weight into the expert *input* (rather than
            # scaling the output) needs different math in SwitchGLU than what
            # this port implements. No known Laguna checkpoint sets this.
            raise NotImplementedError(
                "moe_apply_router_weight_on_input=True is not supported."
            )


def _resolve_rope_config(rope_parameters: Dict[str, Any], attention_type: str) -> Dict[str, Any]:
    """Laguna nests RoPE config by layer type when the model mixes full and
    sliding attention: ``{"full_attention": {...}, "sliding_attention": {...}}``.
    A flat (non-nested) config applies to every layer as-is."""
    sub = rope_parameters.get(attention_type)
    if isinstance(sub, dict):
        return sub
    return rope_parameters
