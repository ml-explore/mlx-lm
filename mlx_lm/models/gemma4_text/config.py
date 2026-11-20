# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Dict, List, Optional

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "gemma4_text"
    hidden_size: int = 1536
    num_hidden_layers: int = 35
    intermediate_size: int = 6144
    num_attention_heads: int = 8
    head_dim: int = 256
    global_head_dim: int = 512
    global_partial_rotary_factor: float = 0.25
    rms_norm_eps: float = 1e-6
    vocab_size: int = 262144
    vocab_size_per_layer_input: int = 262144
    num_key_value_heads: int = 1
    num_global_key_value_heads: Optional[int] = None
    num_kv_shared_layers: int = 20
    pad_token_id: int = 0
    hidden_size_per_layer_input: int = 256
    rope_traditional: bool = False
    partial_rotary_factor: float = 1.0
    rope_parameters: Optional[Dict] = None
    sliding_window: int = 512
    sliding_window_pattern: int = 5
    max_position_embeddings: int = 131072
    attention_k_eq_v: bool = False
    final_logit_softcapping: float = 30.0
    use_double_wide_mlp: bool = True
    enable_moe_block: bool = False
    num_experts: Optional[int] = None
    top_k_experts: Optional[int] = None
    moe_intermediate_size: Optional[int] = None
    layer_types: Optional[List[str]] = None
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.rope_parameters is None:
            self.rope_parameters = {
                "full_attention": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1000000.0,
                    "rope_type": "proportional",
                },
                "sliding_attention": {
                    "partial_rotary_factor": 1.0,
                    "rope_theta": 10000.0,
                    "rope_type": "default",
                },
            }
        if self.layer_types is None:
            pattern = ["sliding_attention"] * (self.sliding_window_pattern - 1) + [
                "full_attention"
            ]
            self.layer_types = (pattern * (self.num_hidden_layers // len(pattern) + 1))[
                : self.num_hidden_layers
            ]
