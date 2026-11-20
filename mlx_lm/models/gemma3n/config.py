# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Dict, List, Optional

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
    "TextConfig",
]


@dataclass
class TextConfig(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    head_dim: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    num_kv_shared_layers: int
    vocab_size_per_layer_input: int
    sliding_window: int
    max_position_embeddings: int
    rope_local_base_freq: float
    rope_theta: float
    final_logit_softcapping: float
    layer_types: List[str]
    activation_sparsity_pattern: List[float]
    hidden_size_per_layer_input: int
    altup_num_inputs: int
    altup_coef_clip: float
    altup_correct_scale: bool
    altup_active_idx: int
    laurel_rank: int
    rope_scaling: Optional[Dict] = None


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict
