# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, Optional

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "glm4_moe_lite"
    vocab_size: int = 154880
    hidden_size: int = 2048
    intermediate_size: int = 10240
    moe_intermediate_size: int = 1536
    num_hidden_layers: int = 47
    num_attention_heads: int = 20
    num_key_value_heads: int = 20
    n_shared_experts: Optional[int] = 1
    n_routed_experts: Optional[int] = 64
    routed_scaling_factor: float = 1.8
    kv_lora_rank: int = 512
    q_lora_rank: int = 768
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 192
    v_head_dim: int = 256
    topk_method: str = "noaux_tc"
    scoring_func: str = "sigmoid"
    norm_topk_prob: bool = True
    n_group: int = 1
    topk_group: int = 1
    num_experts_per_tok: int = 4
    moe_layer_freq: int = 1
    first_k_dense_replace: int = 1
    max_position_embeddings: int = 202752
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1_000_000.0
    rope_scaling: Optional[Dict] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    partial_rotary_factor: float = 1.0
    tie_word_embeddings: bool = False
    num_nextn_predict_layers: int = 1
    quantization: Optional[Dict[str, Any]] = None
