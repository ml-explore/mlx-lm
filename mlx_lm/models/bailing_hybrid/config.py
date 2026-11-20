# Copyright © 2026 Apple Inc.

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
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    num_experts: int
    num_experts_per_tok: int
    num_shared_experts: int
    n_group: int
    topk_group: int
    first_k_dense_replace: int
    layer_group_size: int
    group_norm_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    routed_scaling_factor: float
    head_dim: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    qk_nope_head_dim: int
    v_head_dim: int
    q_lora_rank: Optional[int] = None
    moe_shared_expert_intermediate_size: Optional[int] = None
    rope_interleave: bool = True
    partial_rotary_factor: float = 0.5
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    use_qkv_bias: bool = False
    use_bias: bool = False
    use_qk_norm: bool = True
    score_function: str = "sigmoid"
    norm_topk_prob: bool = True
    moe_router_enable_expert_bias: bool = True
    tie_word_embeddings: bool = False
    num_nextn_predict_layers: int = 0

    def __post_init__(self):
        head_dim = self.head_dim or self.hidden_size // self.num_attention_heads
        dims = int(head_dim * self.partial_rotary_factor)
        if dims <= 0 or dims > head_dim or dims % 2:
            raise ValueError(
                f"Invalid rotary dimension {dims} with head_dim={head_dim}"
            )
