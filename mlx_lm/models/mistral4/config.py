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
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    n_shared_experts: int
    n_routed_experts: int
    routed_scaling_factor: float
    kv_lora_rank: int
    q_lora_rank: int
    norm_topk_prob: bool
    max_position_embeddings: int
    rms_norm_eps: float
    topk_group: int
    num_experts_per_tok: int
    first_k_dense_replace: int
    n_group: int
    qk_rope_head_dim: int
    qk_nope_head_dim: int
    v_head_dim: int
    head_dim: Optional[int] = None
    qk_head_dim: Optional[int] = None
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False
    rope_interleave: Optional[bool] = None
    attention_bias: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    rope_parameters: Optional[Dict] = None

    def __post_init__(self, **kwargs):
        if self.qk_head_dim is None:
            self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        if self.head_dim is None:
            self.head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        if self.rope_parameters is not None:
            self.rope_theta = self.rope_parameters.get("rope_theta", 100000.0)
            self.rope_scaling = self.rope_parameters
