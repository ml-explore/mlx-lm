# Copyright © 2024 Apple Inc.

from dataclasses import dataclass
from typing import Dict, Optional, Union

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
    "TextArgs",
]


@dataclass
class TextArgs(BaseModelArgs):
    vocab_size: int = 102400
    hidden_size: int = 4096
    intermediate_size: int = 11008
    moe_intermediate_size: int = 1407
    num_hidden_layers: int = 30
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    n_shared_experts: Optional[int] = None
    n_routed_experts: Optional[int] = None
    routed_scaling_factor: float = 1.0
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    qk_nope_head_dim: int = 128
    topk_method: str = "noaux_tc"
    scoring_func: str = "sigmoid"
    norm_topk_prob: bool = True
    n_group: int = 1
    topk_group: int = 1
    num_experts_per_tok: int = 1
    moe_layer_freq: int = 1
    first_k_dense_replace: int = 0
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: Dict = None
    attention_bias: bool = False


@dataclass
class ModelArgs(BaseModelArgs):
    text_config: Union[TextArgs, dict]
    model_type: str

    def __post_init__(self):
        self.text_config = TextArgs.from_dict(self.text_config)
