# Copyright © 2023 Apple Inc.

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
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    attention_bias: bool
    moe_topk: int
    num_experts: int
    num_shared_expert: int
    use_mixed_mlp_moe: bool
    use_qk_norm: bool
    rms_norm_eps: float
    rope_theta: float
    use_cla: bool
    cla_share_factor: 2
    moe_intermediate_size: Optional[Union[int, list]] = None
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = False

    def __post_init__(self):

        if self.rope_scaling:
            required_keys = {"factor", "type"}
            if not all(key in self.rope_scaling for key in required_keys):
                raise ValueError(f"rope_scaling must contain keys {required_keys}")
