# Copyright © 2026 Apple Inc.

from dataclasses import dataclass, field
from typing import Any, Dict, List

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]

# The defaults of the Hugging Face configuration for the nested blocks.
_GDN_DEFAULTS = {
    "num_heads": 4,
    "head_dim_k": 64,
    "head_dim_v": 64,
    "short_conv_kernel": 4,
    "conv_bias": False,
    "qk_l2norm": True,
}
_MLA_DEFAULTS = {
    "num_heads": 4,
    "q_lora_rank": 128,
    "kv_lora_rank": 64,
    "qk_nope_head_dim": 48,
    "qk_rope_head_dim": 16,
    "v_head_dim": 64,
    "rope_theta": 100000.0,
}
_MOE_DEFAULTS = {
    "n_shared_experts": 1,
    "num_experts_per_tok": 4,
    "norm_topk_prob": True,
    "routed_scaling_factor": 1.0,
    "group_limited_routing": {"n_group": 1, "topk_group": 1},
}


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 4096
    tie_word_embeddings: bool = True
    gdn_config: Dict[str, Any] = field(default_factory=dict)
    mla_config: Dict[str, Any] = field(default_factory=dict)
    moe_config: Dict[str, Any] = field(default_factory=dict)
    dense_layer_indices: List[int] = field(default_factory=lambda: [0, 9])
    mla_layer_indices: List[int] = field(default_factory=lambda: [3, 7])

    def __post_init__(self):
        self.gdn_config = {**_GDN_DEFAULTS, **self.gdn_config}
        self.mla_config = {**_MLA_DEFAULTS, **self.mla_config}
        # An empty MoE config makes every feed-forward block dense.
        if self.moe_config:
            self.moe_config = {**_MOE_DEFAULTS, **self.moe_config}
