# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    index_head_dim: int
    index_n_heads: int
    index_topk: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    n_shared_experts: Optional[int]
    n_routed_experts: Optional[int]
    routed_scaling_factor: float
    kv_lora_rank: int
    q_lora_rank: int
    qk_rope_head_dim: int
    v_head_dim: int
    qk_nope_head_dim: int
    topk_method: str
    scoring_func: str
    norm_topk_prob: bool
    n_group: int
    topk_group: int
    num_experts_per_tok: int
    moe_layer_freq: int
    first_k_dense_replace: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_parameters: Dict
    attention_bias: bool
    rope_scaling: Dict = None
    rope_theta: Optional[float] = None
    indexer_rope_interleave: bool = True
    indexer_types: Optional[List[str]] = None
    index_topk_pattern: Optional[Any] = None
    index_topk_freq: int = 1
    index_skip_topk_offset: int = 2
    sparse_prefill: bool = True
    sparse_prefill_min_context: int = 16384
    sparse_prefill_fused: bool = True  # sparse prefill attention in one Metal kernel
    sparse_prefill_fused_min_context: int = 0  # with the kernel, sparse from here

    def __post_init__(self):
        self.rope_scaling = self.rope_parameters
        self.rope_theta = self.rope_parameters["rope_theta"]

        if self.indexer_types is None:
            if self.index_topk_pattern is not None:
                pattern = self.index_topk_pattern
                if isinstance(pattern, str):
                    self.indexer_types = [
                        {"F": "full", "S": "shared"}[c] for c in pattern
                    ]
                else:
                    self.indexer_types = list(pattern)
            else:
                freq = max(self.index_topk_freq, 1)
                offset = self.index_skip_topk_offset
                self.indexer_types = [
                    "full" if (max(i - offset + 1, 0) % freq) == 0 else "shared"
                    for i in range(self.num_hidden_layers)
                ]
