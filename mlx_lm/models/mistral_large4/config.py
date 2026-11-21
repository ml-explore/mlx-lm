# Copyright © 2026 Apple Inc.

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
    "TextArgs",
]


@dataclass
class TextArgs(BaseModelArgs):
    vocab_size: int = 204800
    hidden_size: int = 6144
    intermediate_size: int = 24576
    moe_intermediate_size: int = 6144
    num_hidden_layers: int = 72
    num_attention_heads: int = 64
    n_shared_experts: int = 1
    n_routed_experts: int = 128
    routed_scaling_factor: float = 1.0
    kv_lora_rank: int = 512
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 64
    v_head_dim: int = 128
    n_group: int = 1
    topk_group: int = 1
    num_experts_per_tok: int = 4
    rms_norm_eps: float = 1e-5
    rope_parameters: Optional[Dict] = None
    index_topk: int = 2048
    index_head_dim: int = 128
    index_n_heads: int = 32
    first_k_dense_replace: int = 1
    mlp_layer_types: Optional[List[str]] = None
    indexer_types: Optional[List[str]] = None
    no_rope_layers: Optional[List[int]] = None
    no_rope_layer_interval: int = 4
    kv_source_layers: Optional[List[int]] = None
    final_logit_softcapping: Optional[float] = 30.0

    def __post_init__(self):
        self.rope_parameters = {
            "rope_type": "yarn",
            "rope_theta": 10000.0,
            "factor": 2048.0,
            "original_max_position_embeddings": 512,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "attention_factor": 1.0,
            **(self.rope_parameters or {}),
        }
        n = self.num_hidden_layers
        if self.no_rope_layers is None:
            interval = self.no_rope_layer_interval
            self.no_rope_layers = [int((i + 1) % interval != 0) for i in range(n)]
        if self.kv_source_layers is None:
            self.kv_source_layers = list(range(n))
        if self.indexer_types is None:
            self.indexer_types = ["full"] * n
        if self.mlp_layer_types is None:
            dense = min(self.first_k_dense_replace, n)
            self.mlp_layer_types = ["dense"] * dense + ["sparse"] * (n - dense)
        for i, src in enumerate(self.kv_source_layers):
            if src > i or self.kv_source_layers[src] != src:
                raise ValueError(f"Layer {i} cannot share the KV cache of layer {src}.")
        if self.indexer_types[0] != "full":
            raise ValueError("The first layer must run its own indexer.")


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict = field(default_factory=dict)
    quantization_config: Optional[dict] = None

    def __post_init__(self):
        self.text = TextArgs.from_dict(self.text_config)
