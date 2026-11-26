# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, Optional

from mlx_lm.models.base import BaseModelArgs
from mlx_lm.models.kimi_k3.config import TextArgs as KimiK3TextArgs

__all__ = [
    "ModelArgs",
    "TextArgs",
]


@dataclass
class TextArgs(KimiK3TextArgs):
    dsa_config: Optional[Dict[str, Any]] = None
    sparse_prefill: bool = True
    sparse_prefill_min_context: int = 16384
    sparse_prefill_fused: bool = True
    sparse_prefill_fused_min_context: int = 0

    def __post_init__(self):
        super().__post_init__()
        if self.dsa_config is None:
            return
        dsa = self.dsa_config
        if dsa.get("index_rope_dim", 0) != 0:
            raise ValueError("Only index_rope_dim=0 is supported")
        # A frequency of 1 gives each layer its own indexer, so the offset has no effect.
        if dsa.get("index_topk_freq", 1) != 1:
            raise ValueError("Indexer sharing (index_topk_freq > 1) is not supported")
        if dsa.get("index_k_norm_type", "layernorm") != "layernorm":
            raise ValueError("Only a LayerNorm indexer key norm is supported")
        if self.q_lora_rank is None:
            raise ValueError("The indexer queries need q_lora_rank")
        # Both lists are 1-based.
        kda = set(self.linear_attn_config["kda_layers"])
        if kda & set(dsa["dsa_layers"]):
            raise ValueError("dsa_layers must be full attention layers")


@dataclass
class ModelArgs(BaseModelArgs):
    text_config: dict
    model_type: str = "kimi_k31"

    @classmethod
    def from_dict(cls, params):
        if "text_config" not in params:
            return cls(model_type=params["model_type"], text_config=params)
        return super().from_dict(params)

    def __post_init__(self):
        self.text = TextArgs.from_dict(self.text_config)
