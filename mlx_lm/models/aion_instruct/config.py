# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    sliding_window: int
    layer_types: List[str]
    rope_parameters: Dict[str, Dict[str, Any]]
    num_key_value_heads: Optional[int] = None
    head_dim: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    attention_bias: bool = False
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                "layer_types length must match num_hidden_layers: "
                f"{len(self.layer_types)} != {self.num_hidden_layers}"
            )
        unsupported = set(self.layer_types) - {"sliding_attention", "full_attention"}
        if unsupported:
            raise ValueError(f"Unsupported layer types: {sorted(unsupported)}")
        missing_rope = set(self.layer_types) - set(self.rope_parameters)
        if missing_rope:
            raise ValueError(
                f"Missing RoPE parameters for layer types: {sorted(missing_rope)}"
            )
