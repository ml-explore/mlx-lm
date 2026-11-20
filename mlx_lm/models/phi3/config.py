# Copyright © 2023 Apple Inc.

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

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
    num_key_value_heads: Optional[int] = None
    rope_theta: float = 10000
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, List[float]]]] = None
    partial_rotary_factor: float = 1.0
    max_position_embeddings: int = 131072
    original_max_position_embeddings: int = 4096
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.rope_scaling:
            required_keys = {"long_factor", "type"}
            if not all(key in self.rope_scaling for key in required_keys):
                raise ValueError(f"rope_scaling must contain keys {required_keys}")

            if self.rope_scaling["type"] not in ["longrope", "su", "linear"]:
                print(
                    "[WARNING] rope_scaling 'type' currently only supports 'linear', 'su', and 'longrope'; setting rope scaling to false."
                )
                self.rope_scaling = None
