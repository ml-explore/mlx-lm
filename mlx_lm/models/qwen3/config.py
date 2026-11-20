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
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    max_position_embeddings: int
    head_dim: int
    tie_word_embeddings: bool
    rope_theta: Optional[float] = None
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    rope_parameters: Optional[dict] = None

    def __post_init__(self):
        # transformers >= 5 nests rope_theta and rope_scaling under rope_parameters.
        if self.rope_parameters:
            if self.rope_theta is None:
                self.rope_theta = self.rope_parameters.get("rope_theta")
            if self.rope_scaling is None:
                self.rope_scaling = self.rope_parameters
        if self.rope_theta is None:
            raise ValueError("rope_theta is missing from the model config.")
