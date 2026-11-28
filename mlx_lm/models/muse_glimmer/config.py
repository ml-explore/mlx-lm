# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from typing import List, Optional

from mlx_lm.models.base import BaseModelArgs

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "muse_glimmer"
    hidden_size: int = 6656
    num_hidden_layers: int = 52
    intermediate_size: int = 19968
    num_attention_heads: int = 32
    num_key_value_heads: int = 2
    head_dim: int = 128
    vocab_size: int = 202048
    rms_norm_eps: float = 1e-5
    post_norm_eps: float = 1e-8
    sliding_window: int = 2048
    qk_scale_factor: float = 3.87
    hidden_activation: str = "silu"
    max_position_embeddings: int = 131072
    rope_theta: float = 500000.0
    rope_parameters: Optional[dict] = None
    layer_types: Optional[List[str]] = None
    layer_rope_theta: Optional[List[float]] = None
    tie_word_embeddings: bool = False
    final_logit_softcapping: float = 20.0
    output_multiplier: float = 0.19611613513818404

    @classmethod
    def from_dict(cls, params):
        # Meta's original checkpoint nests the language-model fields under
        # "text_config"; the mlx-community conversions flatten them. Accept both.
        if "text_config" in params:
            merged = dict(params)
            merged.update(params["text_config"])
            params = merged
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in params.items() if k in fields})

    def __post_init__(self):
        # rope_theta may live under rope_parameters (as in Meta's config)
        if self.rope_parameters and "rope_theta" in self.rope_parameters:
            self.rope_theta = self.rope_parameters["rope_theta"]
        if self.layer_types is None:
            # default [local, local, local, global] repeating
            pat = ["sliding_attention"] * 3 + ["full_attention"]
            self.layer_types = (pat * (self.num_hidden_layers // 4 + 1))[
                : self.num_hidden_layers
            ]
        if self.layer_rope_theta is None:
            self.layer_rope_theta = [
                0 if t == "full_attention" else self.rope_theta
                for t in self.layer_types
            ]
