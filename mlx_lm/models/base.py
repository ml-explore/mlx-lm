# Copyright © 2023 Apple Inc.

import inspect
from dataclasses import dataclass

from mlx_lm.layers.attention import (
    quantized_scaled_dot_product_attention,
    scaled_dot_product_attention,
)
from mlx_lm.layers.distributed import gather_last_axis
from mlx_lm.layers.masks import (
    create_attention_mask,
    create_causal_mask,
    create_ssm_mask,
)

__all__ = [
    "BaseModelArgs",
    "create_attention_mask",
    "create_causal_mask",
    "create_ssm_mask",
    "gather_last_axis",
    "quantized_scaled_dot_product_attention",
    "scaled_dot_product_attention",
]


@dataclass
class BaseModelArgs:
    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )
