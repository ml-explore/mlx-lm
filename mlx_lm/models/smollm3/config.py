# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Optional

from mlx_lm.models import llama

__all__ = [
    "ModelArgs",
]


@dataclass
class ModelArgs(llama.ModelArgs):
    model_type: str
    no_rope_layer_interval: int = 4
    no_rope_layers: Optional[list[int]] = None

    def __post_init__(self):
        super().__post_init__()
        if self.no_rope_layers is None:
            self.no_rope_layers = [
                int((i + 1) % self.no_rope_layer_interval != 0)
                for i in range(self.num_hidden_layers)
            ]
        elif len(self.no_rope_layers) != self.num_hidden_layers:
            raise ValueError("`no_rope_layers` length mismatch")
