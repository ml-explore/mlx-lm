# Copyright © 2025 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.lille_130m import checkpoint
from mlx_lm.models.lille_130m.config import ModelArgs
from mlx_lm.models.lille_130m.layers import Lille130

__all__ = [
    "Model",
]


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.transformer = Lille130(args)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        return self.transformer(inputs, cache=cache)

    @property
    def layers(self):
        return self.transformer.layers

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)
