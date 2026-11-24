# Copyright © 2023 Apple Inc.

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.olmo.config import ModelArgs
from mlx_lm.models.olmo.layers import Transformer

__all__ = [
    "Model",
    "OlmoModel",
]


class OlmoModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.transformer = Transformer(args)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        return self.transformer(inputs, cache)


class Model(nn.Module):
    model_args = ModelArgs

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model_type = args.model_type
        self.model = OlmoModel(args)
        self.args = args

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        return self.model(inputs, cache)

    @property
    def layers(self):
        return self.model.transformer.blocks
