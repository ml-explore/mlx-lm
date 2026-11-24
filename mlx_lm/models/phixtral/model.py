# Copyright © 2023 Apple Inc.

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.models.phixtral import checkpoint
from mlx_lm.models.phixtral.config import ModelArgs
from mlx_lm.models.phixtral.layers import OutputHead, TransformerDecoder

__all__ = [
    "Model",
]


class Model(nn.Module):
    model_args = ModelArgs

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.model_type = config.model_type
        self.transformer = TransformerDecoder(config)
        self.lm_head = OutputHead(config)
        self.args = config

    def __call__(
        self,
        x: mx.array,
        mask: mx.array = None,
        cache=None,
    ) -> mx.array:

        if cache is None:
            cache = [None] * len(self.transformer.h)

        if mask is None:
            mask = create_attention_mask(x, cache[0])

        y = self.transformer(x, mask, cache)
        return self.lm_head(y)

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    @property
    def layers(self):
        return self.transformer.h
