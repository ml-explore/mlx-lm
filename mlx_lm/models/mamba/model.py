# Copyright © 2024 Apple Inc.

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.mamba import checkpoint
from mlx_lm.models.mamba.config import ModelArgs
from mlx_lm.models.mamba.layers import Mamba

__all__ = [
    "Model",
]


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.backbone = Mamba(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None):
        B, T = inputs.shape

        x = self.backbone(inputs, cache)

        if self.args.tie_word_embeddings:
            logits = self.backbone.embeddings.as_linear(x)
        else:
            logits = self.lm_head(x)

        return logits

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    @property
    def layers(self):
        return self.backbone.layers

    def make_cache(self):
        return [ArraysCache(size=2) for _ in range(len(self.layers))]
