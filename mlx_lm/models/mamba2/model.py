# Copyright © 2025 Apple Inc.

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.mamba2 import checkpoint
from mlx_lm.models.mamba2.config import ModelArgs
from mlx_lm.models.mamba2.layers import Mamba2

__all__ = [
    "Model",
]


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.backbone = Mamba2(args)

        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self, inputs: mx.array, cache: Optional[list[ArraysCache]] = None
    ) -> mx.array:
        hidden = self.backbone(inputs, cache)

        if self.args.tie_word_embeddings:
            logits = self.backbone.embeddings.as_linear(hidden)
        else:
            logits = self.lm_head(hidden)
        return logits

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    @property
    def layers(self):
        return self.backbone.layers

    def make_cache(self, batch_size: int = 1) -> list[ArraysCache]:
        return [ArraysCache(size=2) for _ in range(self.args.num_hidden_layers)]
