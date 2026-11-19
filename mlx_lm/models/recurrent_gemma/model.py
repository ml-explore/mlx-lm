# Copyright © 2023 Apple Inc.

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.cache import ArraysCache, RotatingKVCache
from mlx_lm.models.recurrent_gemma import checkpoint
from mlx_lm.models.recurrent_gemma.layers import Griffin

__all__ = [
    "Model",
]


class Model(nn.Module):

    def __init__(self, config):
        self.args = config
        self.model = Griffin(config)
        self.model_type = config.model_type
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(self, tokens: mx.array, cache=None) -> mx.array:
        logits = self.model(tokens, cache=cache)
        if "lm_head" in self:
            logits = self.lm_head(logits)
        else:
            logits = self.model.embed_tokens.as_linear(logits)

        c = self.args.logits_soft_cap
        if c:
            logits = mx.tanh(logits / c) * c
        return logits

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        cache = []
        for layer in self.layers:
            if layer.temporal_block_type == "recurrent":
                cache.append(ArraysCache(size=2))
            else:
                cache.append(RotatingKVCache(max_size=self.args.attention_window_size))
        return cache
