# Copyright © 2026 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.plamo3 import checkpoint
from mlx_lm.models.plamo3.config import ModelArgs
from mlx_lm.models.plamo3.layers import Plamo3Decoder, RMSNorm

__all__ = [
    "Model",
    "Plamo3Model",
]


class Plamo3Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = Plamo3Decoder(config)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)
            if self.config.scale_embedding:
                h = h * self.config.hidden_size**0.5

        h = self.layers(h, cache)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = Plamo3Model(config)
        self.vocab_size = config.vocab_size

        if not config.tie_word_embeddings:
            self.lm_head: nn.Module = nn.Linear(
                config.hidden_size, self.vocab_size, bias=False
            )

    def sanitize(self, weights: dict[Any, Any]) -> dict[Any, Any]:
        return checkpoint.sanitize(self, weights)

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.full_attn:
                c = KVCache()
            else:
                c = RotatingKVCache(max_size=self.config.attention_window_size)
            caches.append(c)
        return caches

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        outputs = self.model(inputs, cache=cache, input_embeddings=input_embeddings)
        if self.config.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(outputs)
        return self.lm_head(outputs)

    @property
    def layers(self):
        return self.model.layers.layers
