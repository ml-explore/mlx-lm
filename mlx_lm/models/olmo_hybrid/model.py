# Copyright © 2026 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.layers.recurrent import create_ssm_mask
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.olmo_hybrid import checkpoint
from mlx_lm.models.olmo_hybrid.config import ModelArgs
from mlx_lm.models.olmo_hybrid.layers import (
    FullAttentionDecoderLayer,
    LinearAttentionDecoderLayer,
)

__all__ = [
    "Model",
    "OlmoHybridModel",
]


class OlmoHybridModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            (
                LinearAttentionDecoderLayer(args)
                if lt == "linear_attention"
                else FullAttentionDecoderLayer(args)
            )
            for lt in args.layer_types
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        self._fa_idx = next(
            (i for i, lt in enumerate(args.layer_types) if lt == "full_attention"),
            None,
        )
        self._lin_idx = next(
            (i for i, lt in enumerate(args.layer_types) if lt == "linear_attention"),
            None,
        )

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        fa_mask = None
        if self._fa_idx is not None:
            fa_mask = create_attention_mask(h, cache[self._fa_idx])
        ssm_mask = None
        if self._lin_idx is not None:
            ssm_mask = create_ssm_mask(h, cache[self._lin_idx])

        for layer, c in zip(self.layers, cache):
            if isinstance(layer, FullAttentionDecoderLayer):
                h = layer(h, mask=fa_mask, cache=c)
            else:
                h = layer(h, mask=ssm_mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    model_args = ModelArgs

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = OlmoHybridModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for lt in self.args.layer_types:
            if lt == "linear_attention":
                caches.append(ArraysCache(size=4))
            else:
                caches.append(KVCache())
        return caches
