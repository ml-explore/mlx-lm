# Copyright © 2025 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.olmo3.config import ModelArgs
from mlx_lm.models.olmo3.layers import Olmo3DecoderLayer

__all__ = [
    "Model",
    "Olmo3Model",
]


class Olmo3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.sliding_window = args.sliding_window

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Olmo3DecoderLayer(args=args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        self.swa_idx = args.layer_types.index("sliding_attention")
        self.ga_idx = args.layer_types.index("full_attention")
        self.layer_types = args.layer_types

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        full_mask = create_attention_mask(h, cache[self.ga_idx])
        sliding_window_mask = create_attention_mask(
            h, cache[self.swa_idx], window_size=self.sliding_window
        )

        for layer, c, layer_type in zip(self.layers, cache, self.layer_types):
            mask = full_mask if layer_type == "full_attention" else sliding_window_mask
            h = layer(h, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Olmo3Model(args)
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

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for lt in self.model.layer_types:
            if lt == "full_attention":
                caches.append(KVCache())
            else:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window))
        return caches
