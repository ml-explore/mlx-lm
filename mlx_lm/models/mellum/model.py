# Copyright © 2026 Apple Inc.

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.masks import create_attention_mask
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.mellum import checkpoint
from mlx_lm.models.mellum.config import ModelArgs
from mlx_lm.models.mellum.layers import MellumDecoderLayer

__all__ = [
    "MellumModel",
    "Model",
]


class MellumModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            MellumDecoderLayer(args=args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        self._first_full = next(
            i for i, t in enumerate(args.layer_types) if t == "full_attention"
        )
        self._first_sliding = next(
            (i for i, t in enumerate(args.layer_types) if t == "sliding_attention"),
            None,
        )

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        full_mask = create_attention_mask(h, cache[self._first_full])
        if self._first_sliding is not None:
            sliding_mask = create_attention_mask(
                h, cache[self._first_sliding], window_size=self.args.sliding_window
            )
        else:
            sliding_mask = None

        for layer, c, t in zip(self.layers, cache, self.args.layer_types):
            mask = full_mask if t == "full_attention" else sliding_mask
            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = MellumModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("mlp.gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for t in self.args.layer_types:
            if t == "full_attention":
                caches.append(KVCache())
            else:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window))
        return caches
