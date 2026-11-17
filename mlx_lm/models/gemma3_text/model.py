# Copyright © 2025 Apple Inc.

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.masks import create_attention_mask
from mlx_lm.layers.norms import ZeroCenteredRMSNorm
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.gemma3_text import checkpoint
from mlx_lm.models.gemma3_text.config import ModelArgs
from mlx_lm.models.gemma3_text.layers import TransformerBlock

__all__ = [
    "Gemma3TextModel",
    "Model",
    "make_cache",
]


class Gemma3TextModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.window_size = args.sliding_window
        self.sliding_window_pattern = args.sliding_window_pattern
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args, layer_idx=layer_idx)
            for layer_idx in range(args.num_hidden_layers)
        ]
        self.norm = ZeroCenteredRMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)
        h *= mx.array(self.args.hidden_size**0.5, mx.bfloat16).astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        global_mask = create_attention_mask(h, cache[self.sliding_window_pattern - 1])

        if self.sliding_window_pattern > 1:
            sliding_window_mask = create_attention_mask(
                h,
                cache[0],
                window_size=self.window_size,
            )
        else:
            sliding_window_mask = None
        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            is_global = (
                i % self.sliding_window_pattern == self.sliding_window_pattern - 1
            )
            mask = global_mask if is_global else sliding_window_mask
            h = layer(h, mask, c)

        return self.norm(h)


def make_cache(args: ModelArgs):
    """A global cache for every sliding_window_pattern-th layer, sliding ones otherwise."""
    caches = []
    for i in range(args.num_hidden_layers):
        if i % args.sliding_window_pattern == args.sliding_window_pattern - 1:
            caches.append(KVCache())
        else:
            caches.append(RotatingKVCache(max_size=args.sliding_window))
    return caches


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Gemma3TextModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.tie_word_embeddings = False

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        out = self.model(inputs, cache, input_embeddings)
        if self.tie_word_embeddings:
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
        return make_cache(self.args)
