# Copyright © 2023 Apple Inc.

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.masks import create_attention_mask
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.exaone4.config import ModelArgs
from mlx_lm.models.exaone4.layers import TransformerBlock

__all__ = [
    "ExaoneModel",
    "Model",
]


class ExaoneModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        pattern = args.sliding_window_pattern
        self.layers = [
            TransformerBlock(
                args=args,
                is_local=pattern[i % len(pattern)] == "L" if pattern else None,
            )
            for i in range(args.num_hidden_layers)
        ]
        if pattern:
            self.swa_idx = pattern.index("L")
            self.full_idx = pattern.index("G")
        else:
            self.swa_idx = None
            self.full_idx = 0

        self.window_size = args.sliding_window
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)
        global_mask = create_attention_mask(h, cache[self.full_idx])
        if self.swa_idx is not None:
            swa_mask = create_attention_mask(
                h, cache[self.swa_idx], window_size=self.window_size
            )
        else:
            swa_mask = None

        for layer, c in zip(self.layers, cache):
            mask = swa_mask if layer.self_attn.is_local else global_mask
            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = ExaoneModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def make_cache(self):
        return [
            (
                RotatingKVCache(max_size=self.args.sliding_window, keep=0)
                if l.self_attn.is_local
                else KVCache()
            )
            for l in self.layers
        ]

    @property
    def layers(self):
        return self.model.layers
