# Copyright © 2025 Apple Inc.

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.rwkv7 import checkpoint
from mlx_lm.models.rwkv7.config import ModelArgs
from mlx_lm.models.rwkv7.layers import Rwkv7Layer

__all__ = [
    "Model",
    "Rwkv7Model",
]


class Rwkv7Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Rwkv7Layer(args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.LayerNorm(args.hidden_size, eps=args.norm_eps)

    def __call__(self, x: mx.array, cache):
        x = self.embeddings(x)
        if cache is None:
            cache = [None] * len(self.layers)

        v_first = None
        for layer, c in zip(self.layers, cache):
            x, v_first = layer(x, v_first, c)
        return self.norm(x)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Rwkv7Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None):
        x = self.model(inputs, cache)

        if self.args.tie_word_embeddings:
            logits = self.model.embeddings.as_linear(x)
        else:
            logits = self.lm_head(x)

        return logits

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if "lora.2" in path or "embeddings" in path:
                return {"bits": 8}
            return True

        return predicate

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [ArraysCache(size=3) for _ in range(len(self.layers))]
