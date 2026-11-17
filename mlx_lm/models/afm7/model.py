# Copyright © 2025 Apple Inc.

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.models.afm7.config import ModelArgs
from mlx_lm.models.afm7.layers import KVReuseTransformerBlock, TransformerBlock
from mlx_lm.models.cache import ConcatenateKVCache, KVCache

__all__ = [
    "AFMModel",
    "Model",
]


class AFMModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size

        self.embedding = nn.Embedding(args.vocab_size, args.hidden_dim)
        self.layers = [
            TransformerBlock(args)
            for _ in range(args.num_layers - args.num_kv_reuse_layers)
        ]
        self.kv_reuse_layers = [
            KVReuseTransformerBlock(args) for _ in range(args.num_kv_reuse_layers)
        ]
        self.output_norm = nn.RMSNorm(args.hidden_dim, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        h = self.embedding(inputs)

        if cache is None:
            cache = [None] * len(self.layers)
            cache[-1] = ConcatenateKVCache()

        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)

        keys, values = cache[-1].keys_and_values()
        # The cache already holds this step, so go back L to the start offset.
        offset = cache[-1].offset - inputs.shape[1]
        for layer in self.kv_reuse_layers:
            h = layer(h, keys, values, offset, mask)

        return self.output_norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = AFMModel(args)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out = self.model(inputs, cache)
        out = self.model.embedding.as_linear(out)
        return out

    def make_cache(self):
        return [KVCache() for _ in range(len(self.model.layers))]

    @property
    def layers(self):
        return self.model.layers + self.model.kv_reuse_layers
