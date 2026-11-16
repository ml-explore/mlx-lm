# Copyright © 2024 Apple Inc.

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.masks import create_attention_mask
from mlx_lm.models.phimoe import checkpoint
from mlx_lm.models.phimoe.config import ModelArgs
from mlx_lm.models.phimoe.layers import PhiMoEDecoderLayer

__all__ = [
    "Model",
    "PhiMoEModel",
]


class PhiMoEModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [PhiMoEDecoderLayer(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.LayerNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model_type = args.model_type
        self.args = args
        self.model = PhiMoEModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=True)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out = self.model(inputs, cache)
        return self.lm_head(out)

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    @property
    def layers(self):
        return self.model.layers
