# Copyright © 2023 Apple Inc.

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.masks import create_attention_mask
from mlx_lm.models.gemma2.config import ModelArgs
from mlx_lm.models.gemma2.layers import RMSNorm, TransformerBlock

__all__ = [
    "GemmaModel",
    "Model",
]


class GemmaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        h = self.embed_tokens(inputs)
        h = h * (self.args.hidden_size**0.5)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(h, cache[0], return_array=True)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model_type = args.model_type
        self.final_logit_softcapping = args.final_logit_softcapping
        self.model = GemmaModel(args)
        self.args = args

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out = self.model(inputs, cache)
        out = self.model.embed_tokens.as_linear(out)
        out = mx.tanh(out / self.final_logit_softcapping)
        out = out * self.final_logit_softcapping
        return out

    @property
    def layers(self):
        return self.model.layers
