# Copyright © 2025 Apple Inc.

from functools import partial

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.models.nanochat.config import ModelArgs
from mlx_lm.models.nanochat.layers import TransformerBlock, rms_norm

__all__ = [
    "Model",
    "NanoChatModel",
    "softcap",
]


class NanoChatModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.wte = nn.Embedding(args.vocab_size, args.hidden_size)
        self.h = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ) -> mx.array:
        h = self.wte(inputs)
        # Critical: norm after token embedding
        h = rms_norm(h)

        if cache is None:
            cache = [None] * len(self.h)

        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.h, cache):
            h = layer(h, mask=mask, cache=c)

        # Critical: final norm before lm_head
        h = rms_norm(h)

        return h


@partial(mx.compile, shapeless=True)
def softcap(logits, cap=15.0):
    return cap * mx.tanh(logits / cap)


class Model(nn.Module):
    model_args = ModelArgs

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.transformer = NanoChatModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ) -> mx.array:
        out = self.transformer(inputs, cache=cache)
        logits = self.lm_head(out)

        # Critical: logits softcap (nanochat uses softcap=15)
        logits = softcap(logits)

        return logits

    @property
    def layers(self):
        return self.transformer.h
