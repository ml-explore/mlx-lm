# Copyright © 2023 Apple Inc.

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.masks import create_attention_mask
from mlx_lm.models.gpt2 import checkpoint
from mlx_lm.models.gpt2.config import ModelArgs
from mlx_lm.models.gpt2.layers import TransformerBlock

__all__ = [
    "GPT2Model",
    "Model",
]


class GPT2Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_embd = args.n_embd
        self.n_positions = args.n_positions
        self.vocab_size = args.vocab_size
        self.n_layer = args.n_layer
        self.layer_norm_epsilon = args.layer_norm_epsilon
        assert self.vocab_size > 0
        self.wte = nn.Embedding(self.vocab_size, self.n_embd)
        self.wpe = nn.Embedding(self.n_positions, self.n_embd)
        self.h = [TransformerBlock(args=args) for _ in range(self.n_layer)]
        self.ln_f = nn.LayerNorm(self.n_embd, eps=self.layer_norm_epsilon)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        _, L = inputs.shape

        hidden_states = self.wte(inputs)

        if cache is None:
            cache = [None] * len(self.h)

        offset = 0
        if cache[0] is not None:
            offset = cache[0].offset

        offset = mx.array(offset)
        position_ids = mx.arange(L) + offset[..., None]

        hidden_states += self.wpe(position_ids)

        mask = create_attention_mask(hidden_states, cache[0])

        for layer, c in zip(self.h, cache):
            hidden_states = layer(hidden_states, mask, cache=c)

        return self.ln_f(hidden_states)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = GPT2Model(args)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out = self.model(inputs, cache)
        out = self.model.wte.as_linear(out)
        return out

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    @property
    def layers(self):
        return self.model.h
