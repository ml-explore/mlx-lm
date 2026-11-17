# Copyright © 2023 Apple Inc.

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.models.gpt_neox import checkpoint
from mlx_lm.models.gpt_neox.config import ModelArgs
from mlx_lm.models.gpt_neox.layers import TransformerBlock

__all__ = [
    "GPTNeoXModel",
    "Model",
]


class GPTNeoXModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.layer_norm_eps = args.layer_norm_eps
        assert self.vocab_size > 0
        self.embed_in = nn.Embedding(self.vocab_size, self.hidden_size)
        self.embed_out = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.h = [TransformerBlock(args=args) for _ in range(self.num_hidden_layers)]
        self.final_layer_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        _, L = inputs.shape

        hidden_states = self.embed_in(inputs)

        if cache is None:
            cache = [None] * len(self.h)

        mask = create_attention_mask(hidden_states, cache[0])

        for layer, c in zip(self.h, cache):
            hidden_states = layer(hidden_states, mask, cache=c)

        out = self.final_layer_norm(hidden_states)
        out = self.embed_out(out)

        return out


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = GPTNeoXModel(args)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out = self.model(inputs, cache)
        return out

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    @property
    def layers(self):
        return self.model.h
