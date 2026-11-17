# Copyright © 2023 Apple Inc.

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.models.gpt_bigcode.config import ModelArgs
from mlx_lm.models.gpt_bigcode.layers import TransformerBlock

__all__ = [
    "GPTBigCodeModel",
    "Model",
]


class GPTBigCodeModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        assert self.vocab_size > 0
        self.wte = nn.Embedding(args.vocab_size, args.n_embd)
        self.wpe = nn.Embedding(args.n_positions, args.n_embd)
        self.h = [TransformerBlock(args=args) for _ in range(args.n_layer)]
        self.ln_f = nn.LayerNorm(args.n_embd, eps=args.layer_norm_epsilon)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        B, L = inputs.shape

        hidden_states = self.wte(inputs)

        if cache is None:
            cache = [None] * len(self.h)
            position_ids = mx.array(np.arange(L))
        else:
            position_ids = mx.array(np.arange(cache[0].offset, cache[0].offset + L))

        mask = create_attention_mask(hidden_states, cache[0])

        hidden_states += self.wpe(position_ids)

        for layer, c in zip(self.h, cache):
            hidden_states = layer(hidden_states, mask, cache=c)

        return self.ln_f(hidden_states)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.transformer = GPTBigCodeModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out = self.transformer(inputs, cache)
        if self.args.tie_word_embeddings:
            out = self.transformer.wte.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    @property
    def layers(self):
        return self.transformer.h
