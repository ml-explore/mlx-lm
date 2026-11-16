# Copyright © 2026 Apple Inc.

from typing import Any, Optional

from mlx import core as mx
from mlx import nn

from mlx_lm.layers.masks import create_attention_mask
from mlx_lm.models.gptj import checkpoint
from mlx_lm.models.gptj.config import ModelArgs
from mlx_lm.models.gptj.layers import GPTJBlock

__all__ = [
    "GPTJModel",
    "Model",
]


class GPTJModel(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_dim = args.n_embd
        self.vocab_size = args.vocab_size
        self.wte = nn.Embedding(self.vocab_size, self.embed_dim)

        self.h = [GPTJBlock(args=args) for _ in range(args.n_layer)]
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=args.layer_norm_epsilon)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:

        batch_size, input_length = inputs.shape

        if cache is None:
            cache = [None] * len(self.h)

        hidden_states = self.wte(inputs)

        mask = create_attention_mask(hidden_states, cache[0])

        for c, block in zip(cache, self.h):
            hidden_states = block(inputs=hidden_states, mask=mask, cache=c)

        return self.ln_f(hidden_states)


class Model(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = GPTJModel(args=args)
        self.lm_head = nn.Linear(args.n_embd, args.vocab_size, bias=True)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:

        hidden_states = self.model(inputs=inputs, cache=cache)
        return self.lm_head(hidden_states)

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    @property
    def layers(self):
        return self.model.h
