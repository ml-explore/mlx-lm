# Copyright © 2023 Apple Inc.

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.models.phi3small import checkpoint
from mlx_lm.models.phi3small.config import ModelArgs
from mlx_lm.models.phi3small.layers import TransformerBlock

__all__ = [
    "Model",
    "Phi3Model",
]


class Phi3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.mup_embedding_multiplier = args.mup_embedding_multiplier
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args, layer_idx=l)
            for l in range(args.num_hidden_layers)
        ]
        self.final_layernorm = nn.LayerNorm(
            args.hidden_size, eps=args.layer_norm_epsilon
        )

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        h = self.embed_tokens(inputs)
        if self.mup_embedding_multiplier:
            h = self.mup_embedding_multiplier * h

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(h, cache[0], return_array=True)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.final_layernorm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model_type = args.model_type
        self.model = Phi3Model(args)
        self.args = args
        self.mup_width_multiplier = args.mup_width_multiplier
        self._dummy_tokenizer_ids = mx.array(
            [100256, 100258, 100259, 100260, 100264, 100265]
            + list(range(100267, 100352))
        )

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out = self.model(inputs, cache)
        out = self.model.embed_tokens.as_linear(out)
        if self.mup_width_multiplier:
            out = out / self.mup_width_multiplier
        out[..., self._dummy_tokenizer_ids] = -float("inf")
        return out

    @property
    def layers(self):
        return self.model.layers

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)
