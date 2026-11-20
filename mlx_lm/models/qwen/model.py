# Copyright © 2023 Apple Inc.

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.models.qwen.config import ModelArgs
from mlx_lm.models.qwen.layers import TransformerBlock

__all__ = [
    "Model",
    "QwenModel",
]


class QwenModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.wte = nn.Embedding(args.vocab_size, args.hidden_size)
        self.h = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.ln_f = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)

    def __call__(self, inputs, cache=None):
        x = self.wte(inputs)

        if cache is None:
            cache = [None] * len(self.h)
        mask = create_attention_mask(x, cache[0])

        for layer, c in zip(self.h, cache):
            x = layer(x, mask, c)

        return self.ln_f(x)


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.model_type = config.model_type
        self.transformer = QwenModel(config)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=not config.no_bias
        )
        self.args = config

    def __call__(
        self,
        x: mx.array,
        cache=None,
    ) -> mx.array:
        y = self.transformer(x, cache)
        return self.lm_head(y)

    @property
    def layers(self):
        return self.transformer.h
