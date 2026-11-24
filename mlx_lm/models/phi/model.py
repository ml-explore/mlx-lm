# Copyright © 2023 Apple Inc.

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.models.phi.config import ModelArgs
from mlx_lm.models.phi.layers import PhiDecoderLayer

__all__ = [
    "Model",
    "PhiModel",
]


class PhiModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [PhiDecoderLayer(config) for i in range(config.num_hidden_layers)]
        self.final_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def __call__(self, x, cache):
        x = self.embed_tokens(x)

        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(x, cache[0])

        for layer, c in zip(self.layers, cache):
            x = layer(x, mask, c)
        return self.final_layernorm(x)


class Model(nn.Module):
    model_args = ModelArgs

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.model_type = config.model_type
        self.model = PhiModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True)
        self.args = config

    def __call__(
        self,
        x: mx.array,
        cache=None,
    ) -> mx.array:
        y = self.model(x, cache)
        return self.lm_head(y)

    @property
    def layers(self):
        return self.model.layers
