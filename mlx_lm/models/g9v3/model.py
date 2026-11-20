# Copyright © 2026 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.models.g9v3 import checkpoint
from mlx_lm.models.g9v3.config import ModelArgs
from mlx_lm.models.g9v3.layers import G9v3DecodeLayer

__all__ = [
    "G9v3Model",
    "Model",
]


class G9v3Model(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()

        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(self.vocab_size, dims=config.hidden_size)
        self.layers = [
            G9v3DecodeLayer(layer_idx=idx, config=config)
            for idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(dims=config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self, hidden_states: mx.array, cache: Optional[Any] = None
    ) -> mx.array:

        hidden_states = self.embed_tokens(hidden_states)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(hidden_states, cache[0])

        for c, block in zip(cache, self.layers):
            hidden_states = block(hidden_states=hidden_states, cache=c, mask=mask)

        return self.norm(hidden_states)


class Model(nn.Module):

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = G9v3Model(config=config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self, hidden_states: mx.array, cache: Optional[Any] = None
    ) -> mx.array:

        hidden_states = self.model(hidden_states=hidden_states, cache=cache)
        return self.lm_head(hidden_states)

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    @property
    def layers(self):
        return self.model.layers
