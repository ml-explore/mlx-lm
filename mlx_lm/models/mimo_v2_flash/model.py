# Copyright © 2024 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.mimo_v2_flash import checkpoint
from mlx_lm.models.mimo_v2_flash.config import ModelArgs
from mlx_lm.models.mimo_v2_flash.layers import DecoderLayer

__all__ = [
    "LanguageModel",
    "Model",
]


class LanguageModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            DecoderLayer(
                config,
                is_moe=config.moe_layer_freq[idx] == 1,
                is_sliding_window=config.hybrid_layer_pattern[idx] == 1,
            )
            for idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.swa_idx = config.hybrid_layer_pattern.index(1)
        self.ga_idx = config.hybrid_layer_pattern.index(0)
        self.sliding_window_size = config.sliding_window_size

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.embed_tokens(x)

        if cache is None:
            cache = [None] * len(self.layers)

        full_mask = create_attention_mask(x, cache[self.ga_idx])
        swa_mask = create_attention_mask(
            x, cache[self.swa_idx], window_size=self.sliding_window_size
        )

        for l, c in zip(self.layers, cache):
            mask = swa_mask if l.is_sliding_window else full_mask
            h = l(h, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    model_args = ModelArgs

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = LanguageModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ):
        out = self.model(inputs, cache)
        return self.lm_head(out)

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    @property
    def cast_predicate(self):
        def predicate(k):
            return "e_score_correction_bias" not in k

        return predicate

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for l in self.layers:
            if l.is_sliding_window:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window_size))
            else:
                caches.append(KVCache())
        return caches
