# Copyright © 2025 Apple Inc.

from typing import Any, List

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.masks import create_attention_mask
from mlx_lm.models.baichuan_m1 import checkpoint
from mlx_lm.models.baichuan_m1.config import ModelArgs
from mlx_lm.models.baichuan_m1.layers import DecoderLayer
from mlx_lm.models.cache import ArraysCache, CacheList, KVCache, RotatingKVCache

__all__ = [
    "BaichuanModel",
    "Model",
]


class BaichuanModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.sliding_window = config.sliding_window
        self.first_swa_idx = None
        if config.sliding_window_layers:
            self.first_swa_idx = config.sliding_window_layers[0]

        self.first_global_idx = None
        self.swa_layers = set(config.sliding_window_layers)
        for i in range(config.num_hidden_layers):
            if i in self.swa_layers:
                continue
            self.first_global_idx = i
            break

    def __call__(self, inputs: mx.array, cache: Any = None) -> mx.array:
        x = self.embed_tokens(inputs)

        if cache is None:
            cache = [(None, None)] * len(self.layers)

        if self.first_global_idx is None:
            c_global = None
        else:
            c_global = cache[self.first_global_idx][1]

        if self.first_swa_idx is None:
            c_swa = None
        else:
            c_swa = cache[self.first_swa_idx][1]

        global_mask = create_attention_mask(x, c_global)
        swa_mask = create_attention_mask(x, c_swa, window_size=self.sliding_window)

        for l, (layer, c) in enumerate(zip(self.layers, cache)):
            mask = swa_mask if l in self.swa_layers else global_mask
            x = layer(x, mask, c)
        return self.norm(x)


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = BaichuanModel(config)
        self.tie_word_embeddings = config.tie_word_embeddings
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> List[Any]:
        caches = []
        for i in range(len(self.model.layers)):
            is_swa = i in self.config.sliding_window_layers
            conv_cache = ArraysCache(size=2)
            if is_swa:
                kv_cache = RotatingKVCache(max_size=self.config.sliding_window)
            else:
                kv_cache = KVCache()
            caches.append(CacheList(conv_cache, kv_cache))
        return caches

    def sanitize(self, weights: dict) -> dict:
        return checkpoint.sanitize(self, weights)

    def __call__(self, inputs: mx.array, cache: Any = None) -> mx.array:
        outputs = self.model(inputs, cache)
        return self.lm_head(outputs)

    @property
    def layers(self) -> List[nn.Module]:
        return self.model.layers
