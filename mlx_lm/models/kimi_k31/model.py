# Copyright © 2026 Apple Inc.

from typing import Any, List

from mlx_lm.models.cache import ArraysCache, CacheList, KVCache
from mlx_lm.models.kimi_k3.model import KimiK3Model, KimiK3TextModel
from mlx_lm.models.kimi_k3.model import Model as KimiK3
from mlx_lm.models.kimi_k31.config import ModelArgs
from mlx_lm.models.kimi_k31.layers import KimiK31DecoderLayer

__all__ = [
    "KimiK31Model",
    "KimiK31TextModel",
    "Model",
]


class KimiK31TextModel(KimiK3TextModel):
    decoder_layer = KimiK31DecoderLayer


class KimiK31Model(KimiK3Model):
    def __init__(self, args: ModelArgs):
        super().__init__(args)
        self.language_model = KimiK31TextModel(args.text)


class Model(KimiK3):
    """Kimi K3 with DeepSeek sparse attention on the full attention layers.

    shard() is the K3 one: the indexer is not sharded, so each rank keeps all of it.
    """

    def __init__(self, args: ModelArgs):
        super().__init__(args)
        self.model = KimiK31Model(args)

    def make_cache(self):
        caches: List[Any] = []
        for layer in self.layers:
            if layer.is_linear:
                caches.append(ArraysCache(size=2))
            elif layer.is_dsa:
                caches.append(CacheList(KVCache(), KVCache()))
            else:
                caches.append(KVCache())
        return caches
