# Copyright © 2026 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.masks import create_attention_mask
from mlx_lm.models.cache import ArraysCache, CacheList, KVCache
from mlx_lm.models.longcat_flash.layers import LongcatFlashDecoderLayer
from mlx_lm.models.longcat_flash.model import Model as LongcatFlashLM
from mlx_lm.models.longcat_flash_ngram import checkpoint
from mlx_lm.models.longcat_flash_ngram.config import ModelArgs
from mlx_lm.models.longcat_flash_ngram.layers import NgramEmbedding

__all__ = [
    "LongcatFlashNgramModel",
    "Model",
]


class LongcatFlashNgramModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_layers = args.num_layers
        self.ngram_embeddings = NgramEmbedding(args)
        self.layers = [LongcatFlashDecoderLayer(args) for _ in range(args.num_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, args.rms_norm_eps)

    def __call__(
        self,
        input_ids: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if cache is None:
            cache = [None] + [(None, None)] * self.num_layers

        h = self.ngram_embeddings(input_ids, cache=cache[0])

        mask = create_attention_mask(h, cache[1][0], return_array=True)

        for layer, c in zip(self.layers, cache[1:]):
            h = layer(h, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LongcatFlashNgramModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ):
        out = self.model(inputs, cache)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers

    @property
    def quant_predicate(self):
        return LongcatFlashLM.quant_predicate.fget(self)

    @property
    def cast_predicate(self):
        return LongcatFlashLM.cast_predicate.fget(self)

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    def make_cache(self):
        return [ArraysCache(size=1)] + [
            CacheList(KVCache(), KVCache()) for _ in self.model.layers
        ]

    def shard(self, group: Optional[mx.distributed.Group] = None):
        LongcatFlashLM.shard(self, group)
