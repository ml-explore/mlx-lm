# Copyright © 2025 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.layers.recurrent import create_ssm_mask
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.jamba import checkpoint
from mlx_lm.models.jamba.config import ModelArgs
from mlx_lm.models.jamba.layers import JambaDecoderLayer

__all__ = [
    "JambaModel",
    "Model",
]


class JambaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)

        self.layers = [
            JambaDecoderLayer(args, t, idx)
            for idx, t in enumerate(args.layers_block_type)
        ]
        self.final_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.attn_idx = args.layers_block_type.index("attention")
        self.ssm_idx = args.layers_block_type.index("mamba")

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        attn_mask = create_attention_mask(h, cache[self.attn_idx])
        ssm_mask = create_ssm_mask(h, cache[self.ssm_idx])

        for layer, c in zip(self.layers, cache):
            mask = attn_mask if layer.is_attn else ssm_mask
            h = layer(h, mask=mask, cache=c)

        return self.final_layernorm(h)


class Model(nn.Module):
    model_args = ModelArgs

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model_type = args.model_type
        self.args = args
        self.model = JambaModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("router"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for layer in self.model.layers:
            if layer.is_attn:
                caches.append(KVCache())
            else:
                caches.append(ArraysCache(size=2))
        return caches
