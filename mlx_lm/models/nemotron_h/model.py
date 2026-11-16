# Copyright © 2025 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.masks import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.nemotron_h import checkpoint
from mlx_lm.models.nemotron_h.config import ModelArgs
from mlx_lm.models.nemotron_h.layers import NemotronHBlock

__all__ = [
    "Model",
    "NemotronHModel",
]


class NemotronHModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            NemotronHBlock(args, block_type)
            for block_type in args.hybrid_override_pattern
        ]
        self.norm_f = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        self.fa_idx = 0
        self.ssm_idx = 0
        for b in args.hybrid_override_pattern:
            if b == "*":
                break
            elif b == "M":
                self.fa_idx += 1
        for b in args.hybrid_override_pattern:
            if b == "*":
                self.ssm_idx += 1
            elif b == "M":
                break

    def __call__(
        self,
        inputs,
        cache: Optional[Any] = None,
    ):
        hidden_states = self.embeddings(inputs)

        if cache is None:
            cache = [None] * len(self.layers)
        attn_mask = create_attention_mask(hidden_states, cache[self.fa_idx])
        ssm_mask = create_ssm_mask(hidden_states, cache[self.ssm_idx])

        cache_counter = 0
        for layer in self.layers:
            if layer.block_type == "M" or layer.block_type == "*":
                c = cache[cache_counter]
                cache_counter += 1
            else:
                c = None

            if layer.block_type == "*":
                mask = attn_mask
            else:
                mask = ssm_mask
            hidden_states = layer(hidden_states, mask=mask, cache=c)

        return self.norm_f(hidden_states)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.backbone = NemotronHModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.model_type = args.model_type

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ):
        out = self.backbone(inputs, cache=cache)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.backbone.layers

    def make_cache(self):
        caches = []
        for l in self.layers:
            if l.block_type == "M":
                caches.append(ArraysCache(size=2))
            elif l.block_type == "*":
                caches.append(KVCache())
        return caches

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    @property
    def cast_predicate(self):
        def predicate(k):
            return "e_score_correction_bias" not in k and "A_log" not in k

        return predicate
