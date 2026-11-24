# Copyright © 2026 Apple Inc.

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache
from mlx_lm.models.gear import checkpoint
from mlx_lm.models.gear.config import ModelArgs
from mlx_lm.models.gear.layers import DecoderLayer, GearRMSNorm

__all__ = [
    "GearModel",
    "Model",
]


class GearModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.layer_types = args.layer_types
        self.sliding_window = args.sliding_window
        self.embed_scale = math.sqrt(args.hidden_size)

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]
        self.norm = GearRMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        self.fa_idx = self._first_layer_idx("full_attention")
        self.swa_idx = self._first_layer_idx("sliding_attention")
        self.conv_idx = self._first_layer_idx("conv_mixer")

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs) * self.embed_scale

        if cache is None:
            cache = [None] * len(self.layers)

        full_mask = (
            create_attention_mask(h, cache[self.fa_idx])
            if self.fa_idx is not None
            else None
        )
        sliding_mask = (
            create_attention_mask(
                h, cache[self.swa_idx], window_size=self.sliding_window
            )
            if self.swa_idx is not None
            else None
        )
        conv_mask = (
            cache[self.conv_idx].make_mask(h.shape[1])
            if self.conv_idx is not None and cache[self.conv_idx] is not None
            else None
        )

        for layer, c in zip(self.layers, cache):
            if layer.attention_type == "full_attention":
                mask = full_mask
            elif layer.attention_type == "sliding_attention":
                mask = sliding_mask
            else:
                mask = conv_mask
            h = layer(h, mask=mask, cache=c)

        return self.norm(h)

    def _first_layer_idx(self, layer_type: str) -> Optional[int]:
        try:
            return self.layer_types.index(layer_type)
        except ValueError:
            return None


class Model(nn.Module):
    model_args = ModelArgs

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = GearModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        if self.args.final_logit_softcapping:
            cap = self.args.final_logit_softcapping
            out = mx.tanh(out / cap) * cap
        return out

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.attention_type == "full_attention":
                caches.append(KVCache())
            elif layer.attention_type == "sliding_attention":
                caches.append(RotatingKVCache(max_size=self.model.sliding_window))
            else:
                caches.append(ArraysCache(size=2))
        return caches
