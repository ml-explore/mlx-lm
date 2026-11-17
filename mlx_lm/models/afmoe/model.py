# Copyright © 2024 Apple Inc.

import math

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.models.afmoe import checkpoint
from mlx_lm.models.afmoe.config import ModelArgs
from mlx_lm.models.afmoe.layers import DecoderLayer
from mlx_lm.models.cache import KVCache, RotatingKVCache

__all__ = [
    "AfmoeModel",
    "Model",
]


class AfmoeModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.layer_types = args.layer_types
        self.sliding_window = args.sliding_window
        self.mup_enabled = args.mup_enabled
        self.hidden_size = args.hidden_size

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(
                args=args, layer_idx=idx, use_sliding=layer_type == "sliding_attention"
            )
            for idx, layer_type in enumerate(self.layer_types)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        self.fa_idx = self.layer_types.index("full_attention")
        self.swa_idx = None
        for idx, layer in enumerate(self.layers):
            if layer.use_sliding:
                self.swa_idx = idx
                break

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        h = self.embed_tokens(inputs)

        if self.mup_enabled:
            h = h * math.sqrt(self.hidden_size)

        if cache is None:
            cache = [None] * len(self.layers)

        fa_mask = create_attention_mask(h, cache[self.fa_idx])
        swa_mask = None
        if self.swa_idx is not None:
            swa_mask = create_attention_mask(
                h, cache[self.swa_idx], window_size=self.sliding_window
            )

        for layer, c in zip(self.layers, cache):
            mask = swa_mask if layer.use_sliding else fa_mask
            h = layer(h, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = AfmoeModel(args)

        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            (
                RotatingKVCache(max_size=self.model.sliding_window)
                if layer.use_sliding
                else KVCache()
            )
            for layer in self.layers
        ]

    @property
    def cast_predicate(self):
        def predicate(k):
            return "expert_bias" not in k

        return predicate

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if "router.gate" in path:
                return {"group_size": 64, "bits": 8}
            return True

        return predicate
