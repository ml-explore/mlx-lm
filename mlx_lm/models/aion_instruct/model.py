# Copyright © 2026 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.models.aion_instruct.config import ModelArgs
from mlx_lm.models.aion_instruct.layers import DecoderLayer
from mlx_lm.models.cache import KVCache, RotatingKVCache

__all__ = [
    "AionInstructModel",
    "Model",
]


class AionInstructModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.sliding_window = args.sliding_window
        self.layer_types = args.layer_types

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        # A config can have only one of the two layer types.
        self.swa_idx = (
            args.layer_types.index("sliding_attention")
            if "sliding_attention" in args.layer_types
            else None
        )
        self.ga_idx = (
            args.layer_types.index("full_attention")
            if "full_attention" in args.layer_types
            else None
        )

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        full_mask = None
        sliding_window_mask = None
        if self.ga_idx is not None:
            full_mask = create_attention_mask(h, cache[self.ga_idx])
        if self.swa_idx is not None:
            sliding_window_mask = create_attention_mask(
                h, cache[self.swa_idx], window_size=self.sliding_window
            )

        for layer, c in zip(self.layers, cache):
            mask = sliding_window_mask if layer.is_sliding else full_mask
            h = layer(h, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = AionInstructModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights):
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            (
                RotatingKVCache(max_size=self.args.sliding_window)
                if layer.is_sliding
                else KVCache()
            )
            for layer in self.layers
        ]
