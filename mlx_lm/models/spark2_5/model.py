# Copyright © 2026 Apple Inc.

from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.spark2_5 import checkpoint
from mlx_lm.models.spark2_5.config import ModelArgs
from mlx_lm.models.spark2_5.layers import TransformerBlock

__all__ = [
    "Model",
    "Spark2_5Model",
]


class Spark2_5Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embedding = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args, layer_idx)
            for layer_idx in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        self._first_full = next(
            (
                idx
                for idx, layer_type in enumerate(args.layer_types)
                if layer_type == "full_attention"
            ),
            None,
        )
        self._first_sliding = next(
            (
                idx
                for idx, layer_type in enumerate(args.layer_types)
                if layer_type == "sliding_attention"
            ),
            None,
        )

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        hidden_states = (
            input_embeddings if input_embeddings is not None else self.embedding(inputs)
        )

        if cache is None:
            cache = [None] * len(self.layers)

        full_mask = None
        if self._first_full is not None:
            full_mask = create_attention_mask(hidden_states, cache[self._first_full])
        sliding_mask = None
        if self._first_sliding is not None:
            sliding_mask = create_attention_mask(
                hidden_states,
                cache[self._first_sliding],
                window_size=self.args.sliding_window,
            )

        for layer, layer_cache in zip(self.layers, cache):
            mask = sliding_mask if layer.is_sliding else full_mask
            hidden_states = layer(hidden_states, mask, layer_cache)

        return self.norm(hidden_states)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Spark2_5Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        hidden_states = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embedding.as_linear(hidden_states)
        return self.lm_head(hidden_states)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            (
                RotatingKVCache(max_size=self.args.sliding_window, keep=0)
                if layer.is_sliding
                else KVCache()
            )
            for layer in self.layers
        ]

    @property
    def quant_predicate(self):
        # The head-gate is a tiny per-head gating projection; keeping it at
        # source precision preserves the sigmoid gate values.
        def predicate(path, _):
            return not path.endswith("self_attn.g_proj")

        return predicate

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)
