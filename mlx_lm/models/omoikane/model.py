# Copyright © 2026 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.layers.recurrent import create_ssm_mask
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.omoikane import checkpoint
from mlx_lm.models.omoikane.config import ModelArgs
from mlx_lm.models.omoikane.layers import OmoikaneDecoderLayer

__all__ = [
    "Model",
    "OmoikaneModel",
]

# Kept in float32 by mlx_lm.convert: the router scores in float32, and the
# gated delta rule's decay parameters are exponentiated.
_FLOAT32_PARAMS = (
    "mlp.gate.weight",
    "e_score_correction_bias",
    "A_log",
    "dt_bias",
)


class OmoikaneModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            OmoikaneDecoderLayer(args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ssm_idx = next((i for i, l in enumerate(self.layers) if l.is_linear), None)
        self.fa_idx = next(
            (i for i, l in enumerate(self.layers) if not l.is_linear), None
        )

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        fa_mask = None
        if self.fa_idx is not None:
            # MLA adds the RoPE scores through the mask, so it needs an array.
            fa_mask = create_attention_mask(h, cache[self.fa_idx], return_array=True)
        ssm_mask = None
        if self.ssm_idx is not None:
            ssm_mask = create_ssm_mask(h, cache[self.ssm_idx])

        for layer, c in zip(self.layers, cache):
            h = layer(h, ssm_mask if layer.is_linear else fa_mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = OmoikaneModel(args)
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
        return checkpoint.sanitize(self, weights)

    @property
    def cast_predicate(self):
        def predicate(k):
            return not k.endswith(_FLOAT32_PARAMS)

        return predicate

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [ArraysCache(size=2) if l.is_linear else KVCache() for l in self.layers]
