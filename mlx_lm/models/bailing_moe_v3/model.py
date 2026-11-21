# Copyright © 2026 Apple Inc.

from typing import Any

import mlx.core as mx
from mlx import nn

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.layers.recurrent import create_ssm_mask
from mlx_lm.models.bailing_moe_v3 import checkpoint
from mlx_lm.models.bailing_moe_v3.config import ModelArgs
from mlx_lm.models.bailing_moe_v3.layers import BailingDecoderLayer
from mlx_lm.models.cache import ArraysCache, KVCache

__all__ = [
    "BailingModel",
    "Model",
]


class BailingModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            BailingDecoderLayer(args, layer_index)
            for layer_index in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.linear_index = next(
            (index for index, layer in enumerate(self.layers) if layer.is_linear), None
        )
        self.attention_index = next(
            (index for index, layer in enumerate(self.layers) if not layer.is_linear),
            None,
        )

    def __call__(self, inputs: mx.array, cache: Any | None = None) -> mx.array:
        hidden = self.word_embeddings(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        attention_mask = (
            create_attention_mask(
                hidden, cache[self.attention_index], return_array=True
            )
            if self.attention_index is not None
            else None
        )
        linear_mask = (
            create_ssm_mask(hidden, cache[self.linear_index])
            if self.linear_index is not None
            else None
        )
        for layer, layer_cache in zip(self.layers, cache):
            mask = linear_mask if layer.is_linear else attention_mask
            hidden = layer(hidden, mask=mask, cache=layer_cache)
        return self.norm(hidden)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = BailingModel(args)
        self.lm_head = (
            None
            if args.tie_word_embeddings
            else nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        )

    def __call__(self, inputs: mx.array, cache: Any | None = None) -> mx.array:
        hidden = self.model(inputs, cache)
        if self.lm_head is None:
            return self.model.word_embeddings.as_linear(hidden)
        return self.lm_head(hidden)

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path in ("model.word_embeddings", "lm_head"):
                return False
            if path.endswith(
                (
                    "attention.q_a_proj",
                    "attention.q_b_proj",
                    "attention.kv_a_proj_with_mqa",
                    "attention.embed_q",
                    "attention.unembed_out",
                    "attention.b_proj",
                )
            ):
                return False
            if path.endswith(("attention.q_proj", "attention.g_proj")):
                layer_index = int(path.split(".")[2])
                return self.layers[layer_index].is_linear
            return True

        return predicate

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            ArraysCache(size=4) if layer.is_linear else KVCache()
            for layer in self.layers
        ]
