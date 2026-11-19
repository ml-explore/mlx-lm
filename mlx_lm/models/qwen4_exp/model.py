# Copyright © 2026 Apple Inc.

from collections.abc import Callable
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.layers.recurrent import create_ssm_mask
from mlx_lm.models.cache import ArraysCache, CacheList, KVCache
from mlx_lm.models.qwen4_exp import checkpoint
from mlx_lm.models.qwen4_exp.config import ModelArgs, TextArgs
from mlx_lm.models.qwen4_exp.layers import DecoderLayer, GatedResidual

__all__ = [
    "Model",
    "Qwen4ExpModel",
]


class Qwen4ExpModel(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.hc = args.hc_count
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]
        # No final norm: the mixer normalizes the streams before it merges them.
        self.hyper_connection_mixer = GatedResidual(args, use_combine=False)
        self.first_attention = args.layer_types.index("full_attention")
        self.first_linear = args.layer_types.index("linear_attention")

    def __call__(
        self,
        ids: mx.array,
        cache: list[Any] | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        h = self.embed_tokens(ids) if input_embeddings is None else input_embeddings
        if cache is None:
            cache = [None] * len(self.layers)

        # Each kind of layer shares the mask of its first layer.
        attn_cache = cache[self.first_attention]
        mask = create_attention_mask(
            h, cache=attn_cache[0] if attn_cache is not None else None
        )
        # The deltanet and PLE must skip the padding.
        conv_mask = create_ssm_mask(h, cache=cache[self.first_linear])

        h = mx.tile(h, (1, 1, self.hc))
        for layer, c in zip(self.layers, cache, strict=True):
            h = layer(h, mask=mask, conv_mask=conv_mask, cache=c, ids=ids)
        return self.hyper_connection_mixer(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen4ExpModel(args.text)
        if not args.text.tie_word_embeddings:
            self.lm_head = nn.Linear(
                args.text.hidden_size, args.text.vocab_size, bias=False
            )

    def __call__(
        self,
        inputs: mx.array,
        cache: list[Any] | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        out = self.model(inputs, cache, input_embeddings)
        if self.args.text.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        return checkpoint.sanitize(self, weights)

    @property
    def quant_predicate(self) -> Callable[[str, nn.Module], bool]:
        def predicate(path: str, _: nn.Module) -> bool:
            # The MoE router is not quantized.
            return not path.endswith("mlp.gate")

        return predicate

    @property
    def layers(self) -> list[DecoderLayer]:
        return self.model.layers

    def make_cache(self) -> list[Any]:
        caches = []
        for layer in self.layers:
            if layer.layer_type == "full_attention":
                caches.append(CacheList(KVCache(), KVCache()))
            else:
                # The deltanet conv and recurrent state; a PLE layer adds its
                # conv state and n-gram context.
                caches.append(ArraysCache(2 if layer.ple is None else 4))
        return caches
