# Copyright © 2025 Apple Inc.

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.plamo2 import checkpoint
from mlx_lm.models.plamo2.config import ModelArgs
from mlx_lm.models.plamo2.layers import PlamoDecoder, RMSNorm

__all__ = [
    "Model",
    "PlamoModel",
]


class PlamoModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()

        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = PlamoDecoder(config)  # type: ignore
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        batch_size, seq_length = inputs.shape

        h = self.embed_tokens(inputs)

        out = self.layers(
            h,
            cache,
        )

        return self.norm(out)


class Model(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = PlamoModel(config)

        self.vocab_size = config.vocab_size

        if not config.tie_word_embeddings:
            self.lm_head: nn.Module = nn.Linear(
                config.hidden_size, self.vocab_size, bias=False
            )

    def __call__(self, inputs: mx.array, cache=None) -> mx.array:
        outputs = self.model(
            inputs=inputs,
            cache=cache,
        )
        if self.config.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(outputs)
        else:
            logits = self.lm_head(outputs)

        return logits

    def sanitize(self, weights: dict[Any, Any]) -> dict[Any, Any]:
        return checkpoint.sanitize(self, weights)

    @property
    def layers(self):
        return self.model.layers.layers

    def make_cache(self):
        # TODO use RotatingKVCache is not full_attn
        # full_attn = self.layer_idx in self.config.full_attention_idx
        return [ArraysCache(size=2) if l.is_mamba else KVCache() for l in self.layers]
