# Copyright © 2025 Apple Inc.

from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.masks import create_attention_mask
from mlx_lm.models.cache import KVCache
from mlx_lm.models.nemotron_nas import checkpoint
from mlx_lm.models.nemotron_nas.config import ModelArgs
from mlx_lm.models.nemotron_nas.layers import TransformerBlock

__all__ = [
    "Model",
    "NemotronNASModel",
]


class NemotronNASModel(nn.Module):
    """The core Nemotron-NAS style transformer model."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.num_attn_layers = sum(
            1 for layer in self.layers if layer.self_attn is not None
        )

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
    ):
        h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * self.num_attn_layers

        mask = create_attention_mask(h, cache[0])

        cache_idx = 0
        for layer in self.layers:
            if layer.self_attn is not None:
                c = cache[cache_idx]
                cache_idx += 1
            else:
                c = None
            h = layer(h, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = NemotronNASModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        else:
            self.lm_head = None

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out = self.model(inputs, cache=cache)
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
        return [KVCache() for layer in self.layers if layer.self_attn is not None]
