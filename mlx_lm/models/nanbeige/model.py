# Copyright © 2026 Apple Inc.

from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.models.cache import KVCache
from mlx_lm.models.nanbeige import checkpoint
from mlx_lm.models.nanbeige.config import ModelArgs
from mlx_lm.models.nanbeige.layers import TransformerBlock

__all__ = [
    "Model",
    "NanbeigeModel",
]


class NanbeigeModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.num_loops = args.effective_num_loops
        self.skip_loop_final_norm = args.skip_loop_final_norm
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)

        n_layers = len(self.layers)
        if cache is None:
            cache = [None] * (self.num_loops * n_layers)

        mask = create_attention_mask(h, cache[0])

        # The layer stack is applied num_loops times with shared weights.
        # Each loop has its own KV cache entries and, unless
        # skip_loop_final_norm is set, ends with the final RMSNorm.
        for loop_idx in range(self.num_loops):
            loop_cache = cache[loop_idx * n_layers : (loop_idx + 1) * n_layers]
            for layer, c in zip(self.layers, loop_cache):
                h = layer(h, mask, c)
            if not self.skip_loop_final_norm:
                h = self.norm(h)

        if self.skip_loop_final_norm:
            h = self.norm(h)
        return h


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = NanbeigeModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ) -> mx.array:
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
        return [KVCache() for _ in range(self.model.num_loops * len(self.model.layers))]
