# Copyright © 2025 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.layers.recurrent import create_ssm_mask
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.granitemoehybrid import checkpoint
from mlx_lm.models.granitemoehybrid.config import ModelArgs
from mlx_lm.models.granitemoehybrid.layers import GraniteMoeHybridLayer

__all__ = [
    "GraniteMoeHybridModel",
    "Model",
]


class GraniteMoeHybridModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            GraniteMoeHybridLayer(args, layer_type) for layer_type in args.layer_types
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.embedding_multiplier = args.embedding_multiplier

        # Handle hybrid vs non-hybrid mode
        self.fa_idx = (
            args.layer_types.index("attention")
            if "attention" in args.layer_types
            else None
        )
        self.ssm_idx = (
            args.layer_types.index("mamba") if "mamba" in args.layer_types else None
        )

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        hidden_states = self.embed_tokens(inputs) * self.embedding_multiplier

        if cache is None:
            cache = [None] * len(self.layers)

        # Create masks based on what layer types exist
        attn_mask = None
        mamba_mask = None

        if self.fa_idx is not None:
            attn_mask = create_attention_mask(hidden_states, cache[self.fa_idx])
        if self.ssm_idx is not None:
            mamba_mask = create_ssm_mask(hidden_states, cache[self.ssm_idx])

        for layer, c in zip(self.layers, cache):
            mask = attn_mask if layer.layer_type == "attention" else mamba_mask
            hidden_states = layer(hidden_states, mask=mask, cache=c)

        return self.norm(hidden_states)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = GraniteMoeHybridModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.logits_scaling = args.logits_scaling

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        out = self.model(inputs, cache=cache)

        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)

        return out / self.logits_scaling

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if self.args.use_moe and path.endswith("router.layer"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.layer_type == "mamba":
                caches.append(ArraysCache(size=2))
            elif layer.layer_type == "attention":
                caches.append(KVCache())
        return caches
