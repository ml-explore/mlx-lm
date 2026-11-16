# Copyright © 2026 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.masks import create_attention_mask, create_ssm_mask
from mlx_lm.layers.pipeline import PipelineMixin
from mlx_lm.models.bailing_hybrid import checkpoint
from mlx_lm.models.bailing_hybrid.config import ModelArgs
from mlx_lm.models.bailing_hybrid.layers import DecoderLayer
from mlx_lm.models.cache import ArraysCache, KVCache

__all__ = [
    "LanguageModel",
    "Model",
]


class LanguageModel(PipelineMixin, nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.layer_group_size = args.layer_group_size
        self._set_mask_indices()

    def _set_mask_indices(self):
        # One layer of each kind, to size the two masks and read the offset.
        layers = self.pipeline_layers
        self._attn_idx = next((i for i, l in enumerate(layers) if l.is_global), None)
        self._gla_idx = next((i for i, l in enumerate(layers) if not l.is_global), None)

    def pipeline(self, group, split=None):
        if split is None:
            # Every rank needs an MLA layer for the position offset.
            size = group.size()
            n_groups, tail = divmod(len(self.layers), self.layer_group_size)
            base, extra = divmod(n_groups, size)
            split = [
                (base + (1 if r < extra else 0)) * self.layer_group_size
                for r in range(size)
            ]
            split[0] += tail
        super().pipeline(group, split=split)
        self._set_mask_indices()
        if self._attn_idx is None:
            raise ValueError(f"split {split} leaves a rank without an MLA layer")

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.word_embeddings(inputs)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * len(self.pipeline_layers)

        attn_cache = cache[self._attn_idx] if self._attn_idx is not None else None
        attn_mask = create_attention_mask(h, attn_cache, return_array=True)
        gla_mask = (
            create_ssm_mask(h, cache[self._gla_idx])
            if self._gla_idx is not None
            else None
        )
        offset = attn_cache.offset if attn_cache is not None else 0

        # Receive from the previous process in the pipeline
        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        for layer, c in zip(self.pipeline_layers, cache):
            mask = attn_mask if layer.is_global else gla_mask
            h = layer(h, mask, c, offset=offset)

        # Send to the next process in the pipeline
        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            if cache[-1] is not None:
                if hasattr(cache[-1], "keys"):
                    cache[-1].keys = mx.depends(cache[-1].keys, h)
                else:
                    cache[-1][0] = mx.depends(cache[-1][0], h)

        # Broadcast h while keeping it in the graph
        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LanguageModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            return self.model.word_embeddings.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("mlp.gate.gate_proj"):
                return {"group_size": 64, "bits": 8}
            # The absorbed MLA matrices stay unquantized, as in bailing_moe_v3.
            if path.endswith(("attention.embed_q", "attention.unembed_out")):
                return False
            return True

        return predicate

    @property
    def cast_predicate(self):
        def predicate(k):
            return "expert_bias" not in k

        return predicate

    @property
    def layers(self):
        return self.model.pipeline_layers

    def make_cache(self):
        caches = []
        for l in self.layers:
            if l.is_global:
                caches.append(KVCache())
            else:
                caches.append(ArraysCache(size=1))
        return caches
