# Copyright © 2026 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.exaone_moe import checkpoint
from mlx_lm.models.exaone_moe.config import ModelArgs
from mlx_lm.models.exaone_moe.layers import MLP, DecoderLayer

__all__ = [
    "ExaoneMoEModel",
    "Model",
]


class ExaoneMoEModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, idx) for idx in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        self.swa_idx = None
        self.ga_idx = None
        for i, layer in enumerate(self.layers):
            if layer.is_sliding_window and self.swa_idx is None:
                self.swa_idx = i
            if not layer.is_sliding_window and self.ga_idx is None:
                self.ga_idx = i

        self.window_size = args.sliding_window

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        global_mask = create_attention_mask(
            h, cache[self.ga_idx] if self.ga_idx is not None else cache[0]
        )
        swa_mask = create_attention_mask(
            h,
            cache[self.swa_idx] if self.swa_idx is not None else cache[0],
            window_size=self.window_size,
        )

        for layer, c in zip(self.layers, cache):
            mask = swa_mask if layer.is_sliding_window else global_mask
            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = ExaoneMoEModel(args)

        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ):
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    @property
    def cast_predicate(self):
        def predicate(k):
            return "e_score_correction_bias" not in k

        return predicate

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()

        for layer in self.model.layers:
            layer.self_attn.q_proj = shard_linear(
                layer.self_attn.q_proj, "all-to-sharded", group=group
            )
            layer.self_attn.k_proj = shard_linear(
                layer.self_attn.k_proj, "all-to-sharded", group=group
            )
            layer.self_attn.v_proj = shard_linear(
                layer.self_attn.v_proj, "all-to-sharded", group=group
            )
            layer.self_attn.o_proj = shard_linear(
                layer.self_attn.o_proj, "sharded-to-all", group=group
            )
            layer.self_attn.n_heads //= N
            layer.self_attn.n_kv_heads //= N

            if isinstance(layer.mlp, MLP):
                layer.mlp.gate_proj = shard_linear(
                    layer.mlp.gate_proj, "all-to-sharded", group=group
                )
                layer.mlp.down_proj = shard_linear(
                    layer.mlp.down_proj, "sharded-to-all", group=group
                )
                layer.mlp.up_proj = shard_linear(
                    layer.mlp.up_proj, "all-to-sharded", group=group
                )
            else:
                layer.mlp.sharding_group = group
                if layer.mlp.shared_experts is not None:
                    shard_inplace(
                        layer.mlp.shared_experts.gate_proj,
                        "all-to-sharded",
                        group=group,
                    )
                    shard_inplace(
                        layer.mlp.shared_experts.down_proj,
                        "sharded-to-all",
                        group=group,
                    )
                    shard_inplace(
                        layer.mlp.shared_experts.up_proj, "all-to-sharded", group=group
                    )
                shard_inplace(
                    layer.mlp.switch_mlp.gate_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.down_proj, "sharded-to-all", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.up_proj, "all-to-sharded", group=group
                )

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.is_sliding_window:
                caches.append(
                    RotatingKVCache(max_size=self.args.sliding_window, keep=0)
                )
            else:
                caches.append(KVCache())
        return caches
