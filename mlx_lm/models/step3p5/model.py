# Copyright © 2026 Apple Inc.

from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.layers.common import ZeroCenteredRMSNorm
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.step3p5 import checkpoint
from mlx_lm.models.step3p5.config import ModelArgs
from mlx_lm.models.step3p5.layers import Step3p5DecoderLayer

__all__ = [
    "Model",
    "Step3p5Model",
]


class Step3p5Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_layers = args.num_hidden_layers

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Step3p5DecoderLayer(args, layer_idx)
            for layer_idx in range(args.num_hidden_layers)
        ]
        self.norm = ZeroCenteredRMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        self._swa_idx = next(
            (i for i, l in enumerate(self.layers) if l.is_sliding), None
        )
        self._full_idx = next(
            (i for i, l in enumerate(self.layers) if not l.is_sliding), None
        )

    def __call__(
        self,
        x: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        h = self.embed_tokens(x)

        if cache is None:
            cache = [None] * self.num_layers

        full_mask = None
        swa_mask = None

        if self._full_idx is not None:
            full_mask = create_attention_mask(h, cache[self._full_idx])

        if self._swa_idx is not None:
            swa_mask = create_attention_mask(
                h, cache[self._swa_idx], window_size=self.args.sliding_window
            )

        for layer, c in zip(self.layers, cache):
            mask = swa_mask if layer.is_sliding else full_mask
            h = layer(h, mask=mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Step3p5Model(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
    ):
        out = self.model(inputs, cache)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            (
                RotatingKVCache(max_size=self.args.sliding_window)
                if layer.is_sliding
                else KVCache()
            )
            for layer in self.layers
        ]

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    @property
    def cast_predicate(self):
        def predicate(k):
            return "router_bias" not in k

        return predicate

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("moe.gate"):
                return {"group_size": 64, "bits": 8}
            return True

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
            layer.self_attn.num_heads //= N
            layer.self_attn.num_kv_heads //= N

            if layer.self_attn.use_head_wise_attn_gate:
                layer.self_attn.g_proj = shard_linear(
                    layer.self_attn.g_proj, "all-to-sharded", group=group
                )

            if layer.is_moe_layer:
                # In place: the layer sums the routed and shared experts once.
                layer.sharding_group = group
                for mlp in (layer.share_expert, layer.moe):
                    shard_inplace(mlp.gate_proj, "all-to-sharded", group=group)
                    shard_inplace(mlp.up_proj, "all-to-sharded", group=group)
                    shard_inplace(mlp.down_proj, "sharded-to-all", group=group)
            else:
                layer.mlp.gate_proj = shard_linear(
                    layer.mlp.gate_proj, "all-to-sharded", group=group
                )
                layer.mlp.up_proj = shard_linear(
                    layer.mlp.up_proj, "all-to-sharded", group=group
                )
                layer.mlp.down_proj = shard_linear(
                    layer.mlp.down_proj, "sharded-to-all", group=group
                )
