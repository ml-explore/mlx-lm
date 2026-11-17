# Copyright © 2025 Apple Inc.

from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.gpt_oss import checkpoint
from mlx_lm.models.gpt_oss.config import ModelArgs
from mlx_lm.models.gpt_oss.layers import TransformerBlock

__all__ = [
    "GptOssMoeModel",
    "Model",
]


class GptOssMoeModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.norm = nn.RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.layer_types = args.layer_types or [
            "sliding_attention",
            "full_attention",
        ] * (args.num_hidden_layers // 2)
        self.layers = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.window_size = args.sliding_window
        self.swa_idx = self.layer_types.index("sliding_attention")
        self.ga_idx = self.layer_types.index("full_attention")

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        if input_embeddings is not None:
            x = input_embeddings
        else:
            x = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        full_mask = create_attention_mask(x, cache[self.ga_idx])
        swa_mask = create_attention_mask(
            x, cache[self.swa_idx], window_size=self.window_size
        )

        for layer, c, layer_type in zip(self.layers, cache, self.layer_types):
            mask = full_mask if layer_type == "full_attention" else swa_mask
            x = layer(x, mask, c)
        x = self.norm(x)
        return x


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = GptOssMoeModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None):
        return self.lm_head(self.model(inputs, cache))

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()
        R = group.rank()

        for layer in self.model.layers:
            layer.self_attn.q_proj = shard_linear(
                layer.self_attn.q_proj, sharding="all-to-sharded", group=group
            )
            layer.self_attn.k_proj = shard_linear(
                layer.self_attn.k_proj, sharding="all-to-sharded", group=group
            )
            layer.self_attn.v_proj = shard_linear(
                layer.self_attn.v_proj, sharding="all-to-sharded", group=group
            )
            layer.self_attn.o_proj = shard_linear(
                layer.self_attn.o_proj, sharding="sharded-to-all", group=group
            )

            layer.self_attn.num_attention_heads //= N
            layer.self_attn.num_key_value_heads //= N
            layer.self_attn.num_key_value_groups = (
                layer.self_attn.num_attention_heads
                // layer.self_attn.num_key_value_heads
            )

            layer.self_attn.sinks = layer.self_attn.sinks[
                layer.self_attn.num_attention_heads
                * R : layer.self_attn.num_attention_heads
                * (R + 1)
            ]

            shard_inplace(layer.mlp.experts.gate_proj, "all-to-sharded", group=group)
            shard_inplace(layer.mlp.experts.down_proj, "sharded-to-all", group=group)
            layer.mlp.experts.down_proj.bias /= N
            shard_inplace(
                layer.mlp.experts.up_proj, sharding="all-to-sharded", group=group
            )

            layer.mlp.sharding_group = group

    @property
    def layers(self):
        return self.model.layers

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("router"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    def make_cache(self):
        caches = []
        for lt in self.model.layer_types:
            if lt == "full_attention":
                caches.append(KVCache())
            else:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window))
        return caches
