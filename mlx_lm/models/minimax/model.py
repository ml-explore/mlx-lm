# Copyright © 2025 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear

from mlx_lm.layers.masks import create_attention_mask
from mlx_lm.models.minimax import checkpoint
from mlx_lm.models.minimax.config import ModelArgs
from mlx_lm.models.minimax.layers import MiniMaxDecoderLayer, ShardedRMSNorm

__all__ = [
    "MiniMaxModel",
    "Model",
]


class MiniMaxModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)

        self.layers = [
            MiniMaxDecoderLayer(args=args) for _ in range(args.num_hidden_layers)
        ]

        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = MiniMaxModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ):
        out = self.model(inputs=inputs, mask=mask, cache=cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()
        for layer in self.model.layers:
            # Shard the self attention
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
            if layer.self_attn.use_qk_norm:
                layer.self_attn.q_norm = ShardedRMSNorm.from_rms_norm(
                    layer.self_attn.q_norm, group=group
                )
                layer.self_attn.k_norm = ShardedRMSNorm.from_rms_norm(
                    layer.self_attn.k_norm, group=group
                )

            layer.self_attn.num_attention_heads //= N
            layer.self_attn.num_key_value_heads //= N

            # Shard the MLP
            shard_inplace(
                layer.block_sparse_moe.switch_mlp.gate_proj,
                "all-to-sharded",
                group=group,
            )
            shard_inplace(
                layer.block_sparse_moe.switch_mlp.down_proj,
                "sharded-to-all",
                group=group,
            )
            shard_inplace(
                layer.block_sparse_moe.switch_mlp.up_proj,
                "all-to-sharded",
                group=group,
            )
            layer.block_sparse_moe.sharding_group = group

    @property
    def layers(self):
        return self.model.layers

    @property
    def cast_predicate(self):
        def predicate(k):
            return "e_score_correction_bias" not in k

        return predicate

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("block_sparse_moe.gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate
