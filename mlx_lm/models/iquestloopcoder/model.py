# Copyright © 2026 Apple Inc.

from functools import partial
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_linear

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.iquestloopcoder.config import ModelArgs
from mlx_lm.models.iquestloopcoder.layers import LoopGateProjection, TransformerBlock

__all__ = [
    "IQuestLoopCoderModel",
    "Model",
]


@partial(mx.compile, shapeless=True)
def _mix_attention(
    gate: mx.array, attn_global: mx.array, attn_local: mx.array
) -> mx.array:
    return gate * attn_global + (1 - gate) * attn_local


class IQuestLoopCoderModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.loop_num == 2, f"Only loop_num=2 is supported, got {args.loop_num}"
        self.args = args
        self.vocab_size = args.vocab_size
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.gate_projections = [
            LoopGateProjection(args.num_attention_heads, args.head_dim)
            for _ in range(args.num_hidden_layers)
        ]
        self.loop_num = args.loop_num
        self.loop_window_size = args.loop_window_size

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
    ):
        B, L = inputs.shape[:2]
        h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * (2 * len(self.layers))

        mask = create_attention_mask(h, cache[0])
        window_mask = create_attention_mask(
            h, cache[len(self.layers)], window_size=self.loop_window_size
        )

        loop1_kv = []
        for layer, c in zip(self.layers, cache):
            h_norm = layer.input_layernorm(h)
            offset = c.offset if c is not None else 0
            q1, k1, v1 = layer.self_attn.get_qkv(h_norm, offset)

            if c is not None:
                k1, v1 = c.update_and_fetch(k1, v1)
            loop1_kv.append((k1, v1))

            out = layer.self_attn.attention(q1, k1, v1, mask, cache=c)
            r = layer.self_attn.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))
            h = h + r
            r = layer.mlp(layer.post_attention_layernorm(h))
            h = h + r

        for layer, gate_proj, c, (k1, v1) in zip(
            self.layers, self.gate_projections, cache[len(self.layers) :], loop1_kv
        ):
            h_norm = layer.input_layernorm(h)
            offset = c.offset if c is not None else 0
            q2, k2, v2 = layer.self_attn.get_qkv(h_norm, offset)
            gate = gate_proj(q2)
            attn_global = layer.self_attn.attention(q2, k1, v1, mask, cache=c)

            if c is not None:
                k2, v2 = c.update_and_fetch(k2, v2)
            attn_local = layer.self_attn.attention(
                q2,
                k2,
                v2,
                window_mask,
                cache=c,
            )

            mixed = _mix_attention(gate, attn_global, attn_local)
            r = layer.self_attn.o_proj(mixed.transpose(0, 2, 1, 3).reshape(B, L, -1))
            h = h + r
            r = layer.mlp(layer.post_attention_layernorm(h))
            h = h + r

        return self.norm(h)


class Model(nn.Module):
    model_args = ModelArgs

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = IQuestLoopCoderModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()
        rank = group.rank()

        for i, layer in enumerate(self.model.layers):
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

            layer.mlp.gate_proj = shard_linear(
                layer.mlp.gate_proj, "all-to-sharded", group=group
            )
            layer.mlp.down_proj = shard_linear(
                layer.mlp.down_proj, "sharded-to-all", group=group
            )
            layer.mlp.up_proj = shard_linear(
                layer.mlp.up_proj, "all-to-sharded", group=group
            )

            gate_proj = self.model.gate_projections[i]
            heads_per_rank = gate_proj.num_heads // N
            start = rank * heads_per_rank
            end = start + heads_per_rank
            gate_proj.weight = gate_proj.weight[start:end, :]
            gate_proj.bias = gate_proj.bias[start:end]
            gate_proj.num_heads = heads_per_rank

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [KVCache() for _ in self.layers] + [
            RotatingKVCache(max_size=self.args.loop_window_size) for _ in self.layers
        ]
