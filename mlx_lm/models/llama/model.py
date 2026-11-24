# Copyright © 2023 Apple Inc.

from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_linear

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.llama import checkpoint
from mlx_lm.models.llama.config import ModelArgs
from mlx_lm.models.llama.layers import TransformerBlock

__all__ = [
    "LlamaModel",
    "Model",
    "shard_layers",
]


class LlamaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.layer_types = args.layer_types
        self.sliding_window = args.sliding_window
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args, use_sliding=layer_type == "sliding_attention")
            for layer_type in self.layer_types
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.fa_idx = self.layer_types.index("full_attention")
        self.swa_idx = None
        for e, l in enumerate(self.layers):
            if l.use_sliding:
                self.swa_idx = e
                break

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        fa_mask = create_attention_mask(h, cache[self.fa_idx])
        if self.swa_idx is not None:
            swa_mask = create_attention_mask(
                h, cache[self.swa_idx], window_size=self.sliding_window
            )

        for layer, cache in zip(self.layers, cache):
            mask = swa_mask if layer.use_sliding else fa_mask
            h = layer(h, mask, cache=cache)

        return self.norm(h)


def shard_layers(layers, group: mx.distributed.Group):
    """Shard the decoder layers for tensor parallelism."""
    N = group.size()
    for layer in layers:
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
        layer.self_attn.n_heads //= N
        layer.self_attn.n_kv_heads //= N

        # Shard the MLP
        layer.mlp.gate_proj = shard_linear(
            layer.mlp.gate_proj, "all-to-sharded", group=group
        )
        layer.mlp.down_proj = shard_linear(
            layer.mlp.down_proj, "sharded-to-all", group=group
        )
        layer.mlp.up_proj = shard_linear(
            layer.mlp.up_proj, "all-to-sharded", group=group
        )


class Model(nn.Module):
    model_args = ModelArgs

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LlamaModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    def shard(self, group: Optional[mx.distributed.Group] = None):
        shard_layers(self.model.layers, group or mx.distributed.init())

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            (
                RotatingKVCache(max_size=self.model.sliding_window)
                if layer.use_sliding
                else KVCache()
            )
            for layer in self.layers
        ]
