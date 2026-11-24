# Copyright © 2025 Apple Inc.

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.layers.recurrent import create_ssm_mask
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.lfm2_moe import checkpoint
from mlx_lm.models.lfm2_moe.config import ModelArgs
from mlx_lm.models.lfm2_moe.layers import Lfm2DecoderLayer

__all__ = [
    "Lfm2Model",
    "Model",
]


class Lfm2Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Lfm2DecoderLayer(args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]

        self.embedding_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)

        self.fa_idx = args.full_attn_idxs[0]
        self.conv_idx = 0
        for i in range(args.num_hidden_layers):
            if i in args.full_attn_idxs:
                self.conv_idx += 1
            else:
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

        attn_mask = create_attention_mask(h, cache[self.fa_idx])
        conv_mask = create_ssm_mask(h, cache[self.conv_idx])

        for layer, c in zip(self.layers, cache):
            mask = attn_mask if layer.is_attention_layer else conv_mask
            h = layer(h, mask, cache=c)

        return self.embedding_norm(h)


class Model(nn.Module):
    model_args = ModelArgs

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Lfm2Model(args)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        out = self.model(inputs, cache, input_embeddings)
        return self.model.embed_tokens.as_linear(out)

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    @property
    def cast_predicate(self):
        def predicate(k):
            return "expert_bias" not in k

        return predicate

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("feed_forward.gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            KVCache() if l.is_attention_layer else ArraysCache(size=1)
            for l in self.layers
        ]
