# Copyright © 2023 Apple Inc.

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.models.hunyuan import checkpoint
from mlx_lm.models.hunyuan.config import ModelArgs
from mlx_lm.models.hunyuan.layers import DecoderLayer

__all__ = [
    "HunYuanModel",
    "Model",
]


class HunYuanModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(
                args=args,
                kv_proj=(not args.use_cla) or (i % args.cla_share_factor) == 0,
                layer_idx=i,
            )
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(h, cache[0])
        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            if (not self.args.use_cla) or i % self.args.cla_share_factor == 0:
                shared_kv_states = None
            h, shared_kv_states = layer(h, mask, c, shared_kv_states)

        return self.norm(h)


class Model(nn.Module):
    model_args = ModelArgs

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = HunYuanModel(args)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out = self.model(inputs, cache)
        return self.model.embed_tokens.as_linear(out)

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    @property
    def layers(self):
        return self.model.layers
