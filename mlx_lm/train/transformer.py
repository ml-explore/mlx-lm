# Copyright © 2026 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.train.args import ModelArgs
from mlx_lm.train.layers import build_attention, build_mlp

__all__ = ["ModelArgs", "Model"]


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.post_norm = args.layer_norm == "post"
        self.is_linear = (layer_idx + 1) % args.quadratic_attn_interval != 0
        attn_type = args.linear_attn_type if self.is_linear else "full"

        self.self_attn = build_attention(args, attn_type)
        self.mlp = build_mlp(args, args.mlp_type)
        self.attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.mlp_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[Any] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if self.post_norm:
            h = x + self.attention_layernorm(self.self_attn(x, mask, cache))
            return h + self.mlp_layernorm(self.mlp(h))
        h = x + self.self_attn(self.attention_layernorm(x), mask, cache)
        return h + self.mlp(self.mlp_layernorm(h))


class LanguageModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[Any] = None,
        cache: Optional[Any] = None,
    ):
        h = self.embed_tokens(inputs)
        if mask is None and h.shape[1] > 1:
            mask = "causal"
        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            h = layer(h, None if layer.is_linear else mask, c)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model = LanguageModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[Any] = None,
        cache: Optional[Any] = None,
    ):
        out = self.model(inputs, mask=mask, cache=cache)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers

    def init_weights(self):
        std = self.args.init_std
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight = mx.random.normal(shape=module.weight.shape) * std
