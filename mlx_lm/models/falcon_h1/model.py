# Copyright © 2025 Apple Inc.

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.layers.recurrent import create_ssm_mask
from mlx_lm.models.cache import ArraysCache, CacheList, KVCache
from mlx_lm.models.falcon_h1 import checkpoint
from mlx_lm.models.falcon_h1.config import ModelArgs
from mlx_lm.models.falcon_h1.layers import FalconH1DecoderLayer

__all__ = [
    "FalconH1Model",
    "Model",
    "compute_mup_vector",
]


def compute_mup_vector(args):
    intermediate_size = args.mamba_d_ssm
    groups_time_state_size = args.mamba_n_groups * args.mamba_d_state
    num_heads = args.mamba_n_heads
    sizes = [
        intermediate_size,
        intermediate_size,
        groups_time_state_size,
        groups_time_state_size,
        num_heads,
    ]
    return mx.concatenate(
        [
            mx.broadcast_to(mx.array(m), (s,))
            for s, m in zip(sizes, args.ssm_multipliers)
        ]
    )


class FalconH1Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.vocab_size = args.vocab_size
        self.hidden_size = args.hidden_size

        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)

        self._mup_vector = compute_mup_vector(args)
        self.layers = [
            FalconH1DecoderLayer(args) for _ in range(args.num_hidden_layers)
        ]
        self.final_layernorm = nn.RMSNorm(self.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs, cache=None):

        h = self.embed_tokens(inputs)

        if cache is None:
            cache = [(None, None)] * len(self.layers)

        mamba_mask = create_ssm_mask(h, cache[0][0])
        attn_mask = create_attention_mask(h, cache[0][1])

        for layer, c in zip(self.layers, cache):
            h = layer(
                h,
                cache=c,
                attn_mask=attn_mask,
                mamba_mask=mamba_mask,
            )

        return self.final_layernorm(h)


class Model(nn.Module):

    model_args = ModelArgs

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = FalconH1Model(args=args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None):
        hidden_states = self.model(inputs, cache=cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(hidden_states)
            return out * (self.args.lm_head_multiplier / self.args.embedding_multiplier)
        else:
            return self.lm_head(hidden_states)

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            CacheList(ArraysCache(size=2), KVCache())
            for _ in range(self.args.num_hidden_layers)
        ]
