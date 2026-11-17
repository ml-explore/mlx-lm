# Copyright © 2023 Apple Inc.

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.masks import create_attention_mask
from mlx_lm.models.cache import ChunkedKVCache, KVCache
from mlx_lm.models.llama4 import checkpoint
from mlx_lm.models.llama4.config import ModelArgs, TextArgs
from mlx_lm.models.llama4.layers import TransformerBlock

__all__ = [
    "Llama4Model",
    "Llama4TextModel",
    "Model",
]


class Llama4TextModel(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.attention_chunk_size = args.attention_chunk_size

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        h = self.embed_tokens(inputs)

        if cache is not None:
            for idx, c in enumerate(cache):
                if (idx + 1) % 4 != 0:
                    c.maybe_trim_front()
            start = cache[0].start_position
            offset = cache[0].offset
        else:
            start = 0
            offset = 0
        end = offset + h.shape[1]
        linds = mx.arange(start, end)
        rinds = mx.arange(offset, end)[:, None]
        block_pos = mx.abs(
            (linds // self.attention_chunk_size) - (rinds // self.attention_chunk_size)
        )
        token_pos = linds <= rinds
        chunk_mask = (block_pos == 0) & token_pos

        if cache is None:
            cache = [None] * len(self.layers)

        global_mask = create_attention_mask(h, cache[3])

        for idx, (layer, c) in enumerate(zip(self.layers, cache)):
            use_chunked_attention = (idx + 1) % 4 != 0
            mask = chunk_mask if use_chunked_attention else global_mask
            h = layer(h, mask, cache=c)

        return self.norm(h)


class Llama4Model(nn.Module):
    """The language model, named as in the checkpoint."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.language_model = Llama4TextModel(args.text)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Llama4Model(args)
        self.lm_head = nn.Linear(
            args.text.hidden_size, args.text.vocab_size, bias=False
        )

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        return self.lm_head(self.model.language_model(inputs, cache))

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    @property
    def layers(self):
        return self.model.language_model.layers

    def make_cache(self):
        chunk_size = self.args.text.attention_chunk_size
        caches = []
        for i in range(len(self.layers)):
            if (i + 1) % 4 != 0:
                caches.append(ChunkedKVCache(chunk_size))
            else:
                caches.append(KVCache())
        return caches
