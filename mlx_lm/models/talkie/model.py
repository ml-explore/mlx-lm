# Copyright © 2026 Apple Inc.

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.models.talkie import checkpoint
from mlx_lm.models.talkie.config import ModelArgs
from mlx_lm.models.talkie.layers import TransformerBlock, WeightGain, rms_norm

__all__ = [
    "Model",
    "TalkieModel",
]


class TalkieModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.eps = args.rms_norm_eps
        self.embed = nn.Embedding(args.vocab_size, args.hidden_size)
        self.blocks = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed(inputs)
        h = rms_norm(h, self.eps)
        e_x = h

        if cache is None:
            cache = [None] * len(self.blocks)

        mask = create_attention_mask(h, cache[0])

        for block, c in zip(self.blocks, cache):
            h = block(e_x, h, mask, c)

        return rms_norm(h, self.eps)


class Model(nn.Module):
    model_args = ModelArgs

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = TalkieModel(args)
        # The output projection is a bare parameter in the checkpoint (no
        # ".weight" suffix), scaled by a learned scalar gain.
        self.lm_head = mx.zeros((args.vocab_size, args.hidden_size))
        self.lm_head_gain = WeightGain()

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        out = self.model(inputs, cache, input_embeddings)
        return self.lm_head_gain(out @ self.lm_head.T)

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    @property
    def layers(self):
        return self.model.blocks
