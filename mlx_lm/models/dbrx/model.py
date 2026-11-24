# Copyright © 2023 Apple Inc.

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.dbrx import checkpoint
from mlx_lm.models.dbrx.config import ModelArgs
from mlx_lm.models.dbrx.layers import DBRX

__all__ = [
    "Model",
]


class Model(nn.Module):
    model_args = ModelArgs

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model_type = args.model_type
        self.transformer = DBRX(args)
        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self.args = args

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out = self.transformer(inputs, cache)
        return self.lm_head(out)

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    @property
    def layers(self):
        return self.transformer.blocks
