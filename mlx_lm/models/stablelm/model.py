# Copyright © 2023 Apple Inc.

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.stablelm.config import ModelArgs
from mlx_lm.models.stablelm.layers import StableLM

__all__ = [
    "Model",
]


class Model(nn.Module):
    model_args = ModelArgs

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.model_type = config.model_type
        self.model = StableLM(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.args = config

    def __call__(
        self,
        x: mx.array,
        cache=None,
    ) -> mx.array:
        y = self.model(x, cache)
        return self.lm_head(y)

    @property
    def layers(self):
        return self.model.layers
