# Copyright © 2025 Apple Inc.

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models import qwen3_moe
from mlx_lm.models.qwen3_vl_moe import checkpoint
from mlx_lm.models.qwen3_vl_moe.config import ModelArgs

__all__ = [
    "Model",
]


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = qwen3_moe.Model(
            qwen3_moe.ModelArgs.from_dict(args.text_config)
        )

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        return self.language_model(
            inputs, cache=cache, input_embeddings=input_embeddings
        )

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def layers(self):
        return self.language_model.model.layers
