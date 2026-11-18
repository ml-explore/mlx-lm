# Copyright © 2024 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.deepseek_v3.model import DeepseekV3Model
from mlx_lm.models.kimi_vl import checkpoint
from mlx_lm.models.kimi_vl.config import ModelArgs

__all__ = [
    "KimiVLModel",
    "Model",
]


class KimiVLModel(nn.Module):
    """Puts the language model at model.language_model, as the checkpoint names it."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.language_model = DeepseekV3Model(args.text)


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = KimiVLModel(config)
        self.lm_head = nn.Linear(
            config.text.hidden_size, config.text.vocab_size, bias=False
        )

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ):
        return self.lm_head(self.model.language_model(inputs, cache))

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    @property
    def cast_predicate(self):
        def predicate(k):
            return "e_score_correction_bias" not in k

        return predicate

    @property
    def layers(self):
        return self.model.language_model.layers
