# Copyright © 2024 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.deepseek_v3.model import DeepseekV3Model
from mlx_lm.models.kimi_vl import checkpoint
from mlx_lm.models.kimi_vl.config import ModelArgs, TextArgs

__all__ = [
    "LanguageModel",
    "Model",
]


class LanguageModel(nn.Module):
    def __init__(self, config: TextArgs):
        super().__init__()
        self.args = config
        self.model = DeepseekV3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ):
        out = self.model(inputs, cache)
        return self.lm_head(out)


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.language_model = LanguageModel(config.text_config)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ):
        return self.language_model(inputs, cache)

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    @property
    def layers(self):
        return self.language_model.model.layers

    @property
    def cast_predicate(self):
        def predicate(k):
            return "e_score_correction_bias" not in k

        return predicate
