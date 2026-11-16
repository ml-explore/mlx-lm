# Copyright © 2026 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.deepseek_v3.config import ModelArgs as TextConfig
from mlx_lm.models.deepseek_v3.model import DeepseekV3Model
from mlx_lm.models.deepseek_v3.model import Model as DeepseekV3LM
from mlx_lm.models.kimi_k25 import checkpoint
from mlx_lm.models.kimi_k25.config import ModelArgs

__all__ = [
    "LanguageModel",
    "Model",
]


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
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

    def shard(self, group: Optional[mx.distributed.Group] = None):
        DeepseekV3LM.shard(self.language_model, group)

    @property
    def model(self):
        return self.language_model.model

    @property
    def layers(self):
        return self.language_model.model.pipeline_layers

    @property
    def cast_predicate(self):
        def predicate(k):
            return "e_score_correction_bias" not in k

        return predicate
