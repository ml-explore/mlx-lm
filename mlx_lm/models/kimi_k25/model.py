# Copyright © 2026 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.deepseek_v3.model import DeepseekV3Model, shard_layers
from mlx_lm.models.kimi_k25 import checkpoint
from mlx_lm.models.kimi_k25.config import ModelArgs

__all__ = [
    "KimiK25Model",
    "Model",
]


class KimiK25Model(nn.Module):
    """Puts the language model at model.language_model, as the checkpoint names it."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.language_model = DeepseekV3Model(args.text)

    # mlx_lm.utils pipelines model.model with these.
    @property
    def layers(self):
        return self.language_model.layers

    @property
    def start_idx(self):
        return self.language_model.start_idx

    @property
    def end_idx(self):
        return self.language_model.end_idx

    def pipeline(self, group, split=None):
        self.language_model.pipeline(group, split)


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = KimiK25Model(config)
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

    def shard(self, group: Optional[mx.distributed.Group] = None):
        shard_layers(self.model.language_model.layers, group or mx.distributed.init())

    @property
    def layers(self):
        return self.model.language_model.pipeline_layers
