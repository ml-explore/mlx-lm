# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from typing import Any, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from .base import BaseModelArgs
from .deepseek_v3 import DeepseekV3Model
from .deepseek_v3 import Model as DeepseekV3LM
from .deepseek_v3 import ModelArgs as TextConfig


@dataclass
class ModelArgs(BaseModelArgs):
    text_config: Union[TextConfig, dict]
    model_type: str = "kimi_k25"

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)


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
        weights = tree_unflatten(list(weights.items()))
        weights.pop("vision_tower", None)
        weights.pop("vision_model", None)
        weights.pop("multi_modal_projector", None)
        weights.pop("mm_projector", None)
        lm_weights = dict(tree_flatten(weights["language_model"]))
        lm_weights = DeepseekV3LM.sanitize(self.language_model, lm_weights)
        weights["language_model"] = tree_unflatten(list(lm_weights.items()))
        return dict(tree_flatten(weights))

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
