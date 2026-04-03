# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from . import gemma4_text
from .base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "gemma4"
    text_config: dict = None
    vocab_size: int = 262144

    def __post_init__(self):
        if self.text_config is None:
            self.text_config = {}
        self.text_config["vocab_size"] = self.vocab_size
        self.text_config["num_attention_heads"] = self.text_config.get(
            "num_attention_heads", 8
        )
        self.text_config["num_key_value_heads"] = self.text_config.get(
            "num_key_value_heads", 1
        )


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = gemma4_text.Model(
            gemma4_text.ModelArgs.from_dict(args.text_config)
        )

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        per_layer_inputs: Optional[mx.array] = None,
    ):
        return self.language_model(
            inputs,
            cache=cache,
            input_embeddings=input_embeddings,
            per_layer_inputs=per_layer_inputs,
        )

    def sanitize(self, weights):
        # HF weights are prefixed with "model." (Gemma4ForConditionalGeneration.model)
        # Strip it and work with the inner structure
        new_weights = {}
        for k, v in weights.items():
            if k.startswith("model."):
                new_weights[k[len("model.") :]] = v
            else:
                new_weights[k] = v

        new_weights = tree_unflatten(list(new_weights.items()))
        new_weights.pop("vision_tower", None)
        new_weights.pop("multi_modal_projector", None)
        new_weights.pop("audio_tower", None)
        new_weights.pop("embed_audio", None)
        new_weights.pop("embed_vision", None)

        if "language_model" in new_weights:
            # Converted MLX weights already use "model.layers.*" while raw HF
            # checkpoints use "layers.*" under language_model.
            lm_weights = dict(tree_flatten(new_weights["language_model"]))
            if not any(
                k.startswith("model.") or k.startswith("lm_head.") for k in lm_weights
            ):
                lm_weights = {"model." + k: v for k, v in lm_weights.items()}
            lm_weights = self.language_model.sanitize(lm_weights)
            new_weights["language_model"] = tree_unflatten(list(lm_weights.items()))
            return dict(tree_flatten(new_weights))

        return self.language_model.sanitize(dict(tree_flatten(new_weights)))

    @property
    def layers(self):
        return self.language_model.layers

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    def make_cache(self):
        return self.language_model.make_cache()
