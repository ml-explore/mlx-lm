# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from . import qwen3_moe
from .base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict


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
        weights = tree_unflatten(list(weights.items()))
        weights.pop("visual", None)
        weights = dict(
            tree_flatten(
                {
                    "language_model": {
                        "model": weights["language_model"]["model"],
                        "lm_head": weights["language_model"]["lm_head"],
                    }
                }
            )
        )

        for l in range(self.language_model.args.num_hidden_layers):
            prefix = f"language_model.model.layers.{l}.mlp"
            gate_up_key = f"{prefix}.experts.gate_up_proj"
            if gate_up_key in weights:
                gate_up = weights.pop(gate_up_key)
                mid = gate_up.shape[-1] // 2
                weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_up[
                    ..., :mid
                ].swapaxes(-2, -1)
                weights[f"{prefix}.switch_mlp.up_proj.weight"] = gate_up[
                    ..., mid:
                ].swapaxes(-2, -1)
                weights[f"{prefix}.switch_mlp.down_proj.weight"] = weights.pop(
                    f"{prefix}.experts.down_proj"
                ).swapaxes(-2, -1)

        return weights

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def layers(self):
        return self.language_model.model.layers
