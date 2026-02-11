# Copyright © 2026 Apple Inc.

from dataclasses import dataclass

from mlx.utils import tree_flatten, tree_unflatten

from .base import BaseModelArgs
from .qwen3_5 import Model as Qwen3_5Model


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict

    @classmethod
    def from_dict(cls, params):
        if "text_config" not in params:
            return cls(model_type=params["model_type"], text_config=params)
        return super().from_dict(params)


class Model(Qwen3_5Model):

    def sanitize(self, weights):
        weights = tree_unflatten(list(weights.items()))
        weights = dict(tree_flatten(weights))

        new_weights = {}
        for key, value in weights.items():
            if key.startswith("model.visual"):
                continue
            if key.startswith("model.language_model"):
                key = key.replace("model.language_model", "language_model.model")
            elif key.startswith("language_model."):
                pass
            else:
                key = "language_model." + key
            new_weights[key] = value

        for l in range(self.language_model.args.num_hidden_layers):
            prefix = f"language_model.model.layers.{l}.mlp"
            gate_up_key = f"{prefix}.experts.gate_up_proj"
            if gate_up_key in new_weights:
                gate_up = new_weights.pop(gate_up_key)
                mid = gate_up.shape[-2] // 2
                new_weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_up[
                    ..., :mid, :
                ]
                new_weights[f"{prefix}.switch_mlp.up_proj.weight"] = gate_up[
                    ..., mid:, :
                ]
                new_weights[f"{prefix}.switch_mlp.down_proj.weight"] = new_weights.pop(
                    f"{prefix}.experts.down_proj"
                )

        return self.language_model.sanitize(new_weights)
