# Copyright © 2026 Apple Inc.

from dataclasses import dataclass

import mlx.core as mx

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
        new_weights = {}
        for key, value in weights.items():
            if key.startswith("vision_tower") or key.startswith("model.visual"):
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

        # Stack per-expert MTP weights into switch_mlp format.
        # MTP layers use unfused per-expert weights (experts.{i}.gate_proj etc)
        # unlike backbone layers which use fused gate_up_proj.
        mtp_num = getattr(self.language_model.args, "mtp_num_hidden_layers", 0)
        num_experts = self.language_model.args.num_experts
        for l in range(mtp_num):
            prefix = f"language_model.mtp.layers.{l}.mlp"
            test_key = f"{prefix}.experts.0.gate_proj.weight"
            if test_key in new_weights:
                for n in ["gate_proj", "up_proj", "down_proj"]:
                    to_join = [
                        new_weights.pop(f"{prefix}.experts.{e}.{n}.weight")
                        for e in range(num_experts)
                    ]
                    new_weights[f"{prefix}.switch_mlp.{n}.weight"] = mx.stack(to_join)

        return self.language_model.sanitize(new_weights)
