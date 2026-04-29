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


def _unfuse_experts(weights, prefix):
    """Split fused gate_up_proj into per-projection switch_mlp weights (Qwen3.6 format)."""
    gate_up_key = f"{prefix}.experts.gate_up_proj"
    if gate_up_key not in weights:
        return
    gate_up = weights.pop(gate_up_key)
    mid = gate_up.shape[-2] // 2
    weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_up[..., :mid, :]
    weights[f"{prefix}.switch_mlp.up_proj.weight"] = gate_up[..., mid:, :]
    weights[f"{prefix}.switch_mlp.down_proj.weight"] = weights.pop(
        f"{prefix}.experts.down_proj"
    )


def _stack_per_expert(weights, prefix, num_experts):
    """Stack per-expert weights into switch_mlp format (Qwen3.5 format)."""
    for n in ("gate_proj", "up_proj", "down_proj"):
        weights[f"{prefix}.switch_mlp.{n}.weight"] = mx.stack(
            [
                weights.pop(f"{prefix}.experts.{e}.{n}.weight")
                for e in range(num_experts)
            ]
        )


class Model(Qwen3_5Model):

    def sanitize(self, weights):
        new_weights = {}
        for key, value in weights.items():
            if key.startswith("vision_tower") or key.startswith("model.visual"):
                continue
            if key.startswith("model.language_model"):
                key = key.replace("model.language_model", "language_model.model")
            elif not key.startswith("language_model."):
                key = "language_model." + key
            new_weights[key] = value

        # Backbone MoE layers always use fused gate_up_proj (both Qwen3.5 and Qwen3.6).
        for l in range(self.language_model.args.num_hidden_layers):
            _unfuse_experts(new_weights, f"language_model.model.layers.{l}.mlp")

        # MTP layers: fused format (Qwen3.6) or per-expert format (Qwen3.5).
        # Detect format once from the first layer and apply uniformly.
        mtp_num = getattr(self.language_model.args, "mtp_num_hidden_layers", 0)
        if mtp_num > 0:
            num_experts = self.language_model.args.num_experts
            mtp_is_fused = (
                "language_model.mtp.layers.0.mlp.experts.gate_up_proj" in new_weights
            )
            for layer_idx in range(mtp_num):
                prefix = f"language_model.mtp.layers.{layer_idx}.mlp"
                if mtp_is_fused:
                    _unfuse_experts(new_weights, prefix)
                else:
                    _stack_per_expert(new_weights, prefix, num_experts)

        return self.language_model.sanitize(new_weights)
