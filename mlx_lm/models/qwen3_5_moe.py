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


def _normalize_experts(weights, prefix, num_experts):
    """Normalize one Qwen MoE block to MLX ``switch_mlp`` tensors.

    Exporters use three legitimate layouts: a fused ``gate_up_proj`` tensor,
    individual per-expert projections, or already-stacked ``switch_mlp``
    tensors.  Treat a mixture or an incomplete layout as a corrupt artifact;
    silently choosing one representation can make an MTP checkpoint load with
    a different expert ordering than the backbone.
    """

    fused_gate_up = f"{prefix}.experts.gate_up_proj"
    fused_down = f"{prefix}.experts.down_proj"
    stacked = [
        f"{prefix}.switch_mlp.{name}.weight"
        for name in ("gate_proj", "up_proj", "down_proj")
    ]
    per_expert = [
        f"{prefix}.experts.{expert}.{name}.weight"
        for expert in range(num_experts)
        for name in ("gate_proj", "up_proj", "down_proj")
    ]

    has_fused = fused_gate_up in weights or fused_down in weights
    has_stacked = any(key in weights for key in stacked)
    has_per_expert = any(key in weights for key in per_expert)
    formats = sum((has_fused, has_stacked, has_per_expert))
    if formats == 0:
        return
    if formats != 1:
        raise ValueError(f"Mixed MoE expert layouts at {prefix}")

    if has_fused:
        if fused_gate_up not in weights or fused_down not in weights:
            raise ValueError(f"Incomplete fused MoE expert weights at {prefix}")
        gate_up = weights.pop(fused_gate_up)
        if gate_up.shape[-2] % 2:
            raise ValueError(f"Odd fused gate/up dimension at {prefix}")
        middle = gate_up.shape[-2] // 2
        weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_up[..., :middle, :]
        weights[f"{prefix}.switch_mlp.up_proj.weight"] = gate_up[..., middle:, :]
        weights[f"{prefix}.switch_mlp.down_proj.weight"] = weights.pop(fused_down)
        return

    if has_stacked:
        missing = [key for key in stacked if key not in weights]
        if missing:
            raise ValueError(
                f"Incomplete stacked MoE expert weights at {prefix}: {missing[0]}"
            )
        return

    missing = [key for key in per_expert if key not in weights]
    if missing:
        raise ValueError(f"Incomplete per-expert MoE weights at {prefix}: {missing[0]}")
    for name in ("gate_proj", "up_proj", "down_proj"):
        weights[f"{prefix}.switch_mlp.{name}.weight"] = mx.stack(
            [
                weights.pop(f"{prefix}.experts.{expert}.{name}.weight")
                for expert in range(num_experts)
            ]
        )


class Model(Qwen3_5Model):
    def sanitize(self, weights):
        normalized = {}
        for key, value in weights.items():
            if key.startswith("vision_tower") or key.startswith("model.visual"):
                continue
            if key.startswith("model.language_model.mtp."):
                key = key.replace("model.language_model.", "language_model.", 1)
            elif key.startswith("model.language_model"):
                key = key.replace("model.language_model", "language_model.model")
            elif not key.startswith("language_model."):
                key = "language_model." + key
            normalized[key] = value

        args = self.language_model.args
        for layer_idx in range(args.num_hidden_layers):
            _normalize_experts(
                normalized,
                f"language_model.model.layers.{layer_idx}.mlp",
                args.num_experts,
            )
        for layer_idx in range(args.mtp_num_hidden_layers):
            _normalize_experts(
                normalized,
                f"language_model.mtp.layers.{layer_idx}.mlp",
                args.num_experts,
            )

        return self.language_model.sanitize(normalized)
