# Copyright © 2025 Apple Inc.

import mlx.core as mx

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    sanitized_weights = {}
    for name, param in weights.items():
        if "conv.weight" in name:
            if param.shape[-1] > param.shape[1]:
                param = param.transpose(0, 2, 1)
        replacements = {
            "w1.weight": "gate_proj.weight",
            "w2.weight": "down_proj.weight",
            "w3.weight": "up_proj.weight",
        }
        for old, new in replacements.items():
            if old in name:
                name = name.replace(old, new)
        sanitized_weights[name] = param

    for l in range(model.args.num_hidden_layers):
        prefix = f"model.layers.{l}"
        # Only sanitize MoE layer weights
        for n in ["gate_proj", "down_proj", "up_proj"]:
            if f"{prefix}.feed_forward.experts.0.{n}.weight" in sanitized_weights:
                to_join = [
                    sanitized_weights.pop(
                        f"{prefix}.feed_forward.experts.{e}.{n}.weight"
                    )
                    for e in range(model.args.num_experts)
                ]
                sanitized_weights[f"{prefix}.feed_forward.switch_mlp.{n}.weight"] = (
                    mx.stack(to_join)
                )
    return sanitized_weights
