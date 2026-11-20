# Copyright © 2023 Apple Inc.

import mlx.core as mx

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    if "model.layers.0.mlp.experts.0.up_proj.weight" not in weights:
        return weights
    for l in range(model.args.num_hidden_layers):
        prefix = f"model.layers.{l}"
        for n in ["up_proj", "down_proj", "gate_proj"]:
            for k in ["weight", "scales", "biases"]:
                if f"{prefix}.mlp.experts.0.{n}.{k}" in weights:
                    to_join = [
                        weights.pop(f"{prefix}.mlp.experts.{e}.{n}.{k}")
                        for e in range(model.args.num_experts)
                    ]
                    weights[f"{prefix}.mlp.switch_mlp.{n}.{k}"] = mx.stack(to_join)
    return weights
