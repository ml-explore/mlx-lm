# Copyright © 2023 Apple Inc.

import mlx.core as mx

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    if "transformer.h.0.moe.mlp.0.fc1.weight" not in weights:
        return weights
    for l in range(model.args.num_layers):
        prefix = f"transformer.h.{l}"
        for n in ["fc1", "fc2"]:
            for k in ["weight", "scales", "biases", "bias"]:
                if f"{prefix}.moe.mlp.0.{n}.{k}" in weights:
                    to_join = [
                        weights.pop(f"{prefix}.moe.mlp.{e}.{n}.{k}")
                        for e in range(model.args.num_local_experts)
                    ]
                    weights[f"{prefix}.moe.switch_mlp.{n}.{k}"] = mx.stack(to_join)
    return weights
