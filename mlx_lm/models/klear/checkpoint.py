# Copyright © 2025 Apple Inc.

import mlx.core as mx

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    moe_layers = sorted(
        int(k.split(".")[2])
        for k in weights
        if k.startswith("model.layers.")
        and k.endswith(".mlp.experts.0.gate_proj.weight")
    )
    for l in moe_layers:
        prefix = f"model.layers.{l}.mlp.experts"
        for name in ["gate_proj", "up_proj", "down_proj"]:
            stacked = [
                weights.pop(f"{prefix}.{e}.{name}.weight")
                for e in range(model.args.num_experts)
            ]
            weights[f"{prefix}.{name}.weight"] = mx.stack(stacked)

    return weights
