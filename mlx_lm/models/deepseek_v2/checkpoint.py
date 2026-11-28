# Copyright © 2023 Apple Inc.

import mlx.core as mx

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    for l in range(model.args.num_hidden_layers):
        prefix = f"model.layers.{l}"
        for _, m in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]:
            for k in ["weight", "scales", "biases"]:
                if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                    to_join = [
                        weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                        for e in range(model.args.n_routed_experts)
                    ]
                    weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)
    return weights
