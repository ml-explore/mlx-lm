# Copyright © 2024 Apple Inc.

import mlx.core as mx

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    if "model.layers.0.block_sparse_moe.experts.0.w1.weight" not in weights:
        return weights
    for l in range(model.args.num_hidden_layers):
        prefix = f"model.layers.{l}"
        for n, m in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]:
            for k in ["weight", "scales", "biases"]:
                if f"{prefix}.block_sparse_moe.experts.0.{n}.{k}" in weights:
                    to_join = [
                        weights.pop(f"{prefix}.block_sparse_moe.experts.{e}.{n}.{k}")
                        for e in range(model.args.num_local_experts)
                    ]
                    weights[f"{prefix}.block_sparse_moe.switch_mlp.{m}.{k}"] = mx.stack(
                        to_join
                    )

    return weights
