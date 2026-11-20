# Copyright © 2026 Apple Inc.

import mlx.core as mx

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    for layer_idx in range(model.args.num_hidden_layers):

        if layer_idx < model.args.first_k_dense_replace:
            continue

        prefix = f"model.layers.{layer_idx}.mlp.experts"

        for name in ["gate_proj", "up_proj", "down_proj"]:
            key = f"{prefix}.0.{name}.weight"
            if key not in weights:
                continue

            expert_weights = [
                weights.pop(f"{prefix}.{expert_idx}.{name}.weight")
                for expert_idx in range(model.args.n_routed_experts)
            ]
            weights[f"{prefix}.{name}.weight"] = mx.stack(expert_weights)

    return weights
