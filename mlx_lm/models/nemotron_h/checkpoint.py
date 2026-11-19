# Copyright © 2025 Apple Inc.

import mlx.core as mx

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    weights = {k: v for (k, v) in weights.items() if not k.startswith("mtp.")}
    for k, v in weights.items():
        if "conv1d.weight" in k and v.shape[-1] != 1:
            weights[k] = v.moveaxis(2, 1)

    # Stack experts
    for l in range(model.args.num_hidden_layers):
        prefix = f"backbone.layers.{l}.mixer"
        for m, n in [("down_proj", "fc2"), ("up_proj", "fc1")]:
            if f"{prefix}.experts.0.{m}.weight" in weights:
                to_join = [
                    weights.pop(f"{prefix}.experts.{e}.{m}.weight")
                    for e in range(model.args.n_routed_experts)
                ]
                weights[f"{prefix}.switch_mlp.{n}.weight"] = mx.stack(to_join)

    return weights
