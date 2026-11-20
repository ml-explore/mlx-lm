# Copyright © 2023 Apple Inc.

import mlx.core as mx

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    # Split experts into sub matrices
    num_experts = model.args.ffn_config["moe_num_experts"]

    pattern = "experts.mlp"
    new_weights = {k: v for k, v in weights.items() if pattern not in k}
    for k, v in weights.items():
        if pattern in k:
            experts = [
                (k.replace(".mlp", f".{e}") + ".weight", sv)
                for e, sv in enumerate(mx.split(v, num_experts, axis=0))
            ]
            if k.endswith("w2"):
                experts = [(s, sv.T) for s, sv in experts]
            new_weights.update(experts)
    return new_weights
