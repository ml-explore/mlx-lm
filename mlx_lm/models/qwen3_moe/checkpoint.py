# Copyright © 2026 Apple Inc.

import mlx.core as mx

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    if model.args.tie_word_embeddings:
        weights.pop("lm_head.weight", None)
    moe_layers = sorted(
        int(k.split(".")[2])
        for k in weights
        if k.startswith("model.layers.") and k.endswith(".mlp.experts.0.up_proj.weight")
    )
    for l in moe_layers:
        prefix = f"model.layers.{l}"
        for n in ["up_proj", "down_proj", "gate_proj"]:
            to_join = [
                weights.pop(f"{prefix}.mlp.experts.{e}.{n}.weight")
                for e in range(model.args.num_experts)
            ]
            weights[f"{prefix}.mlp.switch_mlp.{n}.weight"] = mx.stack(to_join)
    return weights
