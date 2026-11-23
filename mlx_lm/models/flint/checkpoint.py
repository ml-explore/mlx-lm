# Copyright © 2026 Apple Inc.

import mlx.core as mx

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    if model.args.tie_word_embeddings:
        weights.pop("lm_head.weight", None)
    # RecursiveSI/flint-int4 stores the packed weights as uint8.
    for k, v in weights.items():
        name = k.removesuffix(".weight")
        if v.dtype == mx.uint8 and f"{name}.scales" in weights:
            weights[k] = v.view(mx.uint32)
    for l in range(model.args.num_hidden_layers):
        if l in model.args.mlp_only_layers:
            continue
        prefix = f"model.layers.{l}.mlp"
        for n in ["up_proj", "down_proj", "gate_proj"]:
            if f"{prefix}.experts.0.{n}.weight" in weights:
                to_join = [
                    weights.pop(f"{prefix}.experts.{e}.{n}.weight")
                    for e in range(model.args.num_experts)
                ]
                weights[f"{prefix}.switch_mlp.{n}.weight"] = mx.stack(to_join)
    return weights
