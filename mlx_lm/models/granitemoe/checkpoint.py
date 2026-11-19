# Copyright © 2023 Apple Inc.

import mlx.core as mx

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    if "model.layers.0.block_sparse_moe.input_linear.weight" not in weights:
        return weights
    for l in range(model.args.num_hidden_layers):
        prefix = f"model.layers.{l}.block_sparse_moe"
        key = f"{prefix}.input_linear.weight"
        value = weights.pop(key)
        gate_proj, up_proj = mx.split(value, 2, axis=1)
        weights[key.replace("input_linear", "switch_mlp.gate_proj")] = gate_proj
        weights[key.replace("input_linear", "switch_mlp.up_proj")] = up_proj
        key = f"{prefix}.output_linear.weight"
        weights[key.replace("output_linear", "switch_mlp.down_proj")] = weights.pop(key)
    if model.args.tie_word_embeddings:
        weights.pop("lm_head.weight", None)
    return weights
