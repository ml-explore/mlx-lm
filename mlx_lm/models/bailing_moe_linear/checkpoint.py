# Copyright © 2025 Apple Inc.

import mlx.core as mx

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    if model.args.tie_word_embeddings:
        weights.pop("lm_head.weight", None)

    if model.norm_head:
        w = weights["lm_head.weight"]
        dtype = w.dtype
        weight_norm = mx.linalg.norm(w.astype(mx.float32), axis=0, keepdims=True) + 1e-7
        weights["lm_head.weight"] = (w / weight_norm).astype(dtype)

    for l in range(model.args.num_hidden_layers):
        prefix = f"model.layers.{l}"
        # Handle MoE layers
        if l >= model.args.first_k_dense_replace:
            for m in ["gate_proj", "down_proj", "up_proj"]:
                for k in ["weight", "scales", "biases"]:
                    if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                            for e in range(model.args.num_experts)
                        ]
                        weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)

            if f"{prefix}.mlp.gate.weight" in weights:
                gate_weight = weights.pop(f"{prefix}.mlp.gate.weight")
                weights[f"{prefix}.mlp.gate.gate_proj.weight"] = gate_weight

            if f"{prefix}.mlp.gate.bias" in weights:
                gate_bias = weights.pop(f"{prefix}.mlp.gate.bias")
                weights[f"{prefix}.mlp.gate.gate_proj.bias"] = gate_bias

    return weights
