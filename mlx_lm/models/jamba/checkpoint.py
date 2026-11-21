# Copyright © 2025 Apple Inc.

import mlx.core as mx

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    for k, v in list(weights.items()):
        if "conv1d.weight" in k and v.shape[-1] != 1:
            weights[k] = v.moveaxis(2, 1)

    if model.args.tie_word_embeddings:
        weights.pop("lm_head.weight", None)

    for l in range(model.args.num_hidden_layers):
        base = f"model.layers.{l}.feed_forward"
        if not any(key.startswith(f"{base}.experts.") for key in weights.keys()):
            continue

        for proj in ["gate_proj", "down_proj", "up_proj"]:
            for name in ["weight", "bias", "scales", "biases"]:
                expert_tensors = [
                    weights.pop(f"{base}.experts.{e}.{proj}.{name}")
                    for e in range(len(weights))
                    if f"{base}.experts.{e}.{proj}.{name}" in weights
                ]
                if expert_tensors:
                    weights[f"{base}.switch_mlp.{proj}.{name}"] = mx.stack(
                        expert_tensors
                    )

    return weights
