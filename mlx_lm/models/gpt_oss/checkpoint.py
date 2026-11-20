# Copyright © 2025 Apple Inc.

import mlx.core as mx

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    if any("gate_proj.weight" in k for k in weights.keys()):
        return weights  # already sanitized

    new_weights = {}
    for k, v in weights.items():
        if "gate_up_proj" in k and "bias" not in k:
            if "_blocks" in k:
                v = v.view(mx.uint32).flatten(-2)
                k = k.replace("_blocks", ".weight")
            if "_scales" in k:
                k = k.replace("_scales", ".scales")
            new_weights[k.replace("gate_up_proj", "gate_proj")] = mx.contiguous(
                v[..., ::2, :]
            )
            new_weights[k.replace("gate_up_proj", "up_proj")] = mx.contiguous(
                v[..., 1::2, :]
            )
        elif "down_proj" in k and "bias" not in k:
            if "_blocks" in k:
                v = v.view(mx.uint32).flatten(-2)
                k = k.replace("_blocks", ".weight")
            if "_scales" in k:
                k = k.replace("_scales", ".scales")
            new_weights[k] = v
        elif "gate_up_proj_bias" in k:
            new_weights[k.replace("gate_up_proj_bias", "gate_proj.bias")] = (
                mx.contiguous(v[..., ::2])
            )
            new_weights[k.replace("gate_up_proj_bias", "up_proj.bias")] = mx.contiguous(
                v[..., 1::2]
            )
        elif "down_proj_bias" in k:
            new_weights[k.replace("down_proj_bias", "down_proj.bias")] = v
        else:
            new_weights[k] = v

    return new_weights
