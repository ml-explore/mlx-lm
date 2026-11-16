# Copyright © 2025 Apple Inc.

from __future__ import annotations

import mlx.core as mx

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    moe_layers = sorted(
        int(k.split(".")[2])
        for k in weights
        if k.startswith("model.layers.") and k.endswith(".mlp.experts.0.up_proj.weight")
    )
    if not moe_layers:
        return weights
    weights = {key: value for key, value in weights.items() if "mtp." not in key}

    if model.args.tie_word_embeddings:
        weights.pop("lm_head.weight", None)

    for l in moe_layers:
        prefix = f"model.layers.{l}.mlp"
        for n in ["up_proj", "down_proj", "gate_proj"]:
            to_join = [
                weights.pop(f"{prefix}.experts.{e}.{n}.weight")
                for e in range(model.args.num_experts)
            ]
            weights[f"{prefix}.switch_mlp.{n}.weight"] = mx.stack(to_join)

    norm_keys = (
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        "model.norm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
    )
    for k, v in weights.items():
        if "conv1d.weight" in k and v.shape[-1] != 1:
            weights[k] = v.moveaxis(2, 1)
        if any(k.endswith(sfx) for sfx in norm_keys):
            if v.ndim == 1:
                weights[k] = v + 1.0
    return weights
