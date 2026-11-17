# Copyright © 2025 Apple Inc.

from __future__ import annotations

import mlx.core as mx

__all__ = [
    "sanitize",
]

# The norms whose weight is an offset from 1.
_ZERO_CENTERED_NORMS = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "model.norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)


def sanitize(model, weights):
    if any(".switch_mlp." in k for k in weights):
        weights = _upgrade_mlx_layout(weights)
    weights = {k: v for k, v in weights.items() if "mtp." not in k}

    if model.args.tie_word_embeddings:
        weights.pop("lm_head.weight", None)

    for l in range(model.args.num_hidden_layers):
        prefix = f"model.layers.{l}.mlp.experts"
        for n in ["up_proj", "down_proj", "gate_proj"]:
            if f"{prefix}.0.{n}.weight" in weights:
                to_join = [
                    weights.pop(f"{prefix}.{e}.{n}.weight")
                    for e in range(model.args.num_experts)
                ]
                weights[f"{prefix}.{n}.weight"] = mx.stack(to_join)

    for k, v in weights.items():
        if "conv1d.weight" in k and v.shape[-1] != 1:
            weights[k] = v.moveaxis(2, 1)
    return weights


def _upgrade_mlx_layout(weights):
    """A checkpoint that mlx-lm converted with the experts under switch_mlp and the
    norms storing 1 + weight."""
    upgraded = {}
    for k, v in weights.items():
        k = k.replace(".switch_mlp.", ".experts.")
        upgraded[k] = v - 1.0 if k.endswith(_ZERO_CENTERED_NORMS) else v
    return upgraded
