# Copyright © 2025 Apple Inc.

from __future__ import annotations

import mlx.core as mx

__all__ = [
    "sanitize",
]

# The norms that Hugging Face stores as an offset from 1, and this model as 1 + weight.
_OFFSET_NORMS = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "model.norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)


def _from_hugging_face(model, weights):
    """Stack the experts, which Hugging Face stores one by one, and add 1 to the norms."""
    for l in range(model.args.num_hidden_layers):
        prefix = f"model.layers.{l}.mlp"
        for n in ["up_proj", "down_proj", "gate_proj"]:
            if f"{prefix}.experts.0.{n}.weight" in weights:
                to_join = [
                    weights.pop(f"{prefix}.experts.{e}.{n}.weight")
                    for e in range(model.args.num_experts)
                ]
                weights[f"{prefix}.switch_mlp.{n}.weight"] = mx.stack(to_join)
    for k, v in weights.items():
        if k.endswith(_OFFSET_NORMS):
            weights[k] = v + 1.0
    return weights


def sanitize(model, weights):
    weights = {k: v for k, v in weights.items() if "mtp." not in k}

    if model.args.tie_word_embeddings:
        weights.pop("lm_head.weight", None)

    # MLX conversions already have this model's format.
    if any(".mlp.experts.0." in k for k in weights):
        weights = _from_hugging_face(model, weights)

    for k, v in weights.items():
        if "conv1d.weight" in k and v.shape[-1] != 1:
            weights[k] = v.moveaxis(2, 1)
    return weights
