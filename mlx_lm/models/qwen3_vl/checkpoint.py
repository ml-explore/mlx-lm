# Copyright © 2025 Apple Inc.

import mlx.core as mx

from mlx_lm.models.qwen2_vl.checkpoint import current_name, upgrade_mlx_layout

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    if any(k.startswith("language_model.") for k in weights):
        weights = upgrade_mlx_layout(weights)
    weights = {current_name(k): v for k, v in weights.items()}
    # mlx-lm does not run the vision encoder yet.
    weights = {k: v for k, v in weights.items() if not k.startswith("model.visual.")}
    if model.args.text.tie_word_embeddings:
        weights.pop("lm_head.weight", None)

    # The fused experts are stored [num_experts, input, output].
    for k in list(weights):
        if k.endswith(".experts.gate_up_proj"):
            switch_mlp = k.removesuffix(".experts.gate_up_proj") + ".switch_mlp"
            gate, up = mx.split(weights.pop(k).swapaxes(-2, -1), 2, axis=-2)
            weights[f"{switch_mlp}.gate_proj.weight"] = gate
            weights[f"{switch_mlp}.up_proj.weight"] = up
        elif k.endswith(".experts.down_proj"):
            switch_mlp = k.removesuffix(".experts.down_proj") + ".switch_mlp"
            weights[f"{switch_mlp}.down_proj.weight"] = weights.pop(k).swapaxes(-2, -1)
    return weights
