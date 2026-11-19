# Copyright © 2023 Apple Inc.

import mlx.core as mx

from mlx_lm.models.checkpoint import from_llava_layout

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    # The checkpoints, and older MLX conversions, have the LLaVA layout. Only
    # the language model runs here.
    weights = {
        from_llava_layout(k): v
        for k, v in weights.items()
        if "vision_model" not in k and "multi_modal_projector" not in k
    }

    # Rename expert weights for SwitchGLU
    for l in range(model.args.text.num_hidden_layers):
        prefix = f"model.language_model.layers.{l}.feed_forward.experts"
        if f"{prefix}.gate_up_proj" in weights:
            v = weights.pop(f"{prefix}.gate_up_proj")
            gate_k = f"{prefix}.gate_proj.weight"
            up_k = f"{prefix}.up_proj.weight"
            gate_proj, up_proj = mx.split(v, 2, axis=-1)
            weights[gate_k] = mx.swapaxes(gate_proj, 1, 2)
            weights[up_k] = mx.swapaxes(up_proj, 1, 2)
        if f"{prefix}.down_proj" in weights:
            down_proj = weights.pop(f"{prefix}.down_proj")
            weights[f"{prefix}.down_proj.weight"] = mx.swapaxes(down_proj, 1, 2)
    return weights
