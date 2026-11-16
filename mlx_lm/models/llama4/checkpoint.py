# Copyright © 2023 Apple Inc.

import mlx.core as mx

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    def to_remove(k):
        return "vision_model" in k or "multi_modal_projector" in k

    # Remove vision weights
    weights = {k: v for k, v in weights.items() if not to_remove(k)}

    # Rename expert weights for SwitchGLU
    for l in range(model.args.text_config.num_hidden_layers):
        prefix = f"language_model.model.layers.{l}.feed_forward.experts"
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
