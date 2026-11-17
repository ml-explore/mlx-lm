# Copyright © 2026 Apple Inc.

from mlx_lm.layers.multimodal import from_llava_layout, from_text_layout

__all__ = [
    "sanitize",
]

# This is a text-only port.
_ENCODERS = ("vision_tower", "vision_adapter", "vision_projection")


def sanitize(model, weights):
    out = {}
    for k, v in weights.items():
        # Meta's checkpoints have the LLaVA layout, and older MLX conversions
        # the text model's.
        k = from_llava_layout(k)
        if k.removeprefix("model.").startswith(_ENCODERS):
            continue
        out[from_text_layout(k)] = v
    return out
