# Copyright © 2026 Apple Inc.


__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    out = {}
    for k, v in weights.items():
        # Drop the vision tower — this is a text-only port.
        if k.startswith(("vision_tower", "vision_adapter", "vision_projection")):
            continue
        # Meta/MLX nest the text tower under language_model.*
        if k.startswith("language_model.model."):
            k = "model." + k[len("language_model.model.") :]
        elif k.startswith("language_model.lm_head."):
            k = "lm_head." + k[len("language_model.lm_head.") :]
        elif k.startswith("model.language_model."):
            k = "model." + k[len("model.language_model.") :]
        out[k] = v
    return out
