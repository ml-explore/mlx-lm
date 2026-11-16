# Copyright © 2026 Apple Inc.


__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    lm_weights = {}
    for key, value in weights.items():
        if "vision_tower" in key or "multi_modal_projector" in key:
            continue
        if key.startswith("model.language_model."):
            key = "model." + key.removeprefix("model.language_model.")
        else:
            key = key.removeprefix("language_model.")
        lm_weights[key] = value

    sanitized_lm = model.language_model.sanitize(lm_weights)
    return {"language_model." + k: v for k, v in sanitized_lm.items()}
