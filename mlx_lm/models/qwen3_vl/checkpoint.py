# Copyright © 2025 Apple Inc.

from mlx.utils import tree_flatten, tree_unflatten

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    weights = tree_unflatten(list(weights.items()))
    weights.pop("visual", None)
    weights.pop("vision_tower", None)
    # Newer HF checkpoints nest both towers under a top-level "model".
    if language_model := weights.get("model", {}).get("language_model"):
        weights["model"] = language_model
    weights = dict(tree_flatten(weights))

    sanitized = {}
    for key, value in weights.items():
        if not key.startswith("language_model."):
            key = "language_model." + key
        sanitized[key] = value
    return sanitized
