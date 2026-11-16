# Copyright © 2025 Apple Inc.

from mlx.utils import tree_flatten, tree_unflatten

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    weights = tree_unflatten(list(weights.items()))
    weights.pop("vision_tower", None)
    weights.pop("multi_modal_projector", None)
    lm_weights = dict(tree_flatten(weights["language_model"]))
    lm_weights = model.language_model.sanitize(lm_weights)
    weights["language_model"] = tree_unflatten(list(lm_weights.items()))
    return dict(tree_flatten(weights))
