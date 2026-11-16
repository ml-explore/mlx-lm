# Copyright © 2026 Apple Inc.

from mlx.utils import tree_flatten, tree_unflatten

from mlx_lm.models.deepseek_v3.model import Model as DeepseekV3LM

__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    weights = tree_unflatten(list(weights.items()))
    weights.pop("vision_tower", None)
    weights.pop("vision_model", None)
    weights.pop("multi_modal_projector", None)
    weights.pop("mm_projector", None)
    lm_weights = dict(tree_flatten(weights["language_model"]))
    lm_weights = DeepseekV3LM.sanitize(model.language_model, lm_weights)
    weights["language_model"] = tree_unflatten(list(lm_weights.items()))
    return dict(tree_flatten(weights))
