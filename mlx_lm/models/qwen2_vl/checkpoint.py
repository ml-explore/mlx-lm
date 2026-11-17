# Copyright © 2025 Apple Inc.

import re

__all__ = [
    "current_name",
    "sanitize",
    "upgrade_mlx_layout",
]


def upgrade_mlx_layout(weights):
    """A checkpoint that mlx-lm or mlx-vlm converted with the language model under
    language_model."""
    upgraded = {}
    for k, v in weights.items():
        if k.startswith("vision_tower."):
            continue
        if k.startswith("language_model.model."):
            k = "model.language_model." + k.removeprefix("language_model.model.")
        else:
            k = k.removeprefix("language_model.")
        upgraded[k] = v
    return upgraded


def current_name(key):
    """The name of a weight of Qwen2-VL's checkpoints on the Hub, which have the
    language model under model and the vision encoder under visual, in the current
    layout. transformers renames them the same way."""
    if key.startswith("visual."):
        return "model." + key
    return re.sub(
        r"^model\.(?!language_model\.|visual\.)", "model.language_model.", key
    )


def sanitize(model, weights):
    if any(k.startswith("language_model.") for k in weights):
        weights = upgrade_mlx_layout(weights)
    weights = {current_name(k): v for k, v in weights.items()}
    # mlx-lm does not run the vision encoder yet.
    weights = {k: v for k, v in weights.items() if not k.startswith("model.visual.")}
    if model.args.text.tie_word_embeddings:
        weights.pop("lm_head.weight", None)
    return weights
