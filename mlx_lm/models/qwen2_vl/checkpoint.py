# Copyright © 2025 Apple Inc.

import re

import mlx.core as mx

__all__ = [
    "current_name",
    "sanitize",
    "upgrade_mlx_layout",
]


def upgrade_mlx_layout(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Weights mlx-lm or mlx-vlm saved under language_model, in the current layout."""
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


def current_name(key: str) -> str:
    """The current name of a key in Qwen2-VL's Hub layout, as transformers maps it."""
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
