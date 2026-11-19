# Copyright © 2026 Apple Inc.

from mlx_lm.models.checkpoint import from_llava_layout
from mlx_lm.models.deepseek_v3.checkpoint import sanitize_language_model

__all__ = [
    "sanitize",
]

# Only the language model runs here.
_ENCODERS = ("vision_tower", "vision_model", "multi_modal_projector", "mm_projector")


def sanitize(model, weights):
    # The checkpoints, and older MLX conversions, have the LLaVA layout.
    weights = {from_llava_layout(k): v for k, v in weights.items()}
    weights = {
        k: v
        for k, v in weights.items()
        if not k.removeprefix("model.").startswith(_ENCODERS)
    }
    return sanitize_language_model(
        model.model.language_model, weights, prefix="model.language_model."
    )
