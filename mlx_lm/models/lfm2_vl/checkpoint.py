# Copyright © 2025 Apple Inc.

from mlx_lm.layers.multimodal import from_llava_layout
from mlx_lm.models.lfm2.checkpoint import sanitize_language_model

__all__ = [
    "sanitize",
]

# Only the language model runs here.
_ENCODERS = ("model.vision_tower.", "model.multi_modal_projector.")


def sanitize(model, weights):
    # Older MLX conversions have the LLaVA layout.
    weights = {from_llava_layout(k): v for k, v in weights.items()}
    weights = {k: v for k, v in weights.items() if not k.startswith(_ENCODERS)}
    # The output layer is the embedding.
    weights.pop("lm_head.weight", None)
    return sanitize_language_model(
        model.model.language_model, weights, prefix="model.language_model."
    )
