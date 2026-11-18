# Copyright © 2026 Apple Inc.

from mlx_lm.models.checkpoint import from_llava_layout, from_text_layout
from mlx_lm.models.mistral3.config import language_model_package

__all__ = [
    "sanitize",
]

# Only the language model runs here.
_ENCODERS = ("model.vision_tower.", "model.multi_modal_projector.")


def sanitize(model, weights):
    # Older checkpoints, and older MLX conversions, have the LLaVA layout.
    weights = {from_llava_layout(k): v for k, v in weights.items()}
    weights = {k: v for k, v in weights.items() if not k.startswith(_ENCODERS)}
    # Text-only conversions have the text model's layout.
    weights = {from_text_layout(k): v for k, v in weights.items()}
    if model.args.text.tie_word_embeddings:
        weights.pop("lm_head.weight", None)
    language_model = language_model_package(model.args.text.model_type)
    return language_model.checkpoint.sanitize_language_model(
        model.model.language_model, weights, prefix="model.language_model."
    )
