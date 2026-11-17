# Copyright © 2025 Apple Inc.

from mlx_lm.layers.multimodal import from_llava_layout
from mlx_lm.models.gemma4_text.checkpoint import sanitize_language_model

__all__ = [
    "sanitize",
]

# Only the language model runs here. vision_embedder is gemma4_unified's
# encoder-free vision input.
_ENCODERS = (
    "vision_tower",
    "multi_modal_projector",
    "audio_tower",
    "embed_audio",
    "embed_vision",
    "vision_embedder",
)


def sanitize(model, weights):
    # Older MLX conversions have the LLaVA layout.
    weights = {from_llava_layout(k): v for k, v in weights.items()}
    weights = {
        k: v
        for k, v in weights.items()
        if not k.removeprefix("model.").startswith(_ENCODERS)
    }
    return sanitize_language_model(
        model.model.language_model, weights, prefix="model.language_model."
    )
