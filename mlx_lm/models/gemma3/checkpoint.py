# Copyright © 2025 Apple Inc.

from mlx_lm.layers.multimodal import from_llava_layout

__all__ = [
    "sanitize",
]

# Only the language model runs here.
_ENCODERS = ("model.vision_tower.", "model.multi_modal_projector.")


def sanitize(model, weights):
    # Older checkpoints, and older MLX conversions, have the LLaVA layout.
    weights = {from_llava_layout(k): v for k, v in weights.items()}
    weights = {k: v for k, v in weights.items() if not k.startswith(_ENCODERS)}
    # The output layer is the embedding unless the checkpoint has its own.
    if "lm_head.weight" not in weights:
        model.tie_word_embeddings = True
        model.pop("lm_head")
    return weights
