# Copyright © 2026 Apple Inc.

__all__ = [
    "from_llava_layout",
    "from_text_layout",
]


def from_llava_layout(key: str) -> str:
    """The current name of a key in the LLaVA layout, as transformers maps it."""
    if key.startswith("language_model.lm_head."):
        return key.removeprefix("language_model.")
    if key.startswith("language_model.model."):
        return "model.language_model." + key.removeprefix("language_model.model.")
    if key.startswith(("vision_tower.", "multi_modal_projector.")):
        return "model." + key
    return key


def from_text_layout(key: str) -> str:
    """The name in a multimodal model of a key in a text model's layout.

    Drop the encoders' keys first: they are under model too.
    """
    if key.startswith("model.") and not key.startswith("model.language_model."):
        return "model.language_model." + key.removeprefix("model.")
    return key
