# Copyright © 2026 Apple Inc.

"""What the models that take images, audio or other inputs share."""

import mlx.core as mx

__all__ = [
    "from_llava_layout",
    "splice",
]


def from_llava_layout(key: str) -> str:
    """The current name of a key in the LLaVA layout, as transformers maps it.

    language_model.model.* becomes model.language_model.*, language_model.lm_head.*
    becomes lm_head.*, and the vision tower and the projector move under model.
    Other keys are unchanged.
    """
    if key.startswith("language_model.lm_head."):
        return key.removeprefix("language_model.")
    if key.startswith("language_model.model."):
        return "model.language_model." + key.removeprefix("language_model.model.")
    if key.startswith(("vision_tower.", "multi_modal_projector.")):
        return "model." + key
    return key


def splice(
    embeddings: mx.array,
    inputs: mx.array,
    *,
    rows: mx.array,
    token_id: int,
) -> mx.array:
    """Put the rows, in order, at the positions of the token in the inputs."""
    # A shape check only, so no host sync.
    if rows.shape[0] == 0:
        return embeddings
    mask = inputs == token_id
    # The row of each masked token: the number of masked tokens before it.
    idx = mx.cumsum(mask.flatten()).reshape(mask.shape) - 1
    idx = mx.clip(idx, 0, rows.shape[0] - 1)
    selected = rows[idx].astype(embeddings.dtype)
    return mx.where(mask[..., None], selected, embeddings)
