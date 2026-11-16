# Copyright © 2026 Apple Inc.

"""What the models that take images, audio or other inputs share."""

import mlx.core as mx

__all__ = [
    "splice",
]


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
