"""
Supports:
  - a, b with shape (batch, seqlen, ngroups, K) or (batch, seqlen, K).
  - Optional causal masking (zero out upper-triangle per chunk).

Entrypoint: ``run(a, b, chunk_size, causal=False)``
"""

import math

import mlx.core as mx


def run(
    a: mx.array,
    b: mx.array,
    chunk_size: int,
    causal: bool = False,
) -> mx.array:
    """Naive batched matmul within chunks.

    Args:
        a: (batch, seqlen, ngroups, K) or (batch, seqlen, K)
        b: (batch, seqlen, ngroups, K) or (batch, seqlen, K)  — same shape as a
        chunk_size: int — size of each chunk along the sequence dimension
        causal: bool — if True, zero out positions where row > col within each chunk

    Returns:
        out: (batch, nchunks, ngroups, chunk_size, chunk_size) or
             (batch, nchunks, chunk_size, chunk_size)
    """
    has_groups = a.ndim == 4

    if has_groups:
        batch, seqlen, ngroups, K = a.shape
    else:
        batch, seqlen, K = a.shape
        ngroups = 1

    assert (
        a.shape == b.shape
    ), f"a and b must have the same shape, got {a.shape} vs {b.shape}"

    nchunks = math.ceil(seqlen / chunk_size)

    # Pad sequence to nchunks * chunk_size if needed.
    pad_len = nchunks * chunk_size - seqlen
    if pad_len > 0:
        if has_groups:
            pad_shape = (batch, pad_len, ngroups, K)
        else:
            pad_shape = (batch, pad_len, K)
        z = mx.zeros(pad_shape, dtype=a.dtype)
        a = mx.concatenate([a, z], axis=1)
        b = mx.concatenate([b, z], axis=1)

    # Reshape into chunks.
    if has_groups:
        # (batch, seqlen, ngroups, K) -> (batch, nchunks, chunk_size, ngroups, K)
        a = a.reshape(batch, nchunks, chunk_size, ngroups, K)
        b = b.reshape(batch, nchunks, chunk_size, ngroups, K)
        # -> (batch, nchunks, ngroups, chunk_size, K)
        a = mx.transpose(a, (0, 1, 3, 2, 4))
        b = mx.transpose(b, (0, 1, 3, 2, 4))
        # Matmul: (batch, nchunks, ngroups, chunk_size, K) @ (batch, nchunks, ngroups, K, chunk_size)
        # -> (batch, nchunks, ngroups, chunk_size, chunk_size)
        out = a @ mx.transpose(b, (0, 1, 2, 4, 3))
    else:
        # (batch, seqlen, K) -> (batch, nchunks, chunk_size, K)
        a = a.reshape(batch, nchunks, chunk_size, K)
        b = b.reshape(batch, nchunks, chunk_size, K)
        # Matmul: (batch, nchunks, chunk_size, K) @ (batch, nchunks, K, chunk_size)
        # -> (batch, nchunks, chunk_size, chunk_size)
        out = a @ mx.transpose(b, (0, 1, 3, 2))

    # Apply causal mask: zero out positions where row > col (i.e. keep lower triangle + diagonal).
    if causal:
        row_idx = mx.arange(chunk_size).reshape(chunk_size, 1)
        col_idx = mx.arange(chunk_size).reshape(1, chunk_size)
        mask = row_idx >= col_idx  # lower triangular (including diagonal)
        out = mx.where(mask, out, mx.array(0.0, dtype=out.dtype))

    mx.eval(out)
    return out
