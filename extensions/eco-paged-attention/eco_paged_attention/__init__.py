"""ECO D256 GQA decode extension for MLX 0.32.2."""

import math

import mlx.core as mx

from . import _ext


def _paged_attention(q, k, v, tables, lengths, *, scale, stream=None):
    """Internal entry point for cache-owned, validated page metadata."""
    if not all(isinstance(x, mx.array) for x in (q, k, v, tables, lengths)):
        raise TypeError("Inputs must be MLX arrays")
    if not math.isfinite(scale):
        raise ValueError("Scale must be finite")
    if q.ndim != 4 or q.shape[2:] != (1, 256):
        raise ValueError("Expected queries [batch, heads, 1, 256]")
    if k.ndim != 4 or k.shape != v.shape or k.shape[-1] != 256 or k.shape[1] == 0:
        raise ValueError("Expected matching KV pages [pages, kv_heads, page_size, 256]")
    if q.shape[1] not in (6 * k.shape[1], 8 * k.shape[1]):
        raise ValueError("Only GQA factors 6 and 8 are supported")
    if (
        q.dtype not in (mx.float32, mx.float16, mx.bfloat16)
        or k.dtype != q.dtype
        or v.dtype != q.dtype
    ):
        raise ValueError("Expected matching float32, float16 or bfloat16 inputs")
    if (
        tables.ndim != 2
        or tables.shape[0] != q.shape[0]
        or lengths.shape != (q.shape[0],)
    ):
        raise ValueError("Invalid page metadata shape")
    if tables.dtype != mx.uint32 or lengths.dtype != mx.uint32:
        raise ValueError("Page metadata must use uint32")
    if k.shape[0] == 0 or k.shape[2] == 0 or tables.shape[1] == 0 or q.shape[0] == 0:
        raise ValueError("Empty page pools and batches are unsupported")
    stream = mx.default_stream(mx.gpu) if stream is None else stream
    if not isinstance(stream, mx.Stream) or stream.device != mx.gpu:
        raise ValueError("Expected a GPU stream")
    out = mx.zeros(q.shape, dtype=q.dtype)
    _ext.paged_attention(q, k, v, tables, lengths, scale, out, stream)
    return out


def paged_attention(q, k, v, tables, lengths, *, scale, stream=None):
    """Decode over paged KV. Metadata validation synchronizes with the CPU."""
    # Validate shapes and dtypes before reading metadata. Evaluation remains lazy.
    out = _paged_attention(q, k, v, tables, lengths, scale=scale, stream=stream)
    rows = tables.tolist()
    for row, length in zip(rows, lengths.tolist()):
        if not 0 < length <= len(row) * k.shape[2]:
            raise ValueError("Context lengths must fit the page table and be positive")
        if any(
            page >= k.shape[0]
            for page in row[: (length + k.shape[2] - 1) // k.shape[2]]
        ):
            raise ValueError("Physical page index is outside the pool")
    return out
