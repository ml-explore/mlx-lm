"""Bit-packing for TurboQuant indices.

Packs multiple small-bit indices into uint32 words:
- 1-bit: 32 values per uint32
- 2-bit: 16 values per uint32
- 3-bit: 10 values per uint32 (30/32 bits used)
- 4-bit:  8 values per uint32

For 3-bit with dim=128: 13 uint32s per vector (52 bytes) vs 128 bytes (uint8).
Combined with float32 norm: 56 bytes/vector vs 256 bytes (fp16) = 4.6x compression.
"""

import mlx.core as mx
import math

VALS_PER_WORD = {1: 32, 2: 16, 3: 10, 4: 8}
BIT_MASK = {1: 0x1, 2: 0x3, 3: 0x7, 4: 0xF}


def packed_dim(dim: int, bits: int) -> int:
    """Number of uint32 words needed to pack `dim` values at `bits` each."""
    vpw = VALS_PER_WORD[bits]
    return (dim + vpw - 1) // vpw


def pack_indices(indices: mx.array, bits: int) -> mx.array:
    """Pack uint8 indices into uint32 words.

    Args:
        indices: (..., dim) uint8, values in [0, 2^bits)
        dim: last dimension

    Returns:
        (..., packed_dim) uint32
    """
    vpw = VALS_PER_WORD[bits]
    shape = indices.shape
    dim = shape[-1]
    flat = indices.reshape(-1, dim).astype(mx.uint32)
    n_vecs = flat.shape[0]
    p_dim = packed_dim(dim, bits)

    # Pad to multiple of vpw
    if dim % vpw != 0:
        pad_size = vpw - (dim % vpw)
        flat = mx.concatenate([flat, mx.zeros((n_vecs, pad_size), dtype=mx.uint32)], axis=1)

    # Reshape to (n_vecs, p_dim, vpw) and pack
    flat = flat.reshape(n_vecs, p_dim, vpw)

    # Shift each value by its position and OR together
    packed = mx.zeros((n_vecs, p_dim), dtype=mx.uint32)
    for i in range(vpw):
        packed = packed | (flat[:, :, i] << (i * bits))

    return packed.reshape(*shape[:-1], p_dim)


def unpack_indices(packed: mx.array, bits: int, dim: int) -> mx.array:
    """Unpack uint32 words back to uint8 indices.

    Args:
        packed: (..., packed_dim) uint32
        bits: bit width
        dim: original dimension

    Returns:
        (..., dim) uint8
    """
    vpw = VALS_PER_WORD[bits]
    mask = BIT_MASK[bits]
    shape = packed.shape
    p_dim = shape[-1]
    flat = packed.reshape(-1, p_dim)
    n_vecs = flat.shape[0]

    # Extract each value
    values = []
    for i in range(vpw):
        values.append((flat >> (i * bits)) & mask)

    # Stack and trim to original dim
    result = mx.concatenate(values, axis=-1)  # wrong order, need interleave
    # Actually: values[i] has shape (n_vecs, p_dim) = the i-th value from each word
    # We need to reshape to (n_vecs, p_dim * vpw) then trim
    result = mx.stack(values, axis=-1)  # (n_vecs, p_dim, vpw)
    result = result.reshape(n_vecs, p_dim * vpw)[:, :dim]

    return result.reshape(*shape[:-1], dim).astype(mx.uint8)
