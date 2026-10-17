# Copyright © 2026 Apple Inc.

"""Helpers that the kernels share."""

from functools import lru_cache

import mlx.core as mx

# A sum over all threads of a threadgroup of TG threads.
BLOCK_SUM_HEADER = """
template <int TG>
inline float block_sum(float v, threadgroup float* buf, uint lane, uint sg) {
    v = simd_sum(v);
    if (TG > 32) {
        if (lane == 0) {
            buf[sg] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        v = simd_sum(lane < TG / 32 ? buf[lane] : 0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return v;
}
"""


@lru_cache
def scalar(value: float, dtype: mx.Dtype = mx.float32) -> mx.array:
    """A cached 0-d array, which a kernel gets as a constant."""
    return mx.array(value, dtype=dtype)


def row_threads(dim: int) -> int:
    """Threads for a threadgroup that handles one row of dim values."""
    return min(1024, max(32, dim // 8 // 32 * 32))
