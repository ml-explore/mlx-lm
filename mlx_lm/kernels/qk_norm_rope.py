# Copyright © 2026 Apple Inc.

"""The q and k head norms, the q scale and the rotary embedding, in one
kernel."""

import math

import mlx.core as mx

from ._common import scalar

# qk_norm, then the rotation of mx.fast.rope(traditional=True), which rotates
# the rounded q and k.
_QK_NORM_ROPE_SOURCE = """
    constexpr int R = (D + 31) / 32;
    uint row = thread_position_in_grid.x / 32;
    uint lane = thread_index_in_simdgroup;
    uint n = row / (HQ + HK);
    uint h = row % (HQ + HK);
    auto x = qkvg + size_t(n) * (2 * (HQ + HK) * D) + h * D;

    float v[R];
    float sum_sq = 0.0f;
    for (int i = 0; i < R; i++) {
        int d = lane + 32 * i;
        v[i] = d < D ? static_cast<float>(x[d]) : 0.0f;
        sum_sq += v[i] * v[i];
    }
    float rstd = metal::precise::rsqrt(simd_sum(sum_sq) / D + eps);
    float pos = static_cast<float>(n % seq_len + offset[(n / seq_len) * OFFSET_STRIDE]);
    for (int i = 0; i < R; i++) {
        int d = lane + 32 * i;
        T y = static_cast<T>(v[i] * rstd);
        if (h < HQ) {
            y = static_cast<T>(static_cast<float>(y) * scale);
        }
        float self = static_cast<float>(y);
        float other = simd_shuffle_xor(self, 1);
        float inv_freq = metal::exp2(-(static_cast<float>(d / 2) / (D / 2)) * log2_base);
        float theta = pos * inv_freq;
        float c = metal::fast::cos(theta);
        float s = metal::fast::sin(theta);
        y = static_cast<T>(d % 2 == 0 ? self * c - other * s : other * s + self * c);
        if (d < D) {
            if (h < HQ) {
                q[(size_t(n) * HQ + h) * D + d] = y;
            } else {
                k[(size_t(n) * HK + h - HQ) * D + d] = y;
            }
        }
    }
"""

_qk_norm_rope_kernel = mx.fast.metal_kernel(
    name="qk_norm_rope",
    input_names=["qkvg", "scale", "eps", "offset", "seq_len", "log2_base"],
    output_names=["q", "k"],
    source=_QK_NORM_ROPE_SOURCE,
)


def qk_norm_rope(
    qkvg: mx.array,
    *,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    eps: float,
    query_scale: float,
    rope_base: float,
    offset: int | mx.array,
) -> list[mx.array]:
    """qk_norm, then rotate q and k as mx.fast.rope(traditional=True) does,
    the tokens at positions offset, offset + 1, ... offset is an int, or an
    array with one offset per batch row. Returns q (..., n_heads, head_dim)
    and k (..., n_kv_heads, head_dim)."""
    tokens = qkvg.shape[:-1]
    rows = qkvg.size // qkvg.shape[-1] * (n_heads + n_kv_heads)
    if isinstance(offset, mx.array):
        offset = offset.reshape(-1).astype(mx.int32)
    else:
        offset = mx.array([offset], dtype=mx.int32)
    return _qk_norm_rope_kernel(
        inputs=[
            qkvg,
            scalar(query_scale),
            scalar(eps),
            offset,
            scalar(tokens[-1], mx.int32),
            # As mx.fast.rope, which takes the log2 of the fp32 base.
            scalar(math.log2(rope_base)),
        ],
        template=[
            ("T", qkvg.dtype),
            ("D", head_dim),
            ("HQ", n_heads),
            ("HK", n_kv_heads),
            ("OFFSET_STRIDE", int(offset.size > 1)),
        ],
        grid=(rows * 32, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[tokens + (n_heads, head_dim), tokens + (n_kv_heads, head_dim)],
        output_dtypes=[qkvg.dtype, qkvg.dtype],
    )
