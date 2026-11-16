# Copyright © 2026 Apple Inc.

"""The q and k head norms and the q scale, in one kernel."""

import mlx.core as mx

from ._common import scalar

# One SIMD group per head. q and k are the first two parts of each qkvg row.
_QK_NORM_SOURCE = """
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
    for (int i = 0; i < R; i++) {
        int d = lane + 32 * i;
        if (d < D) {
            T y = static_cast<T>(v[i] * rstd);
            if (h < HQ) {
                q[(size_t(n) * HQ + h) * D + d] = static_cast<T>(static_cast<float>(y) * scale);
            } else {
                k[(size_t(n) * HK + h - HQ) * D + d] = y;
            }
        }
    }
"""

_qk_norm_kernel = mx.fast.metal_kernel(
    name="qk_norm",
    input_names=["qkvg", "scale", "eps"],
    output_names=["q", "k"],
    source=_QK_NORM_SOURCE,
)


def qk_norm(
    qkvg: mx.array,
    *,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    eps: float,
    query_scale: float,
) -> list[mx.array]:
    tokens = qkvg.shape[:-1]
    rows = qkvg.size // qkvg.shape[-1] * (n_heads + n_kv_heads)
    return _qk_norm_kernel(
        inputs=[qkvg, scalar(query_scale), scalar(eps)],
        template=[
            ("T", qkvg.dtype),
            ("D", head_dim),
            ("HQ", n_heads),
            ("HK", n_kv_heads),
        ],
        grid=(rows * 32, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[tokens + (n_heads, head_dim), tokens + (n_kv_heads, head_dim)],
        output_dtypes=[qkvg.dtype, qkvg.dtype],
    )
