# Copyright © 2026 Apple Inc.

import mlx.core as mx

from ._common import row_threads, scalar

# This kernel inputs [tokens * k, D]:
# normalises each input token x_i / sqrt(sum(x_i)**2) * w_i -- rms norm
# and then combine as a weighted sum
_EXPERT_COMBINE_SOURCE = """
    constexpr int R = (D + TG - 1) / TG;
    constexpr int SG = TG / 32;
    threadgroup float buf[K * SG];
    uint t = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    size_t row = threadgroup_position_in_grid.x;

    float w[R];
    for (int i = 0; i < R; i++) {
        int d = t + i * TG;
        w[i] = d < D ? static_cast<float>(norm_w[d]) : 0.0f;
    }
    T x[K][R];
    for (int k = 0; k < K; k++) {
        size_t base = (row * K + k) * D;
        for (int i = 0; i < R; i++) {
            int d = t + i * TG;
            x[k][i] = d < D ? experts[base + d] : static_cast<T>(0.0f);
        }
    }
    float sum_sq[K];
    for (int k = 0; k < K; k++) {
        sum_sq[k] = 0.0f;
        for (int i = 0; i < R; i++) {
            float v = static_cast<float>(x[k][i]) * w[i];
            sum_sq[k] += v * v;
        }
        sum_sq[k] = simd_sum(sum_sq[k]);
    }
    if (TG > 32) {
        if (lane == 0) {
            for (int k = 0; k < K; k++) {
                buf[k * SG + sg] = sum_sq[k];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int k = 0; k < K; k++) {
            sum_sq[k] = simd_sum(lane < SG ? buf[k * SG + lane] : 0.0f);
        }
    }

    float acc[R];
    for (int i = 0; i < R; i++) {
        acc[i] = 0.0f;
    }
    for (int k = 0; k < K; k++) {
        float inv = metal::precise::rsqrt(sum_sq[k] / D + eps);
        float ew = static_cast<T>(weights[row * K + k]);
        for (int i = 0; i < R; i++) {
            float v = static_cast<float>(x[k][i]) * w[i];
            acc[i] = static_cast<T>(acc[i] + v * inv * ew);
        }
    }
    for (int i = 0; i < R; i++) {
        int d = t + i * TG;
        if (d < D) {
            out[row * D + d] = static_cast<T>(acc[i]);
        }
    }
"""

_expert_combine_kernel = mx.fast.metal_kernel(
    name="post_expert_norm_combine",
    input_names=["experts", "weights", "norm_w", "eps"],
    output_names=["out"],
    source=_EXPERT_COMBINE_SOURCE,
)


def post_expert_norm_combine(
    experts: mx.array, weights: mx.array, *, norm_weight: mx.array, eps: float
) -> mx.array:
    """Norm each expert output with norm_weight as its input scale, and add
    the experts with their weights in order. experts is (..., K, D) and
    weights is (..., K); the result is (..., D)."""
    K, D = experts.shape[-2:]
    tg = min(1024, 2 * row_threads(D))
    return _expert_combine_kernel(
        inputs=[experts, weights, norm_weight, scalar(eps)],
        template=[("T", experts.dtype), ("D", D), ("K", K), ("TG", tg)],
        grid=(experts.size // (K * D) * tg, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[experts.shape[:-2] + (D,)],
        output_dtypes=[experts.dtype],
    )[0]
