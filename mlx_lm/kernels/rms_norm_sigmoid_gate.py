# Copyright © 2026 Apple Inc.

"""The attention output norm and sigmoid gate, in one kernel."""

import mlx.core as mx

from ._common import scalar

_NORM_GATE_SOURCE = """
    // One SIMD group per (token, head) row of D values.
    uint row = thread_position_in_grid.x / 32;
    uint lane = thread_index_in_simdgroup;
    uint B = x_shape[0], H = x_shape[1], L = x_shape[2];
    if (row >= B * L * H) {
        return;
    }
    uint h = row % H;
    uint l = (row / H) % L;
    uint b = row / (H * L);
    int xd = x_strides[3], gd = g_strides[2];
    auto xr = x + b * x_strides[0] + h * x_strides[1] + l * x_strides[2];
    auto gr = g + b * g_strides[0] + l * g_strides[1] + h * D * gd;

    constexpr int R = (D + 31) / 32;
    float v[R];
    float sum_sq = 0.0f;
    for (int i = 0; i < R; i++) {
        int d = lane + 32 * i;
        v[i] = d < D ? static_cast<float>(xr[d * xd]) : 0.0f;
        sum_sq += v[i] * v[i];
    }
    float rstd = metal::precise::rsqrt(simd_sum(sum_sq) / D + eps);
    for (int i = 0; i < R; i++) {
        int d = lane + 32 * i;
        if (d < D) {
            float sig = 1.0f / (1.0f + metal::exp(-static_cast<float>(gr[d * gd])));
            out[row * D + d] = static_cast<T>(v[i] * rstd * sig);
        }
    }
"""

_norm_gate_kernel = mx.fast.metal_kernel(
    name="rms_norm_sigmoid_gate",
    input_names=["x", "g", "eps"],
    output_names=["out"],
    source=_NORM_GATE_SOURCE,
    ensure_row_contiguous=False,
)


def rms_norm_sigmoid_gate(x: mx.array, gate: mx.array, *, eps: float) -> mx.array:
    """Return rms_norm(x) * sigmoid(gate) for each head, like the reference's
    fused kernel: fp32 math and one rounding. x is the attention output
    (B, H, L, D), gate is (B, L, H * D), and the result is (B, L, H * D)."""
    B, H, L, D = x.shape
    return _norm_gate_kernel(
        inputs=[x, gate, scalar(eps)],
        template=[("T", x.dtype), ("D", D)],
        grid=(B * L * H * 32, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, L, H * D)],
        output_dtypes=[x.dtype],
    )[0]
