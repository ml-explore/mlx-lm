# Copyright © 2026 Apple Inc.

"""The softmax routing after the router matmul, in one kernel."""

import mlx.core as mx

# One simdgroup per token. The weights are the softmax over all experts, or, with
# NORM, renormalized over the top K (the softmax of the top K logits).
_SOFTMAX_TOPK_SOURCE = """
    constexpr int R = (E + 31) / 32;
    uint token = thread_position_in_grid.x / 32;
    uint lane = thread_index_in_simdgroup;
    auto row = logits + size_t(token) * E;
    float v[R];
    float m = -INFINITY;
    for (int i = 0; i < R; i++) {
        int e = lane + 32 * i;
        v[i] = e < E ? static_cast<float>(row[e]) : -INFINITY;
        m = metal::max(m, v[i]);
    }
    m = simd_max(m);
    float z = 0.0f;
    for (int i = 0; i < R; i++) {
        z += metal::precise::exp(v[i] - m);
    }
    z = simd_sum(z);

    float w[K];
    for (int k = 0; k < K; k++) {
        float best = -INFINITY;
        int best_e = E;
        for (int i = 0; i < R; i++) {
            int e = lane + 32 * i;
            if (v[i] > best || (v[i] == best && e < best_e)) {
                best = v[i];
                best_e = e;
            }
        }
        float top = simd_max(best);
        int winner = simd_min(best == top ? best_e : E);
        for (int i = 0; i < R; i++) {
            if (lane + 32 * i == winner) {
                v[i] = -INFINITY;
            }
        }
        w[k] = metal::precise::exp(top - m);
        if (lane == 0) {
            inds[size_t(token) * K + k] = winner;
        }
    }

    float total = z;
    if (NORM) {
        total = 0.0f;
        for (int k = 0; k < K; k++) {
            total += w[k];
        }
    }
    if (lane == 0) {
        for (int k = 0; k < K; k++) {
            weights[size_t(token) * K + k] = static_cast<WT>(w[k] / total);
        }
    }
"""

_softmax_topk_kernel = mx.fast.metal_kernel(
    name="softmax_topk",
    input_names=["logits"],
    output_names=["inds", "weights"],
    source=_SOFTMAX_TOPK_SOURCE,
)


def softmax_topk(logits: mx.array, *, top_k: int, normalize: bool) -> list[mx.array]:
    """The top ``top_k`` experts of each token and their softmax weights, renormalized
    over the top ``top_k`` if ``normalize``, from the router logits. The weights are
    computed in float32 and returned in the dtype of ``logits``, as ``mx.softmax``."""
    E = logits.shape[-1]
    return _softmax_topk_kernel(
        inputs=[logits],
        template=[("E", E), ("K", top_k), ("NORM", normalize), ("WT", logits.dtype)],
        grid=(logits.size // E * 32, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[logits.shape[:-1] + (top_k,)] * 2,
        output_dtypes=[mx.uint32, logits.dtype],
    )
