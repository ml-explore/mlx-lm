# Copyright © 2026 Apple Inc.

"""The routing after the router matmul, in one kernel."""

import mlx.core as mx

from ._common import scalar

_ROUTING_SOURCE = """
    constexpr int R = (E + 31) / 32;
    uint token = thread_position_in_grid.x / 32;
    uint lane = thread_index_in_simdgroup;
    auto row = logits + size_t(token) * E;

    float score[R];
    float biased[R];
    for (int i = 0; i < R; i++) {
        int e = lane + 32 * i;
        score[i] = 0.0f;
        biased[i] = -INFINITY;
        if (e < E) {
            float x = row[e] * multiplier;
            float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
            score[i] = x < 0 ? y : 1.0f - y;
            biased[i] = score[i] + bias[e];
        }
    }

    float w[K];
    for (int k = 0; k < K; k++) {
        float best = -INFINITY;
        int best_e = E;
        for (int i = 0; i < R; i++) {
            int e = lane + 32 * i;
            if (biased[i] > best || (biased[i] == best && e < best_e)) {
                best = biased[i];
                best_e = e;
            }
        }
        float top = simd_max(best);
        int winner = simd_min(best == top ? best_e : E);
        float s = 0.0f;
        for (int i = 0; i < R; i++) {
            if (lane + 32 * i == winner) {
                s = score[i];
                biased[i] = -INFINITY;
            }
        }
        w[k] = simd_sum(s);
        if (lane == 0) {
            inds[size_t(token) * K + k] = winner;
        }
    }

    float total = 0.0f;
    for (int k = 0; k < K; k++) {
        total = w[k] + total;
    }
    if (lane == 0) {
        for (int k = 0; k < K; k++) {
            weights[size_t(token) * K + k] = w[k] / (total + eps) * scale;
        }
    }
"""

_routing_kernel = mx.fast.metal_kernel(
    name="expert_routing",
    input_names=["logits", "bias", "multiplier", "eps", "scale"],
    output_names=["inds", "weights"],
    source=_ROUTING_SOURCE,
)


def expert_routing(
    logits: mx.array,
    expert_bias: mx.array,
    *,
    top_k: int,
    multiplier: float,
    score_eps: float,
    route_scale: float,
) -> list[mx.array]:
    """The experts and their weights from the router logits, in one kernel."""
    E = logits.shape[-1]
    return _routing_kernel(
        inputs=[
            logits,
            expert_bias,
            scalar(multiplier),
            scalar(score_eps),
            scalar(route_scale),
        ],
        template=[("E", E), ("K", top_k)],
        grid=(logits.size // E * 32, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[logits.shape[:-1] + (top_k,)] * 2,
        output_dtypes=[mx.uint32, mx.float32],
    )
