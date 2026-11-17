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

    // With groups of E / G experts, only the TG groups with the highest sums of
    // their two best biased scores are kept; the others' get the score masked.
    if (G > 1 && TG < G) {
        constexpr int GE = E / G;
        float gscore[G];
        for (int g = 0; g < G; g++) {
            float b1 = -INFINITY;
            float b2 = -INFINITY;
            for (int i = 0; i < R; i++) {
                int e = lane + 32 * i;
                if (e < E && e / GE == g) {
                    if (biased[i] > b1) {
                        b2 = b1;
                        b1 = biased[i];
                    } else if (biased[i] > b2) {
                        b2 = biased[i];
                    }
                }
            }
            float m1 = simd_max(b1);
            uint top = simd_min(b1 == m1 ? lane : 32u);
            gscore[g] = m1 + simd_max(lane == top ? b2 : b1);
        }
        uint kept = 0;
        for (int n = 0; n < TG; n++) {
            int best_g = -1;
            float best = 0.0f;
            for (int g = 0; g < G; g++) {
                if (!((kept >> g) & 1) && (best_g < 0 || gscore[g] > best)) {
                    best = gscore[g];
                    best_g = g;
                }
            }
            kept |= 1u << best_g;
        }
        for (int i = 0; i < R; i++) {
            int e = lane + 32 * i;
            if (e < E && !((kept >> (e / GE)) & 1)) {
                biased[i] = masked;
            }
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
    input_names=["logits", "bias", "multiplier", "eps", "scale", "masked"],
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
    n_group: int = 1,
    topk_group: int = 1,
    masked_score: float = 0.0,
) -> list[mx.array]:
    """The experts and their weights from the router logits, in one kernel. With
    ``n_group`` groups of experts, the experts are picked from the ``topk_group``
    groups with the highest sums of their two best biased scores, as in
    DeepSeek-V3: the biased scores of the other groups' experts become
    ``masked_score``."""
    E = logits.shape[-1]
    return _routing_kernel(
        inputs=[
            logits,
            expert_bias,
            scalar(multiplier),
            scalar(score_eps),
            scalar(route_scale),
            scalar(masked_score),
        ],
        template=[("E", E), ("K", top_k), ("G", n_group), ("TG", topk_group)],
        grid=(logits.size // E * 32, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[logits.shape[:-1] + (top_k,)] * 2,
        output_dtypes=[mx.uint32, mx.float32],
    )
