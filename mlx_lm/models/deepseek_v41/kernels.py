# Copyright © 2026 Apple Inc.

from typing import Callable, Optional

import mlx.core as mx

__all__ = [
    "sinkhorn",
]


def _make_sinkhorn_kernel() -> Optional[Callable]:
    if not mx.metal.is_available():
        return None
    # The loop of sinkhorn in one launch, one thread per matrix, in the same order.
    source = """
        uint n = thread_position_in_grid.x;
        if (n >= comb_shape[0]) {
            return;
        }
        float m[HC * HC];
        for (int k = 0; k < HC * HC; k++) {
            m[k] = comb[n * HC * HC + k];
        }
        float e = eps[0];
        for (int t = 0; t < 2 * ITERS - 1; t++) {
            bool rows = t % 2 == 1;
            for (int a = 0; a < HC; a++) {
                float s = 0.0f;
                for (int b = 0; b < HC; b++) {
                    s += rows ? m[a * HC + b] : m[b * HC + a];
                }
                float d = s + e;
                for (int b = 0; b < HC; b++) {
                    int k = rows ? a * HC + b : b * HC + a;
                    m[k] = m[k] / d;
                }
            }
        }
        for (int k = 0; k < HC * HC; k++) {
            out[n * HC * HC + k] = m[k];
        }
    """
    return mx.fast.metal_kernel(
        name="deepseek_v41_sinkhorn",
        input_names=["comb", "eps"],
        output_names=["out"],
        source=source,
    )


_sinkhorn_kernel = _make_sinkhorn_kernel()


def sinkhorn(comb: mx.array, *, iters: int, eps: mx.array) -> mx.array:
    """Normalize the columns, then the rows and columns iters - 1 more times."""
    if (
        _sinkhorn_kernel is None
        or mx.default_device() != mx.gpu
        or iters < 1
        or comb.size == 0
    ):
        comb = comb / (mx.sum(comb, axis=-2, keepdims=True) + eps)
        for _ in range(iters - 1):
            comb = comb / (mx.sum(comb, axis=-1, keepdims=True) + eps)
            comb = comb / (mx.sum(comb, axis=-2, keepdims=True) + eps)
        return comb
    hc = comb.shape[-1]
    flat = comb.reshape(-1, hc, hc)
    (out,) = _sinkhorn_kernel(
        inputs=[flat, eps],
        template=[("HC", hc), ("ITERS", iters)],
        grid=(flat.shape[0], 1, 1),
        threadgroup=(min(flat.shape[0], 256), 1, 1),
        output_shapes=[flat.shape],
        output_dtypes=[mx.float32],
    )
    return out.reshape(comb.shape)
