"""Metal VJP with compression-aware rank truncation at chunk boundaries.

Combines the 8-11× speed of Metal VJP backward with the 5× extra memory
savings of compression-aware training.

Structure:
  for each CHUNK_SIZE-step chunk:
    y_c, S = metal_vjp_core(q_c, k_c, v_c, g_c, beta_c, S)   # Metal fwd+bwd
    S = power_iter_truncate(S, rank)                         # differentiable

The Metal kernel handles one chunk at a time (forward-with-save + backward
via custom_function). Between chunks, state is projected onto top-r
subspace via power iteration. Backward flows through the projection
(U treated as stop_gradient basis; gradient flows through U·U^T·S).

Result: training compresses at boundary points → 5× memory savings at
backward, while forward/backward inner loop runs on Metal (8-11× faster
than Python VJP).

Activation: MLX_DELTANET_VJP=compress with MLX_DELTANET_COMPRESS_METAL=1
(or set backend='compress' and import from this file instead of
gated_delta_vjp_compressed).
"""

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .gated_delta_vjp_compressed import (
    _LOG_DECAY_CLAMP,
    DEFAULT_POWER_ITERS,
    DEFAULT_RANK,
    _gram_schmidt,
    _power_iter_truncate,
)
from .gated_delta_vjp_metal import _gated_delta_core as _metal_core

CHUNK_SIZE = 64


@mx.compile
def _compute_g(A_log: mx.array, a: mx.array, dt_bias: mx.array) -> mx.array:
    arg = -mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias)
    arg = mx.maximum(arg, _LOG_DECAY_CLAMP)
    return mx.exp(arg).astype(a.dtype)


def gated_delta_update_vjp_metal_compressed(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
    rank: int = DEFAULT_RANK,
    power_iters: int = DEFAULT_POWER_ITERS,
) -> Tuple[mx.array, mx.array]:
    """Metal VJP + compression-aware truncation.

    Drop-in for gated_delta_update_vjp_compressed but backward runs
    on the Metal kernel. Expected ~8× faster than pure-Python compressed.

    mask path: falls back to non-Metal (masked-Metal bwd not implemented).
    """
    if mask is not None:
        raise NotImplementedError(
            "masked + Metal compressed VJP not yet implemented — "
            "fall back to gated_delta_update_vjp_compressed (Python)"
        )

    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]

    beta = mx.sigmoid(b)
    g = _compute_g(A_log, a, dt_bias)

    rf = Hv // Hk
    if rf > 1:
        q = mx.repeat(q, rf, axis=-2)
        k = mx.repeat(k, rf, axis=-2)

    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=q.dtype)

    effective_rank = min(rank, min(Dv, Dk))
    do_compress = rank > 0 and effective_rank < min(Dv, Dk)

    ys = []
    S = state
    for start in range(0, T, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, T)
        # Metal forward+save, backward via custom_function.vjp.
        y_c, S = _metal_core(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            g[:, start:end],
            beta[:, start:end],
            S,
        )
        ys.append(y_c)
        if do_compress:
            # Differentiable rank-r projection; VJP flows through U·U^T·S
            # (U is stop_gradient'd orthonormal basis).
            S = _power_iter_truncate(S, effective_rank, power_iters)
    y = mx.concatenate(ys, axis=1) if len(ys) > 1 else ys[0]
    return y, S
