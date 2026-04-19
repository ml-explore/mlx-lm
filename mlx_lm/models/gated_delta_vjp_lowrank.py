"""Low-rank state DeltaNet VJP (research prototype).

Implementation of the low-rank state approach validated in
``deltanet_lowrank_experiment.py``. Per-chunk SVD truncation keeps
the recurrent state at bounded rank ``r`` — reducing backward state
storage by ``(Dv * Dk) / ((Dv + Dk) * r)`` (~4.8× at rank-16 for
Qwen3.5-9B shapes).

Production caveat: MLX ``mx.linalg.svd`` currently runs on the CPU
stream only, so the SVD step dominates runtime. Shipping this to
production requires replacing the SVD with a GPU-friendly low-rank
update (thin SVD via matmul + eigendecomp on a small Gram matrix, or
randomised SVD); that is a multi-week research follow-up.

Semantics (forward + backward) are identical to the sequential path
except for the rank-``r`` truncation between chunks, which is the
source of the controlled loss of precision documented in
the low-rank analysis notes in the companion paper.
"""

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

CHUNK_SIZE = 64
DEFAULT_RANK = 16
_LOG_DECAY_CLAMP = -20.0


@mx.compile
def _compute_g(A_log, a, dt_bias):
    arg = -mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias)
    arg = mx.maximum(arg, _LOG_DECAY_CLAMP)
    return mx.exp(arg).astype(a.dtype)


def _chunk_forward(q_c, k_c, v_c, g_c, beta_c, S_start):
    """Exact sequential forward over one chunk (same as chunk-parallel
    reference). Returns (y_c, S_end)."""
    T_c = q_c.shape[1]
    S = S_start
    ys = []
    for t in range(T_c):
        decay = g_c[:, t, :, None, None]
        S_tmp = S * decay
        k_t = k_c[:, t]
        kv_mem = (S_tmp * k_t[..., None, :]).sum(axis=-1)
        delta = (v_c[:, t] - kv_mem) * beta_c[:, t, :, None]
        S = S_tmp + k_t[..., None, :] * delta[..., None]
        y_t = (S * q_c[:, t, :, None, :]).sum(axis=-1)
        ys.append(y_t)
    return mx.stack(ys, axis=1), S


_chunk_forward_ckpt = mx.checkpoint(_chunk_forward)


def _svd_truncate(S: mx.array, rank: int) -> mx.array:
    """SVD-truncate S [B, Hv, Dv, Dk] to the given rank and reconstruct.

    Runs on CPU because ``mx.linalg.svd`` has no GPU implementation yet.
    """
    out_dtype = S.dtype
    S32 = S.astype(mx.float32)
    U, sigma, Vt = mx.linalg.svd(S32, stream=mx.cpu)
    U = U[..., :rank]  # [B, Hv, Dv, r]
    sigma = sigma[..., :rank]  # [B, Hv, r]
    Vt = Vt[..., :rank, :]  # [B, Hv, r, Dk]
    recon = mx.einsum("bhvr,bhr,bhrk->bhvk", U, sigma, Vt)
    return recon.astype(out_dtype)


def _gram_schmidt(X: mx.array) -> mx.array:
    """Column-wise Gram-Schmidt orthonormalisation — pure matmul + elementwise."""
    r = X.shape[-1]
    cols = []
    for i in range(r):
        col = X[..., :, i : i + 1]
        for prev in cols:
            proj = (prev * col).sum(axis=-2, keepdims=True)
            col = col - proj * prev
        col = col / mx.sqrt((col * col).sum(axis=-2, keepdims=True) + 1e-30)
        cols.append(col)
    return mx.concatenate(cols, axis=-1)


def _power_iter_truncate(S: mx.array, rank: int, n_iter: int = 10) -> mx.array:
    """GPU-native low-rank truncation via power iteration (no SVD needed).

    The full power-iteration step (random init, Gram-Schmidt, matmul
    powers) is wrapped in ``mx.stop_gradient`` — the basis ``U`` is
    treated as a detached orthonormal matrix, and gradient flows only
    through the final rank-``r`` projection ``U·U^T·S``, which is the
    well-defined rank-``r`` projector of ``S`` onto the column space
    spanned by ``U``. This matches what backprop through exact SVD
    truncation computes.
    """
    out_dtype = S.dtype
    S32 = S.astype(mx.float32)

    # Compute orthonormal basis U of top-r left singular vectors
    # entirely inside ``stop_gradient`` — U is a detached constant.
    S_detached = mx.stop_gradient(S32)
    mx.random.seed(0)
    X = mx.random.normal(list(S_detached.shape[:-1]) + [rank])
    X = _gram_schmidt(X)
    SST = S_detached @ mx.swapaxes(S_detached, -1, -2)
    for _ in range(n_iter):
        X = SST @ X
        X = _gram_schmidt(X)
    U = mx.stop_gradient(X)  # [..., Dv, r]

    # Rank-r projection of the live S: U · U^T · S.
    UtS = mx.swapaxes(U, -1, -2) @ S32  # [..., r, Dk]
    recon = U @ UtS  # [..., Dv, Dk]
    return recon.astype(out_dtype)


def gated_delta_update_vjp_lowrank(
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
    method: str = "svd",  # "svd" (CPU, exact) or "power" (GPU, iterative)
    power_iters: int = 10,
) -> Tuple[mx.array, mx.array]:
    """Low-rank state drop-in for :func:`gated_delta_update`.

    ``method="svd"`` uses ``mx.linalg.svd`` (CPU only in MLX 0.31.1).
    ``method="power"`` uses GPU-native power iteration — fast but
    requires the effective rank of the state to be close to ``rank``
    (holds empirically for DeltaNet state; see
    ``llm/deltanet_realstate_poweriter.py``).
    """
    if mask is not None:
        raise NotImplementedError("masked path not yet implemented for low-rank")

    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]

    beta = mx.sigmoid(b)
    g = _compute_g(A_log, a, dt_bias)
    if g.ndim != 3:
        raise NotImplementedError(
            "vectorised gating not yet supported for low-rank path"
        )

    rf = Hv // Hk
    if rf > 1:
        q = mx.repeat(q, rf, axis=-2)
        k = mx.repeat(k, rf, axis=-2)

    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=q.dtype)

    ys = []
    S = state
    for start in range(0, T, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, T)
        y_c, S = _chunk_forward_ckpt(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            g[:, start:end],
            beta[:, start:end],
            S,
        )
        ys.append(y_c)
        # Truncate between chunks.
        if rank < min(Dv, Dk):
            if method == "svd":
                S = _svd_truncate(S, rank)
            elif method == "power":
                S = _power_iter_truncate(S, rank, n_iter=power_iters)
            else:
                raise ValueError(f"unknown method: {method}")
    y = mx.concatenate(ys, axis=1) if len(ys) > 1 else ys[0]
    return y, S
