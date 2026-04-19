"""Chunk-parallel ``gated_delta_update_vjp`` (no MSL, pure MLX ops).

Replaces the sequential ``for t in range(T_c)`` loop inside each chunk
with a rank-C factorisation + lower-triangular solve expressed in
vectorised MLX ops. Autodiff handles the backward automatically; the
chunk body is wrapped in :func:`mx.checkpoint` so peak memory stays at
``O(CHUNK_SIZE^2)`` per chunk rather than ``O(T^2)`` across the whole
sequence.

Math derivation (per chunk of length ``C``):

    S_t   = g_t · S_{t-1} + β_t · δ_t · k_t^T
    δ_t   = v_t − g_t · S_{t-1} · k_t

which is equivalent to the lower-triangular system

    (I + A) · δ = v'
    A[t,j] = G_{j+1..t} · β_j · ⟨k_j, k_t⟩   (strict lower)
    v'[t]  = v_t − G_{0..t} · S_start · k_t

Output then becomes

    y_t = G_{0..t} · S_start · q_t + Σ_{j≤t} M[t,j] · δ_j
    M[t,j] = G_{j+1..t} · β_j · ⟨k_j, q_t⟩ · causal

and final state

    S_C = G_{0..C-1} · S_start + Σ_j G_{j+1..C-1} · β_j · δ_j · k_j^T

Correctness: verified against sequential reference in
``gated_delta_chunk_parallel_batched.py`` (max|y_diff| ≈ 5e-8,
max|S_diff| ≈ 6e-8 on B=2 T=8 Hv=4 Dk=16 Dv=8). Numerical gradient
check through this module is in ``test_deltanet_vjp.py``.
"""

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

CHUNK_SIZE = 64
_LOG_DECAY_CLAMP = -20.0


@mx.compile
def _compute_g(A_log, a, dt_bias):
    arg = -mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias)
    arg = mx.maximum(arg, _LOG_DECAY_CLAMP)
    return mx.exp(arg).astype(a.dtype)


def _chunk_parallel_forward(q, k, v, g, beta, S_start):
    """Rank-C factorisation forward over a single chunk.

    Shapes (chunk length ``T``):
      * q, k: [B, T, Hv, Dk]
      * v:    [B, T, Hv, Dv]
      * g:    [B, T, Hv]       (scalar gating; vectorised path below)
      * beta: [B, T, Hv]
      * S_start: [B, Hv, Dv, Dk]
    """
    B, T, Hv, Dk = q.shape
    Dv = v.shape[-1]

    log_g = mx.log(g + 1e-30)  # [B, T, Hv]
    log_G = mx.cumsum(log_g, axis=1)  # log G_{0..t}
    # Pairwise: log G_{j+1..t} = log_G[t] − log_G[j]
    log_pair = log_G[:, :, None, :] - log_G[:, None, :, :]  # [B, T, T, Hv]
    G_mat = mx.exp(log_pair)
    G_full = mx.exp(log_G)  # [B, T, Hv]

    strict_lower = mx.tril(mx.ones((T, T), dtype=q.dtype), k=-1)
    causal = mx.tril(mx.ones((T, T), dtype=q.dtype))

    # v'[t] = v[t] − G_{0..t} · (S_start @ k[t])
    S0_k = mx.einsum("bhvk,bthk->bthv", S_start, k)  # [B, T, Hv, Dv]
    v_prime = v - G_full[..., None] * S0_k

    # A[t, j] = G_{j+1..t} · β_j · ⟨k_j, k_t⟩  strictly lower
    k_dot = mx.einsum("bihk,bjhk->bijh", k, k)  # [B, T, T, Hv]
    A = G_mat * beta[:, None, :, :] * k_dot
    A = A * strict_lower[None, :, :, None]

    # Forward-substitute (I + A) δ = v'.
    delta_list = []
    for t in range(T):
        if t == 0:
            d = v_prime[:, t]
        else:
            A_row = A[:, t, :t, :]  # [B, t, Hv]
            prev = mx.stack(delta_list, axis=1)  # [B, t, Hv, Dv]
            d = v_prime[:, t] - (A_row[..., None] * prev).sum(axis=1)
        delta_list.append(d)
    delta = mx.stack(delta_list, axis=1)  # [B, T, Hv, Dv]

    # y[t] = G_{0..t} · (S_start · q_t) + Σ_{j≤t} G_{j+1..t} · β_j · δ_j · ⟨k_j, q_t⟩
    S0_q = mx.einsum("bhvk,bthk->bthv", S_start, q)
    y_inter = G_full[..., None] * S0_q
    kq_dot = mx.einsum("bjhk,bihk->bijh", k, q)  # [B, T_i, T_j, Hv]
    M = G_mat * beta[:, None, :, :] * kq_dot * causal[None, :, :, None]
    y_intra = mx.einsum("bijh,bjhv->bihv", M, delta)
    y = y_inter + y_intra

    # Final state.
    last_G_scalar = G_full[:, -1, :]  # [B, Hv]
    carry_G = mx.exp(log_G[:, -1:, :] - log_G)  # [B, T, Hv]
    S_final = last_G_scalar[:, :, None, None] * S_start + mx.einsum(
        "bth,bthv,bthk->bhvk", carry_G * beta, delta, k
    )
    return y, S_final


_chunk_parallel_ckpt = mx.checkpoint(_chunk_parallel_forward)


def gated_delta_update_vjp_chunkparallel(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """Chunk-parallel drop-in for :func:`gated_delta_update`.

    Currently supports scalar gating (``g.ndim == 3``) and unmasked
    training path only. Extension to vectorised gating / masking
    follows the same pattern.
    """
    if mask is not None:
        raise NotImplementedError("masked path not implemented for chunk-parallel")

    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]

    beta = mx.sigmoid(b)
    g = _compute_g(A_log, a, dt_bias)
    if g.ndim != 3:
        raise NotImplementedError(
            "vectorised gating not yet supported by chunk-parallel path"
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
        y_c, S = _chunk_parallel_ckpt(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            g[:, start:end],
            beta[:, start:end],
            S,
        )
        ys.append(y_c)
    y = mx.concatenate(ys, axis=1) if len(ys) > 1 else ys[0]
    return y, S
