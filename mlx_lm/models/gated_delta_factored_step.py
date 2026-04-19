"""Factored DeltaNet inference step (MLX primitives, no MSL).

Maintains state S = U @ V in factored form during generation. Per step:
  1. V' = g·V − g·β·(V·k)·k^T            (shape [r, Dk], same as V)
  2. y  = U·(V'·q) + β·v·(k·q)           (output, no dense materialisation)
  3. Append to grow rank by 1:
       U ← [U | v·β^(1/2)]
       V ← [V' ; β^(1/2)·k^T]
       (split β between U, V for numerical symmetry; product still β·v·k^T)

Rank grows by 1 per step. Truncation back to rank r is done via the
separate ``rank_truncate`` routine which performs QR-based subspace
iteration. With step-level mx.compile, the factored step is expected
to fuse into efficient Metal ops without hand-written MSL.

For generation with ``N`` tokens from a starting rank-r state:
  compute FMAs per step in factored form:  current_rank × (Dv + Dk)
  vs dense:                                 Dv × Dk
At r=16, Dv=Dk=128: factored 16·256 = 4096 vs dense 16384 — 4× fewer
FMAs per step. Rank growth over 100 tokens: final r=116. Break-even
point where dense is faster: r ≈ 128 (step 112 for starting r=16).
"""

from typing import Optional, Tuple

import mlx.core as mx


@mx.compile
def factored_step_compiled(
    U: mx.array,
    V: mx.array,
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
) -> Tuple[mx.array, mx.array, mx.array]:
    """One factored DeltaNet step. Per-head layout:

    U: [Dv, r]    factored state (left)
    V: [r, Dk]    factored state (right)
    q, k: [Dk]
    v: [Dv]
    g, beta: scalars

    Returns:
      y: [Dv]
      U_new: [Dv, r+1]
      V_new: [r+1, Dk]
    """
    # V' = g · V − g · β · (V @ k) · k^T
    Vk = (V * k[None, :]).sum(axis=-1)  # [r]
    V_prime = g * V - (g * beta) * Vk[:, None] * k[None, :]  # [r, Dk]

    # y = U @ (V' @ q) + β · v · (k · q)
    Vq = (V_prime * q[None, :]).sum(axis=-1)  # [r]
    kq = (k * q).sum()  # scalar
    y = U @ Vq + beta * v * kq  # [Dv]

    # Grow rank by 1:  U_new = [U | v],  V_new = [V' ; β · k^T]
    beta_k = beta * k
    U_new = mx.concatenate([U, v[:, None]], axis=-1)  # [Dv, r+1]
    V_new = mx.concatenate([V_prime, beta_k[None, :]], axis=0)  # [r+1, Dk]
    return y, U_new, V_new


def factored_step_batched(
    U: mx.array,  # [B, Hv, Dv, r]
    V: mx.array,  # [B, Hv, r, Dk]
    q: mx.array,  # [B, Hv, Dk]
    k: mx.array,  # [B, Hv, Dk]
    v: mx.array,  # [B, Hv, Dv]
    g: mx.array,  # [B, Hv]
    beta: mx.array,  # [B, Hv]
):
    """Batched version across heads and batch."""
    # V' = g · V - g·β·(V·k)·k^T
    Vk = (V * k[..., None, :]).sum(axis=-1)  # [B, Hv, r]
    V_prime = (
        g[..., None, None] * V
        - (g * beta)[..., None, None] * Vk[..., None] * k[..., None, :]
    )

    # y = U @ (V'·q) + β · v · (k·q)
    Vq = (V_prime * q[..., None, :]).sum(axis=-1)  # [B, Hv, r]
    kq = (k * q).sum(axis=-1)  # [B, Hv]
    y = (U * Vq[..., None, :]).sum(axis=-1) + beta[..., None] * v * kq[..., None]

    # Grow rank.
    beta_k = beta[..., None] * k  # [B, Hv, Dk]
    U_new = mx.concatenate([U, v[..., None]], axis=-1)  # [B, Hv, Dv, r+1]
    V_new = mx.concatenate([V_prime, beta_k[..., None, :]], axis=-2)  # [B, Hv, r+1, Dk]
    return y, U_new, V_new


def rank_truncate(
    U: mx.array, V: mx.array, target_rank: int
) -> Tuple[mx.array, mx.array]:
    """Truncate factored state (U, V) down to target_rank.

    Uses QR of U, small SVD of R·V, then reassembly. O(r²·(Dv+Dk)) cost.

    U: [..., Dv, r_curr]
    V: [..., r_curr, Dk]
    Returns (U_r, V_r) with shapes [..., Dv, target_rank] and [..., target_rank, Dk].
    """
    r_curr = U.shape[-1]
    if r_curr <= target_rank:
        return U, V

    # Step 1: QR of U. Q: [Dv, r_curr], R: [r_curr, r_curr]
    # mx.linalg.qr is CPU-only but small matrix cheap.
    Q, R = mx.linalg.qr(U, stream=mx.cpu)
    # Step 2: R · V is [r_curr, Dk]. Compute SVD.
    M = R @ V
    Umid, sigma, Vt = mx.linalg.svd(M, stream=mx.cpu)
    # Step 3: truncate.
    Umid_r = Umid[..., :target_rank] * sigma[..., None, :target_rank]  # absorb σ
    Vt_r = Vt[..., :target_rank, :]
    # Step 4: reassemble.
    U_r = Q @ Umid_r
    V_r = Vt_r
    return U_r, V_r


def initial_factored_state(
    B: int,
    Hv: int,
    Dv: int,
    Dk: int,
    rank: int,
    dtype=mx.bfloat16,
) -> Tuple[mx.array, mx.array]:
    """Zero initial state in factored form (rank 0 logically)."""
    # Use rank=1 dummy with zero singular value so shapes match future
    # growth. The zero column means U @ V = 0.
    U = mx.zeros((B, Hv, Dv, 1), dtype=dtype)
    V = mx.zeros((B, Hv, 1, Dk), dtype=dtype)
    return U, V
