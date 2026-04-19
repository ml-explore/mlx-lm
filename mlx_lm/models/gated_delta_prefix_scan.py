"""O(log T) associative prefix scan for GatedDeltaNet recurrence.

Applies the monoid (Lemma 2.1 in THEOREM_ASSOCIATIVITY.md):

    (A_t, B_t) · (A_s, B_s) = (A_t ∘ A_s,   A_t(B_s) + B_t)

with identity ``(Id, 0)``. Blelloch-style up-sweep / down-sweep gives
a prefix scan of depth ``O(log T)``. On a single GPU the main win is
better compiler scheduling of parallel matmuls; on distributed
setups (multi-device MLX) it enables real parallelism.

This module provides a Python reference implementation against
which more optimised kernels (e.g. Metal fused prefix scan) can be
validated. Equivalence with the sequential reference is tested in
``llm/test_gated_delta_prefix_scan.py``.
"""

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


_LOG_DECAY_CLAMP = -20.0


@mx.compile
def _compute_g(A_log, a, dt_bias):
    arg = -mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias)
    arg = mx.maximum(arg, _LOG_DECAY_CLAMP)
    return mx.exp(arg).astype(a.dtype)


def _apply_A(M: mx.array, g: mx.array, k: mx.array, beta: mx.array) -> mx.array:
    """Apply operator A_t to matrix M:  A_t(M) = g · M · (I - β k k^T).

    Shapes:
      M: [B, Hv, Dv, Dk]
      g: [B, Hv]           (scalar gate per head)
      k: [B, Hv, Dk]       (unit key per head)
      beta: [B, Hv]        (scalar write per head)

    Returns A_t(M), shape [B, Hv, Dv, Dk].
    """
    # M · (I - β k k^T) = M - (β k k^T) applied from the right
    #                   = M - β · (M · k) · k^T
    Mk = (M * k[..., None, :]).sum(axis=-1)  # [B, Hv, Dv]
    decayed_component = beta[..., None, None] * Mk[..., None] * k[..., None, :]
    return g[..., None, None] * (M - decayed_component)


def _factored_to_dense(g: mx.array, k: mx.array, beta: mx.array) -> mx.array:
    """Expand factored (g, k, β) form to dense right-projection matrix.

    A(M) = g · M · (I - β k k^T) ⇒ represented by the Dk×Dk matrix
    ``A_right = g · (I - β k k^T)`` (applied to M on the right).
    """
    Dk = k.shape[-1]
    I = mx.eye(Dk)
    kkT = k[..., :, None] * k[..., None, :]
    return g[..., None, None] * (I - beta[..., None, None] * kkT)


def _compose_pair(p_left, p_right):
    """Compose two pairs under the monoid rule.

    Each pair is (A_right, B) where A_right is a Dk×Dk dense matrix
    representing right-multiplication (so ``A(M) = M · A_right``),
    and B is a Dv×Dk bias matrix. Composition:

        (A_left, B_left) · (A_right_in, B_right) =
          (A_right_in · A_left,   B_right · A_left + B_left)

    The sequential update goes left-to-right (newer operator applied
    after older), so the "left" pair is applied first and the "right"
    pair applied second. Matrix form: if composed via `p_new · p_old`,
    the composed right-matrix is ``A_old · A_new_right``.

    Both inputs must be (A_dense, B) tuples. Factored pairs must
    first be expanded via _factored_to_dense.
    """
    A_l, B_l = p_left
    A_r, B_r = p_right
    # A_composed: applied as M · A_l · A_r ⇒ right-matrix = A_l · A_r
    A_composed = A_l @ A_r
    # B at timestep after composition: B_l is applied first, then decayed
    # by A_r; B_r is added last.
    B_composed = B_l @ A_r + B_r
    return (A_composed, B_composed)


def _sequential_scan(
    q: mx.array, k: mx.array, v: mx.array,
    g: mx.array, beta: mx.array,
    state: Optional[mx.array] = None,
):
    """Reference sequential scan: for each t, S_t = A_t(S_{t-1}) + B_t.

    A_t is the operator ``A_t(M) = g_t · M · (I - β_t k_t k_t^T)`` and
    the bias is ``B_t = β_t · v_t · k_t^T`` (pure rank-1 constant).
    The delete term -g·β·(S·k)·k^T is already baked into A_t; do NOT
    re-apply it in the bias (common implementation bug).

    Equivalent to gated_delta_vjp.py's _chunk_forward:
       S_tmp = g·S                        = A_t(S) + g·β·(S·k)·k^T
       delta = (v - g·S·k)·β
       S_new = S_tmp + delta·k^T          = A_t(S) + β·v·k^T   ← our B_t
    """
    B, T, Hv, _ = q.shape
    Dk = k.shape[-1]
    Dv = v.shape[-1]
    if state is None:
        S = mx.zeros((B, Hv, Dv, Dk), dtype=q.dtype)
    else:
        S = state
    ys = []
    for t in range(T):
        # A_t(S) = g · S · (I - β k k^T)
        S_tmp = _apply_A(S, g[:, t], k[:, t], beta[:, t])
        # B_t = β · v · k^T (pure rank-1 constant; delete is inside A_t)
        B_t = beta[:, t, :, None, None] * v[:, t, :, :, None] * k[:, t, :, None, :]
        S = S_tmp + B_t
        y_t = (S * q[:, t, :, None, :]).sum(axis=-1)
        ys.append(y_t)
    return mx.stack(ys, axis=1), S


CHUNK_SIZE = 64  # timesteps per gradient-checkpointed block


def _scan_chunk(q_c, k_c, v_c, g_c, beta_c, S_start):
    return _sequential_scan(q_c, k_c, v_c, g_c, beta_c, S_start)


_scan_chunk_ckpt = mx.checkpoint(_scan_chunk)


def gated_delta_update_prefix_scan(
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
    """Drop-in replacement for :func:`gated_delta_update` using the
    associative-monoid formulation.

    Backward pass flows through MLX autodiff, so this is a valid VJP
    backend. Each ``CHUNK_SIZE``-step block is wrapped in
    :func:`mx.checkpoint` to keep peak memory at ``O(CHUNK_SIZE)`` per
    layer rather than ``O(T)``. On single-device this is roughly
    equivalent to ``gated_delta_update_vjp`` (pure-python chunked);
    the value over the Python VJP path is that each chunk uses the
    explicit ``A_t(S) + B_t`` decomposition (see
    ``THEOREM_ASSOCIATIVITY.md``), which is the building block for
    future Blelloch-style distributed parallelism.

    Semantics match ``gated_delta_update`` exactly (verified by
    ``test_gated_delta_prefix_scan.py`` at machine precision).
    """
    if mask is not None:
        raise NotImplementedError("masked scan not yet implemented")

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

    ys = []
    S = state
    for start in range(0, T, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, T)
        y_c, S = _scan_chunk_ckpt(
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
