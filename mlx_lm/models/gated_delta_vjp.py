"""Gradient-checkpointed gated_delta_update for training.

The Metal kernel in :mod:`gated_delta` has no VJP registered, and the
Python-ops fallback builds a graph with ``O(T)`` intermediate states —
which runs out of memory for training-scale sequences (``T ≥ 2048``)
on Apple Silicon devices with 36 GB unified memory or less.

This module provides :func:`gated_delta_update_vjp`, a drop-in
training-time replacement that runs a pure-Python recurrent forward in
chunks of ``CHUNK_SIZE`` timesteps, each wrapped in ``mx.checkpoint``.
Intermediate state within a chunk is recomputed on the backward pass,
so the peak memory cost is ``O(CHUNK_SIZE)`` per layer rather than
``O(T)``.

Related: https://github.com/ml-explore/mlx-lm/issues/482
"""

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


CHUNK_SIZE = 64  # Timesteps per gradient-checkpointed block.

# Lower bound for the log-domain argument of the decay ``exp`` so that very
# large A_log or softplus(a+dt_bias) do not produce denormal bf16 values that
# poison a long recurrence. ``exp(-20) ≈ 2e-9`` — well within bf16 range.
_LOG_DECAY_CLAMP = -20.0


@mx.compile
def _compute_g(A_log: mx.array, a: mx.array, dt_bias: mx.array) -> mx.array:
    """Decay gate ``g = exp(-exp(A_log) * softplus(a + dt_bias))`` in fp32."""
    arg = -mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias)
    arg = mx.maximum(arg, _LOG_DECAY_CLAMP)
    return mx.exp(arg).astype(a.dtype)


def _chunk_forward(
    q_c: mx.array,      # [B, T_c, Hv, Dk]
    k_c: mx.array,      # [B, T_c, Hv, Dk]
    v_c: mx.array,      # [B, T_c, Hv, Dv]
    g_c: mx.array,      # [B, T_c, Hv] (scalar) or [B, T_c, Hv, Dk] (vectorized)
    beta_c: mx.array,   # [B, T_c, Hv]
    S_start: mx.array,  # [B, Hv, Dv, Dk]
) -> Tuple[mx.array, mx.array]:
    """Recurrent forward over a single unmasked chunk of ``T_c`` timesteps.

    Supports both scalar and per-channel (vectorized) gating. The arithmetic
    runs in the input dtype; fp32 accumulators double the peak memory
    without measurable impact on convergence for bf16 training.
    """
    T_c = q_c.shape[1]
    g_is_scalar = g_c.ndim == 3
    S = S_start
    ys = []
    for t in range(T_c):
        if g_is_scalar:
            decay = g_c[:, t, :, None, None]
        else:
            decay = g_c[:, t, :, None, :]
        S_tmp = S * decay
        k_t = k_c[:, t]
        kv_mem = (S_tmp * k_t[..., None, :]).sum(axis=-1)
        delta = (v_c[:, t] - kv_mem) * beta_c[:, t, :, None]
        S = S_tmp + k_t[..., None, :] * delta[..., None]
        y_t = (S * q_c[:, t, :, None, :]).sum(axis=-1)
        ys.append(y_t)
    y_c = mx.stack(ys, axis=1)
    return y_c, S


def _chunk_forward_masked(
    q_c: mx.array,
    k_c: mx.array,
    v_c: mx.array,
    g_c: mx.array,
    beta_c: mx.array,
    S_start: mx.array,
    mask_c: mx.array,   # [B, T_c] (bool/int, broadcast to state shape)
) -> Tuple[mx.array, mx.array]:
    """Masked variant: when ``mask_c[b, t] == False`` the state is carried
    over unchanged from the previous step (matching the reference ops path).

    ``y_t`` is still produced as if the update had happened, so the output
    shape is unaffected — consumers must themselves ignore the padding
    positions downstream.
    """
    T_c = q_c.shape[1]
    g_is_scalar = g_c.ndim == 3
    S = S_start
    ys = []
    for t in range(T_c):
        if g_is_scalar:
            decay = g_c[:, t, :, None, None]
        else:
            decay = g_c[:, t, :, None, :]
        S_tmp = S * decay
        k_t = k_c[:, t]
        kv_mem = (S_tmp * k_t[..., None, :]).sum(axis=-1)
        delta = (v_c[:, t] - kv_mem) * beta_c[:, t, :, None]
        S_new = S_tmp + k_t[..., None, :] * delta[..., None]
        y_t = (S_new * q_c[:, t, :, None, :]).sum(axis=-1)
        # Pass the prior state through for padded steps.
        m_t = mask_c[:, t, None, None, None]
        S = mx.where(m_t, S_new, S)
        ys.append(y_t)
    y_c = mx.stack(ys, axis=1)
    return y_c, S


_chunk_forward_ckpt = mx.checkpoint(_chunk_forward)
_chunk_forward_masked_ckpt = mx.checkpoint(_chunk_forward_masked)


def gated_delta_update_vjp(
    q: mx.array,           # [B, T, Hk, Dk]
    k: mx.array,           # [B, T, Hk, Dk]
    v: mx.array,           # [B, T, Hv, Dv]
    a: mx.array,           # [B, T, Hv]
    b: mx.array,           # [B, T, Hv]
    A_log: mx.array,       # [Hv]
    dt_bias: mx.array,     # [Hv]
    state: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """Drop-in training replacement for :func:`gated_delta_update`.

    Argument shapes and semantics match the standard forward function.
    Gradients flow through all inputs via :func:`mx.checkpoint`, so both
    the forward and backward pass use ``O(CHUNK_SIZE)`` peak memory per
    layer rather than ``O(T)``.

    ``mask`` is ``[B, T]`` and should be ``True`` for positions that
    participate in the recurrent update. For masked positions the state
    is carried over unchanged — matching the reference ops path.
    """
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]

    beta = mx.sigmoid(b)
    g = _compute_g(A_log, a, dt_bias)

    repeat_factor = Hv // Hk
    if repeat_factor > 1:
        q = mx.repeat(q, repeat_factor, axis=-2)
        k = mx.repeat(k, repeat_factor, axis=-2)

    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=q.dtype)

    # Chunked forward; each chunk is a pure function of the incoming state,
    # so autodiff propagates gradients correctly across the recurrence.
    ys = []
    S = state
    for start in range(0, T, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, T)
        if mask is None:
            y_c, S = _chunk_forward_ckpt(
                q[:, start:end],
                k[:, start:end],
                v[:, start:end],
                g[:, start:end],
                beta[:, start:end],
                S,
            )
        else:
            y_c, S = _chunk_forward_masked_ckpt(
                q[:, start:end],
                k[:, start:end],
                v[:, start:end],
                g[:, start:end],
                beta[:, start:end],
                S,
                mask[:, start:end],
            )
        ys.append(y_c)
    y = mx.concatenate(ys, axis=1) if len(ys) > 1 else ys[0]
    return y, S
