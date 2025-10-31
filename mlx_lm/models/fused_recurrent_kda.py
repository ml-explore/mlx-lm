# Copyright © 2025 Apple Inc.

from typing import Optional, Tuple

import mlx.core as mx

from .gated_delta import _gated_delta_step_ops


@mx.compile
def _fused_recurrent_kda_scan(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    outputs = []
    T = q.shape[1]
    for t in range(T):
        y_t, state = _gated_delta_step_ops(
            q[:, t],
            k[:, t],
            v[:, t],
            g[:, t],
            beta[:, t],
            state,
            None if mask is None else mask[:, t],
        )
        outputs.append(y_t)
    y = mx.stack(outputs, axis=1)
    return y, state


def _ensure_state(
    q: mx.array,
    v: mx.array,
    state: Optional[mx.array],
) -> mx.array:
    if state is not None:
        return state
    B = q.shape[0]
    Hv, Dv = v.shape[-2:]
    Dk = q.shape[-1]
    return mx.zeros((B, Hv, Dv, Dk), dtype=q.dtype)


def fused_recurrent_kda_ops(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    state_buf = _ensure_state(q, v, state)
    return _fused_recurrent_kda_scan(q, k, v, g, beta, state_buf, mask)


def chunked_kda_ops(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
    chunk_size: int = 64,
) -> Tuple[mx.array, mx.array]:
    state_buf = _ensure_state(q, v, state)
    outputs = []
    T = q.shape[1]
    start = 0
    while start < T:
        end = min(start + chunk_size, T)
        y_chunk, state_buf = _fused_recurrent_kda_scan(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            g[:, start:end],
            beta[:, start:end],
            state_buf,
            None if mask is None else mask[:, start:end],
        )
        outputs.append(y_chunk)
        start = end
    return mx.concatenate(outputs, axis=1), state_buf
