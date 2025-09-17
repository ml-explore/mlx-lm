from functools import partial
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


@partial(mx.compile, shapeless=True)
def compute_g(A_log, a, dt_bias):
    return mx.exp(-mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias)).astype(
        A_log.dtype
    )


def _make_gated_delta_kernel_step():
    if not mx.metal.is_available():
        return None
    source = """
        auto n = thread_position_in_grid.z;
        auto g_idx = n / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, Hk, Dk]
        auto q_ = q + g_idx * Dk;
        auto k_ = k + g_idx * Dk;

        // v, y: [B, Hv, Dv]
        auto v_ = v + n * Dv;
        y += n * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        // beta, g: [B, Hv]

        float kv_mem = 0.0f;
        float out_s[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            out_s[i] = static_cast<float>(i_state[s_idx]) * g[n];
            kv_mem += out_s[i] * k_[s_idx];
        }
        kv_mem = simd_sum(kv_mem);

        auto delta = (v_[dv_idx] - kv_mem) * beta[n];

        float out = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            out_s[i] = out_s[i] + k_[s_idx] * delta;
            o_state[s_idx] = static_cast<T>(out_s[i]);
            out += out_s[i] * q_[s_idx];
        }
        out = simd_sum(out);
        if (thread_index_in_simdgroup == 0) {
            y[dv_idx] = static_cast<T>(out);
        }
    """
    return mx.fast.metal_kernel(
        name="gated_delta_step",
        input_names=["q", "k", "v", "g", "beta", "state_in"],
        output_names=["y", "state_out"],
        source=source,
    )


_kernel_step = _make_gated_delta_kernel_step()


def _gated_delta_step_ops(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
) -> Tuple[mx.array, mx.array]:
    """
    Ops-based reference implementation for a single recurrent step.

    Shapes:
      - q, k: [B, 1, H, Dk]
      - v: [B, 1, H, Dv]
      - state: [B, H, Dv, Dkv]
    Returns:
      - y: [B, 1, H, Dv]
      - new_state: [B, H, Dv, Dk]
    """
    q = q.squeeze(1)
    k = k.squeeze(1)
    v = v.squeeze(1)
    g = g.squeeze(1)
    beta = beta.squeeze(1)

    # Decay
    state = state * g[..., None, None]
    kv_mem = (state * k[..., None, :]).sum(axis=-1)  # [B, H, Dv]
    delta = (v - kv_mem) * beta[..., None]  # [B, H, Dv]
    state = state + k[..., None, :] * delta[..., None]
    # Output projection along key dim with q
    y = (state * q[..., None, :]).sum(axis=-1)  # [B, H, Dv]
    return mx.expand_dims(y, 1), state


def gated_delta_kernel(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
) -> Tuple[mx.array, mx.array]:
    B, _, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    input_type = q.dtype
    return _kernel_step(
        inputs=[q, k, v, g, beta, state],
        template=[("T", input_type), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv)],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, 1, Hv, Dv), state.shape],
        output_dtypes=[input_type, input_type],
    )


def gated_delta_prefill(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """
    Ops-based reference implementation for prompt prefill (sequential loop).

    Shapes:
      - q, k: [B, T, Hk, Dk]
      - v: [B, T, Hv, Dv]
      - g, beta: [B, T, Hv]
      - state: [B, Hv, Dk, Dv]
    Returns:
      - y: [B, T, Hv, Dv]
      - state: [B, Hv, Dk, Dv]
    """
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=q.dtype)

    if (repeat_factor := Hv // Hk) > 1:
        q = mx.repeat(q, repeat_factor, -2)
        k = mx.repeat(k, repeat_factor, -2)

    ys = []
    for t in range(T):
        y, state = _gated_delta_step_ops(
            q[:, t : t + 1],
            k[:, t : t + 1],
            v[:, t : t + 1],
            g[:, t : t + 1],
            beta[:, t : t + 1],
            state,
        )
        ys.append(y)
    y = mx.concatenate(ys, axis=1)
    return y, state


def gated_delta_update(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:

    beta = mx.sigmoid(b)
    g = compute_g(A_log, a, dt_bias)

    if (
        q.shape[1] > 1
        or state is None
        or mx.default_device() != mx.gpu
        or not mx.metal.is_available()
    ):
        return gated_delta_prefill(q, k, v, g, beta, state)
    else:
        return gated_delta_kernel(q, k, v, g, beta, state)
