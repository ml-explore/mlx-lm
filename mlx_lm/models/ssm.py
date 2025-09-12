import mlx.core as mx
import mlx.nn as nn


@mx.compile
def compute_dt(dt, dt_bias, time_step_limit):
    dt = nn.softplus(dt + dt_bias)
    return mx.clip(dt, time_step_limit[0], time_step_limit[1])


def ssm_step_ops(
    hidden_states: mx.array,
    A_log: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    dt: mx.array,
    dt_bias: mx.array,
    state: mx.array,
    time_step_limit=(0.001, 100.0),
) -> mx.array:
    batch_size, seq_len, num_heads, head_dim = hidden_states.shape
    state_size = B.shape[-1]
    dt = compute_dt(dt, dt_bias, time_step_limit)
    A = -mx.exp(A_log.astype(mx.float32)).astype(hidden_states.dtype)
    B = B.reshape(batch_size, seq_len, 1, state_size)
    C = C.reshape(batch_size, seq_len, 1, state_size)
    C = mx.repeat(C, num_heads, axis=2)
    dA = mx.exp(dt * A)
    dB = dt[..., None] * mx.repeat(B, num_heads, axis=2)
    dB_h = dB[..., None, :] * hidden_states[..., None]
    hs = []
    for t in range(seq_len):
        state = dA[:, t, :, None, None] * state + dB_h[:, t]
        hs.append(state)
    Dh = D[:, None] * hidden_states
    y = (mx.stack(hs, axis=1) @ C[..., None]).squeeze(-1) + Dh
    return y, state


def make_ssm_kernel():
    source = f"""
        // TODO this needs a fix for batching
        auto n = thread_position_in_grid.z;
        constexpr int n_per_t = Ds / 32;

        auto x = X + n * Dh;
        out += n * Dh;
        auto i_state = state_in + n * Dh * Ds;
        auto o_state = state_out + n * Dh * Ds;

        auto ds_idx = thread_position_in_threadgroup.x;
        auto d_idx = thread_position_in_grid.y;

        auto dt_ = static_cast<float>(dt[n]);
        auto A = -fast::exp(static_cast<float>(A_log[n]));
        auto dA = fast::exp(A * dt_);

        float acc = 0.0;
        auto x_ = static_cast<float>(x[d_idx]);

        for (int i = 0; i < n_per_t; ++i) {{
            auto s_idx = n_per_t * ds_idx + i;
            auto idx = d_idx * Ds + s_idx;
            auto dB_by_x = x_ * dt_ * static_cast<float>(B[s_idx]);
            auto state = dA * i_state[idx] + dB_by_x;
            o_state[idx] = static_cast<T>(state);
            acc += state * C[s_idx];
        }}
        acc = simd_sum(acc);
        if (thread_index_in_simdgroup == 0) {{
            out[d_idx] = static_cast<T>(acc + x_ * D[n]);
        }}
    """
    return mx.fast.metal_kernel(
        name="ssm_kernel",
        input_names=["X", "A_log", "B", "C", "D", "dt", "state_in"],
        output_names=["out", "state_out"],
        source=source,
    )


_ssm_kernel = make_ssm_kernel()


def ssm_step(
    hidden_states: mx.array,
    A_log: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    dt: mx.array,
    dt_bias: mx.array,
    state: mx.array,
    time_step_limit=(0.001, 100.0),
):
    input_type = hidden_states.dtype
    n, _, h, d = hidden_states.shape
    ds = B.shape[-1]
    dt = compute_dt(dt, dt_bias, time_step_limit)
    return _ssm_kernel(
        inputs=[hidden_states, A_log, B, C, D, dt, state],
        template=[("T", input_type), ("Dh", d), ("Ds", ds), ("H", h)],
        grid=(32, d, h * n),
        threadgroup=(32, 8, 1),
        output_shapes=[(n, 1, h, d), state.shape],
        output_dtypes=[input_type, input_type],
    )
