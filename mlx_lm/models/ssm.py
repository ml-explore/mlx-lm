import mlx.core as mx
import mlx.nn as nn


@mx.compile
def compute_dt(dt, dt_bias, time_step_limit):
    dt = nn.softplus(dt + dt_bias)
    return mx.clip(dt, time_step_limit[0], time_step_limit[1])


def ssm_update_ops(
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
    n_group, state_size = B.shape[-2:]
    dt = compute_dt(dt, dt_bias, time_step_limit)
    A = -mx.exp(A_log.astype(mx.float32)).astype(hidden_states.dtype)
    n_repeats = num_heads // n_group
    C = mx.repeat(C, n_repeats, axis=2)
    dA = mx.exp(dt * A)
    dB = dt[..., None] * mx.repeat(B, n_repeats, axis=2)
    dB_h = dB[..., None, :] * hidden_states[..., None]
    hs = []
    for t in range(seq_len):
        state = dA[:, t, :, None, None] * state + dB_h[:, t]
        hs.append(state)
    Dh = D[:, None] * hidden_states
    y = (mx.stack(hs, axis=1) @ C[..., None]).squeeze(-1) + Dh
    return y, state


def make_ssm_kernel():
    if not mx.metal.is_available():
        return None
    source = """
        auto n = thread_position_in_grid.z;
        auto h_idx = n % H;
        auto g_idx = n / G;
        constexpr int n_per_t = Ds / 32;

        auto x = X + n * Dh;
        out += n * Dh;
        auto i_state = state_in + n * Dh * Ds;
        auto o_state = state_out + n * Dh * Ds;

        // C and B have shape [batch, group, state_dim]
        // C and B need to be offset by group size
        auto C_ = C + g_idx * Ds;
        auto B_ = B + g_idx * Ds;

        auto ds_idx = thread_position_in_threadgroup.x;
        auto d_idx = thread_position_in_grid.y;

        auto dt_ = static_cast<float>(dt[n]);
        auto A = -fast::exp(static_cast<float>(A_log[h_idx]));
        auto dA = fast::exp(A * dt_);

        float acc = 0.0;
        auto x_ = static_cast<float>(x[d_idx]);

        for (int i = 0; i < n_per_t; ++i) {{
            auto s_idx = n_per_t * ds_idx + i;
            auto idx = d_idx * Ds + s_idx;
            auto dB_by_x = x_ * dt_ * static_cast<float>(B_[s_idx]);
            auto state = dA * i_state[idx] + dB_by_x;
            o_state[idx] = static_cast<T>(state);
            acc += state * C_[s_idx];
        }}
        acc = simd_sum(acc);
        if (thread_index_in_simdgroup == 0) {{
            out[d_idx] = static_cast<T>(acc + x_ * D[h_idx]);
        }}
    """
    return mx.fast.metal_kernel(
        name="ssm_kernel",
        input_names=["X", "A_log", "B", "C", "D", "dt", "state_in"],
        output_names=["out", "state_out"],
        source=source,
    )


_ssm_kernel = make_ssm_kernel()


def ssm_update_kernel(
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
    n, _, h, d = hidden_states.shape
    input_type = hidden_states.dtype
    hb, ds = B.shape[-2:]
    dt = compute_dt(dt, dt_bias, time_step_limit)
    return _ssm_kernel(
        inputs=[hidden_states, A_log, B, C, D, dt, state],
        template=[("T", input_type), ("Dh", d), ("Ds", ds), ("H", h), ("G", h // hb)],
        grid=(32, d, h * n),
        threadgroup=(32, 8, 1),
        output_shapes=[(n, 1, h, d), state.shape],
        output_dtypes=[input_type, input_type],
    )


def ssm_update(
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
    seq_len = hidden_states.shape[1]
    if seq_len > 1 or mx.default_device() != mx.gpu or not mx.metal.is_available():
        fn = ssm_update_ops
    else:
        fn = ssm_update_kernel
    return fn(
        hidden_states,
        A_log,
        B,
        C,
        D,
        dt,
        dt_bias,
        state,
        time_step_limit,
    )
