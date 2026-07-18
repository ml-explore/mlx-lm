from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


@mx.compile
def compute_dt(dt, dt_bias, time_step_limit):
    dt = dt.astype(mx.float32)
    dt = nn.softplus(dt + dt_bias)
    return mx.clip(dt, time_step_limit[0], time_step_limit[1])


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

        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * ds_idx + i;
            auto idx = d_idx * Ds + s_idx;
            auto dB_by_x = x_ * dt_ * static_cast<float>(B_[s_idx]);
            auto state = dA * i_state[idx] + dB_by_x;
            o_state[idx] = static_cast<U>(state);
            acc += state * C_[s_idx];
        }
        acc = simd_sum(acc);
        if (thread_index_in_simdgroup == 0) {
            out[d_idx] = static_cast<T>(acc + x_ * D[h_idx]);
        }
    """
    return mx.fast.metal_kernel(
        name="ssm_kernel",
        input_names=["X", "A_log", "B", "C", "D", "dt", "state_in"],
        output_names=["out", "state_out"],
        source=source,
    )


_ssm_kernel = make_ssm_kernel()


def make_ssm_seq_kernel():
    """Sequential-scan variant of the step kernel for short sequences
    (speculative verify blocks): one launch, the state slice stays in
    registers across the S positions instead of bouncing through the SSD
    chunked path, which for S in [2, 8] costs ~3x a single step."""
    if not mx.metal.is_available():
        return None
    source = """
        auto n = thread_position_in_grid.z;
        auto h_idx = n % H;
        auto b_idx = n / H;
        auto grp = (h_idx / G);
        constexpr int n_per_t = Ds / 32;
        constexpr int HB = H / G;

        auto i_state = state_in + n * Dh * Ds;
        auto o_state = state_out + n * Dh * Ds;

        auto ds_idx = thread_position_in_threadgroup.x;
        auto d_idx = thread_position_in_grid.y;

        auto A = -fast::exp(static_cast<float>(A_log[h_idx]));
        float st[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
            st[i] = i_state[d_idx * Ds + n_per_t * ds_idx + i];
        }

        for (int s = 0; s < S; ++s) {
            auto x_ = static_cast<float>(
                X[((b_idx * S + s) * H + h_idx) * Dh + d_idx]);
            auto dt_ = static_cast<float>(dt[(b_idx * S + s) * H + h_idx]);
            auto dA = fast::exp(A * dt_);
            auto B_ = B + ((b_idx * S + s) * HB + grp) * Ds;
            auto C_ = C + ((b_idx * S + s) * HB + grp) * Ds;

            float acc = 0.0;
            for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * ds_idx + i;
                auto dB_by_x = x_ * dt_ * static_cast<float>(B_[s_idx]);
                st[i] = dA * st[i] + dB_by_x;
                acc += st[i] * C_[s_idx];
            }
            acc = simd_sum(acc);
            if (thread_index_in_simdgroup == 0) {
                out[((b_idx * S + s) * H + h_idx) * Dh + d_idx] =
                    static_cast<T>(acc + x_ * D[h_idx]);
            }
        }

        for (int i = 0; i < n_per_t; ++i) {
            o_state[d_idx * Ds + n_per_t * ds_idx + i] = static_cast<U>(st[i]);
        }
    """
    return mx.fast.metal_kernel(
        name="ssm_seq_kernel",
        input_names=["X", "A_log", "B", "C", "D", "dt", "state_in"],
        output_names=["out", "state_out"],
        source=source,
    )


_ssm_seq_kernel = make_ssm_seq_kernel()

# Above this length the SSD chunked path wins over the sequential scan.
_SEQ_KERNEL_MAX_LEN = 8


def ssm_update_seq_kernel(
    hidden_states: mx.array,
    A_log: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    dt: mx.array,
    dt_bias: mx.array,
    state: mx.array,
    time_step_limit: Tuple[float, float],
):
    n, S, h, d = hidden_states.shape
    input_type = hidden_states.dtype
    state_type = state.dtype
    hb, ds = B.shape[-2:]
    dt = compute_dt(dt, dt_bias, time_step_limit)
    return _ssm_seq_kernel(
        inputs=[hidden_states, A_log, B, C, D, dt, state],
        template=[
            ("T", input_type),
            ("U", state_type),
            ("Dh", d),
            ("Ds", ds),
            ("H", h),
            ("G", h // hb),
            ("S", S),
        ],
        grid=(32, d, h * n),
        threadgroup=(32, 8, 1),
        output_shapes=[(n, S, h, d), state.shape],
        output_dtypes=[input_type, state_type],
    )


def ssm_update_kernel(
    hidden_states: mx.array,
    A_log: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    dt: mx.array,
    dt_bias: mx.array,
    state: mx.array,
    time_step_limit: Tuple[float, float],
):
    n, _, h, d = hidden_states.shape
    input_type = hidden_states.dtype
    state_type = state.dtype
    hb, ds = B.shape[-2:]
    dt = compute_dt(dt, dt_bias, time_step_limit)
    return _ssm_kernel(
        inputs=[hidden_states, A_log, B, C, D, dt, state],
        template=[
            ("T", input_type),
            ("U", state_type),
            ("Dh", d),
            ("Ds", ds),
            ("H", h),
            ("G", h // hb),
        ],
        grid=(32, d, h * n),
        threadgroup=(32, 8, 1),
        output_shapes=[(n, 1, h, d), state.shape],
        output_dtypes=[input_type, state_type],
    )


def segsum(x, mask=None):
    l = x.shape[-1]
    if mask is not None:
        mask = mx.expand_dims(mask, 1)
        x = x * mask
    x = mx.repeat(x[..., None], l, axis=-1)
    x = mx.tril(x, -1)
    x_segsum = mx.cumsum(x, axis=-2)
    if mask is not None:
        x_segsum = mx.where(
            mask[..., None, :] * mask[..., None], x_segsum, -float("inf")
        )
    return x_segsum


def ssm_attn(
    x: mx.array,
    A_log: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    dt: mx.array,
    dt_bias: mx.array,
    state: Optional[mx.array] = None,
    time_step_limit: Tuple[float, float] = (0.001, 100.0),
    mask: Optional[mx.array] = None,
    lengths: Optional[mx.array] = None,
    step: int = 256,
) -> Tuple[mx.array, mx.array]:
    """SSD-SSM forward pass.

    Args:
        x: Input of shape (batch_size, seq_len, num_heads, head_dim).
        dt: Time deltas of shape (seq_len, num_heads,).
        A_log: State transition of shape (num_heads,).
        B: Input mixing of shape (batch_size, seq_len, num_groups, n).
        C: Output mixing of shape (batch_size, seq_len, num_groups, n).
        D: Residual connection.
        dt_bias: Bias for time deltas of shape (num_heads,).
        time_step_limit: Minimum and maximum value for time deltas.
        mask: Optional multiplicative mask.
        lengths: Optional lenghts of sequences, assumed to be the full length if unspecified.
        step: Step size for processing x.

    Code modified from
    https://github.com/cartesia-ai/edge/blob/main/cartesia-mlx/cartesia_mlx/layers/ssd/ops.py

    """
    b, l, h, dh = x.shape
    _, _, g, d = B.shape

    dt = compute_dt(dt, dt_bias, time_step_limit)
    repeats = h // g
    A = -mx.exp(A_log).astype(dt.dtype)
    dtA = dt * A.reshape(1, 1, -1)
    dtx = dt.reshape(b, l, h, 1) * x

    def _step(dtx, dtA, B, C, state, mask):
        s = dtx.shape[1]
        B = mx.transpose(B, (0, 2, 3, 1))

        CB = mx.swapaxes(C, 1, 2) @ B
        CB = mx.repeat(CB, repeats, axis=1)

        decay = mx.exp(segsum(dtA.swapaxes(1, 2), mask=mask))

        surrogate_attention_matrix = mx.tril(CB * decay, 0)

        y = surrogate_attention_matrix @ dtx.swapaxes(1, 2)
        y = mx.swapaxes(y, 1, 2)

        if lengths is not None:
            pos = mx.maximum(mx.minimum(lengths, step) - 1, 0)
            pos = mx.expand_dims(pos, (1, 2, 3))
            decay = mx.take_along_axis(decay, pos, axis=2)
        else:
            decay = decay[:, :, -1:, :]

        decay = decay.transpose(0, 3, 1, 2)
        B = mx.repeat(B, h // g, axis=1).swapaxes(2, 3)
        dtxdecay = dtx * decay
        dtxdecay = dtxdecay.swapaxes(1, 2).swapaxes(2, 3)

        next_state = dtxdecay @ B

        if state is not None:
            exp_dtA_cumsum = mx.exp(mx.cumsum(dtA, axis=-2))
            next_state += exp_dtA_cumsum[:, -1, :, None, None] * state
            C = C.reshape(b, s, g, 1, d, 1)
            y_prev = (
                (state.reshape((b, 1, g, repeats, dh, d)) @ C).squeeze(-1).flatten(2, 3)
            )
            y += exp_dtA_cumsum[..., None] * y_prev
        if lengths is not None and state is not None:
            next_state = mx.where(
                mx.expand_dims(lengths < 0, (1, 2, 3)), state, next_state
            )

        return y.astype(x.dtype), next_state

    ys = []
    for i in range(0, l, step):
        y, state = _step(
            dtx[:, i : i + step],
            dtA[:, i : i + step],
            B[:, i : i + step],
            C[:, i : i + step],
            state,
            None if mask is None else mask[..., i : i + step],
        )
        if lengths is not None:
            lengths = lengths - step
        ys.append(y)
    y = mx.concatenate(ys, axis=1) + x * D.reshape(1, 1, h, 1)
    return y, state


def ssm_update(
    hidden_states: mx.array,
    A_log: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    dt: mx.array,
    dt_bias: mx.array,
    state: Optional[mx.array] = None,
    time_step_limit: Tuple[float, float] = (0.001, 100.0),
    mask: Optional[mx.array] = None,
    lengths: Optional[mx.array] = None,
):
    seq_len = hidden_states.shape[1]
    state_dim = B.shape[-1]
    on_gpu = mx.default_device() == mx.gpu and mx.metal.is_available()
    if seq_len == 1 and state is not None and on_gpu:
        return ssm_update_kernel(
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
    if (
        1 < seq_len <= _SEQ_KERNEL_MAX_LEN
        and state is not None
        and mask is None
        and lengths is None
        and on_gpu
        and state_dim >= 32
        and state_dim % 32 == 0
    ):
        # Short multi-token updates (speculative verify blocks): sequential
        # scan in one launch instead of the SSD chunked path.
        return ssm_update_seq_kernel(
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
    return ssm_attn(
        hidden_states,
        A_log,
        B,
        C,
        D,
        dt,
        dt_bias,
        state,
        time_step_limit,
        mask=mask,
        lengths=lengths,
    )
