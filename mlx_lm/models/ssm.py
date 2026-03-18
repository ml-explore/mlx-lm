# Copyright © 2026 Apple Inc.

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


@mx.compile
def compute_dt(dt, dt_bias, time_step_limit):
    dt = dt.astype(mx.float32)
    dt = nn.softplus(dt + dt_bias)
    return mx.clip(dt, time_step_limit[0], time_step_limit[1])


def compute_dt_eff(dt: mx.array, lam: mx.array) -> mx.array:
    """Compute the effective dt used in the *input* term of the trapezoidal
    recurrence (Mamba-3, §3 / Proposition 1).

    The 3-term recurrence is:
        h_t = α_t h_{t-1}  +  β_t (B_{t-1} ⊗ x_{t-1})  +  γ_t (B_t ⊗ x_t)
    where:
        α_t = exp(Δt_t · A)          (decay — unchanged from Mamba-2)
        γ_t = λ_t · Δt_t             (current input weight)
        β_t = (1 − λ_t) · Δt_t · α_t (previous input weight, α already in decay)

    Because the α_t factor inside β_t cancels with the accumulated decay in the
    SSD kernel, the parallel SSD only needs a modified *effective* dt for the
    input weighting, while dtA (decay) stays on the original dt:
        dt_eff[t] = λ[t]·Δt[t]  +  (1−λ[t+1])·Δt[t+1]

    (boundary: last position contributes 0 from the t+1 term.)

    Args:
        dt:  processed dt (after softplus+clip), shape (batch, seq_len, num_heads).
        lam: raw λ logits (sigmoid applied here),  shape (batch, seq_len, num_heads).

    Returns:
        dt_eff of the same shape as dt.
    """
    lam = mx.sigmoid(lam) # (b, l, h) ∈ (0,1)
    dt_gamma = lam * dt # γ_t = λ_t Δt_t
    dt_beta_next = mx.concatenate( # (1-λ_{t+1}) Δt_{t+1}
        [(1.0 - lam[:, 1:]) * dt[:, 1:],
         mx.zeros_like(dt[:, :1])],
        axis=1,
    )
    return dt_gamma + dt_beta_next


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
    _dt_eff: Optional[mx.array] = None,
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
        lengths: Optional lengths of sequences, assumed to be the full length if unspecified.
        step: Step size for processing x.
        _dt_eff: Optional pre-computed effective dt for Mamba-3 trapezoidal
            discretization (output of compute_dt_eff).  When supplied, replaces
            dt in the *input* weighting term (dtx) while the *decay* term (dtA)
            continues to use the original processed dt.  Pass None (default) for
            standard Mamba-2 Exponential-Euler behaviour.

    Code modified from
    https://github.com/cartesia-ai/edge/blob/main/cartesia-mlx/cartesia_mlx/layers/ssd/ops.py

    """
    b, l, h, dh = x.shape
    _, _, g, d = B.shape

    dt = compute_dt(dt, dt_bias, time_step_limit)
    repeats = h // g
    A = -mx.exp(A_log).astype(dt.dtype)
    dtA = dt * A.reshape(1, 1, -1) # decay always uses original dt

    # Mamba-3: swap in dt_eff for the input term only; Mamba-2: same as dt
    dt_input = _dt_eff if _dt_eff is not None else dt
    dtx = dt_input.reshape(b, l, h, 1) * x

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
    if (
        seq_len > 1
        or state is None
        or mx.default_device() != mx.gpu
        or not mx.metal.is_available()
    ):
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
    else:
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


# Mamba-3: exponential-trapezoidal SSM update

def _ssm_decode_trap(
    hidden_states: mx.array,
    A_log: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    dt: mx.array, # already processed by compute_dt, shape (b, 1, h)
    lam: mx.array, # raw logits, shape (b, 1, h)
    state: Optional[mx.array],
    prev_Bx: Optional[mx.array],
) -> Tuple[mx.array, mx.array, mx.array]:
    """Single-step trapezoidal recurrence for Mamba-3 autoregressive decode.

    h_t = α_t h_{t-1}  +  γ_t (B_t ⊗ x_t)  +  β_t prev_Bx
    where
        α_t = exp(dt · A),   γ_t = λ_t · dt,   β_t = (1-λ_t) · dt · α_t

    Args:
        hidden_states: (batch, 1, num_heads, head_dim)
        B, C:          (batch, 1, n_groups, state_size)  — BCNorm+bias already applied
        dt:            processed, (batch, 1, num_heads)
        lam:           raw logits, (batch, 1, num_heads)
        state:         (batch, num_heads, head_dim, state_size) or None
        prev_Bx:       (batch, num_heads, head_dim, state_size) or None

    Returns:
        y         (batch, 1, num_heads, head_dim)
        new_state (batch, num_heads, head_dim, state_size)
        new_Bx    (batch, num_heads, head_dim, state_size) — B_t ⊗ x_t for next step
    """
    b, _, h, dh = hidden_states.shape
    _, _, g, d = B.shape
    repeats = h // g

    lam_s = mx.sigmoid(lam[:, 0]) # (b, h)
    dt_s = dt[:, 0] # (b, h)

    A = -mx.exp(A_log.astype(mx.float32)) # (h,)
    alpha = mx.exp(dt_s * A[None, :]) # (b, h)
    gamma = lam_s * dt_s # γ_t  (b, h)
    beta = (1.0 - lam_s) * dt_s * alpha # β_t  (b, h)

    x_s = hidden_states[:, 0] # (b, h, dh)
    B_s = mx.repeat(B[:, 0], repeats, axis=1) # (b, h, d)
    C_s = mx.repeat(C[:, 0], repeats, axis=1) # (b, h, d)

    Bx = x_s[:, :, :, None] * B_s[:, :, None, :] # (b, h, dh, d)  outer product

    alpha_r = alpha[:, :, None, None]
    gamma_r = gamma[:, :, None, None]
    beta_r = beta[:, :, None, None]

    if state   is None: state = mx.zeros((b, h, dh, d), dtype=mx.float32)
    if prev_Bx is None: prev_Bx = mx.zeros_like(state)

    new_state = alpha_r * state + gamma_r * Bx + beta_r * prev_Bx

    y = mx.sum(new_state * C_s[:, :, None, :], axis=-1) # (b, h, dh)
    y = y + x_s * D[None, :, None]
    y = y[:, None] # (b, 1, h, dh)

    return y.astype(hidden_states.dtype), new_state, Bx


def ssm_update_trap(
    hidden_states: mx.array,
    A_log: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    dt: mx.array,
    dt_bias: mx.array,
    lam: mx.array,
    state: Optional[mx.array] = None,
    prev_Bx: Optional[mx.array] = None,
    time_step_limit: Tuple[float, float] = (0.001, 100.0),
    mask: Optional[mx.array] = None,
    lengths: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Mamba-3 SSM update with exponential-trapezoidal discretization.

    Prefill  (seq_len > 1 or state is None):
        Uses the parallel SSD kernel (ssm_attn) with dt_eff injected into the
        input weighting term.  The decay term (dtA) is unchanged from Mamba-2.

    Decode (seq_len == 1 and state is not None):
        Uses the Python 3-term recurrence (_ssm_decode_trap).  A dedicated
        Metal kernel for Mamba-3 decode is not yet implemented.

    Args:
        hidden_states: (batch, seq_len, num_heads, head_dim)
        B, C:         (batch, seq_len, n_groups, state_size) — BCNorm+biases applied
        dt:           raw dt logits,  (batch, seq_len, num_heads)
        dt_bias:      (num_heads,)
        lam:          raw λ logits,   (batch, seq_len, num_heads)
        state:        previous SSM state or None
        prev_Bx:      previous B_{t-1}⊗x_{t-1} for the β term, or None

    Returns:
        y        (batch, seq_len, num_heads, head_dim)
        state    (batch, num_heads, head_dim, state_size)
        prev_Bx  (batch, num_heads, head_dim, state_size)
    """
    b, seq_len, h, dh = hidden_states.shape
    g, d = B.shape[-2:]
    repeats = h // g

    # ---- prefill ----
    if seq_len > 1 or state is None:
        dt_proc = compute_dt(dt, dt_bias, time_step_limit)   # (b, l, h)
        dt_eff = compute_dt_eff(dt_proc, lam)               # (b, l, h)

        y, state = ssm_attn(
            hidden_states, A_log, B, C, D,
            dt, dt_bias, state,
            time_step_limit, mask=mask, lengths=lengths,
            _dt_eff=dt_eff,
        )

        # cache B_{l-1} ⊗ x_{l-1} for the first decode step's β term
        x_last = hidden_states[:, -1]                               # (b, h, dh)
        B_last = mx.repeat(B[:, -1], repeats, axis=1)               # (b, h, d)
        prev_Bx = x_last[:, :, :, None] * B_last[:, :, None, :]     # (b, h, dh, d)

        return y, state, prev_Bx

    # ---- single-step decode ----
    dt_proc = compute_dt(dt, dt_bias, time_step_limit)
    return _ssm_decode_trap(
        hidden_states, A_log, B, C, D, dt_proc, lam, state, prev_Bx
    )
