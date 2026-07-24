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


# Fused chunked-SSD prefill kernels. Same computation as ssm_attn, restructured
# for the GPU: kernel A is chunk-parallel (decayed scores register-resident, the
# intra-chunk output reduced across the lanes sharing a row via
# simd_shuffle_down, plus each chunk's contribution to the carried state);
# kernel B walks chunks sequentially with a threadgroup-resident fp32 state
# block (+1 pad keeps column reads bank-conflict-free). All accumulation is
# fp32. Long sequences run in SSD_SEGMENT slices to bound the U buffer.

SSD_CHUNK = 32
SSD_NTA = None  # kernel-A threadgroup size; None -> min(1024, 4 * SSD_CHUNK)
SSD_DH_BLOCK = 16
SSD_SEGMENT = 8192

_SSD_KERNEL_A_SRC = """
    constexpr int C  = CK;                  // chunk length (internal tile)
    constexpr int NT = NTA;                 // threads per threadgroup
    constexpr int JL = NT / C;              // lanes sharing a score row
    constexpr int JW = C / JL;              // j-slice width per lane
    constexpr int UL = NT / Dh;             // lanes sharing a U d-row
    constexpr int UW = 32 / UL;             // U m-slice width per lane
    const int tid = thread_position_in_threadgroup.x;   // 0..NT-1
    const int c   = threadgroup_position_in_grid.x;      // chunk index
    const int h   = threadgroup_position_in_grid.y;      // ssm head
    const int b   = threadgroup_position_in_grid.z;
    const int g   = h / (H / G);
    const int t0  = c * C;
    const int tt  = min(C, T - t0);
    const int nC  = (T + C - 1) / C;

    threadgroup float lcg_s[C];        // inclusive log-decay cumsum
    threadgroup float wj_s[C];         // exp(lcg_last - lcg_j)
    threadgroup InT  st2[C][33];       // staged 32-wide B-row tiles
    threadgroup InT  xst[C][Dh + 2];   // staged dtx chunk [C][Dh]

    // dtx: [B,T,H,Dh]; Bm,Cm: [B,T,G,Ds]; dtA: [B,T,H]
    // outs: Y0 [B,T,H,Dh]; U [B,H,nC,Dh,Ds]; LCG [B,H,nC,C]

    // P0: log-decay cumsum (sequential, one lane) ------------------------
    if (tid == 0) {
        float acc = 0.0f;
        for (int i = 0; i < C; ++i) {
            float da = (i < tt) ? dtA[((size_t)b * T + t0 + i) * H + h] : 0.0f;
            acc += da;
            lcg_s[i] = acc;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 32) {
        float last = lcg_s[tt - 1];
        for (int j = 0; j < C; ++j) {
            wj_s[j] = (j < tt) ? exp(last - lcg_s[j]) : 0.0f;
        }
    }
    if (tid < C) {
        LCG[(((size_t)b * H + h) * nC + c) * C + tid] = lcg_s[tid];
    }

    // P1: stage dtx chunk (needed by both the Y0 and U passes) ----------
    constexpr bool XEX = (C * Dh) % NT == 0;   // exact division: guard elided
    for (int r = 0; r < (C * Dh + NT - 1) / NT; ++r) {
        int flat = r * NT + tid;
        if (XEX || flat < C * Dh) {
            int j = flat / Dh, d = flat % Dh;
            xst[j][d] = (j < tt)
                ? dtx[(((size_t)b * T + t0 + j) * H + h) * Dh + d]
                : InT(0.0f);
        }
    }

    // Thread mapping for score dots: thread owns (row i, JW-wide j slice).
    const int ai = tid / JL;           // 0..C-1
    const int jg = tid % JL;           // j in [jg*JW, jg*JW+JW)

    // P2: fused tile loop — register-resident score dots AND U ----------
    float dq[JW];
    for (int q = 0; q < JW; ++q) dq[q] = 0.0f;
    for (int st = 0; st < Ds / 32; ++st) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // stage B rows (st2), 32-wide ds tile
        constexpr bool BEX = (C * 32) % NT == 0;
        for (int r = 0; r < (C * 32 + NT - 1) / NT; ++r) {
            int flat = r * NT + tid;
            if (BEX || flat < C * 32) {
                int i = flat / 32, m = flat % 32;
                st2[i][m] = (i < tt)
                    ? Bm[(((size_t)b * T + t0 + i) * G + g) * Ds + st * 32 + m]
                    : InT(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // score dots: C row ai read from device (uniform across JL lanes)
        if (ai < tt) {
            const device InT* crow =
                Cm + (((size_t)b * T + t0 + ai) * G + g) * Ds + st * 32;
            float cm[32];
            for (int m = 0; m < 32; ++m) cm[m] = float(crow[m]);
            for (int q = 0; q < JW; ++q) {
                int j = jg * JW + q;
                if (j <= ai) {
                    float acc = 0.0f;
                    for (int m = 0; m < 32; ++m) {
                        acc += cm[m] * float(st2[j][m]);
                    }
                    dq[q] += acc;
                }
            }
        }
        // U for this ds tile: thread owns (d row, 8-wide m slice); the
        // wj*dtx product is hoisted out of the m loop.
        {
            const int ud = tid / UL, umg = tid % UL;
            float uacc[UW];
            for (int q = 0; q < UW; ++q) uacc[q] = 0.0f;
            for (int j = 0; j < C; ++j) {
                float wxj = wj_s[j] * float(xst[j][ud]);
                for (int q = 0; q < UW; ++q) {
                    uacc[q] += wxj * float(st2[j][umg * UW + q]);
                }
            }
            size_t ub = ((((size_t)b * H + h) * nC + c) * Dh + ud) * Ds + st * 32;
            for (int q = 0; q < UW; ++q) {
                U[ub + umg * UW + q] = InT(uacc[q]);
            }
        }
    }
    // apply decay to register-resident dots
    for (int q = 0; q < JW; ++q) {
        int j = jg * JW + q;
        dq[q] = (j <= ai && ai < tt && j < tt)
            ? dq[q] * exp(lcg_s[ai] - lcg_s[j])
            : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // P3: Y0 = tril(A) @ dtx, reduced across the JL lanes sharing row ai -
    for (int d = 0; d < Dh; ++d) {
        float py = 0.0f;
        for (int q = 0; q < JW; ++q) {
            int j = jg * JW + q;
            py += dq[q] * float(xst[j][d]);
        }
        for (int off = JL / 2; off > 0; off >>= 1) {
            py += simd_shuffle_down(py, off);
        }
        if (jg == 0 && ai < tt) {
            Y0[(((size_t)b * T + t0 + ai) * H + h) * Dh + d] = InT(py);
        }
    }
"""

_SSD_KERNEL_B_SRC = """
    constexpr int C  = CK;
    constexpr int NT = 256;                             // threads per tg
    constexpr int DB = 16;                              // Dh block width
    const int tid  = thread_position_in_threadgroup.x;  // 0..NT-1
    const int dblk = threadgroup_position_in_grid.x;    // Dh block index
    const int h    = threadgroup_position_in_grid.y;
    const int b    = threadgroup_position_in_grid.z;
    const int g    = h / (H / G);
    const int nC   = (T + C - 1) / C;

    threadgroup float S_s[DB][Ds + 1]; // fp32 state block (padded: no bank conflicts)
    threadgroup float lcge_s[C];       // exp(lcg_i)
    threadgroup float gl_s[1];

    // init state block from state_in [B,H,Dh,Ds] fp32
    constexpr bool SEX = (DB * Ds) % NT == 0;
    for (int r = 0; r < (DB * Ds + NT - 1) / NT; ++r) {
        int flat = r * NT + tid;
        if (SEX || flat < DB * Ds) {
            int dd = flat / Ds, s = flat % Ds;
            S_s[dd][s] = state_in[(((size_t)b * H + h) * Dh + dblk * DB + dd) * Ds + s];
        }
    }

    // threads: (i, dd) pairs; i = tid / DB (0..15 per pass), dd = tid % DB
    const int i_lo = tid / DB;   // row offset within a 16-row pass
    const int dd   = tid % DB;

    for (int c = 0; c < nC; ++c) {
        const int t0 = c * C;
        const int tt = min(C, T - t0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < C) {
            lcge_s[tid] = exp(LCG[(((size_t)b * H + h) * nC + c) * C + tid]);
        }
        if (tid == 64) {
            gl_s[0] = exp(LCG[(((size_t)b * H + h) * nC + c) * C + tt - 1]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // y_inter_i[dd] = sum_s S[dd][s] * C_i[s]; C rows read directly from
        // device — uniform address across the DB lanes sharing row i.
        for (int p = 0; p < C / (NT / DB); ++p) {
            int i = p * (NT / DB) + i_lo;
            if (i < tt) {
                const device InT* crow =
                    Cm + (((size_t)b * T + t0 + i) * G + g) * Ds;
                float acc = 0.0f;
                for (int s = 0; s < Ds; ++s) {
                    acc += S_s[dd][s] * float(crow[s]);
                }
                size_t yi = (((size_t)b * T + t0 + i) * H + h) * Dh + dblk * DB + dd;
                y[yi] = InT(float(Y0[yi]) + lcge_s[i] * acc);
            }
        }
        // S <- gl * S + U_c
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float gl = gl_s[0];
        for (int r = 0; r < (DB * Ds + NT - 1) / NT; ++r) {
            int flat = r * NT + tid;
            if (SEX || flat < DB * Ds) {
                int di = flat / Ds, s = flat % Ds;
                float u = float(U[((((size_t)b * H + h) * nC + c) * Dh + dblk * DB + di) * Ds + s]);
                S_s[di][s] = gl * S_s[di][s] + u;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int r = 0; r < (DB * Ds + NT - 1) / NT; ++r) {
        int flat = r * NT + tid;
        if (SEX || flat < DB * Ds) {
            int dd2 = flat / Ds, s = flat % Ds;
            state_out[(((size_t)b * H + h) * Dh + dblk * DB + dd2) * Ds + s] = S_s[dd2][s];
        }
    }
"""


def make_ssd_kernels():
    if not mx.metal.is_available():
        return None, None
    header = """
        #include <metal_stdlib>
        using namespace metal;
    """
    a = mx.fast.metal_kernel(
        name="ssd_prefill_a",
        input_names=["dtx", "Bm", "Cm", "dtA", "T"],
        output_names=["Y0", "U", "LCG"],
        header=header,
        source=_SSD_KERNEL_A_SRC,
    )
    b = mx.fast.metal_kernel(
        name="ssd_prefill_b",
        input_names=["Cm", "Y0", "U", "LCG", "state_in", "T"],
        output_names=["y", "state_out"],
        header=header,
        source=_SSD_KERNEL_B_SRC,
    )
    return a, b


_ssd_kernel_a, _ssd_kernel_b = make_ssd_kernels()


def ssd_supported_shapes(head_dim: int, state_dim: int) -> bool:
    """Shapes the fused SSD kernels handle (thread-count divisibility for the
    CHUNK=64 / 256-thread layout and the 32-wide state tiling). Covers the
    Mamba2 family: head_dim 32/64/128, state_dim 32..256 in steps of 32."""
    return head_dim in (32, 64, 128) and state_dim % 32 == 0 and 32 <= state_dim <= 256


def _ssd_seg(dtx, Bm, Cm, dtA, state):
    b, T, h, dh = dtx.shape
    g, ds = Bm.shape[2:]
    n_chunks = (T + SSD_CHUNK - 1) // SSD_CHUNK
    in_dtype = dtx.dtype
    nta = SSD_NTA or min(1024, 4 * SSD_CHUNK)
    tmpl = [("InT", in_dtype), ("H", h), ("G", g), ("Dh", dh), ("Ds", ds),
            ("CK", SSD_CHUNK)]
    Y0, U, LCG = _ssd_kernel_a(
        inputs=[dtx, Bm, Cm, dtA, T],
        template=tmpl + [("NTA", nta)],
        grid=(nta * n_chunks, h, b),
        threadgroup=(nta, 1, 1),
        output_shapes=[
            (b, T, h, dh),
            (b, h, n_chunks, dh, ds),
            (b, h, n_chunks, SSD_CHUNK),
        ],
        output_dtypes=[in_dtype, in_dtype, mx.float32],
    )
    y, state_out = _ssd_kernel_b(
        inputs=[Cm, Y0, U, LCG, state, T],
        template=tmpl,
        grid=(256 * (dh // SSD_DH_BLOCK), h, b),
        threadgroup=(256, 1, 1),
        output_shapes=[(b, T, h, dh), state.shape],
        output_dtypes=[in_dtype, mx.float32],
    )
    return y, state_out


def ssd_prefill(
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
) -> Tuple[mx.array, mx.array]:
    """Fused-kernel equivalent of ssm_attn on the lengths-free path.

    A masked token is treated as a no-op: its dtA is zeroed (the state passes
    through undecayed) and its dtx is zeroed (no state write, no output
    contribution), so the carried state after a padded batch equals the state
    after running only the valid tokens.
    """
    b, l, h, dh = x.shape
    _, _, g, ds = B.shape
    dtf = compute_dt(dt, dt_bias, time_step_limit)
    A = -mx.exp(A_log.astype(mx.float32))
    dtA = dtf * A.reshape(1, 1, -1)
    dtx = (dtf[..., None] * x).astype(x.dtype)
    if mask is not None:
        mf = mask.astype(mx.float32)
        dtA = dtA * mf[..., None]
        dtx = (dtx * mf[..., None, None].astype(x.dtype)).astype(x.dtype)
    if state is None:
        state = mx.zeros((b, h, dh, ds), dtype=mx.float32)
    else:
        state = state.astype(mx.float32)

    if l <= SSD_SEGMENT:
        y, state = _ssd_seg(dtx, B, C, dtA, state)
    else:
        ys = []
        for s0 in range(0, l, SSD_SEGMENT):
            s1 = min(s0 + SSD_SEGMENT, l)
            y, state = _ssd_seg(
                mx.contiguous(dtx[:, s0:s1]),
                mx.contiguous(B[:, s0:s1]),
                mx.contiguous(C[:, s0:s1]),
                mx.contiguous(dtA[:, s0:s1]),
                state,
            )
            ys.append(y)
        y = mx.concatenate(ys, axis=1)
    y = y + x * D.reshape(1, 1, h, 1)
    return y.astype(x.dtype), state


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
    if (
        seq_len >= 32
        and lengths is None
        and _ssd_kernel_a is not None
        and mx.default_device() == mx.gpu
        and ssd_supported_shapes(hidden_states.shape[3], B.shape[3])
    ):
        return ssd_prefill(
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
        )
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
