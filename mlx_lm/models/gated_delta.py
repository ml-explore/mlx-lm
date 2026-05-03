from functools import partial
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


@partial(mx.compile, shapeless=True)
def compute_g(A_log, a, dt_bias):
    return mx.exp(-mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias))


def _make_gated_delta_kernel(has_mask=False, vectorized=False):
    if not mx.metal.is_available():
        return None
    mask_source = "mask[b_idx * T + t]" if has_mask else "true"

    # Configure g indexing based on whether gating is vectorized
    if vectorized:
        g_comment = "// g: [B, T, Hv, Dk]"
        g_setup = "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;"
        g_access = "g_[s_idx]"
        g_advance = "g_ += Hv * Dk;"
    else:
        g_comment = "// g: [B, T, Hv]"
        g_setup = "auto g_ = g + b_idx * T * Hv;"
        g_access = "g_[hv_idx]"
        g_advance = "g_ += Hv;"

    source = f"""
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }}

        {g_comment}
        {g_setup}
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {{
          if ({mask_source}) {{
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] * {g_access};
              kv_mem += state[i] * k_[s_idx];
            }}
            kv_mem = simd_sum(kv_mem);

            auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] + k_[s_idx] * delta;
              out += state[i] * q_[s_idx];
            }}
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {{
              y[dv_idx] = static_cast<InT>(out);
            }}
          }} else {{
            y[dv_idx] = static_cast<InT>(0);
          }}
          // Increment data pointers to next time step
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          {g_advance}
          beta_ += Hv;
        }}
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }}
    """
    inputs = ["q", "k", "v", "g", "beta", "state_in", "T"]
    if has_mask:
        inputs.append("mask")

    suffix = ""
    if vectorized:
        suffix += "_vec"
    if has_mask:
        suffix += "_mask"

    return mx.fast.metal_kernel(
        name=f"gated_delta_step{suffix}",
        input_names=inputs,
        output_names=["y", "state_out"],
        source=source,
    )


_gated_delta_kernel = _make_gated_delta_kernel(has_mask=False, vectorized=False)
_gated_delta_kernel_masked = _make_gated_delta_kernel(has_mask=True, vectorized=False)
_gated_delta_kernel_vec = _make_gated_delta_kernel(has_mask=False, vectorized=True)
_gated_delta_kernel_vec_masked = _make_gated_delta_kernel(
    has_mask=True, vectorized=True
)


@mx.compile
def _gated_delta_step_ops(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """
    Ops-based reference implementation for a single recurrent step.

    Shapes:
      - q, k: [B, H, Dk]
      - v: [B, H, Dv]
      - g: [B, H] or [B, H, Dk]
      - beta: [B, H]
      - state: [B, H, Dv, Dk]
    Returns:
      - y: [B, H, Dv]
      - new_state: [B, H, Dv, Dk]
    """

    # Decay
    old_state = state
    if g.ndim == 2:
        decay = g[..., None, None]
    elif g.ndim == 3:
        decay = g[..., None, :]
    else:
        raise ValueError(f"Unsupported gating shape {g.shape}")
    state = state * decay
    kv_mem = (state * k[..., None, :]).sum(axis=-1)  # [B, H, Dv]
    delta = (v - kv_mem) * beta[..., None]  # [B, H, Dv]
    state = state + k[..., None, :] * delta[..., None]
    # Output projection along key dim with q
    y = (state * q[..., None, :]).sum(axis=-1)  # [B, H, Dv]

    if mask is not None:
        mask = mx.expand_dims(mask, axis=(1, 2, 3))
        state = mx.where(mask, state, old_state)
    return y.astype(q.dtype), state


def gated_delta_kernel(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    input_type = q.dtype
    state_type = state.dtype
    if g.ndim == 4:
        kernel = _gated_delta_kernel_vec
        inputs = [q, k, v, g, beta, state, T]
        if mask is not None:
            kernel = _gated_delta_kernel_vec_masked
            inputs.append(mask)
    else:
        kernel = _gated_delta_kernel
        inputs = [q, k, v, g, beta, state, T]
        if mask is not None:
            kernel = _gated_delta_kernel_masked
            inputs.append(mask)

    return kernel(
        inputs=inputs,
        template=[
            ("InT", input_type),
            ("StT", state_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape],
        output_dtypes=[input_type, state_type],
    )


def gated_delta_ops(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """
    Ops-based reference implementation for prompt prefill (sequential loop).
    Supports both scalar and vectorized gating.

    Shapes:
      - q, k: [B, T, Hk, Dk]
      - v: [B, T, Hv, Dv]
      - g: [B, T, Hv] (scalar) or [B, T, Hv, Dk] (vectorized)
      - beta: [B, T, Hv]
      - state: [B, Hv, Dv, Dk]
    Returns:
      - y: [B, T, Hv, Dv]
      - state: [B, Hv, Dv, Dk]
    """
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

    if (repeat_factor := Hv // Hk) > 1:
        q = mx.repeat(q, repeat_factor, -2)
        k = mx.repeat(k, repeat_factor, -2)

    ys = []
    for t in range(T):
        y, state = _gated_delta_step_ops(
            q[:, t],
            k[:, t],
            v[:, t],
            g[:, t],
            beta[:, t],
            state,
            None if mask is None else mask[:, t],
        )
        ys.append(y)
    y = mx.stack(ys, axis=1)
    return y, state


# ---------------------------------------------------------------------------
# Chunked prefill (FlashQLA algorithm).
# Ported from https://github.com/QwenLM/FlashQLA (MIT License).
# Replaces the O(T) sequential loop in gated_delta_ops for prefill on Metal.
# ---------------------------------------------------------------------------

def _pad_reshape(x: mx.array, dim: int, C: int) -> mx.array:
    T = x.shape[dim]
    pad = (C - T % C) % C
    if pad:
        pw = [(0, 0)] * x.ndim
        pw[dim] = (0, pad)
        x = mx.pad(x, pw)
    s = list(x.shape)
    s[dim : dim + 1] = [-1, C]
    return x.reshape(s)


def _fill_g_tail(g: mx.array, T: int, C: int) -> mx.array:
    """Broadcast the last valid g value into pad positions of the final chunk."""
    r = T % C
    if r == 0:
        return g
    B = g.shape[0]
    Hv = g.shape[-1]
    tail = g[:, -1]  # [B, C, Hv]
    fill = mx.broadcast_to(tail[:, r - 1 : r, :], (B, C - r, Hv))
    new_tail = mx.concatenate([tail[:, :r], fill], axis=1)
    return mx.concatenate([g[:, :-1], new_tail[:, None]], axis=1)


_CHUNKED_MASKS: dict = {}
_CHUNKED_KKT_KERNELS: dict = {}


def _chunked_mask(C: int, diag: int) -> mx.array:
    key = (C, diag)
    if key not in _CHUNKED_MASKS:
        _CHUNKED_MASKS[key] = mx.triu(mx.ones((C, C), dtype=mx.bool_), k=diag)
    return _CHUNKED_MASKS[key]


def _chunked_kkt_kernel(C: int, H: int):
    key = (C, H)
    if key not in _CHUNKED_KKT_KERNELS:
        HC, CHC = H * C, C * H * C
        src = f"""
            uint b_total = threadgroup_position_in_grid.x;
            uint j = thread_position_in_threadgroup.x;
            uint bn = b_total / {H}u;
            uint h  = b_total % {H}u;
            threadgroup float X_tg[{C * C}];
            for (uint i = 0; i < {C}u; i++)
                X_tg[i * {C}u + j] = (i == j) ? 1.0f : 0.0f;
            for (uint i = 1; i < {C}u; i++) {{
                float acc = X_tg[i * {C}u + j];
                for (uint d = 0; d < i; d++)
                    acc += (float)x_in[bn * {CHC}u + i * {HC}u + h * {C}u + d]
                           * X_tg[d * {C}u + j];
                X_tg[i * {C}u + j] = acc;
            }}
            for (uint i = 0; i < {C}u; i++)
                out[bn * {CHC}u + i * {HC}u + h * {C}u + j] = (T)X_tg[i * {C}u + j];
        """
        _CHUNKED_KKT_KERNELS[key] = mx.fast.metal_kernel(
            name=f"gdr_prefill_kkt_c{C}_h{H}",
            input_names=["x_in"],
            output_names=["out"],
            source=src,
        )
    return _CHUNKED_KKT_KERNELS[key]


def _prefill_local_cumsum(g: mx.array, C: int) -> mx.array:
    B, T, Hv = g.shape
    return mx.cumsum(_pad_reshape(g, 1, C), axis=2).reshape(B, -1, Hv)[:, :T]


def _prefill_kkt(k: mx.array, g: mx.array, beta: mx.array, C: int) -> mx.array:
    B, T, Hk, K = k.shape
    Hv = g.shape[-1]
    gqa = Hv // Hk
    kc = _pad_reshape(k, 1, C)     # [B, N, C, Hk, K]
    gc = _pad_reshape(g, 1, C)     # [B, N, C, Hv]
    bc = _pad_reshape(beta, 1, C)  # [B, N, C, Hv]
    N = kc.shape[1]
    mask = _chunked_mask(C, 0)
    decay = mx.exp(gc[:, :, :, None, :] - gc[:, :, None, :, :])
    decay = mx.where(mask[None, None, :, :, None], mx.zeros_like(decay), decay)
    gram = mx.einsum("bnchk,bndhk->bnchd", kc, kc)  # [B, N, C, Hk, C]
    if gqa > 1:
        bd = (bc[:, :, :, :, None] * mx.swapaxes(decay, -2, -1)).reshape(
            B, N, C, Hk, gqa, C
        )
        A = (gram[:, :, :, :, None, :] * bd).reshape(B, N, C, Hv, C)
    else:
        A = gram * bc[:, :, :, :, None] * mx.swapaxes(decay, -2, -1)
    A = A.reshape(B, -1, Hv, C)[:, :T]
    # Triangular solve (I + L)^{-1} via Metal kernel; layout [B*N, C, Hv, C]
    Ac = -_pad_reshape(A, 1, C)
    BN = B * N
    return _chunked_kkt_kernel(C, Hv)(
        inputs=[Ac.reshape(BN, C, Hv, C)],
        template=[("T", Ac.dtype)],
        grid=(BN * Hv * C, 1, 1),
        threadgroup=(C, 1, 1),
        output_shapes=[(BN, C, Hv, C)],
        output_dtypes=[Ac.dtype],
    )[0].reshape(B, -1, Hv, C)[:, :T]


def _prefill_wu(
    k: mx.array, v: mx.array, beta: mx.array, A: mx.array, g: mx.array, C: int
) -> Tuple[mx.array, mx.array]:
    B, T, Hk, K = k.shape
    _, _, Hv, V = v.shape
    if Hk != Hv:
        k = mx.repeat(k, Hv // Hk, axis=2)
    kb = _pad_reshape(k * (beta * mx.exp(g))[..., None], 1, C)
    vb = _pad_reshape(v * beta[..., None], 1, C)
    Ac = _pad_reshape(A, 1, C)
    w = mx.einsum("bnchd,bndhk->bnchk", Ac, kb).reshape(B, -1, Hv, K)[:, :T]
    u = mx.einsum("bnchd,bndhv->bnchv", Ac, vb).reshape(B, -1, Hv, V)[:, :T]
    return w, u


def _prefill_recurrence(
    k: mx.array,
    w: mx.array,
    u: mx.array,
    g: mx.array,
    state: mx.array,
    C: int,
) -> Tuple[mx.array, mx.array, mx.array]:
    B, T, Hk, K = k.shape
    _, _, Hv, V = u.shape
    if Hk != Hv:
        k = mx.repeat(k, Hv // Hk, axis=2)
    kc = _pad_reshape(k, 1, C)
    wc = _pad_reshape(w, 1, C)
    uc = _pad_reshape(u, 1, C)
    gc = _fill_g_tail(_pad_reshape(g, 1, C), T, C)
    h_list, vn_list = [], []
    for i in range(kc.shape[1]):
        h_list.append(state)
        vn = uc[:, i] - mx.einsum("bchk,bhkv->bchv", wc[:, i], state)
        vn_list.append(vn)
        state = state * mx.exp(gc[:, i, -1])[:, :, None, None]
        state = state + mx.einsum(
            "bchk,bchv->bhkv",
            kc[:, i] * mx.exp(gc[:, i, -1:, :, None] - gc[:, i, :, :, None]),
            vn,
        )
    h = mx.stack(h_list, axis=1)
    vn = mx.stack(vn_list, axis=1).reshape(B, -1, Hv, V)[:, :T]
    return h, vn, state


def _prefill_output(
    q: mx.array, k: mx.array, vn: mx.array, h: mx.array, g: mx.array, C: int
) -> mx.array:
    B, T, Hk, K = k.shape
    _, _, Hv, V = vn.shape
    if Hk != Hv:
        q = mx.repeat(q, Hv // Hk, axis=2)
        k = mx.repeat(k, Hv // Hk, axis=2)
    qc = _pad_reshape(q, 1, C)
    kc = _pad_reshape(k, 1, C)
    vc = _pad_reshape(vn, 1, C)
    gc = _pad_reshape(g, 1, C)
    mask = _chunked_mask(C, 1)
    decay = mx.exp(gc[:, :, :, None, :] - gc[:, :, None, :, :])
    decay = mx.where(mask[None, None, :, :, None], mx.zeros_like(decay), decay)
    attn = mx.einsum("bnchk,bndhk->bncdh", qc, kc) * decay
    inter = mx.einsum("bnchk,bnhkv->bnchv", qc * mx.exp(gc)[..., None], h)
    return (inter + mx.einsum("bncdh,bndhv->bnchv", attn, vc)).reshape(B, -1, Hv, V)[:, :T]


def _gdr_chunked_prefill(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g_log: mx.array,
    beta: mx.array,
    state: mx.array,
    chunk_size: int = 64,
) -> Tuple[mx.array, mx.array]:
    """Parallel chunked prefill for the gated delta rule (FlashQLA algorithm).

    Args:
        q, k:     [B, T, Hk, K]  (already normalized + scaled by caller)
        v:        [B, T, Hv, V]
        g_log:    [B, T, Hv]  log-space decay (negative, = log of multiplicative g)
        beta:     [B, T, Hv]
        state:    [B, Hv, K, V]  initial recurrent state
        chunk_size: tokens per chunk (default 64)

    Returns:
        o:           [B, T, Hv, V]
        final_state: [B, Hv, K, V]
    """
    C = chunk_size
    g = _prefill_local_cumsum(g_log, C)
    A = _prefill_kkt(k=k, g=g, beta=beta, C=C)
    w, u = _prefill_wu(k=k, v=v, beta=beta, A=A, g=g, C=C)
    h, vn, fs = _prefill_recurrence(k=k, w=w, u=u, g=g, state=state, C=C)
    o = _prefill_output(q=q, k=k, vn=vn, h=h, g=g, C=C)
    return o, fs


_gdr_chunked_prefill_compiled = mx.compile(_gdr_chunked_prefill)


def gated_delta_update(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
    use_kernel: bool = True,
) -> Tuple[mx.array, mx.array]:
    beta = mx.sigmoid(b)
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

    # Chunked prefill: parallel over chunks, fast for T > chunk_size.
    # Requires Metal; falls back to sequential ops when mask is set (padded batches).
    if use_kernel and T > 1 and mask is None and mx.metal.is_available():
        g_log = -mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias)
        # State layout: mlx-lm [B, Hv, V, K] <-> FlashQLA [B, Hv, K, V]
        o, fs = _gdr_chunked_prefill_compiled(
            q, k, v, g_log, beta, state.swapaxes(-1, -2)
        )
        return o, fs.swapaxes(-1, -2)

    g = compute_g(A_log, a, dt_bias)
    if not use_kernel or mx.default_device() != mx.gpu or not mx.metal.is_available():
        return gated_delta_ops(q, k, v, g, beta, state, mask)
    return gated_delta_kernel(q, k, v, g, beta, state, mask)
