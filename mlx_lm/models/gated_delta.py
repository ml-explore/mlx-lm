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


CHUNK_SIZE = 64

# Solve block size; bounds fp32 error growth when keys repeat in a chunk.
SUB_BLOCK = 16


def _solve_strict_lower(A: mx.array, b: mx.array, sb: int = SUB_BLOCK) -> mx.array:
    """Solve (I - A) x = b for strictly lower-triangular (nilpotent) A.

    Blocked forward substitution; a global Neumann expansion can
    overflow fp32 when keys repeat within a chunk.
    """
    C = A.shape[-1]

    def doubling(Aii, rhs, n):
        x = rhs
        if n <= 1:
            return x
        P = Aii
        steps = (n - 1).bit_length()
        for s in range(steps):
            x = x + P @ x
            if s != steps - 1:
                P = P @ P
        return x

    if C <= sb:
        return doubling(A, b, C)

    nb = (C + sb - 1) // sb
    blocks = []
    for i in range(nb):
        lo, hi = i * sb, min((i + 1) * sb, C)
        rhs = b[..., lo:hi, :]
        if i > 0:
            prev = blocks[0] if i == 1 else mx.concatenate(blocks, axis=-2)
            rhs = rhs + A[..., lo:hi, :lo] @ prev
        blocks.append(doubling(A[..., lo:hi, lo:hi], rhs, hi - lo))
    return mx.concatenate(blocks, axis=-2)


def _gated_delta_chunk(
    state: mx.array,  # [B, H, Dk, Dv]
    q: mx.array,  # [B, H, C, Dk]
    k: mx.array,  # [B, H, C, Dk]
    v: mx.array,  # [B, H, C, Dv]
    g: mx.array,  # [B, H, C], gating in (0, 1)
    beta: mx.array,  # [B, H, C]
) -> Tuple[mx.array, mx.array]:
    """Run C timesteps as one triangular solve (gated UT/WY transform).

    Exact reformulation of the sequential recurrence; runs in fp32.
    """
    C = q.shape[2]
    orig_dtype = q.dtype

    q = q.astype(mx.float32)
    k = k.astype(mx.float32)
    v = v.astype(mx.float32)
    g = g.astype(mx.float32)
    beta = beta.astype(mx.float32)
    state = state.astype(mx.float32)

    # Log-domain cumulative decay; the clamp keeps -inf out of the cumsum.
    g_log = mx.log(mx.maximum(g, 1e-12))  # [B, H, C]
    g_cumlog = mx.cumsum(g_log, axis=-1)
    g_last = g_cumlog[..., -1:]

    # Zero the upper triangle before exp: it overflows, and inf * 0 = NaN.
    tril_ones = mx.tril(mx.ones((C, C), dtype=mx.float32))
    L_diff = (g_cumlog[..., :, None] - g_cumlog[..., None, :]) * tril_ones
    L_mask = mx.exp(L_diff) * tril_ones

    v_beta = v * beta[..., None]  # [B, H, C, Dv]
    k_beta = k * beta[..., None]  # [B, H, C, Dk]

    strict_lower = mx.tril(mx.ones((C, C), dtype=mx.float32), k=-1)
    A = -(k_beta @ mx.swapaxes(k, -1, -2)) * L_mask * strict_lower

    decay_exp = mx.exp(g_cumlog)[..., None]  # [B, H, C, 1]
    rhs = mx.concatenate([v_beta, k_beta * decay_exp], axis=-1)
    sol = _solve_strict_lower(A, rhs)
    v_corrected, k_cumdecay = mx.split(sol, [v.shape[-1]], axis=-1)

    v_new = v_corrected - k_cumdecay @ state  # [B, H, C, Dv]
    y_inter = (q * decay_exp) @ state  # [B, H, C, Dv]

    attn = (q @ mx.swapaxes(k, -1, -2)) * L_mask
    y = y_inter + attn @ v_new

    state_decay = mx.exp(g_last)[..., None]
    decay_to_end = mx.exp(g_last - g_cumlog)[..., None]
    new_state = state * state_decay + mx.swapaxes(k * decay_to_end, -1, -2) @ v_new

    # The state stays fp32 across chunks; casting at boundaries drifts.
    return y.astype(orig_dtype), new_state


_gated_delta_chunk_checkpointed = mx.checkpoint(_gated_delta_chunk)


def gated_delta_ops_chunked(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
    chunk_size: Optional[int] = None,
) -> Tuple[mx.array, mx.array]:
    """
    Chunk-parallel implementation of prompt prefill for scalar gating.

    Equivalent to gated_delta_ops but processes chunk_size timesteps at a
    time with dense matmuls, each chunk wrapped in mx.checkpoint, so the
    autodiff graph is O(T / chunk_size) instead of O(T).

    Shapes:
      - q, k: [B, T, Hk, Dk]
      - v: [B, T, Hv, Dv]
      - g: [B, T, Hv] (scalar gating only, values in (0, 1))
      - beta: [B, T, Hv]
      - state: [B, Hv, Dv, Dk]
    Returns:
      - y: [B, T, Hv, Dv]
      - state: [B, Hv, Dv, Dk]
    """
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    C = chunk_size or CHUNK_SIZE

    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

    if (repeat_factor := Hv // Hk) > 1:
        q = mx.repeat(q, repeat_factor, -2)
        k = mx.repeat(k, repeat_factor, -2)

    # Masked steps are identities on the state (g = 1, beta = 0).
    if mask is not None:
        m = mask[..., None]
        g = mx.where(m, g, mx.ones_like(g))
        beta = beta * m

    # Pad T to a multiple of C with identity steps (g = 1, beta = 0).
    pad_len = (C - (T % C)) % C
    if pad_len > 0:
        q = mx.pad(q, [(0, 0), (0, pad_len), (0, 0), (0, 0)])
        k = mx.pad(k, [(0, 0), (0, pad_len), (0, 0), (0, 0)])
        v = mx.pad(v, [(0, 0), (0, pad_len), (0, 0), (0, 0)])
        g = mx.concatenate([g, mx.ones((B, pad_len, Hv), dtype=g.dtype)], axis=1)
        beta = mx.pad(beta, [(0, 0), (0, pad_len), (0, 0)])

    num_chunks = (T + pad_len) // C

    # [B, T, H, D] -> [B, H, Nc, C, D]
    q = mx.swapaxes(q, 1, 2).reshape(B, Hv, num_chunks, C, Dk)
    k = mx.swapaxes(k, 1, 2).reshape(B, Hv, num_chunks, C, Dk)
    v = mx.swapaxes(v, 1, 2).reshape(B, Hv, num_chunks, C, Dv)
    g = mx.swapaxes(g, 1, 2).reshape(B, Hv, num_chunks, C)
    beta = mx.swapaxes(beta, 1, 2).reshape(B, Hv, num_chunks, C)

    # [B, Hv, Dv, Dk] -> [B, Hv, Dk, Dv]
    state = mx.swapaxes(state.astype(mx.float32), -1, -2)

    ys = []
    for ci in range(num_chunks):
        y_c, state = _gated_delta_chunk_checkpointed(
            state,
            q[:, :, ci],
            k[:, :, ci],
            v[:, :, ci],
            g[:, :, ci],
            beta[:, :, ci],
        )
        ys.append(y_c)

    y = mx.concatenate(ys, axis=2)
    if pad_len > 0:
        y = y[:, :, :T, :]
    y = mx.swapaxes(y, 1, 2)

    return y, mx.swapaxes(state, -1, -2)


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
    g = compute_g(A_log, a, dt_bias)
    if state is None:
        B, _, Hk, Dk = q.shape
        Hv, Dv = v.shape[-2:]
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

    if not use_kernel or mx.default_device() != mx.gpu or not mx.metal.is_available():
        if g.ndim == 4 or q.shape[1] == 1:
            # Vectorized gating and single-token steps use the sequential path.
            return gated_delta_ops(q, k, v, g, beta, state, mask)
        return gated_delta_ops_chunked(q, k, v, g, beta, state, mask)
    return gated_delta_kernel(q, k, v, g, beta, state, mask)
