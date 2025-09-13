import os
from typing import Optional, Tuple

import mlx.core as mx


def _make_gated_delta_kernel_step():
    if not mx.metal.is_available():
        return None
    source = """
        auto n = thread_position_in_grid.z; // packs (b,h)
        auto h_idx = n % H;
        auto b_idx = n / H;

        auto ds_lane = thread_position_in_threadgroup.x; // 0..31
        auto dv_idx = thread_position_in_grid.y;          // 0..Dv-1
        auto simd_idx = thread_index_in_simdgroup;

        // Shapes: q,k: [B,H,Dk]; v,y: [B,H,Dv]; state: [B,H,Dk,Dv]
        auto q_ = q + n * Dk;
        auto k_ = k + n * Dk;
        auto v_ = v + n * Dv;
        auto y_ = y + n * Dv;
        auto i_state = state_in + n * Dk * Dv;
        auto o_state = state_out + n * Dk * Dv;

        float g_val = static_cast<float>(g[n]);
        float beta_val = static_cast<float>(beta[n]);

        // First pass: decay and accumulate kv_mem over Dk for this dv
        float kv_acc = 0.0f;
        for (int s = ds_lane; s < Dk; s += 32) {
            int idx = s * Dv + dv_idx;
            float st = g_val * static_cast<float>(i_state[idx]);
            float k_val = static_cast<float>(k_[s]);
            kv_acc += st * k_val;
        }
        kv_acc = simd_sum(kv_acc);
        float delta = (static_cast<float>(v_[dv_idx]) - kv_acc) * beta_val;

        // Second pass: update state and accumulate y
        float y_acc = 0.0f;
        for (int s = ds_lane; s < Dk; s += 32) {
            int idx = s * Dv + dv_idx;
            float st = g_val * static_cast<float>(i_state[idx]);
            float k_val = static_cast<float>(k_[s]);
            st = st + k_val * delta;
            o_state[idx] = static_cast<T>(st);
            float q_val = static_cast<float>(q_[s]);
            y_acc += st * q_val;
        }
        y_acc = simd_sum(y_acc);
        if (simd_idx == 0) {
            y_[dv_idx] = static_cast<T>(y_acc);
        }
    """
    return mx.fast.metal_kernel(
        name="gated_delta_step",
        input_names=["q", "k", "v", "g", "beta", "state_in"],
        output_names=["y", "state_out"],
        source=source,
    )


def _make_gated_delta_kernel_prefill():
    # TODO: Implement Metal kernel for prompt prefill loop over T tokens.
    # The kernel should iterate time inside the kernel to avoid O(T) launches
    # and retain only O(1) state.
    # Pseudocode per (b,h,dv):
    #   for t in 0..T-1:
    #     decay state by g[b,h,t]
    #     kv_mem = sum_d state[d,:] * K[b,h,t,d]
    #     delta = (V[b,h,t,dv] - kv_mem[dv]) * beta[b,h,t]
    #     state[:,dv] += K[b,h,t,:] * delta
    #     Y[b,h,t,dv] = sum_d state[d,dv] * Q[b,h,t,d]
    # return mx.fast.metal_kernel(name="gated_delta_prefill", ...)
    return None


_kernel_step = _make_gated_delta_kernel_step()
_kernel_prefill = _make_gated_delta_kernel_prefill()


def _use_kernel() -> bool:
    flag = os.getenv("MLXLM_DELTA_KERNEL", "0").lower() in ("1", "true", "yes")
    return flag and (mx.default_device() == mx.gpu) and mx.metal.is_available()


def gated_delta_step_ops(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
) -> Tuple[mx.array, mx.array]:
    """
    Ops-based reference implementation for a single recurrent step.
    Executes on the current MLX default device (CPU or GPU).

    Shapes:
      - q, k: [B, H, Dk]
      - v: [B, H, Dv]
      - g, beta: [B, H]
      - state: [B, H, Dk, Dv]
    Returns:
      - y: [B, H, Dv]
      - new_state: [B, H, Dk, Dv]
    """
    # Decay
    state = state * g[..., None, None]
    # Memory projection along key dim
    kv_mem = (state * k[..., :, None]).sum(axis=-2)  # [B, H, Dv]
    # Residual update per value dim
    delta = (v - kv_mem) * beta[..., None]  # [B, H, Dv]
    state = state + k[..., :, None] * delta[..., None, :]
    # Output projection along key dim with q
    y = (state * q[..., :, None]).sum(axis=-2)  # [B, H, Dv]
    return y, state


def gated_delta_prefill_ops(
    Q: mx.array,
    K: mx.array,
    V: mx.array,
    G: mx.array,
    BETA: mx.array,
    state: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """
    Ops-based reference implementation for prompt prefill (sequential loop).
    Executes on the current MLX default device (CPU or GPU).

    Shapes:
      - Q, K: [B, H, T, Dk]
      - V: [B, H, T, Dv]
      - G, BETA: [B, H, T]
      - state: [B, H, Dk, Dv]
    Returns:
      - Y: [B, H, T, Dv]
      - new_state: [B, H, Dk, Dv]
    """
    B, H, T, Dk = Q.shape
    Dv = V.shape[-1]
    if state is None:
        state = mx.zeros((B, H, Dk, Dv), dtype=Q.dtype)
    ys = []
    for t in range(T):
        y, state = gated_delta_step_ops(
            Q[..., t, :], K[..., t, :], V[..., t, :], G[..., t], BETA[..., t], state
        )
        ys.append(y)
    Y = mx.stack(ys, axis=2)
    return Y, state


def gated_delta_step(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
) -> Tuple[mx.array, mx.array]:
    """
    Dispatch to Metal kernel if available and enabled; otherwise use the ops-based reference path.
    See gated_delta_step_ops for shapes.
    """
    if _use_kernel() and _kernel_step is not None:
        input_type = q.dtype
        B, H, Dk = q.shape
        Dv = v.shape[-1]
        return _kernel_step(
            inputs=[q, k, v, g, beta, state],
            template=[("T", input_type), ("Dk", Dk), ("Dv", Dv), ("H", H)],
            grid=(32, Dv, B * H),
            threadgroup=(32, 1, 1),
            output_shapes=[(B, H, Dv), state.shape],
            output_dtypes=[input_type, input_type],
        )
    else:
        return gated_delta_step_ops(q, k, v, g, beta, state)


def gated_delta_prefill(
    Q: mx.array,
    K: mx.array,
    V: mx.array,
    G: mx.array,
    BETA: mx.array,
    state: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """
    Dispatch to Metal kernel if available and enabled; otherwise use the ops-based reference path.
    See gated_delta_prefill_ops for shapes.
    """
    if _use_kernel() and _kernel_step is not None:
        # Use the step kernel in a host-side loop to preserve O(1) state.
        input_type = Q.dtype
        B, H, T, Dk = Q.shape
        Dv = V.shape[-1]
        if state is None:
            state = mx.zeros((B, H, Dk, Dv), dtype=input_type)
        ys = []
        for t in range(T):
            y, state = gated_delta_step(
                Q[..., t, :], K[..., t, :], V[..., t, :], G[..., t], BETA[..., t], state
            )
            ys.append(y)
        Y = mx.stack(ys, axis=2)
        return Y, state
    else:
        return gated_delta_prefill_ops(Q, K, V, G, BETA, state)
