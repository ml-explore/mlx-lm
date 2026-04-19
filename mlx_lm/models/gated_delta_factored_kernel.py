"""Metal MSL kernel for factored DeltaNet inference step (T=1 generation).

Factored state: U: [B, Hv, Dv, r], V: [B, Hv, r, Dk].
Per step: compute y = U @ V' @ q + β·v·(k·q), where
V' = g·V - g·β·(V·k)·k^T.

Kernel outputs y and V_prime (= V updated). Host concatenates
v into U and β·k into V_prime outside for rank growth. This keeps
the kernel minimal.

FMAs per head per step at r=16, Dv=Dk=128:
  factored: 3·r·Dk + r·Dv + Dv ≈ 3·16·128 + 16·128 + 128 ≈ 8320
  dense:    Dv·Dk = 16384
  ⇒ 2× fewer FMAs per step, plus less memory bandwidth.
"""

from typing import Tuple

import mlx.core as mx


def _make_factored_step_kernel():
    if not mx.metal.is_available():
        return None

    # Kernel outputs GROWN state: U_out [B,Hv,Dv,R+1], V_out [B,Hv,R+1,Dk].
    # Last column of U_out = v, last row of V_out = β·k. No host concat needed.
    source = """
        constexpr int n_per_t = Dk / 32;

        auto head_flat = thread_position_in_grid.z;
        auto dv_idx    = thread_position_in_grid.y;
        auto dk_thread = thread_position_in_threadgroup.x;

        // Input pointers (rank R).
        auto U_in_h    = U_in    + head_flat * Dv * R;
        auto V_in_h    = V_in    + head_flat * R  * Dk;
        auto q_h       = q       + head_flat * Dk;
        auto k_h       = k       + head_flat * Dk;
        auto v_h       = v       + head_flat * Dv;
        auto y_h       = y       + head_flat * Dv;

        // Output pointers (grown rank R+1).
        auto U_out_h   = U_out   + head_flat * Dv * (R + 1);
        auto V_out_h   = V_out   + head_flat * (R + 1) * Dk;

        float k_local[n_per_t], q_local[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
            int dk = dk_thread * n_per_t + i;
            k_local[i] = static_cast<float>(k_h[dk]);
            q_local[i] = static_cast<float>(q_h[dk]);
        }

        float g_val    = static_cast<float>(g_scalar[head_flat]);
        float beta_val = static_cast<float>(beta_scalar[head_flat]);

        // kq = Σ k·q (SIMD reduce).
        float kq_partial = 0.0f;
        for (int i = 0; i < n_per_t; ++i)
            kq_partial += k_local[i] * q_local[i];
        float kq = simd_sum(kq_partial);

        float Vq_acc[R];

        // For each j of R: compute Vk_j, Vq_orig_j (SIMD reductions),
        // then write V_prime row into V_out[j,:].
        for (int j = 0; j < R; ++j) {
            float v_k_partial = 0.0f, v_q_partial = 0.0f;
            float V_local[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
                int dk = dk_thread * n_per_t + i;
                V_local[i] = static_cast<float>(V_in_h[j * Dk + dk]);
                v_k_partial += V_local[i] * k_local[i];
                v_q_partial += V_local[i] * q_local[i];
            }
            float Vk_j = simd_sum(v_k_partial);
            float Vq_orig_j = simd_sum(v_q_partial);
            Vq_acc[j] = g_val * Vq_orig_j - g_val * beta_val * Vk_j * kq;

            // Write V_out row j = V_prime for this dk_thread's slice.
            float scale_sub = g_val * beta_val * Vk_j;
            for (int i = 0; i < n_per_t; ++i) {
                int dk = dk_thread * n_per_t + i;
                float val = g_val * V_local[i] - scale_sub * k_local[i];
                V_out_h[j * Dk + dk] = static_cast<InT>(val);
            }
        }

        // Write V_out last row = β · k (for rank growth).
        for (int i = 0; i < n_per_t; ++i) {
            int dk = dk_thread * n_per_t + i;
            V_out_h[R * Dk + dk] = static_cast<InT>(beta_val * k_local[i]);
        }

        // Write y[dv_idx] = Σ_j U[dv_idx,j]·Vq_j + β·v[dv_idx]·kq.
        // And copy U row + append v at slot R.
        if (dk_thread == 0) {
            float v_val = static_cast<float>(v_h[dv_idx]);
            float y_val = beta_val * v_val * kq;
            for (int j = 0; j < R; ++j) {
                float U_val = static_cast<float>(U_in_h[dv_idx * R + j]);
                y_val += U_val * Vq_acc[j];
                // Copy-through U row.
                U_out_h[dv_idx * (R + 1) + j] = U_in_h[dv_idx * R + j];
            }
            // Append v as last column of U row.
            U_out_h[dv_idx * (R + 1) + R] = v_h[dv_idx];
            y_h[dv_idx] = static_cast<InT>(y_val);
        }
    """

    return mx.fast.metal_kernel(
        name="gated_delta_factored_step_grown",
        input_names=["U_in", "V_in", "q", "k", "v", "g_scalar", "beta_scalar"],
        output_names=["y", "U_out", "V_out"],
        source=source,
    )


_factored_step_kernel = _make_factored_step_kernel()


def gated_delta_factored_step_metal(
    U: mx.array,      # [B, Hv, Dv, R]    bf16
    V: mx.array,      # [B, Hv, R, Dk]    bf16
    q: mx.array,      # [B, 1, Hv, Dk]    bf16 (single token, T=1)
    k: mx.array,      # [B, 1, Hv, Dk]    bf16
    v: mx.array,      # [B, 1, Hv, Dv]    bf16
    g: mx.array,      # [B, 1, Hv]        bf16
    beta: mx.array,   # [B, 1, Hv]        bf16
) -> Tuple[mx.array, mx.array, mx.array]:
    """Metal kernel wrapper for T=1 factored step.

    Returns:
      y:       [B, 1, Hv, Dv]     bf16
      U_new:   [B, Hv, Dv, R+1]   bf16 (grown rank)
      V_new:   [B, Hv, R+1, Dk]   bf16
    """
    assert q.shape[1] == 1, "Factored kernel is T=1 specialised"
    if _factored_step_kernel is None:
        raise RuntimeError("Metal unavailable — cannot use factored kernel")

    B, _, Hv, Dk = q.shape
    Dv = v.shape[-1]
    R = U.shape[-1]
    dtype = U.dtype

    # Squeeze T=1.
    q_flat = q[:, 0]           # [B, Hv, Dk]
    k_flat = k[:, 0]
    v_flat = v[:, 0]
    g_flat = g[:, 0]           # [B, Hv]
    beta_flat = beta[:, 0]

    y_out, U_new, V_new = _factored_step_kernel(
        inputs=[U, V, q_flat, k_flat, v_flat, g_flat, beta_flat],
        template=[
            ("InT", dtype),
            ("Dk", Dk),
            ("Dv", Dv),
            ("R", R),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, Hv, Dv), (B, Hv, Dv, R + 1), (B, Hv, R + 1, Dk)],
        output_dtypes=[dtype, dtype, dtype],
    )

    # Restore T=1 dim in y.
    y = y_out[:, None, :, :]  # [B, 1, Hv, Dv]
    return y, U_new, V_new
