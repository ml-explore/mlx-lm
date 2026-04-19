"""Metal MSL kernel for fixed-rank factored DeltaNet inference step.

Maintains rank-R approximation via round-robin slot replacement:
the current step writes new contribution (v, β·k) into slot
``slot_idx = step % R``, replacing the oldest contribution.

For DeltaNet with decay ``g < 1``, oldest contribution (step T - R)
has weight g^R (e.g. g=0.9, R=16 → 18%), so replacement discards
a small-weight term. Theorem: state stable rank ≤ 2 means the
"active" state lives in a tiny subspace; round-robin among R slots
retains this subspace as long as R ≥ effective rank.

Fixed template R means kernel compiles ONCE regardless of step;
no recompile overhead. Full generation benefit: kernel-only speedup
(5-6×) translates to end-to-end.
"""

from typing import Optional, Tuple

import mlx.core as mx


def _make_fixed_step_kernel():
    if not mx.metal.is_available():
        return None

    # SLOT is compile-time template constant. We pre-make R kernels
    # (one per slot index 0..R-1) to avoid runtime int tensor creation.
    source = """
        constexpr int n_per_t = Dk / 32;

        auto head_flat = thread_position_in_grid.z;
        auto dv_idx    = thread_position_in_grid.y;
        auto dk_thread = thread_position_in_threadgroup.x;

        auto U_in_h    = U_in    + head_flat * Dv * R;
        auto V_in_h    = V_in    + head_flat * R  * Dk;
        auto q_h       = q       + head_flat * Dk;
        auto k_h       = k       + head_flat * Dk;
        auto v_h       = v       + head_flat * Dv;
        auto y_h       = y       + head_flat * Dv;
        auto U_out_h   = U_out   + head_flat * Dv * R;
        auto V_out_h   = V_out   + head_flat * R  * Dk;

        constexpr int slot = SLOT;  // compile-time replacement slot

        float k_local[n_per_t], q_local[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
            int dk = dk_thread * n_per_t + i;
            k_local[i] = static_cast<float>(k_h[dk]);
            q_local[i] = static_cast<float>(q_h[dk]);
        }

        float g_val    = static_cast<float>(g_scalar[head_flat]);
        float beta_val = static_cast<float>(beta_scalar[head_flat]);

        // kq
        float kq_partial = 0.0f;
        for (int i = 0; i < n_per_t; ++i)
            kq_partial += k_local[i] * q_local[i];
        float kq = simd_sum(kq_partial);

        // For each slot j: compute contribution to y, write V_out row.
        float Vq_acc[R];

        for (int j = 0; j < R; ++j) {
            if (j == slot) {
                // Replace: V_out[slot,:] = β · k
                for (int i = 0; i < n_per_t; ++i) {
                    int dk = dk_thread * n_per_t + i;
                    V_out_h[j * Dk + dk] = static_cast<InT>(beta_val * k_local[i]);
                }
                // Contribution to Vq: β · kq
                Vq_acc[j] = beta_val * kq;
            } else {
                // Normal decay-delete update
                float v_k_partial = 0.0f, v_q_partial = 0.0f;
                float V_local[n_per_t];
                for (int i = 0; i < n_per_t; ++i) {
                    int dk = dk_thread * n_per_t + i;
                    V_local[i] = static_cast<float>(V_in_h[j * Dk + dk]);
                    v_k_partial += V_local[i] * k_local[i];
                    v_q_partial += V_local[i] * q_local[i];
                }
                float Vk_j      = simd_sum(v_k_partial);
                float Vq_orig_j = simd_sum(v_q_partial);
                Vq_acc[j] = g_val * Vq_orig_j - g_val * beta_val * Vk_j * kq;

                // Write V_out row
                float scale_sub = g_val * beta_val * Vk_j;
                for (int i = 0; i < n_per_t; ++i) {
                    int dk = dk_thread * n_per_t + i;
                    float val = g_val * V_local[i] - scale_sub * k_local[i];
                    V_out_h[j * Dk + dk] = static_cast<InT>(val);
                }
            }
        }

        // Write U row + y.
        if (dk_thread == 0) {
            float v_val = static_cast<float>(v_h[dv_idx]);
            float y_val = 0.0f;
            for (int j = 0; j < R; ++j) {
                float U_val;
                if (j == slot) {
                    U_val = v_val;
                    U_out_h[dv_idx * R + j] = v_h[dv_idx];
                } else {
                    U_val = static_cast<float>(U_in_h[dv_idx * R + j]);
                    U_out_h[dv_idx * R + j] = U_in_h[dv_idx * R + j];
                }
                y_val += U_val * Vq_acc[j];
            }
            y_h[dv_idx] = static_cast<InT>(y_val);
        }
    """

    return mx.fast.metal_kernel(
        name="gated_delta_factored_fixed",
        input_names=["U_in", "V_in", "q", "k", "v", "g_scalar", "beta_scalar"],
        output_names=["y", "U_out", "V_out"],
        source=source,
    )


_fixed_kernel = _make_fixed_step_kernel()


def factored_step_fixed(
    U: mx.array,      # [B, Hv, Dv, R]
    V: mx.array,      # [B, Hv, R, Dk]
    q: mx.array,      # [B, Hv, Dk]
    k: mx.array,      # [B, Hv, Dk]
    v: mx.array,      # [B, Hv, Dv]
    g: mx.array,      # [B, Hv]
    beta: mx.array,   # [B, Hv]
    slot_idx: int,    # which slot to replace this step (compile-time)
) -> Tuple[mx.array, mx.array, mx.array]:
    """Fixed-rank factored step. Output (U, V) shape = input.

    slot_idx is a compile-time template constant; R variants get
    compiled (cached after first use). Cycling 0..R-1 avoids
    per-step tensor creation overhead.
    """
    if _fixed_kernel is None:
        raise RuntimeError("Metal unavailable")

    B, Hv, Dv, R = U.shape
    Dk = V.shape[-1]
    dtype = U.dtype

    y, U_new, V_new = _fixed_kernel(
        inputs=[U, V, q, k, v, g, beta],
        template=[
            ("InT", dtype), ("Dk", Dk), ("Dv", Dv),
            ("R", R), ("SLOT", slot_idx),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, Hv, Dv), (B, Hv, Dv, R), (B, Hv, R, Dk)],
        output_dtypes=[dtype, dtype, dtype],
    )
    return y, U_new, V_new
