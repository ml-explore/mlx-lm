"""T=1 specialized gated_delta kernel for generation inference.

Generation does one token at a time. Generic kernel has outer ``for
(int t = 0; t < T; ++t)`` loop that compiler can unroll for T=1 but
still keeps full template machinery. A specialized kernel без loop +
tighter scheduling saves launch overhead и kernel compile time.

Framework contribution: new mlx_lm/models/gated_delta_t1.py. Used via
auto-switch в gated_delta_kernel wrapper when T=1 detected.
"""

from typing import Optional, Tuple

import mlx.core as mx


def _make_t1_kernel():
    """Specialized T=1 kernel: no loop, inline compute_g + sigmoid(b)
    + rms_norm(q) + rms_norm(k) + inv_scale.

    Saves ~5 kernel launches per layer vs generic path."""
    if not mx.metal.is_available():
        return None

    source = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto q_raw = q_in + b_idx * Hk * Dk + hk_idx * Dk;
        auto k_raw = k_in + b_idx * Hk * Dk + hk_idx * Dk;
        auto v_    = v    + b_idx * Hv * Dv + hv_idx * Dv;
        y += b_idx * Hv * Dv + hv_idx * Dv;

        auto dk_thread = thread_position_in_threadgroup.x;
        auto dv_idx    = thread_position_in_grid.y;

        auto i_state = state_in  + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        // --- RMS norm q и k inline, with inv_scale baked in ---
        // q = inv_scale^2 * rms_norm(q_raw)
        // k = inv_scale   * rms_norm(k_raw)
        // rms_norm(x) = x / sqrt(mean(x^2) + eps)
        // For Dk=128: compute SIMD-wide sum of squares, broadcast.
        float q_local[n_per_t], k_local[n_per_t];
        float q_sq_partial = 0.0f, k_sq_partial = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_thread + i;
            float qv = static_cast<float>(q_raw[s_idx]);
            float kv = static_cast<float>(k_raw[s_idx]);
            q_local[i] = qv;
            k_local[i] = kv;
            q_sq_partial += qv * qv;
            k_sq_partial += kv * kv;
        }
        float q_sq = simd_sum(q_sq_partial);
        float k_sq = simd_sum(k_sq_partial);
        float q_rms = 1.0f / sqrt(q_sq / (float)Dk + 1e-6f);
        float k_rms = 1.0f / sqrt(k_sq / (float)Dk + 1e-6f);
        float inv_scale = 1.0f / sqrt((float)Dk);
        float q_scale = inv_scale * inv_scale * q_rms;
        float k_scale = inv_scale * k_rms;
        for (int i = 0; i < n_per_t; ++i) {
            q_local[i] *= q_scale;
            k_local[i] *= k_scale;
        }

        // Load state.
        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_thread + i;
            state[i] = static_cast<float>(i_state[s_idx]);
        }

        // Inline compute_g + sigmoid(b).
        float A_log_val = static_cast<float>(A_log[hv_idx]);
        float dt_bias_val = static_cast<float>(dt_bias[hv_idx]);
        float a_val = static_cast<float>(a_raw[b_idx * Hv + hv_idx]);
        float b_val = static_cast<float>(b_raw[b_idx * Hv + hv_idx]);

        float sp_in = a_val + dt_bias_val;
        float sp = (sp_in > 20.0f) ? sp_in : log(1.0f + exp(sp_in));
        float g_arg = -exp(A_log_val) * sp;
        if (g_arg < -20.0f) g_arg = -20.0f;
        float g_val = exp(g_arg);

        float beta_val = (b_val > 0.0f)
            ? 1.0f / (1.0f + exp(-b_val))
            : exp(b_val) / (1.0f + exp(b_val));

        // DeltaNet step.
        float kv_mem = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
            state[i] = state[i] * g_val;
            kv_mem += state[i] * k_local[i];
        }
        kv_mem = simd_sum(kv_mem);
        auto delta = (v_[dv_idx] - kv_mem) * beta_val;

        float out = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
            state[i] = state[i] + k_local[i] * delta;
            out += state[i] * q_local[i];
        }
        out = simd_sum(out);
        if (thread_index_in_simdgroup == 0) {
            y[dv_idx] = static_cast<InT>(out);
        }

        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_thread + i;
            o_state[s_idx] = static_cast<InT>(state[i]);
        }
    """

    return mx.fast.metal_kernel(
        name="gated_delta_t1_rms",
        input_names=["q_in", "k_in", "v", "a_raw", "b_raw", "A_log", "dt_bias", "state_in"],
        output_names=["y", "state_out"],
        source=source,
    )


_t1_kernel = _make_t1_kernel()


def gated_delta_kernel_t1(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: mx.array,
) -> Tuple[mx.array, mx.array]:
    """T=1 mega-fused kernel for generation inference.

    Takes RAW q, k (pre-rms_norm), inlines:
      1. rms_norm(q) + inv_scale²  (was 2 separate kernel launches)
      2. rms_norm(k) + inv_scale   (was 2 separate launches)
      3. compute_g (was 1 launch)
      4. sigmoid(b) (was 1 launch)
      5. gated_delta recurrence (was 1 launch)

    Total fusion: 5 kernel launches → 1. Per-token savings:
      24 layers × 4 saved launches × ~5 μs = ~480 μs/token
      = ~3% end-to-end on Qwen3.5-9B.

    Caller must provide q, k as pre-GQA-expanded (Hv heads both).
    """
    B, T, Hv, Dk = k.shape
    Dv = v.shape[-1]
    assert T == 1, "T=1 specialized kernel"
    dtype = q.dtype
    # For template: Hk stays as num_k_heads (before expansion);
    # q/k fed here are already Hv heads, so Hk=Hv for this call.

    q_f = q.reshape(B * Hv * Dk)
    k_f = k.reshape(B * Hv * Dk)
    v_f = v.reshape(B * Hv * Dv)
    a_f = a.reshape(B * Hv)
    b_f = b.reshape(B * Hv)

    y, state_new = _t1_kernel(
        inputs=[q_f, k_f, v_f, a_f, b_f, A_log, dt_bias, state],
        template=[
            ("InT", dtype), ("Dk", Dk), ("Dv", Dv),
            ("Hk", Hv),  # q/k are already expanded to Hv
            ("Hv", Hv),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, Hv, Dv), state.shape],
        output_dtypes=[dtype, dtype],
    )

    y = y.reshape(B, 1, Hv, Dv)
    return y, state_new
