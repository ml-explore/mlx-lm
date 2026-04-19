"""Metal-accelerated VJP for gated_delta_update.

Two Metal kernels:

* ``_fwd_save_kernel``  — forward recurrence that additionally writes the
  full state history ``S[t]`` (after every update) into a B×Hv×T×Dv×Dk
  output tensor so that the backward kernel can reuse it.
* ``_bwd_kernel``       — reverse-order sweep over the saved states,
  producing ``dq``, ``dk``, ``dv``, ``dg``, ``dbeta`` and ``dS_initial``.

Python orchestration runs the kernels in fixed-size chunks to bound the
``state_history`` memory footprint — typical chunk is 64 timesteps, so
storage per chunk is ``B×Hv×64×Dv×Dk×dtype_bytes`` (≈ 400 MB per chunk
for Qwen3.5-9B shapes at bf16).

Thread layout matches the upstream forward kernel:
    grid       = (32, Dv, B * Hv)
    threadgroup= (32, 4, 1)
so that a SIMD group of 32 threads collectively owns one ``(b, hv, dv)``
state row and uses ``simd_sum`` for ``Dk`` reductions.
"""

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .gated_delta import gated_delta_kernel


CHUNK_SIZE = 64


# -- Decay computation -------------------------------------------------------

_LOG_DECAY_CLAMP = -20.0


@mx.compile
def _compute_g(A_log: mx.array, a: mx.array, dt_bias: mx.array) -> mx.array:
    arg = -mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias)
    arg = mx.maximum(arg, _LOG_DECAY_CLAMP)
    return mx.exp(arg).astype(a.dtype)


# -- Metal kernels -----------------------------------------------------------

def _make_fwd_save_kernel(vectorized: bool = False):
    """Forward recurrence that also persists the state history for backward.

    ``vectorized`` toggles per-channel gating (``g`` shape ``[B, T, Hv, Dk]``
    vs the scalar ``[B, T, Hv]``) used by Kimi-Linear.
    """
    if not mx.metal.is_available():
        return None
    if vectorized:
        g_setup = "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;"
        g_advance = "g_ += Hv * Dk;"
        g_decay = "state[i] = state[i] * static_cast<float>(g_[n_per_t * dk_idx + i]);"
    else:
        g_setup = "auto g_ = g + b_idx * T * Hv;"
        g_advance = "g_ += Hv;"
        g_decay = "state[i] = state[i] * static_cast<float>(g_[hv_idx]);"

    source = f"""
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        auto y_out = y + b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;
        auto h_state = state_history
                     + ((b_idx * Hv + hv_idx) * T) * Dv * Dk
                     + dv_idx * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {{
          state[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
        }}

        {g_setup}
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {{
          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {{
            auto s_idx = n_per_t * dk_idx + i;
            {g_decay}
            kv_mem += state[i] * static_cast<float>(k_[s_idx]);
          }}
          kv_mem = simd_sum(kv_mem);

          auto delta = (static_cast<float>(v_[dv_idx]) - kv_mem)
                     * static_cast<float>(beta_[hv_idx]);

          float out_val = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {{
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + static_cast<float>(k_[s_idx]) * delta;
            out_val += state[i] * static_cast<float>(q_[s_idx]);
          }}
          out_val = simd_sum(out_val);
          if (thread_index_in_simdgroup == 0) {{
            y_out[dv_idx] = static_cast<InT>(out_val);
          }}

          for (int i = 0; i < n_per_t; ++i) {{
            h_state[n_per_t * dk_idx + i] = static_cast<InT>(state[i]);
          }}

          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y_out += Hv * Dv;
          {g_advance}
          beta_ += Hv;
          h_state += Dv * Dk;
        }}

        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<InT>(state[i]);
        }}
    """
    return mx.fast.metal_kernel(
        name="gated_delta_fwd_save_vec" if vectorized else "gated_delta_fwd_save",
        input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
        output_names=["y", "state_out", "state_history"],
        source=source,
    )


def _make_bwd_kernel():
    if not mx.metal.is_available():
        return None
    # Each SIMD group owns one ``(b, hv, dv_idx)`` slice of state and walks
    # t = T-1 .. 0 using the saved history. Within a threadgroup four SIMD
    # groups (one per ``dv_idx`` in {0..3}) cooperate via shared memory to
    # reduce their contributions to ``dq``/``dk``/``dg``/``dbeta`` into a
    # single value per threadgroup — so the output carries only the grid.y
    # dimension (``Dv / 4``) rather than the full ``Dv``. The caller then
    # performs the final deterministic sum over grid.y in Python. This
    # avoids ``atomic_fetch_add`` (order-dependent, hurts precision) while
    # keeping the output tensor 4× smaller than the per-Dv layout.
    source = r"""
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto q_base = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_base = k + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto v_base = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        auto g_base = g + b_idx * T * Hv + hv_idx;
        auto beta_base = beta + b_idx * T * Hv + hv_idx;
        auto dy_base = dy + b_idx * T * Hv * Dv + hv_idx * Dv;
        auto hist_base = state_history
                       + ((b_idx * Hv + hv_idx) * T) * Dv * Dk;
        auto s_initial = state_initial + (n * Dv + 0) * Dk;  // base for (b,hv)

        // Outputs carry a per-threadgroup slot (Dv_groups = Dv / 4); caller
        // sums along that axis in Python.
        constexpr int Dv_groups = Dv / 4;
        auto tg_y = threadgroup_position_in_grid.y;
        auto dq_base = dq + ((b_idx * T * Hv + hv_idx) * Dv_groups) * Dk;     // [B,T,Hv,Dvg,Dk]
        auto dk_base = dk_out + ((b_idx * T * Hv + hv_idx) * Dv_groups) * Dk; // [B,T,Hv,Dvg,Dk]
        auto dv_base = dv_out + b_idx * T * Hv * Dv + hv_idx * Dv;            // [B,T,Hv,Dv]
        auto dg_base = dg + (b_idx * T * Hv + hv_idx) * Dv_groups;            // [B,T,Hv,Dvg]
        auto dbeta_base = dbeta + (b_idx * T * Hv + hv_idx) * Dv_groups;      // [B,T,Hv,Dvg]

        // Shared-memory tile for intra-threadgroup reduction across four
        // SIMD groups (one per dv offset 0..3).
        threadgroup float dq_tile[4 * Dk];
        threadgroup float dk_tile[4 * Dk];
        threadgroup float dg_tile[4];
        threadgroup float dbeta_tile[4];
        auto simd_y = thread_position_in_threadgroup.y;  // 0..3

        auto ds_final_ptr = dS_final + (n * Dv + 0) * Dk;
        auto ds_init_ptr = dS_initial + (n * Dv + 0) * Dk;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // Running gradient wrt current S_t (per Dv row, Dk chunk).
        float dS[n_per_t];
        {
          auto ds_row = ds_final_ptr + dv_idx * Dk;
          for (int i = 0; i < n_per_t; ++i) {
            dS[i] = static_cast<float>(ds_row[n_per_t * dk_idx + i]);
          }
        }

        for (int t = T - 1; t >= 0; --t) {
          auto q_t = q_base + t * Hk * Dk;
          auto k_t = k_base + t * Hk * Dk;
          auto v_t = v_base + t * Hv * Dv;
          auto dy_t = dy_base + t * Hv * Dv;
          float beta_t = static_cast<float>(*(beta_base + t * Hv));
          float g_t    = static_cast<float>(*(g_base    + t * Hv));

          // S_new[dv_idx, :] is stored at history[t]; S_prev is history[t-1]
          // for t>0, otherwise state_initial.
          auto S_new_row = hist_base + t * Dv * Dk + dv_idx * Dk;
          float S_new[n_per_t];
          for (int i = 0; i < n_per_t; ++i) {
            S_new[i] = static_cast<float>(S_new_row[n_per_t * dk_idx + i]);
          }

          const device InT* S_prev_row;
          if (t == 0) {
            S_prev_row = s_initial + dv_idx * Dk;
          } else {
            S_prev_row = hist_base + (t - 1) * Dv * Dk + dv_idx * Dk;
          }
          float S_prev[n_per_t];
          for (int i = 0; i < n_per_t; ++i) {
            S_prev[i] = static_cast<float>(S_prev_row[n_per_t * dk_idx + i]);
          }

          // Recover S_tmp = g_t * S_prev (pre-update state used for kv_mem).
          float S_tmp[n_per_t];
          for (int i = 0; i < n_per_t; ++i) {
            S_tmp[i] = g_t * S_prev[i];
          }

          // kv_mem = sum_dk(S_tmp * k_t)
          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            kv_mem += S_tmp[i] * static_cast<float>(k_t[n_per_t * dk_idx + i]);
          }
          kv_mem = simd_sum(kv_mem);

          float v_val = static_cast<float>(v_t[dv_idx]);
          float delta = (v_val - kv_mem) * beta_t;
          float dy_val = static_cast<float>(dy_t[dv_idx]);

          // (1) y = S_new @ q  =>  dS_t += outer(dy_t, q_t); dq_t = S_new.T @ dy_t.
          float dq_contrib[n_per_t];
          for (int i = 0; i < n_per_t; ++i) {
            float q_val = static_cast<float>(q_t[n_per_t * dk_idx + i]);
            dS[i] += dy_val * q_val;
            dq_contrib[i] = S_new[i] * dy_val;
          }

          // (2) S_new = S_tmp + outer(delta, k_t)
          float k_regs[n_per_t];
          for (int i = 0; i < n_per_t; ++i) {
            k_regs[i] = static_cast<float>(k_t[n_per_t * dk_idx + i]);
          }
          float ddelta = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            ddelta += dS[i] * k_regs[i];
          }
          ddelta = simd_sum(ddelta);

          float dk_a[n_per_t];
          for (int i = 0; i < n_per_t; ++i) {
            dk_a[i] = dS[i] * delta;
          }

          // (3) delta = (v - kv_mem) * beta
          float dv_val  = ddelta * beta_t;
          float dkv_mem = -ddelta * beta_t;
          float dbeta_t = ddelta * (v_val - kv_mem);

          // (4) kv_mem = sum_dk(S_tmp * k_t)
          // dS_tmp += outer(dkv_mem, k_t); dk_t += S_tmp.T @ dkv_mem.
          float dk_b[n_per_t];
          for (int i = 0; i < n_per_t; ++i) {
            dS[i] += dkv_mem * k_regs[i];        // dS_tmp += dkv_mem * k
            dk_b[i] = S_tmp[i] * dkv_mem;
          }

          // (5) S_tmp = g_t * S_prev
          //     dS_prev += g_t * dS_tmp;  dg_t += sum(dS_tmp * S_prev)
          float dg_local = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            dg_local += dS[i] * S_prev[i];
          }
          dg_local = simd_sum(dg_local);

          for (int i = 0; i < n_per_t; ++i) {
            dS[i] = g_t * dS[i];  // propagate to dS_prev for next iteration
          }

          // Stash each SIMD group's contribution in shared memory, then
          // reduce across the four SIMD groups into a single result per
          // threadgroup (one slot in the Dv_groups axis).
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            dq_tile[simd_y * Dk + s_idx] = dq_contrib[i];
            dk_tile[simd_y * Dk + s_idx] = dk_a[i] + dk_b[i];
          }
          if (thread_index_in_simdgroup == 0) {
            dg_tile[simd_y]    = dg_local;
            dbeta_tile[simd_y] = dbeta_t;
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          if (simd_y == 0) {
            auto dq_slot = dq_base + t * Hv * Dv_groups * Dk + tg_y * Dk;
            auto dk_slot = dk_base + t * Hv * Dv_groups * Dk + tg_y * Dk;
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              float sum_q = dq_tile[0 * Dk + s_idx] + dq_tile[1 * Dk + s_idx]
                          + dq_tile[2 * Dk + s_idx] + dq_tile[3 * Dk + s_idx];
              float sum_k = dk_tile[0 * Dk + s_idx] + dk_tile[1 * Dk + s_idx]
                          + dk_tile[2 * Dk + s_idx] + dk_tile[3 * Dk + s_idx];
              dq_slot[s_idx] = static_cast<OutT>(sum_q);
              dk_slot[s_idx] = static_cast<OutT>(sum_k);
            }
            if (thread_index_in_simdgroup == 0) {
              float sum_dg = dg_tile[0] + dg_tile[1] + dg_tile[2] + dg_tile[3];
              float sum_db = dbeta_tile[0] + dbeta_tile[1]
                           + dbeta_tile[2] + dbeta_tile[3];
              dg_base[t * Hv * Dv_groups + tg_y]    = static_cast<OutT>(sum_dg);
              dbeta_base[t * Hv * Dv_groups + tg_y] = static_cast<OutT>(sum_db);
            }
          }
          // dv is unique per (b,hv,t,dv_idx) — no reduction needed.
          if (thread_index_in_simdgroup == 0) {
            dv_base[t * Hv * Dv + dv_idx] = static_cast<InT>(dv_val);
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Write dS_initial for the chunk (same (b,hv,dv) slice).
        auto ds_init_row = ds_init_ptr + dv_idx * Dk;
        for (int i = 0; i < n_per_t; ++i) {
          ds_init_row[n_per_t * dk_idx + i] = static_cast<InT>(dS[i]);
        }
    """
    return mx.fast.metal_kernel(
        name="gated_delta_bwd",
        input_names=[
            "q", "k", "v", "g", "beta",
            "state_initial", "state_history", "dy", "dS_final", "T",
        ],
        output_names=[
            "dq", "dk_out", "dv_out", "dg", "dbeta", "dS_initial",
        ],
        source=source,
    )


def _make_bwd_kernel_vec():
    """Backward kernel for vectorised gating (``g`` shape [B,T,Hv,Dk]).

    Differs from the scalar kernel in three places:
      * ``g_t`` is loaded as a per-Dk vector (``g_t_vec[n_per_t]``) rather
        than a scalar read.
      * ``S_tmp = g_t * S_prev`` becomes element-wise: ``g_t_vec[i] * S_prev[i]``.
      * ``dg`` carries an extra ``Dk`` axis: output shape
        ``[B, T, Hv, Dv/4, Dk]`` (vs ``[B, T, Hv, Dv/4]`` for scalar).
        The intra-threadgroup reduction tile grows to ``4 * Dk`` floats for
        ``dg``; scalar values per ``(b, hv, t)`` are retained for ``dbeta``.
    """
    if not mx.metal.is_available():
        return None
    source = r"""
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto q_base = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_base = k + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto v_base = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        auto g_base = g + (b_idx * T * Hv + hv_idx) * Dk;   // [B,T,Hv,Dk]
        auto beta_base = beta + b_idx * T * Hv + hv_idx;
        auto dy_base = dy + b_idx * T * Hv * Dv + hv_idx * Dv;
        auto hist_base = state_history
                       + ((b_idx * Hv + hv_idx) * T) * Dv * Dk;
        auto s_initial = state_initial + (n * Dv + 0) * Dk;

        constexpr int Dv_groups = Dv / 4;
        auto tg_y = threadgroup_position_in_grid.y;
        auto dq_base = dq + ((b_idx * T * Hv + hv_idx) * Dv_groups) * Dk;     // [B,T,Hv,Dvg,Dk]
        auto dk_base = dk_out + ((b_idx * T * Hv + hv_idx) * Dv_groups) * Dk; // [B,T,Hv,Dvg,Dk]
        auto dv_base = dv_out + b_idx * T * Hv * Dv + hv_idx * Dv;            // [B,T,Hv,Dv]
        // dg is per-Dk now: [B,T,Hv,Dvg,Dk].
        auto dg_base = dg + ((b_idx * T * Hv + hv_idx) * Dv_groups) * Dk;
        auto dbeta_base = dbeta + (b_idx * T * Hv + hv_idx) * Dv_groups;      // [B,T,Hv,Dvg]

        threadgroup float dq_tile[4 * Dk];
        threadgroup float dk_tile[4 * Dk];
        threadgroup float dg_tile[4 * Dk];
        threadgroup float dbeta_tile[4];
        auto simd_y = thread_position_in_threadgroup.y;

        auto ds_final_ptr = dS_final + (n * Dv + 0) * Dk;
        auto ds_init_ptr = dS_initial + (n * Dv + 0) * Dk;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        float dS[n_per_t];
        {
          auto ds_row = ds_final_ptr + dv_idx * Dk;
          for (int i = 0; i < n_per_t; ++i) {
            dS[i] = static_cast<float>(ds_row[n_per_t * dk_idx + i]);
          }
        }

        for (int t = T - 1; t >= 0; --t) {
          auto q_t = q_base + t * Hk * Dk;
          auto k_t = k_base + t * Hk * Dk;
          auto v_t = v_base + t * Hv * Dv;
          auto dy_t = dy_base + t * Hv * Dv;
          auto g_t_row = g_base + t * Hv * Dk;
          float beta_t = static_cast<float>(*(beta_base + t * Hv));

          // Load per-Dk g_t into registers.
          float g_t_vec[n_per_t];
          for (int i = 0; i < n_per_t; ++i) {
            g_t_vec[i] = static_cast<float>(g_t_row[n_per_t * dk_idx + i]);
          }

          auto S_new_row = hist_base + t * Dv * Dk + dv_idx * Dk;
          float S_new[n_per_t];
          for (int i = 0; i < n_per_t; ++i) {
            S_new[i] = static_cast<float>(S_new_row[n_per_t * dk_idx + i]);
          }

          const device InT* S_prev_row;
          if (t == 0) {
            S_prev_row = s_initial + dv_idx * Dk;
          } else {
            S_prev_row = hist_base + (t - 1) * Dv * Dk + dv_idx * Dk;
          }
          float S_prev[n_per_t];
          for (int i = 0; i < n_per_t; ++i) {
            S_prev[i] = static_cast<float>(S_prev_row[n_per_t * dk_idx + i]);
          }

          // S_tmp = g_t_vec (per-Dk) * S_prev
          float S_tmp[n_per_t];
          for (int i = 0; i < n_per_t; ++i) {
            S_tmp[i] = g_t_vec[i] * S_prev[i];
          }

          // kv_mem = sum_dk(S_tmp * k_t)
          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            kv_mem += S_tmp[i] * static_cast<float>(k_t[n_per_t * dk_idx + i]);
          }
          kv_mem = simd_sum(kv_mem);

          float v_val = static_cast<float>(v_t[dv_idx]);
          float delta = (v_val - kv_mem) * beta_t;
          float dy_val = static_cast<float>(dy_t[dv_idx]);

          // (1) y = S_new @ q
          float dq_contrib[n_per_t];
          for (int i = 0; i < n_per_t; ++i) {
            float q_val = static_cast<float>(q_t[n_per_t * dk_idx + i]);
            dS[i] += dy_val * q_val;
            dq_contrib[i] = S_new[i] * dy_val;
          }

          // (2) S_new = S_tmp + outer(delta, k_t)
          float k_regs[n_per_t];
          for (int i = 0; i < n_per_t; ++i) {
            k_regs[i] = static_cast<float>(k_t[n_per_t * dk_idx + i]);
          }
          float ddelta = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            ddelta += dS[i] * k_regs[i];
          }
          ddelta = simd_sum(ddelta);

          float dk_a[n_per_t];
          for (int i = 0; i < n_per_t; ++i) {
            dk_a[i] = dS[i] * delta;
          }

          // (3) delta = (v - kv_mem) * beta
          float dv_val  = ddelta * beta_t;
          float dkv_mem = -ddelta * beta_t;
          float dbeta_t = ddelta * (v_val - kv_mem);

          // (4) kv_mem = sum_dk(S_tmp * k_t)
          float dk_b[n_per_t];
          for (int i = 0; i < n_per_t; ++i) {
            dS[i] += dkv_mem * k_regs[i];
            dk_b[i] = S_tmp[i] * dkv_mem;
          }

          // (5) S_tmp = g_t_vec (per-Dk) * S_prev
          //     dS_prev[i] = g_t_vec[i] * dS[i]   (per-i)
          //     dg_vec[i]  = dS[i] * S_prev[i]     (per-i, scalar per element)
          float dg_local_vec[n_per_t];
          for (int i = 0; i < n_per_t; ++i) {
            dg_local_vec[i] = dS[i] * S_prev[i];   // per-Dk dg contribution
            dS[i] = g_t_vec[i] * dS[i];            // propagate to dS_prev
          }

          // Stash contributions into shared memory for intra-threadgroup sum.
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            dq_tile[simd_y * Dk + s_idx] = dq_contrib[i];
            dk_tile[simd_y * Dk + s_idx] = dk_a[i] + dk_b[i];
            dg_tile[simd_y * Dk + s_idx] = dg_local_vec[i];
          }
          if (thread_index_in_simdgroup == 0) {
            dbeta_tile[simd_y] = dbeta_t;
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          if (simd_y == 0) {
            auto dq_slot = dq_base + t * Hv * Dv_groups * Dk + tg_y * Dk;
            auto dk_slot = dk_base + t * Hv * Dv_groups * Dk + tg_y * Dk;
            auto dg_slot = dg_base + t * Hv * Dv_groups * Dk + tg_y * Dk;
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              float sum_q = dq_tile[0 * Dk + s_idx] + dq_tile[1 * Dk + s_idx]
                          + dq_tile[2 * Dk + s_idx] + dq_tile[3 * Dk + s_idx];
              float sum_k = dk_tile[0 * Dk + s_idx] + dk_tile[1 * Dk + s_idx]
                          + dk_tile[2 * Dk + s_idx] + dk_tile[3 * Dk + s_idx];
              float sum_g = dg_tile[0 * Dk + s_idx] + dg_tile[1 * Dk + s_idx]
                          + dg_tile[2 * Dk + s_idx] + dg_tile[3 * Dk + s_idx];
              dq_slot[s_idx] = static_cast<OutT>(sum_q);
              dk_slot[s_idx] = static_cast<OutT>(sum_k);
              dg_slot[s_idx] = static_cast<OutT>(sum_g);
            }
            if (thread_index_in_simdgroup == 0) {
              float sum_db = dbeta_tile[0] + dbeta_tile[1]
                           + dbeta_tile[2] + dbeta_tile[3];
              dbeta_base[t * Hv * Dv_groups + tg_y] = static_cast<OutT>(sum_db);
            }
          }
          if (thread_index_in_simdgroup == 0) {
            dv_base[t * Hv * Dv + dv_idx] = static_cast<InT>(dv_val);
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        auto ds_init_row = ds_init_ptr + dv_idx * Dk;
        for (int i = 0; i < n_per_t; ++i) {
          ds_init_row[n_per_t * dk_idx + i] = static_cast<InT>(dS[i]);
        }
    """
    return mx.fast.metal_kernel(
        name="gated_delta_bwd_vec",
        input_names=[
            "q", "k", "v", "g", "beta",
            "state_initial", "state_history", "dy", "dS_final", "T",
        ],
        output_names=[
            "dq", "dk_out", "dv_out", "dg", "dbeta", "dS_initial",
        ],
        source=source,
    )


_fwd_save_kernel = _make_fwd_save_kernel(vectorized=False)
_fwd_save_kernel_vec = _make_fwd_save_kernel(vectorized=True)
_bwd_kernel = _make_bwd_kernel()
_bwd_kernel_vec = _make_bwd_kernel_vec()


# -- Python-level wrappers ---------------------------------------------------

def _fwd_save(q, k, v, g, beta, state_in):
    """Metal forward over full T. Routes to the scalar or vectorised
    kernel based on ``g.ndim`` (3 = scalar, 4 = per-Dk vectorised).
    """
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    input_type = q.dtype
    kernel = _fwd_save_kernel_vec if g.ndim == 4 else _fwd_save_kernel
    return kernel(
        inputs=[q, k, v, g, beta, state_in, T],
        template=[
            ("InT", input_type),
            ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[
            (B, T, Hv, Dv),              # y
            state_in.shape,              # state_out
            (B, Hv, T, Dv, Dk),          # state_history
        ],
        output_dtypes=[input_type, input_type, input_type],
    )


def _bwd(q, k, v, g, beta, state_initial, state_history, dy, dS_final):
    """Metal backward. For vectorised gating ``dg`` is per-Dk and
    therefore carries an extra axis: output shape
    ``[B, T, Hv, Dv/4, Dk]`` vs ``[B, T, Hv, Dv/4]`` for scalar.
    """
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    input_type = q.dtype
    out_type = mx.float32
    if g.ndim == 4:
        kernel = _bwd_kernel_vec
        dg_shape = (B, T, Hv, Dv // 4, Dk)
    else:
        kernel = _bwd_kernel
        dg_shape = (B, T, Hv, Dv // 4)
    return kernel(
        inputs=[
            q, k, v, g, beta,
            state_initial, state_history, dy, dS_final, T,
        ],
        template=[
            ("InT", input_type),
            ("OutT", out_type),
            ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[
            (B, T, Hv, Dv // 4, Dk),  # dq per threadgroup-y
            (B, T, Hv, Dv // 4, Dk),  # dk per threadgroup-y
            (B, T, Hv, Dv),           # dv
            dg_shape,                  # dg (scalar) or per-Dk (vectorised)
            (B, T, Hv, Dv // 4),      # dbeta per threadgroup-y
            state_initial.shape,      # dS_initial
        ],
        output_dtypes=[
            out_type, out_type, input_type,
            out_type, out_type, input_type,
        ],
    )


@mx.custom_function
def _gated_delta_core(q, k, v, g, beta, state_in):
    # Forward path uses the upstream single-pass Metal kernel — it is the
    # fastest implementation available and does not pay for the extra
    # ``state_history`` buffer that backward needs. The VJP below runs a
    # dedicated forward-with-save kernel only when gradients are required.
    return gated_delta_kernel(q, k, v, g, beta, state_in)


@_gated_delta_core.vjp
def _gated_delta_core_vjp(primals, cotangents, outputs):
    q, k, v, g, beta, state_in = primals
    dy, dS_final = cotangents
    # Recompute the forward to recover the state history.
    _, _, history = _fwd_save(q, k, v, g, beta, state_in)
    dq_dv, dk_dv, dv_, dg_dv, dbeta_dv, dS_init = _bwd(
        q, k, v, g, beta, state_in, history, dy, dS_final
    )
    # Deterministic reduction over the Dv axis (added by the kernel to
    # avoid ``atomic_fetch_add`` ordering noise).
    dq = dq_dv.sum(axis=3).astype(q.dtype)         # [B, T, Hv, Dk]
    dk_ = dk_dv.sum(axis=3).astype(k.dtype)        # [B, T, Hv, Dk]
    dg = dg_dv.sum(axis=3).astype(g.dtype)         # [B, T, Hv]
    dbeta = dbeta_dv.sum(axis=3).astype(beta.dtype)  # [B, T, Hv]
    return dq, dk_, dv_, dg, dbeta, dS_init


# -- Public drop-in replacement ----------------------------------------------

def gated_delta_update_vjp_metal(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """Metal-accelerated drop-in for :func:`gated_delta_update`.

    Semantically identical to the pure-Python ``gated_delta_update_vjp``;
    moves the recurrent forward and backward into dedicated Metal kernels.
    ``mask`` is not supported yet (training default).
    """
    if mask is not None:
        raise NotImplementedError("masked recurrence not implemented for Metal VJP")

    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]

    beta = mx.sigmoid(b)
    g = _compute_g(A_log, a, dt_bias)

    rf = Hv // Hk
    if rf > 1:
        q = mx.repeat(q, rf, axis=-2)
        k = mx.repeat(k, rf, axis=-2)

    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=q.dtype)

    # Chunked to bound state_history memory.
    ys = []
    S = state
    for start in range(0, T, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, T)
        y_c, S = _gated_delta_core(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            g[:, start:end],
            beta[:, start:end],
            S,
        )
        ys.append(y_c)
    y = mx.concatenate(ys, axis=1) if len(ys) > 1 else ys[0]
    return y, S
