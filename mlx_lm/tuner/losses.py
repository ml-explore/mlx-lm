# Copyright © 2025 Apple Inc.

import mlx.core as mx
import mlx.nn as nn


def _make_kl_forward_kernel():
    source = """
    constexpr int M = 4;
    constexpr int block = 1024 * M;
    constexpr int full_blocks = V / block;
    constexpr int extra = V - full_blocks * block;

    threadgroup float shared[32 * 2];

    uint out_idx = threadgroup_position_in_grid.y;
    uint simd_lane_id = thread_index_in_simdgroup;
    uint simd_group_id = simdgroup_index_in_threadgroup;

    logits_q += out_idx * V;
    logits_p += out_idx * V;
    out += out_idx;

    float lse_q_minus_p;
    float lse_p;

    {
        float max_q = -1e30;
        float max_p = -1e30;
        float sum_exp_q = 0;
        float sum_exp_p = 0;

        int offset = thread_index_in_threadgroup * M;
        for (int i = 0; i < full_blocks; i++) {
            // Read and update q and p
            float vals_q[M];
            float vals_p[M];
            for (int j=0; j<M; j++) {
                vals_q[j] = logits_q[offset + j];
                vals_p[j] = logits_p[offset + j];
            }
            float prev_max_q = max_q;
            float prev_max_p = max_p;
            for (int j=0; j<M; j++) {
                max_q = max(max_q, vals_q[j]);
                max_p = max(max_p, vals_p[j]);
            }
            sum_exp_q *= metal::fast::exp(prev_max_q - max_q);
            sum_exp_p *= metal::fast::exp(prev_max_p - max_p);
            for (int j=0; j<M; j++) {
                sum_exp_q += metal::fast::exp(vals_q[j] - max_q);
                sum_exp_p += metal::fast::exp(vals_p[j] - max_p);
            }

            // Move to the next block
            offset += block;
        }
        if (extra > 0) {
            // Read and update q and p
            float vals_q[M];
            float vals_p[M];
            for (int j=0; j < M; j++) {
                vals_q[j] = (offset + j < V) ? logits_q[offset + j] : -1e30;
                vals_p[j] = (offset + j < V) ? logits_p[offset + j] : -1e30;
            }
            float prev_max_q = max_q;
            float prev_max_p = max_p;
            for (int j=0; j<M; j++) {
                max_q = max(max_q, vals_q[j]);
                max_p = max(max_p, vals_p[j]);
            }
            sum_exp_q *= metal::fast::exp(prev_max_q - max_q);
            sum_exp_p *= metal::fast::exp(prev_max_p - max_p);
            for (int j=0; j<M; j++) {
                sum_exp_q += metal::fast::exp(vals_q[j] - max_q);
                sum_exp_p += metal::fast::exp(vals_p[j] - max_p);
            }
        }

        // Share the maxs across the threadgroup
        float prev_max_q = max_q;
        float prev_max_p = max_p;
        max_q = simd_max(max_q);
        max_p = simd_max(max_p);
        if (simd_lane_id == 0) {
            shared[simd_group_id * 2 + 0] = max_q;
            shared[simd_group_id * 2 + 1] = max_p;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        max_q = shared[simd_lane_id * 2 + 0];
        max_p = shared[simd_lane_id * 2 + 1];
        max_q = simd_max(max_q);
        max_p = simd_max(max_p);

        // Share the sum_exp across the threadgroup
        sum_exp_q *= metal::fast::exp(prev_max_q - max_q);
        sum_exp_p *= metal::fast::exp(prev_max_p - max_p);
        sum_exp_q = simd_sum(sum_exp_q);
        sum_exp_p = simd_sum(sum_exp_p);
        if (simd_lane_id == 0) {
            shared[simd_group_id * 2 + 0] = sum_exp_q;
            shared[simd_group_id * 2 + 1] = sum_exp_p;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum_exp_q = shared[simd_lane_id * 2 + 0];
        sum_exp_p = shared[simd_lane_id * 2 + 1];
        sum_exp_q = simd_sum(sum_exp_q);
        sum_exp_p = simd_sum(sum_exp_p);

        lse_p = max_p + metal::fast::log(sum_exp_p);
        lse_q_minus_p = max_q + metal::fast::log(sum_exp_q) - lse_p;
    }

    threadgroup_barrier(mem_flags::mem_none);

    {
        float kl = 0;

        int offset = thread_index_in_threadgroup * M;
        for (int i = 0; i < full_blocks; i++) {
            // Read and add to the kl
            float vals_q[M];
            float vals_p[M];
            for (int j=0; j<M; j++) {
                vals_q[j] = logits_q[offset + j];
                vals_p[j] = logits_p[offset + j];
            }

            for (int j=0; j<M; j++) {
                kl += metal::fast::exp(vals_p[j] - lse_p) * (vals_p[j] - vals_q[j] + lse_q_minus_p);
            }

            // Move to the next block
            offset += block;
        }
        if (extra > 0) {
            float vals_q[M];
            float vals_p[M];
            for (int j=0; j<M; j++) {
                vals_q[j] = (offset + j < V) ? logits_q[offset + j] : -1e30;
                vals_p[j] = (offset + j < V) ? logits_p[offset + j] : -1e30;
            }

            for (int j=0; j<M; j++) {
                kl += metal::fast::exp(vals_p[j] - lse_p) * (vals_p[j] - vals_q[j] + lse_q_minus_p);
            }
        }

        // Add the kl across the threadgroup
        kl = simd_sum(kl);
        if (simd_lane_id == 0) {
            shared[simd_group_id] = kl;
        }
        threadgroup_barrier(mem_flags::mem_none);
        kl = shared[simd_lane_id];
        kl = simd_sum(kl);

        if (thread_index_in_threadgroup == 0) {
            out[0] = static_cast<T>(kl);
        }
    }
    """

    return mx.fast.metal_kernel(
        name="kl_forward",
        input_names=["logits_q", "logits_p"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


def _make_kl_backward_kernel():
    source = """
    constexpr int M = 4;
    constexpr int block = 1024 * M;
    constexpr int full_blocks = V / block;
    constexpr int extra = V - full_blocks * block;

    threadgroup float shared[32 * 2];

    uint out_idx = threadgroup_position_in_grid.y;
    uint simd_lane_id = thread_index_in_simdgroup;
    uint simd_group_id = simdgroup_index_in_threadgroup;

    logits_q += out_idx * V;
    logits_p += out_idx * V;
    out += out_idx * V;
    cotan += out_idx;

    float lse_q;
    float lse_p;

    {
        float max_q = -1e30;
        float max_p = -1e30;
        float sum_exp_q = 0;
        float sum_exp_p = 0;

        int offset = thread_index_in_threadgroup * M;
        for (int i = 0; i < full_blocks; i++) {
            // Read and update q and p
            float vals_q[M];
            float vals_p[M];
            for (int j=0; j<M; j++) {
                vals_q[j] = logits_q[offset + j];
                vals_p[j] = logits_p[offset + j];
            }
            float prev_max_q = max_q;
            float prev_max_p = max_p;
            for (int j=0; j<M; j++) {
                max_q = max(max_q, vals_q[j]);
                max_p = max(max_p, vals_p[j]);
            }
            sum_exp_q *= metal::fast::exp(prev_max_q - max_q);
            sum_exp_p *= metal::fast::exp(prev_max_p - max_p);
            for (int j=0; j<M; j++) {
                sum_exp_q += metal::fast::exp(vals_q[j] - max_q);
                sum_exp_p += metal::fast::exp(vals_p[j] - max_p);
            }

            // Move to the next block
            offset += block;
        }
        if (extra > 0) {
            // Read and update q and p
            float vals_q[M];
            float vals_p[M];
            for (int j=0; j < M; j++) {
                vals_q[j] = (offset + j < V) ? logits_q[offset + j] : -1e30;
                vals_p[j] = (offset + j < V) ? logits_p[offset + j] : -1e30;
            }
            float prev_max_q = max_q;
            float prev_max_p = max_p;
            for (int j=0; j<M; j++) {
                max_q = max(max_q, vals_q[j]);
                max_p = max(max_p, vals_p[j]);
            }
            sum_exp_q *= metal::fast::exp(prev_max_q - max_q);
            sum_exp_p *= metal::fast::exp(prev_max_p - max_p);
            for (int j=0; j<M; j++) {
                sum_exp_q += metal::fast::exp(vals_q[j] - max_q);
                sum_exp_p += metal::fast::exp(vals_p[j] - max_p);
            }
        }

        // Share the maxs across the threadgroup
        float prev_max_q = max_q;
        float prev_max_p = max_p;
        max_q = simd_max(max_q);
        max_p = simd_max(max_p);
        if (simd_lane_id == 0) {
            shared[simd_group_id * 2 + 0] = max_q;
            shared[simd_group_id * 2 + 1] = max_p;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        max_q = shared[simd_lane_id * 2 + 0];
        max_p = shared[simd_lane_id * 2 + 1];
        max_q = simd_max(max_q);
        max_p = simd_max(max_p);

        // Share the sum_exp across the threadgroup
        sum_exp_q *= metal::fast::exp(prev_max_q - max_q);
        sum_exp_p *= metal::fast::exp(prev_max_p - max_p);
        sum_exp_q = simd_sum(sum_exp_q);
        sum_exp_p = simd_sum(sum_exp_p);
        if (simd_lane_id == 0) {
            shared[simd_group_id * 2 + 0] = sum_exp_q;
            shared[simd_group_id * 2 + 1] = sum_exp_p;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum_exp_q = shared[simd_lane_id * 2 + 0];
        sum_exp_p = shared[simd_lane_id * 2 + 1];
        sum_exp_q = simd_sum(sum_exp_q);
        sum_exp_p = simd_sum(sum_exp_p);

        lse_p = max_p + metal::fast::log(sum_exp_p);
        lse_q = max_q + metal::fast::log(sum_exp_q);
    }

    threadgroup_barrier(mem_flags::mem_none);

    {
        float kl = 0;
        float c = cotan[0];

        int offset = thread_index_in_threadgroup * M;
        for (int i = 0; i < full_blocks; i++) {
            // Read and add to the kl
            float vals_q[M];
            float vals_p[M];
            for (int j=0; j<M; j++) {
                vals_q[j] = logits_q[offset + j];
                vals_p[j] = logits_p[offset + j];
            }

            for (int j=0; j<M; j++) {
                out[offset + j] = static_cast<T>(
                    c * (metal::fast::exp(vals_q[j] - lse_q) - metal::fast::exp(vals_p[j] - lse_p)));
            }

            // Move to the next block
            offset += block;
        }
        if (extra > 0) {
            float vals_q[M];
            float vals_p[M];
            for (int j=0; j<M; j++) {
                vals_q[j] = (offset + j < V) ? logits_q[offset + j] : -1e30;
                vals_p[j] = (offset + j < V) ? logits_p[offset + j] : -1e30;
            }

            for (int j=0; j<M; j++) {
                if (offset + j < V) {
                    out[offset + j] = static_cast<T>(
                        c * (metal::fast::exp(vals_q[j] - lse_q) - metal::fast::exp(vals_p[j] - lse_p)));
                }
            }
        }
    }
    """

    return mx.fast.metal_kernel(
        name="kl_backward",
        input_names=["logits_q", "logits_p", "cotan"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


_kl_forward_kernel = _make_kl_forward_kernel()
_kl_backward_kernel = _make_kl_backward_kernel()


@mx.custom_function
def _kl_div_loss(logits_q, logits_p):
    n_outs = logits_q.size // logits_q.shape[-1]
    dt = logits_q.dtype

    return _kl_forward_kernel(
        inputs=[logits_q, logits_p],
        output_shapes=[logits_q.shape[:-1]],
        output_dtypes=[dt],
        template=[("T", dt), ("V", logits_q.shape[-1])],
        grid=(1024, n_outs, 1),
        threadgroup=(1024, 1, 1),
    )[0]


@_kl_div_loss.vjp
def _kl_div_loss(primals, cotangent, output):
    logits_q, logits_p = primals
    dt = logits_q.dtype

    dp = mx.zeros_like(logits_p)
    dq = _kl_backward_kernel(
        inputs=[logits_q, logits_p, cotangent],
        output_shapes=[logits_q.shape],
        output_dtypes=[dt],
        template=[("T", dt), ("V", logits_q.shape[-1])],
        grid=(1024, cotangent.size, 1),
        threadgroup=(1024, 1, 1),
    )[0]

    return dq, dp


def kl_div_loss(logits_q, logits_p):
    if mx.metal.is_available():
        return _kl_div_loss(logits_q, logits_p)
    else:
        return nn.losses.kl_div_loss(
            logits_q - mx.logsumexp(logits_q, axis=-1, keepdims=True),
            logits_p - mx.logsumexp(logits_p, axis=-1, keepdims=True),
            axis=-1,
            reduction="none",
        )
