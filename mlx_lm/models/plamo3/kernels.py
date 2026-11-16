# Copyright © 2026 Apple Inc.

import mlx.core as mx


def _make_qk_norm_rope_kernel():
    if not mx.metal.is_available():
        return None
    return mx.fast.metal_kernel(
        name="plamo3_qk_norm_rope",
        input_names=["qkv", "qw", "kw", "freqs", "offset", "eps", "mscale", "base"],
        output_names=["q", "k"],
        source="""
            uint lane = thread_index_in_simdgroup;
            uint head = threadgroup_position_in_grid.x;
            uint start = head * 128;
            float sum = 0;
            for (uint i = 0; i < 4; ++i) {
                float x = float(qkv[start + lane * 4 + i]);
                sum += x * x;
            }
            float inv = metal::precise::rsqrt(simd_sum(sum) / 128 + eps);
            for (uint i = 0; i < 2; ++i) {
                uint d = lane + i * 32;
                T w1 = head < NQ ? qw[d] : kw[d];
                T w2 = head < NQ ? qw[d + 64] : kw[d + 64];
                // Keep the intermediate rounding of RMSNorm and YaRN.
                T x1 = T(float(qkv[start + d]) * inv) * w1;
                T x2 = T(float(qkv[start + d + 64]) * inv) * w2;
                x1 = T(x1 * T(mscale));
                x2 = T(x2 * T(mscale));
                float freq = YARN ? 1.0f / freqs[d]
                                  : metal::exp2(-float(d) / 64 * base);
                float angle = float(offset) * freq;
                float co = metal::fast::cos(angle);
                float si = metal::fast::sin(angle);
                T y1 = T(float(x1) * co - float(x2) * si);
                T y2 = T(float(x1) * si + float(x2) * co);
                if (head < NQ) {
                    q[head * 128 + d] = y1;
                    q[head * 128 + d + 64] = y2;
                } else {
                    k[(head - NQ) * 128 + d] = y1;
                    k[(head - NQ) * 128 + d + 64] = y2;
                }
            }
        """,
    )


_qk_norm_rope_kernel = _make_qk_norm_rope_kernel()


def _make_residual_norm_kernel():
    if not mx.metal.is_available():
        return None
    return mx.fast.metal_kernel(
        name="plamo3_residual_norm",
        input_names=["x", "residual", "post", "pre", "eps"],
        output_names=["h", "normalized"],
        source="""
            uint lane = thread_index_in_simdgroup;
            uint group = simdgroup_index_in_threadgroup;
            uint start = thread_position_in_threadgroup.x * 4;
            threadgroup float sums[32];
            threadgroup float inv;
            float values[4];
            for (uint i = 0; i < 4; ++i) values[i] = float(x[start + i]);
            for (uint pass = 0; pass < 2; ++pass) {
                float sum = 0;
                for (uint i = 0; i < 4; ++i) sum += values[i] * values[i];
                sum = simd_sum(sum);
                if (group == 0) sums[lane] = 0;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (lane == 0) sums[group] = sum;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (group == 0) {
                    sum = simd_sum(sums[lane]);
                    if (lane == 0) inv = metal::precise::rsqrt(sum / D + eps);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint i = 0; i < 4; ++i) {
                    uint d = start + i;
                    T value = T(values[i] * inv);
                    if (pass == 0) {
                        T scaled = value * post[d];
                        T added = residual[d] + scaled;
                        h[d] = added;
                        values[i] = float(added);
                    } else {
                        normalized[d] = value * pre[d];
                    }
                }
            }
        """,
    )


_residual_norm_kernel = _make_residual_norm_kernel()
