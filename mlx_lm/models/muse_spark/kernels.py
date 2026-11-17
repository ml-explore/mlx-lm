# Copyright © 2026 Apple Inc.

"""Muse Spark's fused Metal kernels, and their launch functions."""

import math

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.kernels.common import row_threads, scalar

__all__ = [
    "norm_add_norm",
    "post_expert_norm_combine",
    "pre_norm",
    "qk_norm",
    "qk_norm_rope",
    "rms_norm_sigmoid_gate",
]


# The q and k head norms and the q scale, in one kernel.

# One SIMD group per head. q and k are the first two parts of each qkvg row.
_QK_NORM_SOURCE = """
    constexpr int R = (D + 31) / 32;
    uint row = thread_position_in_grid.x / 32;
    uint lane = thread_index_in_simdgroup;
    uint n = row / (HQ + HK);
    uint h = row % (HQ + HK);
    auto x = qkvg + size_t(n) * (2 * (HQ + HK) * D) + h * D;

    float v[R];
    float sum_sq = 0.0f;
    for (int i = 0; i < R; i++) {
        int d = lane + 32 * i;
        v[i] = d < D ? static_cast<float>(x[d]) : 0.0f;
        sum_sq += v[i] * v[i];
    }
    float rstd = metal::precise::rsqrt(simd_sum(sum_sq) / D + eps);
    for (int i = 0; i < R; i++) {
        int d = lane + 32 * i;
        if (d < D) {
            T y = static_cast<T>(v[i] * rstd);
            if (h < HQ) {
                q[(size_t(n) * HQ + h) * D + d] = static_cast<T>(static_cast<float>(y) * scale);
            } else {
                k[(size_t(n) * HK + h - HQ) * D + d] = y;
            }
        }
    }
"""

_qk_norm_kernel = mx.fast.metal_kernel(
    name="qk_norm",
    input_names=["qkvg", "scale", "eps"],
    output_names=["q", "k"],
    source=_QK_NORM_SOURCE,
)


def qk_norm(
    qkvg: mx.array,
    *,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    eps: float,
    query_scale: float,
) -> list[mx.array]:
    tokens = qkvg.shape[:-1]
    rows = qkvg.size // qkvg.shape[-1] * (n_heads + n_kv_heads)
    return _qk_norm_kernel(
        inputs=[qkvg, scalar(query_scale), scalar(eps)],
        template=[
            ("T", qkvg.dtype),
            ("D", head_dim),
            ("HQ", n_heads),
            ("HK", n_kv_heads),
        ],
        grid=(rows * 32, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[tokens + (n_heads, head_dim), tokens + (n_kv_heads, head_dim)],
        output_dtypes=[qkvg.dtype, qkvg.dtype],
    )


# The q and k head norms, the q scale and the rotary embedding, in one kernel.

# qk_norm, then the rotation of mx.fast.rope(traditional=True), which rotates
# the rounded q and k.
_QK_NORM_ROPE_SOURCE = """
    constexpr int R = (D + 31) / 32;
    uint row = thread_position_in_grid.x / 32;
    uint lane = thread_index_in_simdgroup;
    uint n = row / (HQ + HK);
    uint h = row % (HQ + HK);
    auto x = qkvg + size_t(n) * (2 * (HQ + HK) * D) + h * D;

    float v[R];
    float sum_sq = 0.0f;
    for (int i = 0; i < R; i++) {
        int d = lane + 32 * i;
        v[i] = d < D ? static_cast<float>(x[d]) : 0.0f;
        sum_sq += v[i] * v[i];
    }
    float rstd = metal::precise::rsqrt(simd_sum(sum_sq) / D + eps);
    float pos = static_cast<float>(n % seq_len + offset[(n / seq_len) * OFFSET_STRIDE]);
    for (int i = 0; i < R; i++) {
        int d = lane + 32 * i;
        T y = static_cast<T>(v[i] * rstd);
        if (h < HQ) {
            y = static_cast<T>(static_cast<float>(y) * scale);
        }
        float self = static_cast<float>(y);
        float other = simd_shuffle_xor(self, 1);
        float inv_freq = metal::exp2(-(static_cast<float>(d / 2) / (D / 2)) * log2_base);
        float theta = pos * inv_freq;
        float c = metal::fast::cos(theta);
        float s = metal::fast::sin(theta);
        y = static_cast<T>(d % 2 == 0 ? self * c - other * s : other * s + self * c);
        if (d < D) {
            if (h < HQ) {
                q[(size_t(n) * HQ + h) * D + d] = y;
            } else {
                k[(size_t(n) * HK + h - HQ) * D + d] = y;
            }
        }
    }
"""

_qk_norm_rope_kernel = mx.fast.metal_kernel(
    name="qk_norm_rope",
    input_names=["qkvg", "scale", "eps", "offset", "seq_len", "log2_base"],
    output_names=["q", "k"],
    source=_QK_NORM_ROPE_SOURCE,
)


def qk_norm_rope(
    qkvg: mx.array,
    *,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    eps: float,
    query_scale: float,
    rope_base: float,
    offset: int | mx.array,
) -> list[mx.array]:
    """qk_norm, then rotate q and k as mx.fast.rope(traditional=True) does,
    the tokens at positions offset, offset + 1, ... offset is an int, or an
    array with one offset per batch row. Returns q (..., n_heads, head_dim)
    and k (..., n_kv_heads, head_dim)."""
    tokens = qkvg.shape[:-1]
    rows = qkvg.size // qkvg.shape[-1] * (n_heads + n_kv_heads)
    if isinstance(offset, mx.array):
        offset = offset.reshape(-1).astype(mx.int32)
    else:
        offset = mx.array([offset], dtype=mx.int32)
    return _qk_norm_rope_kernel(
        inputs=[
            qkvg,
            scalar(query_scale),
            scalar(eps),
            offset,
            scalar(tokens[-1], mx.int32),
            # As mx.fast.rope, which takes the log2 of the fp32 base.
            scalar(math.log2(rope_base)),
        ],
        template=[
            ("T", qkvg.dtype),
            ("D", head_dim),
            ("HQ", n_heads),
            ("HK", n_kv_heads),
            ("OFFSET_STRIDE", int(offset.size > 1)),
        ],
        grid=(rows * 32, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[tokens + (n_heads, head_dim), tokens + (n_kv_heads, head_dim)],
        output_dtypes=[qkvg.dtype, qkvg.dtype],
    )


# The attention output norm and sigmoid gate, in one kernel.

_NORM_GATE_SOURCE = """
    // One SIMD group per (token, head) row of D values.
    uint row = thread_position_in_grid.x / 32;
    uint lane = thread_index_in_simdgroup;
    uint B = x_shape[0], H = x_shape[1], L = x_shape[2];
    if (row >= B * L * H) {
        return;
    }
    uint h = row % H;
    uint l = (row / H) % L;
    uint b = row / (H * L);
    int xd = x_strides[3], gd = g_strides[2];
    auto xr = x + b * x_strides[0] + h * x_strides[1] + l * x_strides[2];
    auto gr = g + b * g_strides[0] + l * g_strides[1] + h * D * gd;

    constexpr int R = (D + 31) / 32;
    float v[R];
    float sum_sq = 0.0f;
    for (int i = 0; i < R; i++) {
        int d = lane + 32 * i;
        v[i] = d < D ? static_cast<float>(xr[d * xd]) : 0.0f;
        sum_sq += v[i] * v[i];
    }
    float rstd = metal::precise::rsqrt(simd_sum(sum_sq) / D + eps);
    for (int i = 0; i < R; i++) {
        int d = lane + 32 * i;
        if (d < D) {
            float sig = 1.0f / (1.0f + metal::exp(-static_cast<float>(gr[d * gd])));
            out[row * D + d] = static_cast<T>(v[i] * rstd * sig);
        }
    }
"""

_norm_gate_kernel = mx.fast.metal_kernel(
    name="rms_norm_sigmoid_gate",
    input_names=["x", "g", "eps"],
    output_names=["out"],
    source=_NORM_GATE_SOURCE,
    ensure_row_contiguous=False,
)


def rms_norm_sigmoid_gate(x: mx.array, gate: mx.array, *, eps: float) -> mx.array:
    """Return rms_norm(x) * sigmoid(gate) for each head, like the reference's
    fused kernel: fp32 math and one rounding. x is the attention output
    (B, H, L, D), gate is (B, L, H * D), and the result is (B, L, H * D)."""
    B, H, L, D = x.shape
    return _norm_gate_kernel(
        inputs=[x, gate, scalar(eps)],
        template=[("T", x.dtype), ("D", D)],
        grid=(B * L * H * 32, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, L, H * D)],
        output_dtypes=[x.dtype],
    )[0]


_BLOCK_SUM_ONCE_HEADER = """
template <int TG>
inline float block_sum_once(float v, threadgroup float* buf, uint lane, uint sg) {
    v = simd_sum(v);
    if (TG > 32) {
        if (lane == 0) {
            buf[sg] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        v = simd_sum(lane < TG / 32 ? buf[lane] : 0.0f);
    }
    return v;
}
"""


_HEAD = """
    constexpr int R = (D + TG - 1) / TG;
    threadgroup float buf[2][32];
    uint t = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    size_t base = size_t(threadgroup_position_in_grid.x) * D;
"""

# The pre-norm of the stream values in v, with their sum of squares in sum_sq.
_PRE_NORM = """
    rstd = metal::precise::rsqrt(block_sum_once<TG>(sum_sq, buf[1], lane, sg) / D + pre_eps);
    for (int i = 0; i < R; i++) {
        int d = t + i * TG;
        if (d < D) {
            normed[base + d] = static_cast<T>(v[i] * rstd * pw[i]);
        }
    }
"""


def _source(pre_norm: bool) -> str:
    pw = ", pw[R]" if pre_norm else ""
    load_pw = "pw[i] = pre_w[d];" if pre_norm else ""
    source = _HEAD + f"""
    float v[R], st[R], al[R], be[R]{pw};
    for (int i = 0; i < R; i++) {{
        int d = t + i * TG;
        v[i] = st[i] = al[i] = be[i] = 0.0f;
        if (d < D) {{
            v[i] = static_cast<float>(branch[base + d]);
            st[i] = stream[base + d];
            al[i] = alpha[d];
            be[i] = beta[d];
            {load_pw}
            if (POST_SCALE) {{
                // The weight scales the input in T, like the norm module.
                float w = static_cast<T>(post_w[d]);
                v[i] = static_cast<T>(w * v[i]);
            }}
        }}
    }}

    float sum_sq = 0.0f;
    for (int i = 0; i < R; i++) {{
        sum_sq += v[i] * v[i];
    }}
    float rstd = metal::precise::rsqrt(block_sum_once<TG>(sum_sq, buf[0], lane, sg) / D + post_eps);

    sum_sq = 0.0f;
    for (int i = 0; i < R; i++) {{
        int d = t + i * TG;
        float s = 0.0f;
        if (d < D) {{
            float n = static_cast<T>(v[i] * rstd);
            s = al[i] * st[i] + be[i] * n;
            new_stream[base + d] = s;
        }}
        v[i] = s;
        sum_sq += s * s;
    }}
"""
    if pre_norm:
        source += _PRE_NORM
    return source


# Only the pre-norm, with the same sums as in norm_add_norm.
_PRE_NORM_ONLY = _HEAD + """
    float v[R], pw[R];
    float sum_sq = 0.0f;
    for (int i = 0; i < R; i++) {
        int d = t + i * TG;
        v[i] = pw[i] = 0.0f;
        if (d < D) {
            v[i] = stream[base + d];
            pw[i] = pre_w[d];
        }
        sum_sq += v[i] * v[i];
    }
    float rstd;
""" + _PRE_NORM


_NORM_ADD_INPUTS = ["branch", "stream", "alpha", "beta", "post_w", "post_eps"]

_norm_add_norm_kernel = mx.fast.metal_kernel(
    name="norm_add_norm",
    input_names=_NORM_ADD_INPUTS + ["pre_w", "pre_eps"],
    output_names=["new_stream", "normed"],
    source=_source(pre_norm=True),
    header=_BLOCK_SUM_ONCE_HEADER,
)

_norm_add_kernel = mx.fast.metal_kernel(
    name="norm_add",
    input_names=_NORM_ADD_INPUTS,
    output_names=["new_stream"],
    source=_source(pre_norm=False),
    header=_BLOCK_SUM_ONCE_HEADER,
)

_pre_norm_kernel = mx.fast.metal_kernel(
    name="norm_add_norm_pre_norm",
    input_names=["stream", "pre_w", "pre_eps"],
    output_names=["normed"],
    source=_PRE_NORM_ONLY,
    header=_BLOCK_SUM_ONCE_HEADER,
)


def norm_add_norm(
    branch: mx.array,
    stream: mx.array,
    *,
    alpha: mx.array,
    beta: mx.array,
    post_weight: mx.array | None,
    post_eps: float,
    next_norm: nn.RMSNorm | None,
) -> tuple[mx.array, mx.array | None]:
    """Post-norm the branch, add it to the fp32 stream with the gate weights,
    and pre-norm the sum with next_norm.
    Returns the new stream and the normed sum (None without next_norm)."""
    D = branch.shape[-1]
    tg = row_threads(D)
    has_scale = post_weight is not None
    inputs = [
        branch,
        stream,
        alpha,
        beta,
        post_weight if has_scale else alpha,
        scalar(post_eps),
    ]
    kwargs = {
        "template": [
            ("T", branch.dtype),
            ("D", D),
            ("TG", tg),
            ("POST_SCALE", has_scale),
        ],
        "grid": (branch.size // D * tg, 1, 1),
        "threadgroup": (tg, 1, 1),
    }
    if next_norm is None:
        new_stream = _norm_add_kernel(
            inputs=inputs,
            output_shapes=[stream.shape],
            output_dtypes=[mx.float32],
            **kwargs,
        )[0]
        return new_stream, None
    new_stream, normed = _norm_add_norm_kernel(
        inputs=inputs + [next_norm.weight, scalar(next_norm.eps)],
        output_shapes=[stream.shape, branch.shape],
        output_dtypes=[mx.float32, branch.dtype],
        **kwargs,
    )
    return new_stream, normed


def pre_norm(stream: mx.array, norm: nn.RMSNorm, *, dtype: mx.Dtype) -> mx.array:
    """Pre-norm the fp32 stream with norm, as norm_add_norm does.
    For a stream from norm_add_norm, the values are equal to its normed sum."""
    D = stream.shape[-1]
    tg = row_threads(D)
    return _pre_norm_kernel(
        inputs=[stream, norm.weight, scalar(norm.eps)],
        template=[("T", dtype), ("D", D), ("TG", tg)],
        grid=(stream.size // D * tg, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[stream.shape],
        output_dtypes=[dtype],
    )[0]


# This kernel inputs [tokens * k, D]:
# normalises each input token x_i / sqrt(sum(x_i)**2) * w_i -- rms norm
# and then combine as a weighted sum
_EXPERT_COMBINE_SOURCE = """
    constexpr int R = (D + TG - 1) / TG;
    constexpr int SG = TG / 32;
    threadgroup float buf[K * SG];
    uint t = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    size_t row = threadgroup_position_in_grid.x;

    float w[R];
    for (int i = 0; i < R; i++) {
        int d = t + i * TG;
        w[i] = d < D ? static_cast<float>(norm_w[d]) : 0.0f;
    }
    T x[K][R];
    for (int k = 0; k < K; k++) {
        size_t base = (row * K + k) * D;
        for (int i = 0; i < R; i++) {
            int d = t + i * TG;
            x[k][i] = d < D ? experts[base + d] : static_cast<T>(0.0f);
        }
    }
    float sum_sq[K];
    for (int k = 0; k < K; k++) {
        sum_sq[k] = 0.0f;
        for (int i = 0; i < R; i++) {
            float v = static_cast<float>(x[k][i]) * w[i];
            sum_sq[k] += v * v;
        }
        sum_sq[k] = simd_sum(sum_sq[k]);
    }
    if (TG > 32) {
        if (lane == 0) {
            for (int k = 0; k < K; k++) {
                buf[k * SG + sg] = sum_sq[k];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int k = 0; k < K; k++) {
            sum_sq[k] = simd_sum(lane < SG ? buf[k * SG + lane] : 0.0f);
        }
    }

    float acc[R];
    for (int i = 0; i < R; i++) {
        acc[i] = 0.0f;
    }
    for (int k = 0; k < K; k++) {
        float inv = metal::precise::rsqrt(sum_sq[k] / D + eps);
        float ew = static_cast<T>(weights[row * K + k]);
        for (int i = 0; i < R; i++) {
            float v = static_cast<float>(x[k][i]) * w[i];
            acc[i] = static_cast<T>(acc[i] + v * inv * ew);
        }
    }
    for (int i = 0; i < R; i++) {
        int d = t + i * TG;
        if (d < D) {
            out[row * D + d] = static_cast<T>(acc[i]);
        }
    }
"""

_expert_combine_kernel = mx.fast.metal_kernel(
    name="post_expert_norm_combine",
    input_names=["experts", "weights", "norm_w", "eps"],
    output_names=["out"],
    source=_EXPERT_COMBINE_SOURCE,
)


def post_expert_norm_combine(
    experts: mx.array, weights: mx.array, *, norm_weight: mx.array, eps: float
) -> mx.array:
    """Norm each expert output with norm_weight as its input scale, and add
    the experts with their weights in order. experts is (..., K, D) and
    weights is (..., K); the result is (..., D)."""
    K, D = experts.shape[-2:]
    tg = min(1024, 2 * row_threads(D))
    return _expert_combine_kernel(
        inputs=[experts, weights, norm_weight, scalar(eps)],
        template=[("T", experts.dtype), ("D", D), ("K", K), ("TG", tg)],
        grid=(experts.size // (K * D) * tg, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[experts.shape[:-2] + (D,)],
        output_dtypes=[experts.dtype],
    )[0]
