# Copyright © 2026 Apple Inc.

import mlx.core as mx
from mlx import nn

from ._common import row_threads, scalar

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
