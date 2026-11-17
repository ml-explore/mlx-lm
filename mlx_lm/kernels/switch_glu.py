# Copyright © 2026 Apple Inc.

"""A quantized SwitchGLU for one token, with its experts' outputs weighted and
summed, in two kernels."""

from functools import lru_cache

import mlx.core as mx

# x[16] as floats and their sum. The weights of a row are a little-endian stream of
# BITS-bit values; x[i] is prescaled by 2^-s, s the offset of value i's lowest bit in
# the byte (uint16 for 4 bits) that qdot16 masks it from, so that the masked weights
# need no shift (as mlx's qmv).
_HEADER = """
template <int BITS>
inline float prescale(int i) {
    if (BITS == 2) {
        return 1.0f / float(1 << (2 * (i % 4)));
    }
    if (BITS == 3) {
        constexpr int s[8] = {0, 3, 6, 1, 4, 7, 2, 5};
        return 1.0f / float(1 << s[i % 8]);
    }
    if (BITS == 4) {
        return 1.0f / float(1 << (4 * (i % 4)));
    }
    if (BITS == 5) {
        constexpr int s[8] = {0, 5, 2, 7, 4, 1, 6, 3};
        return 1.0f / float(1 << s[i % 8]);
    }
    if (BITS == 6) {
        constexpr int s[4] = {0, 6, 4, 2};
        return 1.0f / float(1 << s[i % 4]);
    }
    return 1.0f;
}

template <int BITS, typename U>
inline float load16(const device U* x, thread float* xs) {
    float s = 0.0f;
    for (int i = 0; i < 16; i += 4) {
        float a = x[i], b = x[i + 1], c = x[i + 2], d = x[i + 3];
        s += a + b + c + d;
        xs[i] = a * prescale<BITS>(i);
        xs[i + 1] = b * prescale<BITS>(i + 1);
        xs[i + 2] = c * prescale<BITS>(i + 2);
        xs[i + 3] = d * prescale<BITS>(i + 3);
    }
    return s;
}

template <int ACT>
inline float glu_act(float g) {
    if (ACT == 1) {
        return 0.5f * g * (1.0f + metal::precise::tanh(0.7978845608028654f * (g + 0.044715f * g * g * g)));
    }
    return g / (1.0f + metal::exp(-g));
}

template <int BITS>
inline float qdot16(const device uint8_t* w, thread const float* xs) {
    float a = 0.0f;
    if (BITS == 2) {
        for (int i = 0; i < 4; i++) {
            uint8_t q = w[i];
            a += xs[4 * i] * (q & 0x03) + xs[4 * i + 1] * (q & 0x0c) +
                xs[4 * i + 2] * (q & 0x30) + xs[4 * i + 3] * (q & 0xc0);
        }
    } else if (BITS == 3) {
        // Two packs of 8 values in 3 bytes; values 2 and 5 straddle two bytes.
        for (int p = 0; p < 2; p++) {
            auto q = w + 3 * p;
            auto v = xs + 8 * p;
            a += (q[0] & 0x07) * v[0] + (q[0] & 0x38) * v[1] + (q[0] & 0xc0) * v[2] +
                (q[1] & 0x01) * (v[2] * 256.0f) + (q[1] & 0x0e) * v[3] + (q[1] & 0x70) * v[4] +
                (q[1] & 0x80) * v[5] + (q[2] & 0x03) * (v[5] * 256.0f) + (q[2] & 0x1c) * v[6] +
                (q[2] & 0xe0) * v[7];
        }
    } else if (BITS == 5) {
        // Two packs of 8 values in 5 bytes.
        for (int p = 0; p < 2; p++) {
            auto q = w + 5 * p;
            auto v = xs + 8 * p;
            a += (q[0] & 0x1f) * v[0] + (q[0] & 0xe0) * v[1] + (q[1] & 0x03) * (v[1] * 256.0f) +
                (q[1] & 0x7c) * v[2] + (q[1] & 0x80) * v[3] + (q[2] & 0x0f) * (v[3] * 256.0f) +
                (q[2] & 0xf0) * v[4] + (q[3] & 0x01) * (v[4] * 256.0f) + (q[3] & 0x3e) * v[5] +
                (q[3] & 0xc0) * v[6] + (q[4] & 0x07) * (v[6] * 256.0f) + (q[4] & 0xf8) * v[7];
        }
    } else if (BITS == 6) {
        // Four packs of 4 values in 3 bytes.
        for (int p = 0; p < 4; p++) {
            auto q = w + 3 * p;
            auto v = xs + 4 * p;
            a += (q[0] & 0x3f) * v[0] + (q[0] & 0xc0) * v[1] + (q[1] & 0x0f) * (v[1] * 256.0f) +
                (q[1] & 0xf0) * v[2] + (q[2] & 0x03) * (v[2] * 256.0f) + (q[2] & 0xfc) * v[3];
        }
    } else if (BITS == 4) {
        auto ws = (const device uint16_t*)w;
        for (int i = 0; i < 4; i++) {
            uint16_t q = ws[i];
            a += xs[4 * i] * float(q & 0x000f) + xs[4 * i + 1] * float(q & 0x00f0) +
                xs[4 * i + 2] * float(q & 0x0f00) + xs[4 * i + 3] * float(q & 0xf000);
        }
    } else {
        auto ws = (const device uchar4*)w;
        for (int i = 0; i < 4; i++) {
            uchar4 q = ws[i];
            a += xs[4 * i] * q.x + xs[4 * i + 1] * q.y + xs[4 * i + 2] * q.z + xs[4 * i + 3] * q.w;
        }
    }
    return a;
}
"""

# The gate and up rows of the K experts (K * N rows of D values), the GLU, times
# each expert's weight. A threadgroup takes RPS * SPT rows of one expert: RPS rows
# per simdgroup, 16 values per lane.
_GATE_UP_SOURCE = """
    constexpr int ROWS = RPS * SPT;
    constexpr int TPE = (N + ROWS - 1) / ROWS;
    constexpr int RB = D * BITS / 8;
    constexpr int NG = D / GS;
    uint lane = thread_index_in_simdgroup;
    uint slot = threadgroup_position_in_grid.x / TPE;
    int row0 = (threadgroup_position_in_grid.x % TPE) * ROWS + simdgroup_index_in_threadgroup * RPS;
    if (row0 >= N) {
        return;
    }
    size_t base = size_t(inds[slot]) * N + row0;
    auto gq = (const device uint8_t*)gw;
    auto uq = (const device uint8_t*)uw;
    float ga[RPS] = {0.0f};
    float ua[RPS] = {0.0f};
    float xs[16];
    for (int k0 = lane * 16; k0 < D; k0 += 512) {
        float xsum = load16<BITS>(x + k0, xs);
        for (int r = 0; r < RPS; r++) {
            size_t row = base + r;
            size_t g = row * NG + k0 / GS;
            ga[r] += float(gs[g]) * qdot16<BITS>(gq + row * RB + k0 * BITS / 8, xs) + float(gb[g]) * xsum;
            ua[r] += float(us[g]) * qdot16<BITS>(uq + row * RB + k0 * BITS / 8, xs) + float(ub[g]) * xsum;
        }
    }
    float we = weights[slot];
    for (int r = 0; r < RPS; r++) {
        float g = simd_sum(ga[r]);
        float u = simd_sum(ua[r]);
        if (lane == 0) {
            h[slot * N + row0 + r] = static_cast<T>(glu_act<ACT>(g) * u * we);
        }
    }
"""

# The down rows of the K experts side by side are one row of K * N values, and h
# (K x N) the matching vector: the weighted sum over the experts is one dot
# product per output row. A lane's 16 values are of one expert (N % 16 == 0).
_DOWN_SOURCE = """
    constexpr int RB = N * BITS / 8;
    constexpr int NG = N / GS;
    uint lane = thread_index_in_simdgroup;
    int row0 = (threadgroup_position_in_grid.x * SPT + simdgroup_index_in_threadgroup) * RPS;
    if (row0 >= D) {
        return;
    }
    auto dq = (const device uint8_t*)dw;
    float acc[RPS] = {0.0f};
    float xs[16];
    for (int j0 = lane * 16; j0 < K * N; j0 += 512) {
        int slot = j0 / N;
        int j = j0 - slot * N;
        float xsum = load16<BITS>(h + j0, xs);
        size_t base = size_t(inds[slot]) * D + row0;
        for (int r = 0; r < RPS; r++) {
            size_t row = base + r;
            size_t g = row * NG + j / GS;
            acc[r] += float(ds[g]) * qdot16<BITS>(dq + row * RB + j * BITS / 8, xs) + float(db[g]) * xsum;
        }
    }
    for (int r = 0; r < RPS; r++) {
        float a = simd_sum(acc[r]);
        if (lane == 0) {
            y[row0 + r] = static_cast<O>(a);
        }
    }
"""

_gate_up_kernel = mx.fast.metal_kernel(
    name="switch_glu_gate_up",
    input_names=["x", "inds", "weights", "gw", "gs", "gb", "uw", "us", "ub"],
    output_names=["h"],
    header=_HEADER,
    source=_GATE_UP_SOURCE,
)

_down_kernel = mx.fast.metal_kernel(
    name="switch_glu_down_combine",
    input_names=["h", "inds", "dw", "ds", "db"],
    output_names=["y"],
    header=_HEADER,
    source=_DOWN_SOURCE,
)

# Rows per simdgroup and simdgroups per threadgroup, the fastest of those tried
# with the experts of ten MoE models (D from 1536 to 2816, N from 512 to 1792).
_GATE_UP_SHAPE = (1, 2)
_DOWN_SHAPE = (1, 4)


def can_combine(switch_glu, x: mx.array) -> bool:
    """Whether ``switch_glu_combine`` takes ``x``: one token, bf16 or fp16, on Metal,
    not training, and a SwitchGLU whose activation has a ``metal_act`` (SwiGLU, GeGLU)
    and whose experts are affine 2- to 8-bit (group size 32, 64 or 128, scales in the
    dtype of ``x``, no bias)."""
    projs = (switch_glu.gate_proj, switch_glu.up_proj, switch_glu.down_proj)
    if not (
        mx.metal.is_available()
        and mx.default_device() == mx.gpu
        and x.size == x.shape[-1]
        and x.dtype in (mx.bfloat16, mx.float16)
        and not switch_glu.training
        and getattr(switch_glu.activation, "metal_act", None) in (0, 1)
    ):
        return False
    bits, group_size = getattr(projs[0], "bits", None), getattr(
        projs[0], "group_size", None
    )
    return (
        bits in (2, 3, 4, 5, 6, 8)
        and group_size in (32, 64, 128)
        and all(
            getattr(p, "mode", None) == "affine"
            and getattr(p, "bits", None) == bits
            and getattr(p, "group_size", None) == group_size
            and p.weight.ndim == 3
            and p.scales.dtype == x.dtype
            and p.get("biases") is not None
            and "bias" not in p
            and getattr(p, "global_scale", None) is None
            for p in projs
        )
        and x.shape[-1] % 16 == 0
        and switch_glu.gate_proj.weight.shape[1] % 16 == 0
    )


@lru_cache
def _combine_fn(act: int, group_size: int, bits: int):
    """The two kernels for one configuration, compiled: shapes come from the inputs, so a
    model's layers share one compiled function, and a call costs less on the host."""

    def combine(x, inds, weights, gw, gs, gb, uw, us, ub, dw, ds, db):
        K, N, D = inds.size, gw.shape[1], x.shape[-1]
        inds = inds.reshape(-1).astype(mx.uint32)
        weights = weights.reshape(-1)
        out_dtype = mx.result_type(x.dtype, weights.dtype)
        template = [
            ("T", x.dtype),
            ("O", out_dtype),
            ("ACT", act),
            ("D", D),
            ("N", N),
            ("K", K),
            ("GS", group_size),
            ("BITS", bits),
        ]
        rps, spt = _GATE_UP_SHAPE
        tpe = -(-N // (rps * spt))
        h = _gate_up_kernel(
            inputs=[x, inds, weights, gw, gs, gb, uw, us, ub],
            template=template + [("RPS", rps), ("SPT", spt)],
            grid=(K * tpe * 32 * spt, 1, 1),
            threadgroup=(32 * spt, 1, 1),
            output_shapes=[(K, N)],
            output_dtypes=[x.dtype],
        )[0]
        rps, spt = _DOWN_SHAPE
        return _down_kernel(
            inputs=[h, inds, dw, ds, db],
            template=template + [("RPS", rps), ("SPT", spt)],
            grid=(-(-D // (rps * spt)) * 32 * spt, 1, 1),
            threadgroup=(32 * spt, 1, 1),
            output_shapes=[x.shape],
            output_dtypes=[out_dtype],
        )[0]

    return mx.compile(combine)


def switch_glu_combine(
    switch_glu, x: mx.array, inds: mx.array, weights: mx.array
) -> mx.array:
    """The sum of ``switch_glu``'s experts ``inds`` on the one token ``x``, weighted by
    ``weights``: ``(switch_glu(x, inds) * weights[..., None]).sum(-2)`` and of its
    dtype, in two kernels. Needs ``can_combine(switch_glu, x)``."""
    gp, up, dp = switch_glu.gate_proj, switch_glu.up_proj, switch_glu.down_proj
    fn = _combine_fn(switch_glu.activation.metal_act, gp.group_size, gp.bits)
    return fn(
        x,
        inds,
        weights,
        gp.weight,
        gp.scales,
        gp.biases,
        up.weight,
        up.scales,
        up.biases,
        dp.weight,
        dp.scales,
        dp.biases,
    )
