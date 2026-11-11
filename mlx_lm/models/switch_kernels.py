# Copyright © 2026 Apple Inc.

"""Metal kernels for quantized SwitchGLU at small batch (decode, batched decode, short verify steps).

For a few tokens per step, the generic path reads each (token, expert) slot's weights separately. Here:
- T = 1: one simdgroup per expert row computes gate and up together, and the down projection is split over
  (slot, row block) so all k experts are in flight at once.
- T > 1: slots are grouped by expert into tiles of up to 8 and multiplied with simdgroup MMA, so each
  expert's weights are read and dequantized once per tile instead of once per slot.

Supported: affine 4-bit, group size 64, no bias, bf16/fp16 activations, SwiGLU or tanh-GELU GLU.
The output has the same per-slot shape as SwitchGLU, so model routing code is unchanged. Gradients use the
gather_qmm path (custom_function vjp), so training through SwitchGLU keeps working.
"""

from functools import partial

import mlx.core as mx
import mlx.nn as nn

HEADER = """
// GLU activation on the gate (ACT 0 = SiLU as mlx-lm swiglu, 1 = tanh-GELU as nn.gelu_approx, Gemma-4 geglu)
template <int A> inline float glu_act(float g) {
    if (A == 1) return 0.5f * g * (1.f + metal::precise::tanh(0.7978845608028654f * (g + 0.044715f * g * g * g)));
    return g / (1.f + metal::exp(-g));
}
// load 32 activations, prescale by 1/16^(i%8), return sum(x)
template <typename T>
inline float load_x32(const device T* xp, thread float* xs) {
    float s = 0.f;
    for (uint j = 0; j < 4; ++j) {
        float sc = 1.f;
        for (uint i = 0; i < 8; ++i) {
            float v = float(xp[j * 8 + i]);
            s += v;
            xs[j * 8 + i] = v * sc;
            sc *= 0.0625f;
        }
    }
    return s;
}
inline float dot_q4x32(thread const float* xs, uint4 w) {
    float acc = 0.f;
    for (uint j = 0; j < 4; ++j) {
        uint word = w[j];
        for (uint i = 0; i < 8; ++i) acc += xs[j * 8 + i] * float(word & (0xFu << (4 * i)));
    }
    return acc;
}
// two tokens share one weight decode: and + cvt once, two fma
inline void dot2_q4x32(thread const float* xa, thread const float* xb, uint4 w, thread float& da, thread float& db) {
    float a = 0.f, b = 0.f;
    for (uint j = 0; j < 4; ++j) {
        uint word = w[j];
        for (uint i = 0; i < 8; ++i) { float q = float(word & (0xFu << (4 * i))); a += xa[j * 8 + i] * q; b += xb[j * 8 + i] * q; }
    }
    da += a; db += b;
}
// A-fragment pair for MMA step s: P interleaves two words (low 16 bits: word a nibbles, high: word b).
// Raw codes plus an exact offset: (q | 0x6400) as half = 1024 + q, (q | 0x4300) as bfloat = 128 + q.
// Scale and bias are applied in fp32 per scale group (bias corrected by -OFF * scale).
template <typename M> inline vec<M, 2> ar_qo2(uint P, uint s);
template <> inline half2 ar_qo2<half>(uint P, uint s) { return as_type<half2>(((P >> (4 * s)) & 0x000F000Fu) | 0x64006400u); }
template <> inline vec<bfloat, 2> ar_qo2<bfloat>(uint P, uint s) { return as_type<vec<bfloat, 2>>(((P >> (4 * s)) & 0x000F000Fu) | 0x43004300u); }
// per-token sums of one 64-wide activation group for this lane's fragment cols (fn, fn+1):
// lane-local sum over its 8 values, then over the 8 fragment rows (lane bits 1, 2, 4).
template <typename T>
inline float2 ar_colsum(vec<T, 4> a0, vec<T, 4> a1, vec<T, 4> b0, vec<T, 4> b1) {
    float2 v = float2(0.f);
    for (uint i = 0; i < 4; ++i) { v.x += float(a0[i]) + float(a1[i]); v.y += float(b0[i]) + float(b1[i]); }
    v += simd_shuffle_xor(v, 2); v += simd_shuffle_xor(v, 4); v += simd_shuffle_xor(v, 16);
    return v;
}
// read the tile built by the map kernel: n slots (0 = no tile here), ids into sl[0..n)
inline uint ar_tile(const device uint* tl, uint p, thread uint* sl, uint VT) {
    const device uint* t = tl + p * (VT + 1);
    uint n = t[0];
    for (uint i = 0; i < VT; ++i) sl[i] = (i < n) ? t[1 + i] : 0u;
    return n;
}
"""

GEMV_GATE_UP = """
    uint lane = thread_index_in_simdgroup;
    uint row0 = (thread_position_in_grid.x / 32) * R;
    uint slot = thread_position_in_grid.y;
    if (row0 >= F) return;
    constexpr uint NW = D / 8, NG = D / GS;
    uint e = idx[slot];
    const device T* xr = x + (slot / K) * D;
    size_t r0 = (size_t)e * F + row0;
    float ag[R] = {0.f}, au[R] = {0.f};
    float xs[32];
    for (uint c = lane; c < NW / 4; c += 32) {
        float xsum = load_x32(xr + c * 32, xs);
        uint grp = (c * 32) / GS;
        for (uint r = 0; r < R; ++r) {
            size_t rr = r0 + r;
            float dg = dot_q4x32(xs, ((const device uint4*)(gq + rr * NW))[c]);
            float du = dot_q4x32(xs, ((const device uint4*)(uq + rr * NW))[c]);
            ag[r] += float(gs[rr * NG + grp]) * dg + float(gb[rr * NG + grp]) * xsum;
            au[r] += float(us[rr * NG + grp]) * du + float(ub[rr * NG + grp]) * xsum;
        }
    }
    for (uint r = 0; r < R; ++r) {
        float g = simd_sum(ag[r]), u = simd_sum(au[r]);
        if (lane == 0) h[slot * F + row0 + r] = T(glu_act<ACT>(g) * u);
    }
"""

GEMV_DOWN = """
    // threadgroup = (32, K): simdgroup j computes the rows of slot t*K+j and writes y[slot].
    constexpr uint RB = R * 32 / LPR;  // rows per simdgroup
    uint lane = thread_index_in_simdgroup;
    uint sub = lane % LPR;
    uint j = thread_position_in_threadgroup.y;
    uint rb0 = (thread_position_in_grid.x / 32) * RB;
    uint row0 = rb0 + (lane / LPR) * R;
    uint t = thread_position_in_grid.y / K;
    uint slot = t * K + j;
    bool ok = row0 < D;
    constexpr uint NW = F / 8, NG = F / GS;
    size_t r0 = (size_t)idx[slot] * D + (ok ? row0 : 0);
    float acc[R] = {0.f};
    float hs[32];
    for (uint c = sub; ok && c < NW / 4; c += LPR) {
        float hsum = load_x32(h + slot * F + c * 32, hs);
        uint grp = (c * 32) / GS;
        for (uint r = 0; r < R; ++r) {
            size_t rr = r0 + r;
            float d = dot_q4x32(hs, ((const device uint4*)(dq + rr * NW))[c]);
            acc[r] += float(ds[rr * NG + grp]) * d + float(db[rr * NG + grp]) * hsum;
        }
    }
    for (uint r = 0; r < R; ++r) {
        float a = acc[r];
        for (uint o = LPR / 2; o > 0; o >>= 1) a += simd_shuffle_xor(a, o);
        if (sub == 0 && ok) y[slot * D + row0 + r] = T(a);
    }
"""

MAP = """
    uint p = thread_position_in_grid.x;
    if (p >= TK) return;
    int e = idx[p];
    uint rank = 0;
    for (uint q = 0; q < p; ++q) rank += (idx[q] == e) ? 1u : 0u;
    device uint* t = tl + p * (VT + 1);
    if (rank % VT != 0) { t[0] = 0; return; }
    uint n = 0;
    for (uint q = p; q < TK && n < VT; ++q) if (idx[q] == e) t[1 + n++] = q;
    t[0] = n;
"""

TILE_GATE_UP = """
    uint lane = thread_index_in_simdgroup;
    uint ks = simdgroup_index_in_threadgroup;   // K split: SGK simdgroups share 8 rows
    uint row0 = threadgroup_position_in_grid.x * 8;
    uint p = thread_position_in_grid.y;
    threadgroup float red[SGK * 128];
    const device uint* tt = tl + p * (VT + 1);
    uint n = tt[0];
    if (n == 0 || row0 >= NR) return;
    size_t ex = idx[tt[1]];
    uint rw0 = row0 + ks * RS;  // small tiles: v31 per-slot GEMV, RS = 8 / SGK rows per simdgroup
    if (n == 2 && NSMALL >= 2) {   // 2-slot tile: scalar GEMV, weights decoded once for both tokens
        constexpr uint NWs = D / 8, NGs = D / GS;
        uint sa = tt[1], sb = tt[2];
        const device T* xra = x + (sa / K) * D;
        const device T* xrb = x + (sb / K) * D;
        float ga[RS] = {0.f}, gb_[RS] = {0.f}, ua[RS] = {0.f}, ub_[RS] = {0.f}, xs[32], xt[32];
        for (uint c = lane; c < NWs / 4; c += 32) {
            float xsa = load_x32(xra + c * 32, xs), xsb = load_x32(xrb + c * 32, xt);
            uint grp = (c * 32) / GS;
            for (uint r = 0; r < RS; ++r) {
                size_t q = ex * F + rw0 + r;
                float dga = 0.f, dgb = 0.f, dua = 0.f, dub = 0.f;
                dot2_q4x32(xs, xt, ((const device uint4*)(gq + q * NWs))[c], dga, dgb);
                dot2_q4x32(xs, xt, ((const device uint4*)(uq + q * NWs))[c], dua, dub);
                float sg = float(gs[q * NGs + grp]), bg = float(gb[q * NGs + grp]);
                float su = float(us[q * NGs + grp]), bu = float(ub[q * NGs + grp]);
                ga[r] += sg * dga + bg * xsa; gb_[r] += sg * dgb + bg * xsb;
                ua[r] += su * dua + bu * xsa; ub_[r] += su * dub + bu * xsb;
            }
        }
        for (uint r = 0; r < RS; ++r) {
            float g = simd_sum(ga[r]), u = simd_sum(ua[r]);
            float g2 = simd_sum(gb_[r]), u2 = simd_sum(ub_[r]);
            if (lane == 0) {
                h[sa * F + rw0 + r] = T(glu_act<ACT>(g) * u);
                h[sb * F + rw0 + r] = T(glu_act<ACT>(g2) * u2);
            }
        }
        return;
    }
    if (n == 1) {
        constexpr uint NWs = D / 8, NGs = D / GS;
        for (uint t = 0; t < n; ++t) {
            uint slot = tt[1 + t];
            const device T* xr = x + (slot / K) * D;
            float ag[RS] = {0.f}, au[RS] = {0.f}, xs[32];
            for (uint c = lane; c < NWs / 4; c += 32) {
                float xsum = load_x32(xr + c * 32, xs);
                uint grp = (c * 32) / GS;
                for (uint r = 0; r < RS; ++r) {
                    size_t q = ex * F + rw0 + r;
                    float dg = dot_q4x32(xs, ((const device uint4*)(gq + q * NWs))[c]);
                    float du = dot_q4x32(xs, ((const device uint4*)(uq + q * NWs))[c]);
                    ag[r] += float(gs[q * NGs + grp]) * dg + float(gb[q * NGs + grp]) * xsum;
                    au[r] += float(us[q * NGs + grp]) * du + float(ub[q * NGs + grp]) * xsum;
                }
            }
            for (uint r = 0; r < RS; ++r) {
                float g = simd_sum(ag[r]), u = simd_sum(au[r]);
                if (lane == 0) h[slot * F + rw0 + r] = T(glu_act<ACT>(g) * u);
            }
        }
        return;
    }

    uint qid = lane / 4;
    uint fm = (qid & 4) + ((lane / 2) % 4);   // fragment row of this lane
    uint fn = (qid & 2) * 2 + (lane % 2) * 2; // fragment cols fn, fn+1
    uint s0 = tt[1 + min(fn, n - 1)], s1 = tt[1 + min(fn + 1, n - 1)];
    size_t rr = (size_t)idx[tt[1]] * NR + row0 + fm;
    constexpr uint NW = NK / 8, NB = NK / 64;

    const device T* xa = x + (s0 / K) * D + fm * 8;
    const device T* xb = x + (s1 / K) * D + fm * 8;
    const device uint2* g2 = (const device uint2*)(gq + rr * NW) + fn / 2;
    const device uint2* u2 = (const device uint2*)(uq + rr * NW) + fn / 2;
    simdgroup_matrix<float, 8, 8> Cg = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_matrix<float, 8, 8> Cu = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_matrix<float, 8, 8> Pg, Pu;
    simdgroup_matrix<MT, 8, 8> Ag, Au, B;
    for (uint g = ks; g < NB; g += SGK) {   // one 64-wide scale group: col c, step s <-> k = 8c + s
        uint2 wg = g2[g * 4], wu = u2[g * 4];
        uint pg[2] = {(wg.x & 0xFFFFu) | (wg.y << 16), (wg.x >> 16) | (wg.y & 0xFFFF0000u)};
        uint pu[2] = {(wu.x & 0xFFFFu) | (wu.y << 16), (wu.x >> 16) | (wu.y & 0xFFFF0000u)};
        vec<T, 4> a0 = ((const device vec<T, 4>*)(xa + g * 64))[0], a1 = ((const device vec<T, 4>*)(xa + g * 64))[1];
        vec<T, 4> b0 = ((const device vec<T, 4>*)(xb + g * 64))[0], b1 = ((const device vec<T, 4>*)(xb + g * 64))[1];
        for (uint s = 0; s < 8; ++s) {
            { vec<MT, 2> v_ = ar_qo2<MT>(pg[s / 4], s % 4); Ag.thread_elements()[0] = v_.x; Ag.thread_elements()[1] = v_.y; }
            { vec<MT, 2> v_ = ar_qo2<MT>(pu[s / 4], s % 4); Au.thread_elements()[0] = v_.x; Au.thread_elements()[1] = v_.y; }
            B.thread_elements()[0] = MT(s < 4 ? a0[s] : a1[s - 4]); B.thread_elements()[1] = MT(s < 4 ? b0[s] : b1[s - 4]);
            if (s == 0) { simdgroup_multiply(Pg, Ag, B); simdgroup_multiply(Pu, Au, B); }
            else { simdgroup_multiply_accumulate(Pg, Ag, B, Pg); simdgroup_multiply_accumulate(Pu, Au, B, Pu); }
        }
        float2 sx = ar_colsum(a0, a1, b0, b1);
        uint grp = rr * NB + g;
        float sg_ = float(gs[grp]), su = float(us[grp]);
        float bg = float(gb[grp]) - OFF * sg_, bu = float(ub[grp]) - OFF * su;   // A holds q + OFF
        for (uint i = 0; i < 2; ++i) {
            Cg.thread_elements()[i] = fma(sg_, Pg.thread_elements()[i], fma(bg, sx[i], Cg.thread_elements()[i]));
            Cu.thread_elements()[i] = fma(su, Pu.thread_elements()[i], fma(bu, sx[i], Cu.thread_elements()[i]));
        }
    }
    for (uint i = 0; i < 2; ++i) {
        red[ks * 128 + lane * 2 + i] = Cg.thread_elements()[i];
        red[ks * 128 + 64 + lane * 2 + i] = Cu.thread_elements()[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ks != 0) return;
    for (uint i = 0; i < 2; ++i) {
        float g = 0.f, u = 0.f;
        for (uint j = 0; j < SGK; ++j) { g += red[j * 128 + lane * 2 + i]; u += red[j * 128 + 64 + lane * 2 + i]; }
        if (fn + i < n) h[(i ? s1 : s0) * F + row0 + fm] = T(glu_act<ACT>(g) * u);
    }
"""

TILE_DOWN = """
    uint lane = thread_index_in_simdgroup;
    uint ks = simdgroup_index_in_threadgroup;   // K split: SGK simdgroups share 8 rows
    uint row0 = threadgroup_position_in_grid.x * 8;
    uint p = thread_position_in_grid.y;
    threadgroup float red[SGK * 128];
    const device uint* tt = tl + p * (VT + 1);
    uint n = tt[0];
    if (n == 0 || row0 >= NR) return;
    size_t ex = idx[tt[1]];
    uint rw0 = row0 + ks * RS;  // small tiles: v31 per-slot GEMV, RS = 8 / SGK rows per simdgroup
    if (n == 2 && NSMALL >= 2) {   // 2-slot tile: LPR GEMV, weights decoded once for both slots
        constexpr uint NWs = F / 8, NGs = F / GS;
        constexpr uint RR = RS * LPR / 32;
        uint sa = tt[1], sb = tt[2];
        uint sub = lane % LPR, rg = rw0 + (lane / LPR) * RR;
        float aa[RR] = {0.f}, ab[RR] = {0.f}, hs[32], ht[32];
        for (uint c = sub; c < NWs / 4; c += LPR) {
            float hsa = load_x32(h + sa * F + c * 32, hs), hsb = load_x32(h + sb * F + c * 32, ht);
            uint grp = (c * 32) / GS;
            for (uint r = 0; r < RR; ++r) {
                size_t q = ex * D + rg + r;
                float da = 0.f, db_ = 0.f;
                dot2_q4x32(hs, ht, ((const device uint4*)(dq + q * NWs))[c], da, db_);
                float sc = float(ds[q * NGs + grp]), bi = float(db[q * NGs + grp]);
                aa[r] += sc * da + bi * hsa; ab[r] += sc * db_ + bi * hsb;
            }
        }
        float wa = wts[sa], wb = wts[sb];
        for (uint r = 0; r < RR; ++r) {
            float a = aa[r], b = ab[r];
            for (uint o = LPR / 2; o > 0; o >>= 1) { a += simd_shuffle_xor(a, o); b += simd_shuffle_xor(b, o); }
            if (sub == 0) { y[sa * D + rg + r] = wa * a; y[sb * D + rg + r] = wb * b; }
        }
        return;
    }
    if (n == 1) {
        constexpr uint NWs = F / 8, NGs = F / GS;
        for (uint t = 0; t < n; ++t) {
            uint slot = tt[1 + t];
            // LPR lanes per row (v4 trick): 32 / LPR row groups of RR rows, all lanes busy when F/32 < 32
            constexpr uint RR = RS * LPR / 32;
            uint sub = lane % LPR, rg = rw0 + (lane / LPR) * RR;
            float acc[RR] = {0.f}, hs[32];
            for (uint c = sub; c < NWs / 4; c += LPR) {
                float hsum = load_x32(h + slot * F + c * 32, hs);
                uint grp = (c * 32) / GS;
                for (uint r = 0; r < RR; ++r) {
                    size_t q = ex * D + rg + r;
                    float d = dot_q4x32(hs, ((const device uint4*)(dq + q * NWs))[c]);
                    acc[r] += float(ds[q * NGs + grp]) * d + float(db[q * NGs + grp]) * hsum;
                }
            }
            float w = wts[slot];
            for (uint r = 0; r < RR; ++r) {
                float a = acc[r];
                for (uint o = LPR / 2; o > 0; o >>= 1) a += simd_shuffle_xor(a, o);
                if (sub == 0) y[slot * D + rg + r] = w * a;
            }
        }
        return;
    }

    uint qid = lane / 4;
    uint fm = (qid & 4) + ((lane / 2) % 4);   // fragment row of this lane
    uint fn = (qid & 2) * 2 + (lane % 2) * 2; // fragment cols fn, fn+1
    uint s0 = tt[1 + min(fn, n - 1)], s1 = tt[1 + min(fn + 1, n - 1)];
    size_t rr = (size_t)idx[tt[1]] * NR + row0 + fm;
    constexpr uint NW = NK / 8, NB = NK / 64;

    const device T* ha = h + s0 * F + fm * 8;
    const device T* hb = h + s1 * F + fm * 8;
    const device uint2* d2 = (const device uint2*)(dq + rr * NW) + fn / 2;
    simdgroup_matrix<float, 8, 8> C = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    simdgroup_matrix<float, 8, 8> P;
    simdgroup_matrix<MT, 8, 8> A, B;
    for (uint g = ks; g < NB; g += SGK) {
        uint2 w = d2[g * 4];
        uint pw[2] = {(w.x & 0xFFFFu) | (w.y << 16), (w.x >> 16) | (w.y & 0xFFFF0000u)};
        vec<T, 4> a0 = ((const device vec<T, 4>*)(ha + g * 64))[0], a1 = ((const device vec<T, 4>*)(ha + g * 64))[1];
        vec<T, 4> b0 = ((const device vec<T, 4>*)(hb + g * 64))[0], b1 = ((const device vec<T, 4>*)(hb + g * 64))[1];
        for (uint s = 0; s < 8; ++s) {
            { vec<MT, 2> v_ = ar_qo2<MT>(pw[s / 4], s % 4); A.thread_elements()[0] = v_.x; A.thread_elements()[1] = v_.y; }
            B.thread_elements()[0] = MT(s < 4 ? a0[s] : a1[s - 4]); B.thread_elements()[1] = MT(s < 4 ? b0[s] : b1[s - 4]);
            if (s == 0) simdgroup_multiply(P, A, B);
            else simdgroup_multiply_accumulate(P, A, B, P);
        }
        float2 sx = ar_colsum(a0, a1, b0, b1);
        uint grp = rr * NB + g;
        float sc = float(ds[grp]), bi = float(db[grp]) - OFF * sc;
        for (uint i = 0; i < 2; ++i) C.thread_elements()[i] = fma(sc, P.thread_elements()[i], fma(bi, sx[i], C.thread_elements()[i]));
    }
    for (uint i = 0; i < 2; ++i) red[ks * 128 + lane * 2 + i] = C.thread_elements()[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ks != 0) return;
    for (uint i = 0; i < 2; ++i) {
        float a = 0.f;
        for (uint j = 0; j < SGK; ++j) a += red[j * 128 + lane * 2 + i];
        uint slot = i ? s1 : s0;
        if (fn + i < n) y[slot * D + row0 + fm] = wts[slot] * a;
    }
"""


R_UP, R_DOWN, SG_PER_TG = (
    1,
    2,
    4,
)  # GEMV rows per simdgroup (gate/up, down), simdgroups per threadgroup
VT, SGK_UP, NSMALL = (
    8,
    8,
    2,
)  # tile width, K-split simdgroups (gate/up), tiles <= NSMALL slots use GEMV


def _make_kernels():
    if not mx.metal.is_available():
        return None
    mk = partial(mx.fast.metal_kernel, header=HEADER)
    w = ["gq", "gs", "gb", "uq", "us", "ub"]
    return {
        "gemv_gate_up": mk(
            name="switch_glu_gemv_gate_up",
            input_names=["x", *w, "idx"],
            output_names=["h"],
            source=GEMV_GATE_UP,
        ),
        "gemv_down": mk(
            name="switch_glu_gemv_down",
            input_names=["h", "dq", "ds", "db", "idx"],
            output_names=["y"],
            source=GEMV_DOWN,
        ),
        "map": mx.fast.metal_kernel(
            name="switch_glu_tile_map",
            input_names=["idx"],
            output_names=["tl"],
            source=MAP,
        ),
        "tile_gate_up": mk(
            name="switch_glu_tile_gate_up",
            input_names=["x", *w, "idx", "tl"],
            output_names=["h"],
            source=TILE_GATE_UP,
        ),
        "tile_down": mk(
            name="switch_glu_tile_down",
            input_names=["h", "dq", "ds", "db", "idx", "wts", "tl"],
            output_names=["y"],
            source=TILE_DOWN,
        ),
    }


_kernels = _make_kernels()


def _lanes_per_row(chunks, iters=3):
    """Smallest power of 2 >= chunks / iters, at most 32."""
    lanes = 1
    while lanes < 32 and lanes * iters < chunks:
        lanes *= 2
    return lanes


def _forward(x, indices, weights, act):
    """x: [T, D], indices: [T, k] -> per-slot expert outputs [T * k, D]."""
    gq, gs, gb, uq, us, ub, dq, ds, db = weights
    T, k = indices.shape
    F, D = gq.shape[1], dq.shape[1]
    w = [gq, gs, gb, uq, us, ub]
    if T == 1:
        base = [
            ("T", x.dtype),
            ("S", gs.dtype),
            ("D", D),
            ("F", F),
            ("K", k),
            ("GS", 64),
            ("ACT", act),
        ]
        h = _kernels["gemv_gate_up"](
            inputs=[x, *w, indices],
            template=base + [("R", R_UP)],
            grid=(32 * F // R_UP, k, 1),
            threadgroup=(32 * SG_PER_TG, 1, 1),
            output_shapes=[(k, F)],
            output_dtypes=[x.dtype],
        )[0]
        lpr = _lanes_per_row(F // 32)
        rb = R_DOWN * 32 // lpr
        return _kernels["gemv_down"](
            inputs=[h, dq, ds, db, indices],
            template=base + [("R", R_DOWN), ("LPR", lpr)],
            grid=(32 * (-(-D // rb)), k, 1),
            threadgroup=(32, k, 1),
            output_shapes=[(k, D)],
            output_dtypes=[x.dtype],
        )[0]
    # T > 1: every slot is its own row (K = 1); the tile map groups rows by expert
    n = T * k
    x = mx.broadcast_to(x[:, None], (T, k, D)).reshape(n, D)
    indices = indices.reshape(n)
    bf = x.dtype == mx.bfloat16
    tm = [
        ("T", x.dtype),
        ("S", gs.dtype),
        ("D", D),
        ("F", F),
        ("K", 1),
        ("GS", 64),
        ("ACT", act),
        ("VT", VT),
        ("TK", n),
        ("MT", mx.bfloat16 if bf else mx.float16),
        ("OFF", 128 if bf else 1024),
    ]
    tl = _kernels["map"](
        inputs=[indices],
        template=[("VT", VT), ("TK", n)],
        grid=(n, 1, 1),
        threadgroup=(min(n, 256), 1, 1),
        output_shapes=[(n * (VT + 1),)],
        output_dtypes=[mx.uint32],
    )[0]
    h = _kernels["tile_gate_up"](
        inputs=[x, *w, indices, tl],
        template=tm
        + [
            ("NR", F),
            ("NK", D),
            ("SGK", SGK_UP),
            ("NSMALL", NSMALL),
            ("RS", 8 // SGK_UP),
        ],
        grid=(32 * SGK_UP * F // 8, n, 1),
        threadgroup=(32 * SGK_UP, 1, 1),
        output_shapes=[(n, F)],
        output_dtypes=[x.dtype],
    )[0]
    lpr = _lanes_per_row(F // 32)
    sgk = max(1, lpr // (4 * R_DOWN))
    lpr = max(lpr, 4 * sgk)
    y = _kernels["tile_down"](
        inputs=[h, dq, ds, db, indices, mx.ones((n,), mx.float32), tl],
        template=tm
        + [
            ("NR", D),
            ("NK", F),
            ("SGK", sgk),
            ("NSMALL", NSMALL),
            ("RS", 8 // sgk),
            ("LPR", lpr),
        ],
        grid=(32 * sgk * D // 8, n, 1),
        threadgroup=(32 * sgk, 1, 1),
        output_shapes=[(n, D)],
        output_dtypes=[mx.float32],
    )[0]
    return y.astype(x.dtype)


def _reference(x, indices, weights, act):
    """Same math through gather_qmm (used for gradients)."""
    gq, gs, gb, uq, us, ub, dq, ds, db = weights
    xx = mx.expand_dims(x, (-2, -3))
    mm = partial(
        mx.gather_qmm, rhs_indices=indices, transpose=True, group_size=64, bits=4
    )
    g, u = mm(xx, gq, gs, gb), mm(xx, uq, us, ub)
    a = nn.silu(g) if act == 0 else nn.gelu_approx(g)
    return mm(a * u, dq, ds, db).squeeze(-2).reshape(-1, x.shape[-1])


def _make_op(act):
    @mx.custom_function
    def op(x, indices, *weights):
        return _forward(x, indices, weights, act)

    @op.vjp
    def op(primals, cotangent, output):
        x, indices, *weights = primals
        _, (dx,) = mx.vjp(
            lambda x: _reference(x, indices, weights, act), [x], [cotangent]
        )
        return (dx, mx.zeros_like(indices), *(mx.zeros_like(w) for w in weights))

    return op


_ops = {0: _make_op(0), 1: _make_op(1)}


def _nax_gpu():
    """M5 and later, where gather_qmm runs the NAX kernels from 4 tokens per expert."""
    gen = mx.device_info().get("architecture", "")[-3:-1]
    return gen.isdigit() and int(gen) >= 17


_nax = _kernels is not None and _nax_gpu()


def supported_load(load, D, F):
    """Whether these kernels are used at `load` = average tokens per expert (T*k/E).

    M4 Pro: the measured crossover against gather_qmm. M5 and later: gather_qmm runs the NAX kernels
    from 4; these kernels were not measured against them, so there they only replace gather_qmv.
    """
    if _nax:
        return load < 4
    return load <= (24 if D * F >= 3 << 19 else 8)


def switch_glu(glu, x, indices):
    """Per-slot output of SwitchGLU `glu` (as SwitchGLU.__call__), or None if this path does not apply."""
    act = getattr(glu.activation, "metal_act", None)
    if _kernels is None or act is None or x.dtype not in (mx.bfloat16, mx.float16):
        return None
    projs = (glu.gate_proj, glu.up_proj, glu.down_proj)
    for p in projs:
        fmt = (
            getattr(p, "bits", None),
            getattr(p, "group_size", None),
            getattr(p, "mode", "affine"),
        )
        if fmt != (4, 64, "affine") or "bias" in p or p.scales.dtype != x.dtype:
            return None
    k = indices.shape[-1]
    T = indices.size // k
    (E, F), D = glu.gate_proj.weight.shape[:2], x.shape[-1]
    # k <= 8: the T = 1 down kernel uses a (32, k) threadgroup
    if T == 0 or k > 8 or not supported_load(T * k / E, D, F):
        return None
    weights = [a for p in projs for a in (p.weight, p.scales, p.biases)]
    y = _ops[act](x.reshape(T, D), indices.reshape(T, k).astype(mx.uint32), *weights)
    return y.reshape(*indices.shape, D)
