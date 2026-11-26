# Copyright © 2026 Apple Inc.

"""Fused sparse prefill attention: DSA (deepseek_v32, glm_moe_dsa), QSA (qwen4_exp).

``indexed_attention`` computes, for every query, attention over its own selected key rows,
without a gathered ``[L, topk, ...]`` copy. Two layouts use it:

- MLA: the absorbed latent space (keys are latent + rope, values are the latent), as
  ``layers.attention.sparse_attention`` does. All heads read the same cache rows.
- GQA: separate key and value rows and no rope part, as the gather path of ``qwen4_exp``
  does. The heads of a kv head read that kv head's rows.

The kernel:

- One threadgroup per (query, group of at most 16 heads that read one kv head) with WN
  simdgroups. The heads are the rows of the Q.K^T and P.V tiles, so each selected key row
  is read once per group.
- Keys are processed in blocks of 32 * WN selected indices. For Q.K^T each simdgroup scores
  its own 32 keys over all d key dims and the rope part (if any), reading the rows by index
  straight from the key and rope caches. The simdgroups agree on the row max and row sum
  through threadgroup memory (online softmax, float32) and share P; for P.V each simdgroup
  accumulates d / WN value columns over all keys of the block.
- Only the final ``[heads, L, d]`` output is written.

Two variants share this kernel and differ only in their matrix fragments (``SmlaFrag``):

- ``tensor_ops``: Metal 4 tensor ops (``matmul2d`` 16x32x16 on one simdgroup, 16x16
  fragments as in mlx's NAX kernels). GPU architecture generation 17 (M5) and later (18
  for p GPUs), on macOS 26.2 or later.
- ``simdgroup_matrix``: 8x8 ``simdgroup_matrix`` products, available on every Apple GPU
  since M1.

Both accumulate in float32.
"""

import platform
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import mlx.core as mx

__all__ = [
    "SIMDGROUP_MATRIX",
    "TENSOR_OPS",
    "VARIANTS",
    "Variant",
    "default_variant",
    "indexed_attention",
    "is_available",
    "supports",
]

# Shared by both variants. A variant defines SmlaFrag: the fragment size and lane layout,
# which rows and columns of the cache a lane loads for Q.K^T, and the matrix products.
_HEADER = r"""
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

#define SMLA_UNROLL _Pragma("clang loop unroll(full)")
#define SMLA_CONST static constant constexpr const

// Lane layout of the fragments: a lane holds rows fm + 8 * i (i < F::kRows) and columns fn + j (j < F::kCols)
// of an F::kSize x F::kSize fragment, as value i * F::kCols + j.
template <typename F>
METAL_FUNC short2 smla_coord(ushort lane) {
  const short qid = lane >> 2;
  const short fm = (qid & 4) | ((lane >> 1) & 3);
  const short fn = ((qid & 2) | (lane & 1)) * F::kCols;
  return short2{fn, fm};
}

// The lane's part of a 16 x COLS float32 tile (the scores s or the output o) whose rows are
// the threadgroup's heads (G used, the others padding). Each lane holds 2 rows and
// COLS / 4 columns of it:
// lane row r < 2 is head fm + 8 * r and lane column c < kCols is value c % F::kCols of
// fragment column c / F::kCols.
template <typename F, short COLS>
struct SmlaTile {
  SMLA_CONST short kM = 16 / F::kSize;  // fragments along the heads
  SMLA_CONST short kN = COLS / F::kSize;  // fragments along the columns
  SMLA_CONST short kCols = COLS / 4;  // columns per lane

  typename F::template frag<float> f[kM][kN];

  // The lane's value at its row r and column c: finds the fragment and the element within it.
  METAL_FUNC thread float& at(short r, short c) {
    thread float* v = reinterpret_cast<thread float*>(&f[r / F::kRows][c / F::kCols]);
    return v[(r % F::kRows) * F::kCols + c % F::kCols];
  }

  // The tile column (pick or latent column) that the lane's column c corresponds to.
  METAL_FUNC static short col(short c, short fn) {
    return (c / F::kCols) * F::kSize + fn + c % F::kCols;
  }
};

// One step of Q.K^T, dims c .. c + kStep: s[16 x 32] += q[16 x kStep] * k^T[kStep x 32].
//   s: the score tile, 16 heads x the simdgroup's 32 picks.
//   q: the lane's 2 query rows (heads fm, fm + 8), offset by F::q_offset to its first dim.
//   k: the lane's F::kKeys pick rows of the cache, offset by F::k_offset to its first dim.
//   c: the first dim of the step, the same for all lanes.
// Each lane loads only its own values; F::score puts them into fragments.
template <typename F, typename T>
METAL_FUNC void smla_score_step(
    thread SmlaTile<F, 32>& s,
    const thread metal::array<const device T*, 2>& q,
    const thread metal::array<const device T*, F::kKeys>& k,
    int c) {
  vec<T, F::kQVals> qv[2];
  vec<T, F::kKVals> kv[F::kKeys];
  SMLA_UNROLL for (short i = 0; i < 2; i++) {
    SMLA_UNROLL for (short j = 0; j < F::kQVals; j++) qv[i][j] = q[i][c + j];
  }
  SMLA_UNROLL for (short n = 0; n < F::kKeys; n++) {
    SMLA_UNROLL for (short j = 0; j < F::kKVals; j++) kv[n][j] = k[n][c + j];
  }
  F::score(s, qv, kv);
}
"""

# Metal 4 tensor ops: one 16x32x16 matmul2d per step.
_TENSOR_OPS = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

struct SmlaFrag {
  // 16x16 fragments: a lane holds rows fm, fm + 8 and columns fn .. fn + 3, 8 values.
  SMLA_CONST short kSize = 16;
  SMLA_CONST short kRows = 2;
  SMLA_CONST short kCols = 4;
  template <typename U>
  using frag = vec<U, kRows * kCols>;

  // Q.K^T in steps of 16 dims. The lane loads dims fn .. fn + 3 of its 2 query rows and of
  // 4 picks: rows fm and fm + 8 of both 16-pick fragments.
  SMLA_CONST short kStep = 16;
  SMLA_CONST short kKeys = 4;  // picks per lane
  SMLA_CONST short kQVals = 4;  // dims per query row
  SMLA_CONST short kKVals = 4;  // dims per pick
  // Score column (of the simdgroup's 32 picks) of the lane's pick n.
  METAL_FUNC static short key(short n, short fn, short fm) {
    return (n / 2) * 16 + fm + 8 * (n % 2);
  }
  METAL_FUNC static short q_offset(short fn) { return fn; }
  METAL_FUNC static short k_offset(short fn, short fm) { return fn; }

  // Latent column (in the simdgroup's slice) of lane column c of the output tile.
  METAL_FUNC static short out_col(short c, short fn) {
    return SmlaTile<SmlaFrag, 32>::col(c, fn);
  }

  // c[16 x 32] += a[16 x 16] * b[16 x 32]: one matmul2d by the whole simdgroup, each lane
  // passing its 8 values of every fragment. b and c come as two 16x16 fragments along the
  // columns (b0, b1 and c0, c1).
  template <bool TB, typename AT, typename BT>
  METAL_FUNC static void mma(
      thread frag<float>& c0,
      thread frag<float>& c1,
      const thread frag<AT>& a,
      const thread frag<BT>& b0,
      const thread frag<BT>& b1) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, false, TB, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ca = op.template get_left_input_cooperative_tensor<AT, BT, float>();
    auto cb = op.template get_right_input_cooperative_tensor<AT, BT, float>();
    auto cc = op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(ca)>,
        metal::remove_addrspace_t<decltype(cb)>,
        float>();
    SMLA_UNROLL for (short i = 0; i < 8; i++) {
      ca[i] = a[i];
      cb[i] = b0[i];
      cb[8 + i] = b1[i];
      cc[i] = c0[i];
      cc[8 + i] = c1[i];
    }
    op.run(ca, cb, cc);
    SMLA_UNROLL for (short i = 0; i < 8; i++) {
      c0[i] = cc[i];
      c1[i] = cc[8 + i];
    }
  }

  // s[16 x 32] += q[16 x 16] * k^T[16 x 32]: one product.
  //   s: the score tile, 16 heads x the simdgroup's 32 picks.
  //   q: dims fn .. fn + 3 of the lane's 2 query rows (heads fm, fm + 8).
  //   k: the same 4 dims of the lane's 4 picks (rows fm, fm + 8 of both pick fragments).
  template <typename T>
  METAL_FUNC static void score(
      thread SmlaTile<SmlaFrag, 32>& s,
      const thread vec<T, 4> (&q)[2],
      const thread vec<T, 4> (&k)[4]) {
    // Fragments: a is all 16 heads; b0, b1 are picks 0-15 and 16-31, as rows of k.
    frag<T> a, b0, b1;
    SMLA_UNROLL for (short i = 0; i < 2; i++) {
      SMLA_UNROLL for (short j = 0; j < 4; j++) {
        a[i * 4 + j] = q[i][j];
        b0[i * 4 + j] = k[i][j];
        b1[i * 4 + j] = k[2 + i][j];
      }
    }
    mma<true>(s.f[0][0], s.f[0][1], a, b0, b1);
  }

  // o[16 x 32] += p[16 x 16] * v[16 x 32]: 32 latent columns from 16 picks, one product.
  //   o:  the output tile, 16 heads x the simdgroup's latent columns; updates its fragment
  //       columns n0, n0 + 1.
  //   p:  the probabilities of the 16 heads for the 16 picks, one fragment.
  //   v:  the 16 picks' rows of the latent cache, 32 columns as 2 fragments.
  template <typename T, short N>
  METAL_FUNC static void accumulate(
      thread frag<float> (&o)[1][N],
      short n0,
      const thread frag<float> (&p)[1],
      const thread frag<T> (&v)[2]) {
    mma<false>(o[0][n0], o[0][n0 + 1], p[0], v[0], v[1]);
  }
};
"""

# simdgroup_matrix: 8x8x8 products.
_SIMDGROUP_MATRIX = r"""
struct SmlaFrag {
  // 8x8 fragments: a lane holds row fm and columns fn, fn + 1, 2 values.
  SMLA_CONST short kSize = 8;
  SMLA_CONST short kRows = 1;
  SMLA_CONST short kCols = 2;
  template <typename U>
  using frag = vec<U, kRows * kCols>;

  // Q.K^T in steps of 8 dims. The lane loads dims fn, fn + 1 of its 2 query rows (its
  // values of Q) and dim fm of 8 picks, which are its score columns (its values of K^T:
  // row fm, columns fn, fn + 1 of each 8-pick fragment).
  SMLA_CONST short kStep = 8;
  SMLA_CONST short kKeys = 8;  // picks per lane
  SMLA_CONST short kQVals = 2;  // dims per query row
  SMLA_CONST short kKVals = 1;  // dims per pick
  // Score column (of the simdgroup's 32 picks) of the lane's pick n.
  METAL_FUNC static short key(short n, short fn, short fm) {
    return SmlaTile<SmlaFrag, 32>::col(n, fn);
  }
  METAL_FUNC static short q_offset(short fn) { return fn; }
  METAL_FUNC static short k_offset(short fn, short fm) { return fm; }

  // Latent column (in the simdgroup's slice) of lane column c of the output tile.
  METAL_FUNC static short out_col(short c, short fn) {
    return SmlaTile<SmlaFrag, 32>::col(c, fn);
  }

  // c[8 x 8] += a[8 x 8] * b[8 x 8]: one simdgroup_matrix product by the whole simdgroup,
  // each lane passing its 2 values of a, b and c.
  METAL_FUNC static void mma(
      thread frag<float>& c,
      const thread frag<float>& a,
      const thread frag<float>& b) {
    simdgroup_matrix<float, 8, 8> ma, mb, mc;
    reinterpret_cast<thread frag<float>&>(ma.thread_elements()) = a;
    reinterpret_cast<thread frag<float>&>(mb.thread_elements()) = b;
    reinterpret_cast<thread frag<float>&>(mc.thread_elements()) = c;
    simdgroup_multiply_accumulate(mc, ma, mb, mc);
    c = reinterpret_cast<thread frag<float>&>(mc.thread_elements());
  }

  // s[16 x 32] += q[16 x 8] * k^T[8 x 32]: 2 x 4 products.
  //   s: the score tile, 16 heads x the simdgroup's 32 picks.
  //   q: dims fn, fn + 1 of the lane's 2 query rows (heads fm, fm + 8).
  //   k: dim fm of the lane's 8 picks.
  template <typename T>
  METAL_FUNC static void score(
      thread SmlaTile<SmlaFrag, 32>& s,
      const thread vec<T, 2> (&q)[2],
      const thread vec<T, 1> (&k)[8]) {
    // Fragments: a[i] is heads 8i .. 8i + 7, b[n] picks 8n .. 8n + 7 (two per lane).
    frag<float> a[2], b[4];
    SMLA_UNROLL for (short i = 0; i < 2; i++) a[i] = frag<float>(q[i]);
    SMLA_UNROLL for (short n = 0; n < 4; n++) {
      b[n] = frag<float>(float(k[2 * n][0]), float(k[2 * n + 1][0]));
    }
    // One product per (head fragment i, pick fragment n).
    SMLA_UNROLL for (short i = 0; i < 2; i++) {
      SMLA_UNROLL for (short n = 0; n < 4; n++) mma(s.f[i][n], a[i], b[n]);
    }
  }

  // o[16 x 32] += p[16 x 8] * v[8 x 32]: 32 latent columns from 8 picks, 2 x 4 products.
  //   o:  the output tile, 16 heads x the simdgroup's latent columns; updates its fragment
  //       columns n0 .. n0 + 3.
  //   p:  the probabilities of the 16 heads for the 8 picks, one fragment per 8 heads.
  //   v:  the 8 picks' rows of the latent cache, 32 columns as 4 fragments.
  template <typename T, short N>
  METAL_FUNC static void accumulate(
      thread frag<float> (&o)[2][N],
      short n0,
      const thread frag<float> (&p)[2],
      const thread frag<T> (&v)[4]) {
    // One product per (latent fragment n, head fragment i).
    SMLA_UNROLL for (short n = 0; n < 4; n++) {
      const frag<float> b = frag<float>(v[n]);
      SMLA_UNROLL for (short i = 0; i < 2; i++) mma(o[i][n0 + n], p[i], b);
    }
  }
};
"""

# Inputs: q [H, L, R], q_pe [H, L, RP], kv [KVH * N, R], k_pe [KVH * N, RP], v [KVH * N, R],
# idx [L, TOPK] (entries outside [0, N) are skipped), scale [1]. Output: out [H, L, R].
# Each group of G heads (the tile rows; rows G .. 15 are not used) reads one kv head of N
# rows. The values are the rows of v if SEPV, else the rows of kv (MLA). RP is 0 without a
# rope part. The latent part of Q.K^T is unrolled by QK_UNROLL (full unrolling hoists all its
# loads and spills), the P.V key loop by PV_UNROLL.
_SOURCE = r"""
  using F = SmlaFrag;
  using ScoreTile = SmlaTile<F, 32>;
  using OutTile = SmlaTile<F, R / WN>;
  constexpr int LATW = R / WN;
  constexpr int BLK = 32 * WN;
  constexpr short FE = F::kSize * F::kSize;
  constexpr short FV = FE / 32;
  constexpr short TM = ScoreTile::kM;

  const ushort lane = thread_index_in_simdgroup;
  const short g = simdgroup_index_in_threadgroup;
  const int L = q_shape[1];
  const uint N = uint(kv_shape[0]) / KVH;
  const int nhg = q_shape[0] / G;
  const int tgid = threadgroup_position_in_grid.y;
  const int qi = tgid / nhg;
  const int h0 = (tgid % nhg) * G;
  const short2 sc = smla_coord<F>(lane);
  const short fn = sc.x, fm = sc.y;
  // The first row of the group's kv head.
  const long kvh = long((tgid % nhg) * KVH / nhg) * N;
  const device T* vh = (SEPV ? v : kv) + kvh * R;

  metal::array<const device T*, 2> qrow, prow;
  SMLA_UNROLL for (short i = 0; i < 2; i++) {
    const short hr = fm + 8 * i;
    const long row = (long)(h0 + (hr < G ? hr : 0)) * L + qi;
    qrow[i] = q + row * R + F::q_offset(fn);
    prow[i] = q_pe + row * RP + F::q_offset(fn);
  }
  // Online softmax state for the lane's 2 heads, carried over all blocks: the unnormalized
  // output o (each block adds its picks' P.V, after rescaling o to the new row max), the
  // row max m (of the scores in log2 units) and the row sum l. Set once, here.
  OutTile o;
  SMLA_UNROLL for (short i = 0; i < TM; i++) {
    SMLA_UNROLL for (short n = 0; n < OutTile::kN; n++) o.f[i][n] = 0;
  }
  float2 m = float2(-FLT_MAX);
  float2 l = 0;
  threadgroup float pbuf[BLK * 16];
  threadgroup float2 mbuf[WN * 32];
  threadgroup float2 lbuf[WN * 32];
  const device int* ip = idx + (long)qi * TOPK;
  const float sl2 = scale[0] * M_LOG2E_F;

  for (int kb = 0; kb < TOPK; kb += BLK) {
    const int k0 = kb + 32 * g;
    metal::array<const device T*, F::kKeys> kr, kpr;
    SMLA_UNROLL for (short n = 0; n < F::kKeys; n++) {
      const int r = ip[k0 + F::key(n, fn, fm)];
      const long row = kvh + (uint(r) < N ? r : 0);
      kr[n] = kv + row * R + F::k_offset(fn, fm);
      kpr[n] = k_pe + row * RP + F::k_offset(fn, fm);
    }
    // s holds only this block's scores, so it starts at zero in every block (the steps
    // below add into it, one step of dims at a time).
    ScoreTile s;
    SMLA_UNROLL for (short i = 0; i < TM; i++) {
      SMLA_UNROLL for (short n = 0; n < ScoreTile::kN; n++) s.f[i][n] = 0;
    }
    QK_UNROLL for (int c = 0; c < R; c += F::kStep) {
      smla_score_step<F, T>(s, qrow, kr, c);
    }
    SMLA_UNROLL for (int c = 0; c < RP; c += F::kStep) {
      smla_score_step<F, T>(s, prow, kpr, c);
    }

    bool valid[ScoreTile::kCols];
    SMLA_UNROLL for (short c = 0; c < ScoreTile::kCols; c++) {
      valid[c] = uint(ip[k0 + ScoreTile::col(c, fn)]) < N;
    }
    float2 rmax = float2(-FLT_MAX);
    SMLA_UNROLL for (short r = 0; r < 2; r++) {
      SMLA_UNROLL for (short c = 0; c < ScoreTile::kCols; c++) {
        s.at(r, c) = valid[c] ? s.at(r, c) * sl2 : -FLT_MAX;
        rmax[r] = max(rmax[r], s.at(r, c));
      }
      rmax[r] = max(rmax[r], simd_shuffle_xor(rmax[r], ushort(1)));
      rmax[r] = max(rmax[r], simd_shuffle_xor(rmax[r], ushort(8)));
    }
    mbuf[g * 32 + lane] = rmax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float2 mn = m;
    SMLA_UNROLL for (short w = 0; w < WN; w++) mn = max(mn, mbuf[w * 32 + lane]);
    const float2 fac = float2(fast::exp2(m[0] - mn[0]), fast::exp2(m[1] - mn[1]));
    m = mn;
    float2 rs = 0;
    SMLA_UNROLL for (short r = 0; r < 2; r++) {
      SMLA_UNROLL for (short c = 0; c < ScoreTile::kCols; c++) {
        const float p = valid[c] ? fast::exp2(s.at(r, c) - mn[r]) : 0.f;
        s.at(r, c) = p;
        rs[r] += p;
      }
      rs[r] += simd_shuffle_xor(rs[r], ushort(1));
      rs[r] += simd_shuffle_xor(rs[r], ushort(8));
    }
    SMLA_UNROLL for (short n = 0; n < ScoreTile::kN; n++) {
      SMLA_UNROLL for (short i = 0; i < TM; i++) {
        threadgroup float* pw = pbuf + ((g * ScoreTile::kN + n) * TM + i) * FE + lane * FV;
        SMLA_UNROLL for (short e = 0; e < FV; e++) pw[e] = s.f[i][n][e];
      }
    }
    lbuf[g * 32 + lane] = rs;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float2 rsum = 0;
    SMLA_UNROLL for (short w = 0; w < WN; w++) rsum += lbuf[w * 32 + lane];
    l = l * fac + rsum;
    SMLA_UNROLL for (short r = 0; r < 2; r++) {
      SMLA_UNROLL for (short c = 0; c < OutTile::kCols; c++) o.at(r, c) *= fac[r];
    }

    PV_UNROLL for (short t = 0; t < BLK / F::kSize; t++) {
      typename F::template frag<float> p[TM];
      SMLA_UNROLL for (short i = 0; i < TM; i++) {
        const threadgroup float* pr = pbuf + (t * TM + i) * FE + lane * FV;
        SMLA_UNROLL for (short e = 0; e < FV; e++) p[i][e] = pr[e];
      }
      const device T* vr[F::kRows];
      SMLA_UNROLL for (short i = 0; i < F::kRows; i++) {
        const int r = ip[kb + t * F::kSize + fm + 8 * i];
        vr[i] = vh + (uint(r) < N ? long(r) : 0) * R + g * LATW;
      }
      SMLA_UNROLL for (short c0 = 0; c0 < OutTile::kCols; c0 += 8) {
        typename F::template frag<T> v[8 / F::kCols];
        SMLA_UNROLL for (short c = 0; c < 8; c++) {
          SMLA_UNROLL for (short i = 0; i < F::kRows; i++) {
            v[c / F::kCols][i * F::kCols + c % F::kCols] = vr[i][F::out_col(c0 + c, fn)];
          }
        }
        F::accumulate(o.f, c0 / F::kCols, p, v);
      }
    }
  }

  SMLA_UNROLL for (short r = 0; r < 2; r++) {
    if (fm + 8 * r >= G) {
      continue;
    }
    const float inv = l[r] > 0 ? 1.f / l[r] : 0.f;
    device T* orow = out + ((long)(h0 + fm + 8 * r) * L + qi) * R + g * LATW;
    SMLA_UNROLL for (short c = 0; c < OutTile::kCols; c++) {
      orow[F::out_col(c, fn)] = static_cast<T>(o.at(r, c) * inv);
    }
  }
"""


@dataclass(frozen=True)
class Variant:
    name: str
    header: str
    step: int  # Q.K^T dims per step; the rope width must be a multiple
    qk_unroll: int
    pv_unroll: int


# Loop unroll counts tuned at GLM-5.3 sizes (tensor_ops on M5 Ultra, simdgroup_matrix on
# M3 Ultra).
TENSOR_OPS = Variant("tensor_ops", _TENSOR_OPS, 16, qk_unroll=4, pv_unroll=2)
SIMDGROUP_MATRIX = Variant(
    "simdgroup_matrix", _SIMDGROUP_MATRIX, 8, qk_unroll=2, pv_unroll=1
)
VARIANTS = {v.name: v for v in (TENSOR_OPS, SIMDGROUP_MATRIX)}


@lru_cache(maxsize=None)
def _kernel(variant: Variant, separate_values: bool):
    pragma = '_Pragma("clang loop unroll_count({})")'
    source = _SOURCE.replace("QK_UNROLL", pragma.format(variant.qk_unroll)).replace(
        "PV_UNROLL", pragma.format(variant.pv_unroll)
    )
    return mx.fast.metal_kernel(
        name=f"{'sparse_gqa' if separate_values else 'dsa_sparse_mla'}_{variant.name}"
        f"_u{variant.qk_unroll}_{variant.pv_unroll}",
        input_names=["q", "q_pe", "kv", "k_pe", "v", "idx", "scale"],
        output_names=["out"],
        header=_HEADER + variant.header,
        source=source,
    )


@lru_cache(maxsize=None)
def _gpu_architecture() -> str:
    # For example applegpu_g17d (M5 Ultra); empty without a Metal GPU.
    if not mx.metal.is_available():
        return ""
    return mx.device_info(mx.gpu).get("architecture", "")


def _gpu_generation() -> int:
    # For example 17 for applegpu_g17d; 0 without a Metal GPU.
    gen = re.search(r"applegpu_g(\d+)", _gpu_architecture())
    return int(gen.group(1)) if gen else 0


@lru_cache(maxsize=None)
def _macos_version() -> tuple:
    # For example (26, 2); (0,) on other systems.
    version = platform.mac_ver()[0]
    return tuple(int(v) for v in version.split(".")) if version else (0,)


def is_available(variant: str) -> bool:
    """Whether this device can run ``variant`` (``tensor_ops``: M5 and later, macOS 26.2+)."""
    if variant == TENSOR_OPS.name:
        # The condition mlx uses for its NAX kernels (generation 18 for p GPUs).
        min_gen = 18 if _gpu_architecture().endswith("p") else 17
        return _gpu_generation() >= min_gen and _macos_version() >= (26, 2)
    return variant in VARIANTS and mx.metal.is_available()


def default_variant() -> Optional[str]:
    """The variant ``indexed_attention`` uses on this device, or ``None``."""
    for variant in (TENSOR_OPS.name, SIMDGROUP_MATRIX.name):
        if is_available(variant):
            return variant
    return None


def _simdgroups(dim: int) -> Optional[int]:
    # Each simdgroup accumulates dim / WN value columns, 32 at a time.
    for wn in (8, 4, 2):
        if dim % (32 * wn) == 0:
            return wn
    return None


def _group(heads: int, kv_heads: Optional[int]) -> Optional[int]:
    # The heads of one threadgroup (the tile rows), all of one kv head, or None: groups of
    # 16 (for k [N, d] only these, as in #2027), or for k [kv_heads, N, d] all heads of a
    # kv head when it has fewer.
    if kv_heads is None:
        group, kv_heads = 16, 1
    elif kv_heads > 0 and heads % kv_heads == 0:
        group = min(heads // kv_heads, 16)
    else:
        return None
    return group if heads > 0 and heads % (kv_heads * group) == 0 else None


def supports(
    heads: int,
    dim: int,
    *dtypes: mx.Dtype,
    kv_heads: Optional[int] = None,
    rope: int = 0,
    variant: Optional[str] = None,
) -> bool:
    """Whether ``indexed_attention`` runs for these sizes and dtypes on this device.

    ``dtypes`` are the dtypes of the inputs: all bfloat16 or all float16. ``kv_heads`` is
    ``None`` for ``k`` [N, d], else ``k.shape[0]``; ``rope`` is the width of ``q_pe`` (0
    without a rope part). ``indexed_attention`` also checks its arrays.
    """
    variant = variant or default_variant()
    return (
        variant in VARIANTS
        and is_available(variant)
        and set(dtypes) in ({mx.bfloat16}, {mx.float16})
        and _group(heads, kv_heads) is not None
        and rope % VARIANTS[variant].step == 0
        and _simdgroups(dim) is not None
        and mx.default_device() == mx.gpu
    )


def indexed_attention(
    q: mx.array,
    k: mx.array,
    v: Optional[mx.array],
    indices: mx.array,
    scale: float,
    *,
    q_pe: Optional[mx.array] = None,
    k_pe: Optional[mx.array] = None,
    variant: Optional[str] = None,
) -> Optional[mx.array]:
    """Attention of each query over its selected key rows, or ``None`` when unsupported.

    ``q`` [heads, L, d]; ``k`` [N, d] or [kv_heads, N, d] (query head h reads kv head
    h // (heads // kv_heads)); ``v`` like ``k``, or ``None`` when the values are the key rows
    (MLA latent); ``q_pe`` [heads, L, rope] and ``k_pe`` like ``k`` with width rope, or
    ``None`` for no rope part; ``indices`` [L, topk] the selected rows per query, where
    entries outside ``[0, N)`` (for example -1 for masked keys) are skipped. Returns
    [heads, L, d] in the input dtype. ``variant`` (``tensor_ops`` or ``simdgroup_matrix``)
    defaults to ``default_variant()``.
    """
    variant = variant or default_variant()
    if q.ndim != 3 or k.ndim not in (2, 3) or (q_pe is None) != (k_pe is None):
        return None
    heads, L, dim = q.shape
    kv_heads = k.shape[0] if k.ndim == 3 else None
    rope = 0 if q_pe is None else q_pe.shape[-1]
    arrays = [q, k] + [x for x in (v, q_pe, k_pe) if x is not None]
    dtypes = [x.dtype for x in arrays]
    if not supports(heads, dim, *dtypes, kv_heads=kv_heads, rope=rope, variant=variant):
        return None
    if (
        any(x.size == 0 for x in arrays)
        or k.shape[-1] != dim
        or (v is not None and v.shape != k.shape)
        or (q_pe is not None and q_pe.shape != (heads, L, rope))
        or (k_pe is not None and k_pe.shape != k.shape[:-1] + (rope,))
        or indices.ndim != 2
        or indices.shape[0] != L
        or indices.shape[-1] == 0
    ):
        return None
    group = _group(heads, kv_heads)
    wn = _simdgroups(dim)
    # The kernel does not read the unused inputs below. A one-element input becomes a
    # scalar in the kernel: use 64.
    unused = mx.zeros((64,), q.dtype)
    if q_pe is not None:
        k_pe = k_pe.reshape(-1, rope)
    else:
        q_pe = k_pe = unused
    k = k.reshape(-1, dim)
    indices = indices.astype(mx.int32)
    block = 32 * wn
    if indices.shape[-1] % block:
        pad = block - indices.shape[-1] % block
        indices = mx.concatenate([indices, mx.full((L, pad), -1, mx.int32)], axis=-1)
    return _kernel(VARIANTS[variant], v is not None)(
        inputs=[
            q,
            q_pe,
            k,
            k_pe,
            unused if v is None else v.reshape(-1, dim),
            indices,
            mx.array([scale], mx.float32),
        ],
        template=[
            ("T", q.dtype),
            ("R", dim),
            ("RP", rope),
            ("TOPK", indices.shape[-1]),
            ("WN", wn),
            ("G", group),
            ("KVH", kv_heads or 1),
            ("SEPV", v is not None),
        ],
        grid=(32 * wn, L * (heads // group), 1),
        threadgroup=(32 * wn, 1, 1),
        output_shapes=[(heads, L, dim)],
        output_dtypes=[q.dtype],
    )[0]
