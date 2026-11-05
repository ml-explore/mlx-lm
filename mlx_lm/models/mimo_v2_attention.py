# Copyright © 2026 Apple Inc.

"""Fused attention for MiMo-V2's global layers: 192-wide queries/keys, 128-wide values.

``mx.fast.scaled_dot_product_attention`` has fused kernels only when queries/keys and
values share one supported head width, so MiMo-V2's global-attention layers (192/128)
take the unfused path: prefill materializes the whole heads x queries x keys score
matrix, and decode handles every query head separately, so the 16 query heads that share
a KV head each read that head's keys and values again.

* ``mimo_prefill_attention``: causal flash attention for a prefill chunk. Scores and
  probabilities stay in registers, so there is no transient beyond the output, and the
  probabilities meet the values in float32. Long contexts are split into several launches
  so no single GPU command runs for seconds.
* ``mimo_decode_attention``: split-context flash decoding for one query per sequence,
  with the query heads of a KV head as the rows of the matrix products, so each KV head's
  keys and values are read once. An optional boolean mask (a batch cache's left padding)
  is honoured per key.

Both read keys and values in place through their strides (a cache view needs no copy)
at their native widths, in bfloat16.

The fragment layout and the row-reduction lane pattern follow MLX's steel attention
kernel (``mlx/backend/metal/kernels/steel/attn``).
"""

import mlx.core as mx

KEY_WIDTH = 192
VALUE_WIDTH = 128

# Prefill tiling: SIMDGROUPS x FRAGMENTS x 8 query rows per threadgroup, KEY_BLOCK keys
# per step. At most MAX_SCORES_PER_DISPATCH heads x queries x keys per launch (~0.3 s).
_PREFILL_SIMDGROUPS = 8
_PREFILL_FRAGMENTS = 1
_KEY_BLOCK = 16
_PREFILL_QUERY_BLOCK = _PREFILL_SIMDGROUPS * _PREFILL_FRAGMENTS * 8
_MAX_SCORES_PER_DISPATCH = 1 << 33

# Decode tiling: SIMDGROUPS x FRAGMENTS x 8 must equal the query heads per KV head. The
# context is split into slices of at least MIN_SLICE keys, aiming at TARGET_THREADGROUPS
# threadgroups across (sequence, KV head, slice); a second pass combines the slices.
_DECODE_SIMDGROUPS = 2
_DECODE_FRAGMENTS = 1
DECODE_GROUP = _DECODE_SIMDGROUPS * _DECODE_FRAGMENTS * 8
_DECODE_TARGET_THREADGROUPS = 512
_DECODE_MIN_SLICE = 256

_HEADER = """
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

#define MIMO_UNROLL _Pragma("clang loop unroll(full)")

typedef simdgroup_matrix<float, 8, 8> mimo_frag;
"""

_PREFILL_SOURCE = """
  constexpr int DQK = 192;
  constexpr int DV = 128;
  constexpr int BQ = WM * TQ * 8;
  constexpr int TK = BK / 8;
  constexpr int TDQ = DQK / 8;
  constexpr int TDV = DV / 8;
  constexpr int LDK = DQK + 8;
  constexpr int LDV = DV + 8;
  constexpr int NT = WM * 32;

  threadgroup bfloat Ks[BK * LDK];
  threadgroup bfloat Vs[BK * LDV];

  const uint sg = simdgroup_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const uint tid = thread_index_in_threadgroup;
  const int q_start = int(threadgroup_position_in_grid.x) * BQ;
  const int head = int(threadgroup_position_in_grid.y);
  const int batch = int(threadgroup_position_in_grid.z);

  const int L = params[0];
  const int S = params[1];
  const int q_offset = params[2];
  const int gqa = params[3];
  const int n_heads = params[4];
  const int L_padded = params[5];
  const float log2_scale = scale[0] * 1.4426950408889634f;

  const int kv_head = head / gqa;
  const device ushort* Kg =
      (const device ushort*)k + batch * k_strides[0] + kv_head * k_strides[1];
  const device ushort* Vg =
      (const device ushort*)v + batch * v_strides[0] + kv_head * v_strides[1];
  const long k_row = k_strides[2];
  const long v_row = v_strides[2];

  const short qid = short(lane / 4);
  const short fm = (qid & 4) + short((lane / 2) % 4);
  const short fn = (qid & 2) * 2 + short(lane % 2) * 2;
  const int sg_row = q_start + int(sg) * TQ * 8;

  // This simdgroup's query rows, kept in registers (the queries are padded to L_padded rows).
  const device bfloat* Qg =
      (const device bfloat*)q + ((long(batch) * n_heads + head) * long(L_padded) + sg_row) * DQK;
  simdgroup_matrix<bfloat, 8, 8> Qf[TQ][TDQ];
  MIMO_UNROLL
  for (int iq = 0; iq < TQ; ++iq) {
    MIMO_UNROLL
    for (int dd = 0; dd < TDQ; ++dd) {
      simdgroup_load(Qf[iq][dd], Qg + iq * 8 * DQK + dd * 8, DQK);
    }
  }

  const int q_last = min(q_start + BQ, L) - 1;
  const int key_end = min(S, q_offset + q_last + 1);
  const int n_kb = (key_end + BK - 1) / BK;
  const int first_masked_kb = (q_offset + q_start) / BK;

  mimo_frag O[TQ][TDV];
  float row_max[TQ];
  float row_sum[TQ];
  MIMO_UNROLL
  for (int iq = 0; iq < TQ; ++iq) {
    row_max[iq] = -3.0e38f;
    row_sum[iq] = 0.0f;
    MIMO_UNROLL
    for (int id = 0; id < TDV; ++id) {
      O[iq][id] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
  }

  for (int kb = 0; kb < n_kb; ++kb) {
    const int k0 = kb * BK;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = int(tid); i < BK * (DQK / 8); i += NT) {
      const int kr = i / (DQK / 8);
      const int c = (i % (DQK / 8)) * 8;
      uint4 bits = uint4(0);
      if (k0 + kr < S) {
        bits = *(const device uint4*)(Kg + long(k0 + kr) * k_row + c);
      }
      *(threadgroup uint4*)(Ks + kr * LDK + c) = bits;
    }
    for (int i = int(tid); i < BK * (DV / 8); i += NT) {
      const int kr = i / (DV / 8);
      const int c = (i % (DV / 8)) * 8;
      uint4 bits = uint4(0);
      if (k0 + kr < S) {
        bits = *(const device uint4*)(Vg + long(k0 + kr) * v_row + c);
      }
      *(threadgroup uint4*)(Vs + kr * LDV + c) = bits;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    mimo_frag Sf[TQ][TK];
    MIMO_UNROLL
    for (int iq = 0; iq < TQ; ++iq) {
      MIMO_UNROLL
      for (int ik = 0; ik < TK; ++ik) {
        Sf[iq][ik] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
      }
    }
    MIMO_UNROLL
    for (int dd = 0; dd < TDQ; ++dd) {
      MIMO_UNROLL
      for (int ik = 0; ik < TK; ++ik) {
        simdgroup_matrix<bfloat, 8, 8> b;
        simdgroup_load(b, Ks + ik * 8 * LDK + dd * 8, LDK, ulong2(0, 0), true);
        MIMO_UNROLL
        for (int iq = 0; iq < TQ; ++iq) {
          simdgroup_multiply_accumulate(Sf[iq][ik], Qf[iq][dd], b, Sf[iq][ik]);
        }
      }
    }

    const bool needs_mask = kb >= first_masked_kb || k0 + BK > S;
    MIMO_UNROLL
    for (int iq = 0; iq < TQ; ++iq) {
      const int q_pos = q_offset + sg_row + iq * 8 + fm;
      float new_max = row_max[iq];
      MIMO_UNROLL
      for (int ik = 0; ik < TK; ++ik) {
        MIMO_UNROLL
        for (int e = 0; e < 2; ++e) {
          float s = Sf[iq][ik].thread_elements()[e] * log2_scale;
          if (needs_mask) {
            const int k_pos = k0 + ik * 8 + fn + e;
            if (k_pos > q_pos || k_pos >= S) {
              s = -3.0e38f;
            }
          }
          Sf[iq][ik].thread_elements()[e] = s;
          new_max = max(new_max, s);
        }
      }
      new_max = max(new_max, simd_shuffle_xor(new_max, ushort(1)));
      new_max = max(new_max, simd_shuffle_xor(new_max, ushort(8)));
      const float factor = fast::exp2(row_max[iq] - new_max);
      row_max[iq] = new_max;
      float block_sum = 0.0f;
      MIMO_UNROLL
      for (int ik = 0; ik < TK; ++ik) {
        MIMO_UNROLL
        for (int e = 0; e < 2; ++e) {
          const float p = fast::exp2(Sf[iq][ik].thread_elements()[e] - new_max);
          Sf[iq][ik].thread_elements()[e] = p;
          block_sum += p;
        }
      }
      block_sum += simd_shuffle_xor(block_sum, ushort(1));
      block_sum += simd_shuffle_xor(block_sum, ushort(8));
      row_sum[iq] = row_sum[iq] * factor + block_sum;
      MIMO_UNROLL
      for (int id = 0; id < TDV; ++id) {
        O[iq][id].thread_elements()[0] *= factor;
        O[iq][id].thread_elements()[1] *= factor;
      }
    }

    MIMO_UNROLL
    for (int ik = 0; ik < TK; ++ik) {
      MIMO_UNROLL
      for (int id = 0; id < TDV; ++id) {
        simdgroup_matrix<bfloat, 8, 8> b;
        simdgroup_load(b, Vs + ik * 8 * LDV + id * 8, LDV);
        MIMO_UNROLL
        for (int iq = 0; iq < TQ; ++iq) {
          simdgroup_multiply_accumulate(O[iq][id], Sf[iq][ik], b, O[iq][id]);
        }
      }
    }
  }

  MIMO_UNROLL
  for (int iq = 0; iq < TQ; ++iq) {
    const int row = sg_row + iq * 8 + fm;
    if (row < L) {
      const float inverse = 1.0f / row_sum[iq];
      device bfloat16_t* out_row =
          o + ((long(batch) * n_heads + head) * long(L) + row) * DV + fn;
      MIMO_UNROLL
      for (int id = 0; id < TDV; ++id) {
        out_row[id * 8] = bfloat16_t(O[iq][id].thread_elements()[0] * inverse);
        out_row[id * 8 + 1] = bfloat16_t(O[iq][id].thread_elements()[1] * inverse);
      }
    }
  }
"""

_DECODE_SOURCE = """
  constexpr int DQK = 192;
  constexpr int DV = 128;
  constexpr int G = WM * TQ * 8;
  constexpr int TK = BK / 8;
  constexpr int TDQ = DQK / 8;
  constexpr int TDV = DV / 8;
  constexpr int LDK = DQK + 8;
  constexpr int LDV = DV + 8;
  constexpr int NT = WM * 32;

  threadgroup bfloat Ks[BK * LDK];
  threadgroup bfloat Vs[BK * LDV];

  const uint sg = simdgroup_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const uint tid = thread_index_in_threadgroup;
  const int split = int(threadgroup_position_in_grid.x);
  const int kv_head = int(threadgroup_position_in_grid.y);
  const int batch = int(threadgroup_position_in_grid.z);

  const int S = params[0];
  const int slice = params[1];
  const int n_heads = params[2];
  const int n_splits = params[3];
  const int has_mask = params[4];
  const int n_kv_heads = params[5];
  const float log2_scale = scale[0] * 1.4426950408889634f;

  const int k_begin = split * slice;
  const int k_end = min(S, k_begin + slice);

  const device ushort* Kg =
      (const device ushort*)k + batch * k_strides[0] + kv_head * k_strides[1];
  const device ushort* Vg =
      (const device ushort*)v + batch * v_strides[0] + kv_head * v_strides[1];
  const long k_row = k_strides[2];
  const long v_row = v_strides[2];

  const short qid = short(lane / 4);
  const short fm = (qid & 4) + short((lane / 2) % 4);
  const short fn = (qid & 2) * 2 + short(lane % 2) * 2;
  const int sg_head = int(sg) * TQ * 8;

  const device bfloat* Qg = (const device bfloat*)q
      + (long(batch) * n_heads + long(kv_head) * G + sg_head) * DQK;
  simdgroup_matrix<bfloat, 8, 8> Qf[TQ][TDQ];
  MIMO_UNROLL
  for (int iq = 0; iq < TQ; ++iq) {
    MIMO_UNROLL
    for (int dd = 0; dd < TDQ; ++dd) {
      simdgroup_load(Qf[iq][dd], Qg + iq * 8 * DQK + dd * 8, DQK);
    }
  }

  mimo_frag O[TQ][TDV];
  float row_max[TQ];
  float row_sum[TQ];
  MIMO_UNROLL
  for (int iq = 0; iq < TQ; ++iq) {
    row_max[iq] = -1.0e30f;
    row_sum[iq] = 0.0f;
    MIMO_UNROLL
    for (int id = 0; id < TDV; ++id) {
      O[iq][id] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
  }

  for (int k0 = k_begin; k0 < k_end; k0 += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = int(tid); i < BK * (DQK / 8); i += NT) {
      const int kr = i / (DQK / 8);
      const int c = (i % (DQK / 8)) * 8;
      uint4 bits = uint4(0);
      if (k0 + kr < k_end) {
        bits = *(const device uint4*)(Kg + long(k0 + kr) * k_row + c);
      }
      *(threadgroup uint4*)(Ks + kr * LDK + c) = bits;
    }
    for (int i = int(tid); i < BK * (DV / 8); i += NT) {
      const int kr = i / (DV / 8);
      const int c = (i % (DV / 8)) * 8;
      uint4 bits = uint4(0);
      if (k0 + kr < k_end) {
        bits = *(const device uint4*)(Vg + long(k0 + kr) * v_row + c);
      }
      *(threadgroup uint4*)(Vs + kr * LDV + c) = bits;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    mimo_frag Sf[TQ][TK];
    MIMO_UNROLL
    for (int iq = 0; iq < TQ; ++iq) {
      MIMO_UNROLL
      for (int ik = 0; ik < TK; ++ik) {
        Sf[iq][ik] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
      }
    }
    MIMO_UNROLL
    for (int dd = 0; dd < TDQ; ++dd) {
      MIMO_UNROLL
      for (int ik = 0; ik < TK; ++ik) {
        simdgroup_matrix<bfloat, 8, 8> b;
        simdgroup_load(b, Ks + ik * 8 * LDK + dd * 8, LDK, ulong2(0, 0), true);
        MIMO_UNROLL
        for (int iq = 0; iq < TQ; ++iq) {
          simdgroup_multiply_accumulate(Sf[iq][ik], Qf[iq][dd], b, Sf[iq][ik]);
        }
      }
    }

    MIMO_UNROLL
    for (int iq = 0; iq < TQ; ++iq) {
      float new_max = row_max[iq];
      MIMO_UNROLL
      for (int ik = 0; ik < TK; ++ik) {
        MIMO_UNROLL
        for (int e = 0; e < 2; ++e) {
          const int k_pos = k0 + ik * 8 + fn + e;
          bool visible = k_pos < k_end;
          if (has_mask != 0 && visible) {
            visible = bool(mask[batch * mask_strides[0] + long(k_pos) * mask_strides[3]]);
          }
          const float s = visible ? Sf[iq][ik].thread_elements()[e] * log2_scale : -1.0e30f;
          Sf[iq][ik].thread_elements()[e] = s;
          new_max = max(new_max, s);
        }
      }
      new_max = max(new_max, simd_shuffle_xor(new_max, ushort(1)));
      new_max = max(new_max, simd_shuffle_xor(new_max, ushort(8)));
      // Until a row has seen a visible key its statistics stay empty.
      const bool seen = new_max > -1.0e29f;
      const float factor = seen ? fast::exp2(row_max[iq] - new_max) : 1.0f;
      row_max[iq] = new_max;
      float block_sum = 0.0f;
      MIMO_UNROLL
      for (int ik = 0; ik < TK; ++ik) {
        MIMO_UNROLL
        for (int e = 0; e < 2; ++e) {
          const float p =
              seen ? fast::exp2(Sf[iq][ik].thread_elements()[e] - new_max) : 0.0f;
          Sf[iq][ik].thread_elements()[e] = p;
          block_sum += p;
        }
      }
      block_sum += simd_shuffle_xor(block_sum, ushort(1));
      block_sum += simd_shuffle_xor(block_sum, ushort(8));
      row_sum[iq] = row_sum[iq] * factor + block_sum;
      MIMO_UNROLL
      for (int id = 0; id < TDV; ++id) {
        O[iq][id].thread_elements()[0] *= factor;
        O[iq][id].thread_elements()[1] *= factor;
      }
    }

    MIMO_UNROLL
    for (int ik = 0; ik < TK; ++ik) {
      MIMO_UNROLL
      for (int id = 0; id < TDV; ++id) {
        simdgroup_matrix<bfloat, 8, 8> b;
        simdgroup_load(b, Vs + ik * 8 * LDV + id * 8, LDV);
        MIMO_UNROLL
        for (int iq = 0; iq < TQ; ++iq) {
          simdgroup_multiply_accumulate(O[iq][id], Sf[iq][ik], b, O[iq][id]);
        }
      }
    }
  }

  MIMO_UNROLL
  for (int iq = 0; iq < TQ; ++iq) {
    const int head = sg_head + iq * 8 + fm;
    const long row = ((long(batch) * n_kv_heads + kv_head) * n_splits + split) * G + head;
    device float* out_row = partial_out + row * DV + fn;
    MIMO_UNROLL
    for (int id = 0; id < TDV; ++id) {
      out_row[id * 8] = O[iq][id].thread_elements()[0];
      out_row[id * 8 + 1] = O[iq][id].thread_elements()[1];
    }
    if (fn == 0) {
      partial_max[row] = row_max[iq] * 0.6931471805599453f;
      partial_sum[row] = row_sum[iq];
    }
  }
"""


def _make_kernel(name, input_names, output_names, source):
    if not mx.metal.is_available():
        return None
    return mx.fast.metal_kernel(
        name=name,
        input_names=input_names,
        output_names=output_names,
        source=source,
        header=_HEADER,
        ensure_row_contiguous=False,
    )


_prefill_kernel = _make_kernel(
    "mimo_v2_prefill_attention",
    ["q", "k", "v", "params", "scale"],
    ["o"],
    _PREFILL_SOURCE,
)
_decode_kernel = _make_kernel(
    "mimo_v2_decode_attention",
    ["q", "k", "v", "mask", "params", "scale"],
    ["partial_out", "partial_max", "partial_sum"],
    _DECODE_SOURCE,
)


def _widths_supported(queries, keys, values):
    return (
        queries.ndim == 4
        and queries.shape[-1] == KEY_WIDTH
        and keys.shape[-1] == KEY_WIDTH
        and values.shape[-1] == VALUE_WIDTH
        and queries.dtype == mx.bfloat16
        and keys.dtype == mx.bfloat16
        and values.dtype == mx.bfloat16
        and queries.shape[1] % keys.shape[1] == 0
    )


def prefill_supported(queries, keys, values):
    """Whether ``mimo_prefill_attention`` takes this causal prefill call."""
    return (
        _prefill_kernel is not None
        and _widths_supported(queries, keys, values)
        and queries.shape[2] > 1
        and keys.shape[2] >= queries.shape[2]
    )


def decode_supported(queries, keys, values, mask=None):
    """Whether ``mimo_decode_attention`` takes this decode call."""
    if (
        _decode_kernel is None
        or not _widths_supported(queries, keys, values)
        or queries.shape[2] != 1
        or queries.shape[1] != keys.shape[1] * DECODE_GROUP
    ):
        return False
    if mask is None:
        return True
    return isinstance(mask, mx.array) and mask.dtype == mx.bool_ and mask.ndim <= 4


def mimo_prefill_attention(queries, keys, values, scale):
    """Causal attention of ``queries`` [B, H, L, 192] over ``keys`` [B, Hkv, S, 192] and
    ``values`` [B, Hkv, S, 128]; the queries are the last L of the S positions.

    Returns [B, H, L, 128] bfloat16. Check the call with ``prefill_supported`` first.
    """
    batch, n_heads, length, _ = queries.shape
    n_keys = keys.shape[2]
    rows = _MAX_SCORES_PER_DISPATCH // (batch * n_heads * n_keys)
    rows = max(
        _PREFILL_QUERY_BLOCK, rows // _PREFILL_QUERY_BLOCK * _PREFILL_QUERY_BLOCK
    )
    if rows >= length:
        return _prefill_dispatch(queries, keys, values, n_keys, scale)
    outputs = []
    for start in range(0, length, rows):
        stop = min(start + rows, length)
        outputs.append(
            _prefill_dispatch(
                queries[:, :, start:stop], keys, values, n_keys - length + stop, scale
            )
        )
    return mx.concatenate(outputs, axis=2)


def _prefill_dispatch(queries, keys, values, visible_keys, scale):
    """One launch: ``queries`` are the last rows of the first ``visible_keys`` positions."""
    batch, n_heads, length, _ = queries.shape
    n_kv_heads = keys.shape[1]
    threads = _PREFILL_SIMDGROUPS * 32
    blocks = (length + _PREFILL_QUERY_BLOCK - 1) // _PREFILL_QUERY_BLOCK
    padded_length = blocks * _PREFILL_QUERY_BLOCK
    if padded_length != length:
        queries = mx.pad(queries, [(0, 0), (0, 0), (0, padded_length - length), (0, 0)])
    params = mx.array(
        [
            length,
            visible_keys,
            visible_keys - length,
            n_heads // n_kv_heads,
            n_heads,
            padded_length,
        ],
        dtype=mx.int32,
    )
    (output,) = _prefill_kernel(
        inputs=[
            mx.contiguous(queries),
            keys,
            values,
            params,
            mx.array([scale], dtype=mx.float32),
        ],
        template=[
            ("WM", _PREFILL_SIMDGROUPS),
            ("TQ", _PREFILL_FRAGMENTS),
            ("BK", _KEY_BLOCK),
        ],
        grid=(blocks * threads, n_heads, batch),
        threadgroup=(threads, 1, 1),
        output_shapes=[(batch, n_heads, length, VALUE_WIDTH)],
        output_dtypes=[mx.bfloat16],
    )
    return output


def mimo_decode_attention(queries, keys, values, scale, mask=None):
    """Attention of one query per sequence, ``queries`` [B, H, 1, 192], over ``keys``
    [B, Hkv, S, 192] and ``values`` [B, Hkv, S, 128], with an optional boolean ``mask``
    broadcastable to [B, 1, 1, S].

    Returns [B, H, 1, 128] bfloat16. Check the call with ``decode_supported`` first.
    """
    batch, n_heads, _, _ = queries.shape
    _, n_kv_heads, n_keys, _ = keys.shape
    per_sequence_head = max(1, _DECODE_TARGET_THREADGROUPS // (batch * n_kv_heads))
    n_splits = max(1, min(per_sequence_head, -(-n_keys // _DECODE_MIN_SLICE)))
    slice_length = -(-n_keys // n_splits)
    slice_length = -(-slice_length // _KEY_BLOCK) * _KEY_BLOCK
    n_splits = -(-n_keys // slice_length)
    if mask is None:
        mask_input = mx.ones((1, 1, 1, 1), dtype=mx.bool_)
        has_mask = 0
    else:
        mask_input = mx.broadcast_to(mask, (batch, 1, 1, n_keys))
        has_mask = 1
    params = mx.array(
        [n_keys, slice_length, n_heads, n_splits, has_mask, n_kv_heads],
        dtype=mx.int32,
    )
    threads = _DECODE_SIMDGROUPS * 32
    partial_out, partial_max, partial_sum = _decode_kernel(
        inputs=[
            mx.contiguous(queries),
            keys,
            values,
            mask_input,
            params,
            mx.array([scale], dtype=mx.float32),
        ],
        template=[
            ("WM", _DECODE_SIMDGROUPS),
            ("TQ", _DECODE_FRAGMENTS),
            ("BK", _KEY_BLOCK),
        ],
        grid=(n_splits * threads, n_kv_heads, batch),
        threadgroup=(threads, 1, 1),
        output_shapes=[
            (batch, n_kv_heads, n_splits, DECODE_GROUP, VALUE_WIDTH),
            (batch, n_kv_heads, n_splits, DECODE_GROUP),
            (batch, n_kv_heads, n_splits, DECODE_GROUP),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    top = partial_max.max(axis=2, keepdims=True)
    weights = mx.exp(partial_max - top)
    total = (partial_sum * weights).sum(axis=2)
    combined = (partial_out * weights[..., None]).sum(axis=2) / total[..., None]
    return combined.reshape(batch, n_heads, 1, VALUE_WIDTH).astype(mx.bfloat16)
