"""Fast partial attention for decoding (one query token) on a long KV shard.

Same job as ``distributed_attention.local_partial_attention`` for L == 1, but
runs as a Metal kernel modelled on MLX's own ``sdpa_vector_2pass_1``: the keys
are split into blocks, each block yields (max, sum of exp, weighted V), and the
blocks are merged with the online-softmax rule. MLX's public fused attention
does not return those statistics, which the cross-machine merge needs, so we
build the kernel with ``mx.fast.metal_kernel`` (no MLX rebuild required).

The kernel reads K and V straight from the (possibly sliced, non-contiguous)
cache buffers through their strides, so no copy of the cache is made.
"""

import os

import mlx.core as mx

ENABLED = os.environ.get("MLX_LM_SHARD_FAST_DECODE", "1") != "0"
MIN_KEYS = 256  # below this the plain path is as fast

_HEADER = "#include <metal_simdgroup>\n"

_SOURCE = r"""
constexpr int BD = 32;
constexpr int qk_per_thread = D / BD;
constexpr int v_per_thread = D / BD;
typedef float U;

thread U q[qk_per_thread];
thread U o[v_per_thread];
for (int i = 0; i < v_per_thread; i++) {
  o[i] = 0;
}

const int kv_head_idx = threadgroup_position_in_grid.x;
const int batch_idx = threadgroup_position_in_grid.y;
const int block_idx = threadgroup_position_in_grid.z;
const int gqa_factor = threads_per_threadgroup.y;
const int q_seq_len = threads_per_threadgroup.z;
const int q_seq_idx = thread_position_in_threadgroup.z;
const int q_head_idx = gqa_factor * kv_head_idx + thread_position_in_threadgroup.y;
const int num_kv_heads = threadgroups_per_grid.x;
const int num_q_heads = num_kv_heads * gqa_factor;
const int blocks = threadgroups_per_grid.z;
const int lane = thread_index_in_simdgroup;
const int N = keys_shape[2];
const int q_batch_head_idx = batch_idx * num_q_heads + q_head_idx;
const int o_offset = q_batch_head_idx * q_seq_len + q_seq_idx;

const device T* qp = queries + o_offset * D + lane * qk_per_thread;
const device T* kp = keys + batch_idx * keys_strides[0] + kv_head_idx * keys_strides[1]
    + block_idx * keys_strides[2] + lane * qk_per_thread;
const device T* vp = values + batch_idx * values_strides[0] + kv_head_idx * values_strides[1]
    + block_idx * values_strides[2] + lane * v_per_thread;
const long kstep = blocks * keys_strides[2];
const long vstep = blocks * values_strides[2];

const U scale = static_cast<U>(scale_in[0]);
for (int i = 0; i < qk_per_thread; i++) {
  q[i] = scale * static_cast<U>(qp[i]);
}

U max_score = -3.4028234e38f;
U sum_exp = 0;
for (int i = block_idx; i < N; i += blocks) {
  U score = 0;
  for (int j = 0; j < qk_per_thread; j++) {
    score += q[j] * static_cast<U>(kp[j]);
  }
  score = simd_sum(score);
  U new_max = max(max_score, score);
  U factor = metal::fast::exp(max_score - new_max);
  U exp_score = metal::fast::exp(score - new_max);
  max_score = new_max;
  sum_exp = sum_exp * factor + exp_score;
  for (int j = 0; j < v_per_thread; j++) {
    o[j] = o[j] * factor + exp_score * static_cast<U>(vp[j]);
  }
  kp += kstep;
  vp += vstep;
}

device float* op = partial + (o_offset * blocks + block_idx) * D + lane * v_per_thread;
for (int j = 0; j < v_per_thread; j++) {
  op[j] = o[j];
}
if (lane == 0) {
  sums[o_offset * blocks + block_idx] = sum_exp;
  maxs[o_offset * blocks + block_idx] = max_score;
}
"""

_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name="partial_decode_attention",
            input_names=["queries", "keys", "values", "scale_in"],
            output_names=["partial", "sums", "maxs"],
            source=_SOURCE,
            header=_HEADER,
            ensure_row_contiguous=False,
        )
    return _kernel


def _pick_blocks(n_keys: int, group: int) -> int:
    """Number of key blocks (tuned on M3 Max: 256 up to ~90k keys, 1024 beyond)."""
    return 256 if n_keys < 90_000 else 1024


def supported(queries, keys_shard, values_shard, mask) -> bool:
    if not ENABLED or mask is not None:
        return False
    if isinstance(keys_shard, tuple) or keys_shard is None:
        return False
    if mx.default_device() != mx.gpu:
        return False
    B, H, L, D = queries.shape
    KVH = keys_shard.shape[1]
    if D not in (64, 128) or values_shard.shape[-1] != D or H % KVH != 0:
        return False
    if keys_shard.dtype != queries.dtype or values_shard.dtype != queries.dtype:
        return False
    if keys_shard.shape[2] < MIN_KEYS:
        return False
    return (H // KVH) * L <= 32  # threadgroup holds gqa_factor * L simdgroups


def fast_partial_attention(queries, keys_shard, values_shard, scale, blocks=None):
    """(max, sum of exp, weighted V) of ``queries`` against a KV shard.

    Shapes match ``local_partial_attention``: (B,H,L,1), (B,H,L,1), (B,H,L,D).
    """
    B, H, L, D = queries.shape
    KVH = keys_shard.shape[1]
    G = H // KVH
    N = keys_shard.shape[2]
    blocks = blocks or _pick_blocks(N, G)
    q = mx.contiguous(queries)
    partial, sums, maxs = _get_kernel()(
        inputs=[q, keys_shard, values_shard, mx.array([scale], dtype=mx.float32)],
        template=[("T", queries.dtype), ("D", D)],
        grid=(KVH * 32, B * G, blocks * L),
        threadgroup=(32, G, L),
        output_shapes=[(B, H, L, blocks, D), (B, H, L, blocks), (B, H, L, blocks)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    m = mx.max(maxs, axis=-1, keepdims=True)
    f = mx.exp(maxs - m)
    total = mx.sum(sums * f, axis=-1, keepdims=True)
    wv = mx.sum(partial * f[..., None], axis=-2)
    dt = queries.dtype
    return m.astype(dt), total.astype(dt), wv.astype(dt)
