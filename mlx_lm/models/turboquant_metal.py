"""Fused Metal quantize kernel: raw fp16 vector → packed uint32 + norm.

Replaces the Python path: upcast → norm → normalize → signs → WHT → scale →
nearest centroid → pack. All in one Metal dispatch per batch of vectors.

Also includes fp16-output dequant for decode buffer writes.
"""

import mlx.core as mx
import math

# Fused quantize: one threadgroup per vector (dim threads)
# Input: fp16 vectors. Output: packed uint32 indices + float32 norms.
FUSED_QUANTIZE_KERNEL = """
    uint pos = threadgroup_position_in_grid.x;
    uint elem = thread_position_in_threadgroup.x;
    uint dim = dims[0];
    uint bits = dims[1];
    uint vals_per_word = dims[2];
    uint packed_dim = dims[3];
    uint n_centroids = dims[4];

    // Load input vector into shared memory as float32
    threadgroup float shared[256];
    shared[elem] = (float)inp[pos * dim + elem];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 1: Compute L2 norm via parallel reduction
    threadgroup float norm_shared[256];
    norm_shared[elem] = shared[elem] * shared[elem];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = dim / 2; stride > 0; stride >>= 1) {
        if (elem < stride) {
            norm_shared[elem] += norm_shared[elem + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float vec_norm = sqrt(norm_shared[0]);
    float safe_norm = max(vec_norm, 1e-8f);

    // Step 2: Normalize
    shared[elem] = shared[elem] / safe_norm;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: Apply signs (randomized Hadamard = signs * WHT)
    shared[elem] = shared[elem] * signs[elem];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4: WHT butterfly
    uint h = 1;
    while (h < dim) {
        uint block = elem / (2 * h);
        uint offset = elem % (2 * h);
        if (offset < h) {
            uint j = block * 2 * h + offset;
            float a = shared[j];
            float b = shared[j + h];
            shared[j] = a + b;
            shared[j + h] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        h *= 2;
    }

    // After raw butterfly (no 1/sqrt(d) normalization), values are already
    // in N(0,1) space: butterfly(x_unit * signs) ≈ N(0, 1)
    // No additional scaling needed — butterfly output matches codebook directly
    float scaled = shared[elem];

    // Step 6: Nearest centroid (count boundaries exceeded)
    uint idx = 0;
    for (uint b = 0; b < n_centroids - 1; b++) {
        if (scaled > boundaries[b]) {
            idx++;
        }
    }

    // Step 7: Pack indices - thread 0 of each pack group collects and packs
    // First store indices to shared memory
    threadgroup uint idx_shared[256];
    idx_shared[elem] = idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread responsible for one packed word writes it
    uint word_idx = elem / vals_per_word;
    uint pos_in_word = elem % vals_per_word;

    if (pos_in_word == 0 && word_idx < packed_dim) {
        uint word = 0;
        for (uint i = 0; i < vals_per_word && (word_idx * vals_per_word + i) < dim; i++) {
            word |= (idx_shared[word_idx * vals_per_word + i] & ((1u << bits) - 1u)) << (i * bits);
        }
        packed_out[pos * packed_dim + word_idx] = word;
    }

    // Thread 0 writes the norm
    if (elem == 0) {
        norms_out[pos] = vec_norm;
    }
"""

# fp16-output dequant: same as v3 but outputs half precision
DEQUANT_FP16_KERNEL = """
    uint pos = threadgroup_position_in_grid.x;
    uint elem = thread_position_in_threadgroup.x;
    uint dim = dims[0];
    uint bits = dims[1];
    uint vals_per_word = dims[2];
    uint packed_dim = dims[3];
    uint bit_mask = (1u << bits) - 1u;

    uint word_idx = elem / vals_per_word;
    uint pos_in_word = elem % vals_per_word;
    uint word = packed[pos * packed_dim + word_idx];
    uint idx = (word >> (pos_in_word * bits)) & bit_mask;

    float val = centroids[idx] * scale[0];

    threadgroup float shared[256];
    shared[elem] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint h = 1;
    while (h < dim) {
        uint block = elem / (2 * h);
        uint offset = elem % (2 * h);
        if (offset < h) {
            uint j = block * 2 * h + offset;
            float a = shared[j];
            float b = shared[j + h];
            shared[j] = a + b;
            shared[j + h] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        h *= 2;
    }

    float result = shared[elem] * scale[0] * signs[elem] * norms[pos];
    out[pos * dim + elem] = (half)result;
"""

_fused_quantize_kernel = None
_dequant_fp16_kernel = None


def fused_quantize(
    vectors: mx.array,
    signs: mx.array,
    boundaries: mx.array,
    dim: int,
    bits: int,
) -> tuple:
    """Fused Metal quantize: raw vectors → packed uint32 + norms.

    Args:
        vectors: (n_vecs, dim) fp16/fp32 input
        signs: (dim,) rotation signs
        boundaries: (n_centroids-1,) decision boundaries
        dim: head dimension
        bits: quantization bits

    Returns:
        (packed, norms): packed uint32 (n_vecs, packed_dim), norms float32 (n_vecs,)
    """
    global _fused_quantize_kernel
    if _fused_quantize_kernel is None:
        _fused_quantize_kernel = mx.fast.metal_kernel(
            name="tq_fused_quantize",
            input_names=["inp", "signs", "boundaries", "dims"],
            output_names=["packed_out", "norms_out"],
            source=FUSED_QUANTIZE_KERNEL,
        )

    from mlx_lm.models.turboquant_packing import packed_dim as calc_packed_dim, VALS_PER_WORD
    n_vecs = vectors.shape[0]
    vpw = VALS_PER_WORD[bits]
    p_dim = calc_packed_dim(dim, bits)
    n_centroids = len(boundaries) + 1

    dims_arr = mx.array([dim, bits, vpw, p_dim, n_centroids], dtype=mx.uint32)

    outputs = _fused_quantize_kernel(
        inputs=[
            vectors.reshape(n_vecs * dim).astype(mx.float32),
            signs.astype(mx.float32),
            boundaries.astype(mx.float32),
            dims_arr,
        ],
        template=[],
        grid=(n_vecs * dim, 1, 1),
        threadgroup=(dim, 1, 1),
        output_shapes=[(n_vecs * p_dim,), (n_vecs,)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    return outputs[0].reshape(n_vecs, p_dim), outputs[1]


def dequant_fp16(
    packed: mx.array,
    norms: mx.array,
    centroids: mx.array,
    signs: mx.array,
    dim: int,
    bits: int,
) -> mx.array:
    """Dequantize from packed to fp16 directly (no float32 intermediate)."""
    global _dequant_fp16_kernel
    if _dequant_fp16_kernel is None:
        _dequant_fp16_kernel = mx.fast.metal_kernel(
            name="tq_dequant_fp16",
            input_names=["packed", "norms", "centroids", "signs", "scale", "dims"],
            output_names=["out"],
            source=DEQUANT_FP16_KERNEL,
        )

    from mlx_lm.models.turboquant_packing import packed_dim as calc_packed_dim, VALS_PER_WORD
    seq_len = norms.shape[0]
    vpw = VALS_PER_WORD[bits]
    p_dim = calc_packed_dim(dim, bits)
    scale = mx.array([1.0 / math.sqrt(dim)], dtype=mx.float32)
    dims_arr = mx.array([dim, bits, vpw, p_dim], dtype=mx.uint32)

    outputs = _dequant_fp16_kernel(
        inputs=[packed.astype(mx.uint32).reshape(-1), norms.astype(mx.float32), centroids, signs, scale, dims_arr],
        template=[],
        grid=(seq_len * dim, 1, 1),
        threadgroup=(dim, 1, 1),
        output_shapes=[(seq_len, dim)],
        output_dtypes=[mx.float16],
    )
    return outputs[0]
