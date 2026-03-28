"""TurboQuantKVCache: PolarQuant KV cache compression with fused Metal kernels.

Implements TurboQuant (arXiv 2504.19874, ICLR 2026) for MLX KV cache compression.
4.6x compression via randomized Hadamard rotation + Lloyd-Max quantization.
Bit-packed uint32 storage with fused Metal quantize/dequantize kernels.
"""

import mlx.core as mx
import math
from mlx_lm.models.turboquant_rotation import random_diagonal_sign
from mlx_lm.models.turboquant_packing import pack_indices, unpack_indices, packed_dim, VALS_PER_WORD
from mlx_lm.models.turboquant_metal import fused_quantize, dequant_fp16
from mlx_lm.models.turboquant_kernels import packed_dequantize


def _compute_gaussian_codebook(bits):
    codebooks = {
        1: [-0.7979, 0.7979],
        2: [-1.5104, -0.4528, 0.4528, 1.5104],
        3: [-2.1520, -1.3440, -0.7560, -0.2451,
             0.2451, 0.7560, 1.3440, 2.1520],
        4: [-2.7326, -2.0690, -1.6180, -1.2562,
            -0.9423, -0.6568, -0.3881, -0.1284,
             0.1284, 0.3881, 0.6568, 0.9423,
             1.2562, 1.6180, 2.0690, 2.7326],
    }
    return mx.array(codebooks[bits], dtype=mx.float32)


def _compute_boundaries(centroids):
    return (centroids[:-1] + centroids[1:]) / 2.0


class _Quantizer:
    def __init__(self, dim, bits, seed):
        self.dim = dim
        self.bits = bits
        self.signs = random_diagonal_sign(dim, seed=seed)
        self.centroids = _compute_gaussian_codebook(bits)
        self.boundaries = _compute_boundaries(self.centroids)


class TurboQuantKVCache:
    """TurboQuant KV cache — drop-in replacement for KVCache.

    Compresses KV vectors using PolarQuant (Hadamard rotation + Lloyd-Max
    codebook quantization). Stores bit-packed indices in uint32 + float32 norms.

    Uses fused Metal kernels for quantize and dequantize operations.
    Maintains an incremental decode buffer for O(1) per-step dequantization.
    """

    step = 256

    def __init__(self, bits: int = 3, seed: int = 42):
        self.quant_bits = bits
        self.seed = seed
        self.offset = 0

        self.k_packed = None
        self.k_norms = None
        self.v_packed = None
        self.v_norms = None

        self._k_deq_buf = None
        self._v_deq_buf = None
        self._deq_offset = 0
        self._deq_alloc = 0

        self._k_q = None
        self._v_q = None
        self._k_dim = None
        self._v_dim = None
        self._k_pdim = None
        self._v_pdim = None

    def _ensure_quantizer(self, k_dim, v_dim):
        if self._k_q is None:
            self._k_q = _Quantizer(k_dim, self.quant_bits, self.seed)
            self._k_dim = k_dim
            self._k_pdim = packed_dim(k_dim, self.quant_bits)
        if self._v_q is None:
            self._v_q = _Quantizer(v_dim, self.quant_bits, self.seed + 1)
            self._v_dim = v_dim
            self._v_pdim = packed_dim(v_dim, self.quant_bits)

    def _ensure_storage(self, B, H, num_new):
        prev = self.offset
        needed = prev + num_new
        if self.k_packed is None or needed > self.k_packed.shape[2]:
            n = ((needed + self.step - 1) // self.step) * self.step
            new_kp = mx.zeros((B, H, n, self._k_pdim), dtype=mx.uint32)
            new_kn = mx.zeros((B, H, n), dtype=mx.float32)
            new_vp = mx.zeros((B, H, n, self._v_pdim), dtype=mx.uint32)
            new_vn = mx.zeros((B, H, n), dtype=mx.float32)
            if self.k_packed is not None:
                self.k_packed = mx.concatenate([self.k_packed[..., :prev, :], new_kp], axis=2)
                self.k_norms = mx.concatenate([self.k_norms[..., :prev], new_kn], axis=2)
                self.v_packed = mx.concatenate([self.v_packed[..., :prev, :], new_vp], axis=2)
                self.v_norms = mx.concatenate([self.v_norms[..., :prev], new_vn], axis=2)
            else:
                self.k_packed, self.k_norms = new_kp, new_kn
                self.v_packed, self.v_norms = new_vp, new_vn

    def _full_dequant(self, packed, norms, q, dim, B, H, total, dtype):
        flat_p = packed[..., :total, :].reshape(-1, packed.shape[-1])
        flat_n = norms[..., :total].reshape(-1)
        out = packed_dequantize(flat_p, flat_n, q.centroids, q.signs, dim, self.quant_bits)
        return out.reshape(B, H, total, dim).astype(dtype)

    def update_and_fetch(self, keys, values):
        B, H, S, k_dim = keys.shape
        v_dim = values.shape[3]
        self._ensure_quantizer(k_dim, v_dim)
        self._ensure_storage(B, H, S)
        prev = self.offset

        # Fused Metal quantize
        k_pk, k_nrm = fused_quantize(keys.reshape(-1, k_dim), self._k_q.signs, self._k_q.boundaries, k_dim, self.quant_bits)
        k_pk = k_pk.reshape(B, H, S, self._k_pdim)
        v_pk, v_nrm = fused_quantize(values.reshape(-1, v_dim), self._v_q.signs, self._v_q.boundaries, v_dim, self.quant_bits)
        v_pk = v_pk.reshape(B, H, S, self._v_pdim)

        self.k_packed[..., prev:prev+S, :] = k_pk
        self.k_norms[..., prev:prev+S] = k_nrm.reshape(B, H, S)
        self.v_packed[..., prev:prev+S, :] = v_pk
        self.v_norms[..., prev:prev+S] = v_nrm.reshape(B, H, S)
        self.offset += S
        total = self.offset

        # Incremental decode
        if S <= 4 and self._v_deq_buf is not None and self._deq_offset == prev:
            if total > self._deq_alloc:
                na = ((total + self.step - 1) // self.step) * self.step
                self._k_deq_buf = mx.concatenate([self._k_deq_buf[..., :self._deq_offset, :],
                    mx.zeros((B, H, na - self._deq_alloc, k_dim), dtype=keys.dtype)], axis=2)
                self._v_deq_buf = mx.concatenate([self._v_deq_buf[..., :self._deq_offset, :],
                    mx.zeros((B, H, na - self._deq_alloc, v_dim), dtype=values.dtype)], axis=2)
                self._deq_alloc = na

            nk = dequant_fp16(k_pk.reshape(-1, self._k_pdim), k_nrm, self._k_q.centroids, self._k_q.signs, k_dim, self.quant_bits).reshape(B, H, S, k_dim)
            nv = dequant_fp16(v_pk.reshape(-1, self._v_pdim), v_nrm, self._v_q.centroids, self._v_q.signs, v_dim, self.quant_bits).reshape(B, H, S, v_dim)
            self._k_deq_buf[..., prev:total, :] = nk
            self._v_deq_buf[..., prev:total, :] = nv
            self._deq_offset = total
            return self._k_deq_buf[..., :total, :], self._v_deq_buf[..., :total, :]

        # Full dequant (prefill)
        all_k = self._full_dequant(self.k_packed, self.k_norms, self._k_q, k_dim, B, H, total, keys.dtype)
        all_v = self._full_dequant(self.v_packed, self.v_norms, self._v_q, v_dim, B, H, total, values.dtype)
        alloc = ((total + self.step - 1) // self.step) * self.step
        self._k_deq_buf = mx.zeros((B, H, alloc, k_dim), dtype=keys.dtype)
        self._v_deq_buf = mx.zeros((B, H, alloc, v_dim), dtype=values.dtype)
        self._k_deq_buf[..., :total, :] = all_k
        self._v_deq_buf[..., :total, :] = all_v
        self._deq_offset = total
        self._deq_alloc = alloc
        return all_k, all_v

    def empty(self):
        return self.k_packed is None

    @property
    def nbytes(self):
        if self.k_packed is None:
            return 0
        return (self.k_packed[..., :self.offset, :].nbytes + self.v_packed[..., :self.offset, :].nbytes +
                self.k_norms[..., :self.offset].nbytes + self.v_norms[..., :self.offset].nbytes)

    @property
    def state(self):
        if self.k_packed is None:
            return []
        return [self.k_packed[..., :self.offset, :], self.k_norms[..., :self.offset],
                self.v_packed[..., :self.offset, :], self.v_norms[..., :self.offset]]

    @state.setter
    def state(self, v):
        if not v:
            return
        self.k_packed, self.k_norms, self.v_packed, self.v_norms = v
        self.offset = self.k_packed.shape[2]

    @property
    def meta_state(self):
        return f"{self.offset},{self.quant_bits},{self.seed},{self._k_dim or 0},{self._v_dim or 0}"

    @meta_state.setter
    def meta_state(self, v):
        parts = v.split(",")
        self.offset, self.quant_bits, self.seed = int(parts[0]), int(parts[1]), int(parts[2])
        self._k_dim = int(parts[3]) or None
        self._v_dim = int(parts[4]) or None

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def size(self):
        return self.offset

    def make_mask(self, *args, **kwargs):
        from mlx_lm.models.cache import create_attention_mask
        return create_attention_mask(*args, offset=self.offset, **kwargs)
