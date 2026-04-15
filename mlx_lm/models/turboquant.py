"""
TurboQuant KV cache compression (experimental).

PolarQuant from "TurboQuant: Redefining AI Efficiency with Extreme
Compression" (Google, ICLR 2026, https://arxiv.org/abs/2504.19874).

Data-oblivious KV cache quantization at 2-4 bits per coordinate via
random orthogonal rotation followed by Lloyd-Max optimal scalar
quantization. No calibration data needed.
"""

import math

import mlx.core as mx

from .cache import _BaseCache, create_attention_mask

# fmt: off
# Lloyd-Max optimal centroids and boundaries for N(0,1).
# Scaled by 1/sqrt(head_dim) at runtime.
_CENTROIDS = {
    2: [-1.5104, -0.4528,  0.4528,  1.5104],
    3: [-2.1519, -1.3439, -0.7560, -0.2451, 0.2451, 0.7560, 1.3439, 2.1519],
    4: [-2.7331, -2.0698, -1.6189, -1.2570, -0.9431, -0.6573,
        -0.3884, -0.1285,  0.1285,  0.3884,  0.6573,  0.9431,
         1.2570,  1.6189,  2.0698,  2.7331],
}
_BOUNDARIES = {
    2: [-5.0, -0.9816, 0.0, 0.9816, 5.0],
    3: [-5.0, -1.7479, -1.0499, -0.5005, 0.0, 0.5005, 1.0499, 1.7479, 5.0],
    4: [-5.0, -2.4015, -1.8443, -1.4380, -1.1001, -0.8002,
        -0.5229, -0.2585,  0.0,    0.2585,  0.5229,  0.8002,
         1.1001,  1.4380,  1.8443,  2.4015, 5.0],
}
# fmt: on


def _rotation_matrix(dim, seed=42):
    """Haar-distributed random orthogonal matrix via QR of Gaussian."""
    key = mx.random.key(seed)
    g = mx.random.normal(shape=(dim, dim), key=key)
    q, r = mx.linalg.qr(g, stream=mx.cpu)
    sign = mx.sign(mx.diag(r))
    sign = mx.where(sign == 0, 1, sign)
    return q * sign


def _load_codebook(bits, dim):
    s = 1.0 / math.sqrt(dim)
    c = mx.array(_CENTROIDS[bits], dtype=mx.float32) * s
    b = mx.array(_BOUNDARIES[bits], dtype=mx.float32) * s
    return c, b


def _quantize(vectors, rotation_t, boundaries):
    norms = mx.linalg.norm(vectors, axis=-1, keepdims=True)
    rotated = (vectors / mx.maximum(norms, 1e-8)) @ rotation_t
    inner = boundaries[1:-1]
    indices = mx.zeros(rotated.shape, dtype=mx.uint8)
    for b in range(inner.shape[0]):
        indices = indices + (rotated > inner[b]).astype(mx.uint8)
    return indices, norms


def _dequantize(indices, norms, rotation, centroids):
    return centroids[indices] @ rotation * norms


def _pack(indices, bits):
    """Pack b-bit indices into uint32."""
    shape = indices.shape
    dim = shape[-1]
    vpi = 32 // bits
    n_packed = (dim + vpi - 1) // vpi
    pad_size = n_packed * vpi - dim
    if pad_size > 0:
        indices = mx.concatenate(
            [indices, mx.zeros((*shape[:-1], pad_size), dtype=indices.dtype)],
            axis=-1,
        )
    reshaped = indices.reshape(*shape[:-1], n_packed, vpi).astype(mx.uint32)
    shifts = mx.arange(vpi, dtype=mx.uint32) * bits
    shifted = reshaped << shifts
    packed = shifted[..., 0]
    for i in range(1, vpi):
        packed = packed | shifted[..., i]
    return packed


def _unpack(packed, bits, dim):
    """Unpack uint32 back to b-bit indices."""
    shape = packed.shape
    vpi = 32 // bits
    mask = (1 << bits) - 1
    shifts = mx.arange(vpi, dtype=mx.uint32) * bits
    extracted = (packed[..., None] >> shifts) & mask
    return extracted.reshape(*shape[:-1], shape[-1] * vpi)[..., :dim].astype(mx.uint8)


class TurboQuantKVCache(_BaseCache):
    """KV cache compressed with PolarQuant (experimental).

    Data-oblivious compression: random orthogonal rotation maps KV vectors
    to coordinates with a known Gaussian distribution, then Lloyd-Max
    optimal scalar quantizers compress each coordinate independently.
    Bit-packed into uint32 for storage, dequantized on fetch.

    Args:
        bits (int): Bits per coordinate (2, 3, or 4). Default: ``4``.
    """

    step = 256

    def __init__(self, bits: int = 4):
        if bits not in (2, 3, 4):
            raise ValueError(f"bits must be 2, 3, or 4, got {bits}")
        self.turbo_bits = bits
        self.offset = 0
        self._head_dim = None
        self._k_indices = None
        self._k_norms = None
        self._v_indices = None
        self._v_norms = None
        self._centroids = None
        self._boundaries = None
        self._rotation = None
        self._rotation_t = None

    def _init_codebook(self, head_dim):
        self._head_dim = head_dim
        self._centroids, self._boundaries = _load_codebook(
            self.turbo_bits, head_dim
        )
        self._rotation = _rotation_matrix(head_dim)
        self._rotation_t = self._rotation.T

    def update_and_fetch(self, keys, values):
        B, n_kv_heads, num_steps, head_dim = keys.shape
        prev = self.offset

        if self._centroids is None:
            self._init_codebook(head_dim)

        k_idx, k_norms = _quantize(keys, self._rotation_t, self._boundaries)
        v_idx, v_norms = _quantize(values, self._rotation_t, self._boundaries)
        pk = _pack(k_idx, self.turbo_bits)
        pv = _pack(v_idx, self.turbo_bits)

        if self._k_indices is None or (prev + num_steps) > self._k_indices.shape[2]:
            self._expand(B, n_kv_heads, num_steps, keys.dtype, pk.shape[-1])

        self._k_indices[..., prev : prev + num_steps, :] = pk
        self._k_norms[..., prev : prev + num_steps, :] = k_norms
        self._v_indices[..., prev : prev + num_steps, :] = pv
        self._v_norms[..., prev : prev + num_steps, :] = v_norms
        self.offset += num_steps

        all_k = _dequantize(
            _unpack(self._k_indices[..., :self.offset, :], self.turbo_bits, head_dim),
            self._k_norms[..., :self.offset, :],
            self._rotation,
            self._centroids,
        )
        all_v = _dequantize(
            _unpack(self._v_indices[..., :self.offset, :], self.turbo_bits, head_dim),
            self._v_norms[..., :self.offset, :],
            self._rotation,
            self._centroids,
        )
        return all_k, all_v

    def _expand(self, B, n_kv_heads, new_steps, dtype, packed_dim):
        alloc = ((self.step + new_steps - 1) // self.step) * self.step
        shape = (B, n_kv_heads, alloc)

        def _new():
            return (
                mx.zeros((*shape, packed_dim), dtype=mx.uint32),
                mx.zeros((*shape, 1), dtype=dtype),
                mx.zeros((*shape, packed_dim), dtype=mx.uint32),
                mx.zeros((*shape, 1), dtype=dtype),
            )

        if self._k_indices is not None and self.offset > 0:
            old = (
                self._k_indices[..., :self.offset, :],
                self._k_norms[..., :self.offset, :],
                self._v_indices[..., :self.offset, :],
                self._v_norms[..., :self.offset, :],
            )
            self._k_indices, self._k_norms, self._v_indices, self._v_norms = (
                mx.concatenate([o, n], axis=2) for o, n in zip(old, _new())
            )
        else:
            self._k_indices, self._k_norms, self._v_indices, self._v_norms = _new()

    def size(self):
        return self.offset

    @property
    def state(self):
        if self._k_indices is None:
            return []
        return [
            self._k_indices[..., :self.offset, :],
            self._k_norms[..., :self.offset, :],
            self._v_indices[..., :self.offset, :],
            self._v_norms[..., :self.offset, :],
        ]

    @state.setter
    def state(self, v):
        if v is not None and v:
            self._k_indices, self._k_norms, self._v_indices, self._v_norms = v
            self.offset = self._k_indices.shape[2]

    @property
    def meta_state(self):
        return tuple(map(str, (self.offset, self.turbo_bits, self._head_dim or 0)))

    @meta_state.setter
    def meta_state(self, v):
        self.offset, self.turbo_bits = int(v[0]), int(v[1])
        head_dim = int(v[2])
        if head_dim > 0:
            self._init_codebook(head_dim)

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self):
        return self._k_indices is None

    @property
    def nbytes(self):
        if self._k_indices is None:
            return 0
        return sum(
            a[..., :self.offset, :].nbytes
            for a in (self._k_indices, self._k_norms, self._v_indices, self._v_norms)
        )
