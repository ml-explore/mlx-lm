# Copyright © 2024 Apple Inc.
"""TurboQuant KV cache for mlx-lm.

Integrates TurboQuant (arXiv 2504.19874) via the generic
mx.fast.quantized_scaled_dot_product_attention API (modes "turbo3" / "turbo4").

Encoding scheme (per key/value vector of dimension D):
  1. k_rot  = WHT(k) / sqrt(D)           [normalized Walsh-Hadamard transform]
  2. norm   = ||k_rot||
  3. k_scaled = k_rot / norm * sqrt(D)   [→ N(0,1) for Lloyd-Max codebook]
  4. idx    = argmin |k_scaled - codebook|
  5. key_scale = norm / sqrt(D)           [stored alongside packed indices]

SDPA at generation time (Lq == 1):
  q_rot = WHT(q) / sqrt(D)
  mx.fast.quantized_scaled_dot_product_attention(
      q_rot, k_packed, k_scales, v_packed, v_scales,
      scale=1/sqrt(D), mode="turbo3", group_size=D)

The math: scale * key_scale * dot(q_rot, codebook[idx])
         = (1/√D) * (norm/√D) * dot(q_rot, codebook[idx])
         ≈ (1/√D) * dot(q_rot, k_rot)   [WHT orthogonality: dot = q·k]
"""

import math
from typing import Optional

import mlx.core as mx

from .cache import _BaseCache

# N(0,1) Lloyd-Max 3-bit codebook (8 levels)
_CB3 = mx.array(
    [-1.7481, -1.0498, -0.5012, -0.1624, 0.1624, 0.5012, 1.0498, 1.7481],
    dtype=mx.float32,
)

# N(0,1) equal-probability 4-bit codebook (16 levels)
_CB4 = mx.array(
    [
        -1.9672,
        -1.3305,
        -1.0130,
        -0.7811,
        -0.5714,
        -0.4053,
        -0.2382,
        -0.0784,
        0.0784,
        0.2382,
        0.4053,
        0.5714,
        0.7811,
        1.0130,
        1.3305,
        1.9672,
    ],
    dtype=mx.float32,
)

_CODEBOOKS: dict[int, mx.array] = {3: _CB3, 4: _CB4}

# Supported head dimensions (must have a QUANT_SDPA_DISPATCH entry in MLX core)
_SUPPORTED_DIMS: frozenset[int] = frozenset({64, 128, 256})

def _hadamard_transform(x: mx.array, scale: float = 1.0) -> mx.array:
    """Normalized WHT. Dispatches to the Metal kernel when available.

    Falls back to an iterative butterfly in MLX ops on non-Metal devices.
    D must be a power of 2. Result dtype matches x (Metal path) or float32
    (fallback path — caller must cast if needed).
    """
    try:
        from .turbo_metal import is_available, wht_rotate_metal
        if is_available():
            return wht_rotate_metal(x, scale)
    except Exception:
        pass

    # Fallback: iterative butterfly in MLX ops
    *batch, D = x.shape
    assert D & (D - 1) == 0, f"head_dim {D} must be a power of 2 for WHT"
    h = x.astype(mx.float32)
    stride = 1
    while stride < D:
        h = h.reshape(*batch, D // (2 * stride), 2, stride)
        a, b = h[..., 0, :], h[..., 1, :]
        h = mx.concatenate([a + b, a - b], axis=-1).reshape(*batch, D)
        stride *= 2
    if scale != 1.0:
        h = h * scale
    return h

def _turbo_encode_python(
    x: mx.array,
    codebook: mx.array,
    bits: int,
    rotate: bool = True,
) -> tuple[mx.array, mx.array]:
    """Pure-Python fallback encode (CPU numpy packing). Used on non-Metal."""
    *batch, D = x.shape
    inv_sqrt_d = 1.0 / math.sqrt(D)

    x_f32 = x.astype(mx.float32)
    x_preq = _hadamard_transform(x_f32, scale=inv_sqrt_d) if rotate else x_f32

    norm = mx.sqrt(mx.sum(x_preq * x_preq, axis=-1, keepdims=True))
    x_scaled = x_preq / (norm + 1e-8) * math.sqrt(D)

    diff = x_scaled[..., None] - codebook
    indices = mx.argmin(diff * diff, axis=-1).astype(mx.uint32)

    packed = _pack_indices(indices, bits)
    scales = (norm * inv_sqrt_d).astype(x.dtype)
    return packed, scales

def _turbo_encode(
    x: mx.array,
    codebook: mx.array,
    bits: int,
    rotate: bool = True,
) -> tuple[mx.array, mx.array]:
    """Encode a float16/bf16 K or V tensor to TurboQuant packed format.

    Args:
        x: Input tensor, shape [B, H, L, D].
        codebook: Lloyd-Max codebook of shape [2**bits].
        bits: Quantization bit-width (3 or 4).
        rotate: If True (default, use for K), apply a normalized WHT before
            quantization to make the distribution Gaussian, matching the
            Lloyd-Max codebook assumption.  Set to False for V so that the
            kernel output is in the original V space without any inverse
            transform on the caller side.

    Returns:
        packed: uint32 array of shape [B, H, L, D*bits//32].
        scales: float16/bfloat16 array of shape [B, H, L, 1].
    """
    try:
        from .turbo_metal import is_available, turbo_encode_metal
        if is_available():
            return turbo_encode_metal(x, codebook, bits, rotate)
    except Exception:
        pass
    return _turbo_encode_python(x, codebook, bits, rotate)

def _pack_indices(indices: mx.array, bits: int) -> mx.array:
    """Pack bit-width indices into uint32 words using PackReader<bits> layout.

    For bits=3: 8 indices → 3 bytes → D*3//32 uint32 words per vector.
    For bits=4: 8 indices → 1 uint32 word (lower nibble first) → D//8 words.

    Uses numpy for reliable bit manipulation, then converts back to MLX.
    The packing happens once per generation step (O(Lk × D)), not in the
    hot inner loop, so numpy overhead is acceptable.
    """
    import numpy as np

    # Materialise indices to numpy (this triggers MLX evaluation)
    indices_np = np.array(indices, copy=False).astype(np.int64)
    *batch, D = indices_np.shape

    if bits == 4:
        # 8 × 4-bit per uint32, lower nibble first
        n_u32 = D // 8
        flat = indices_np.reshape(-1, D)
        n_tok = flat.shape[0]
        u32 = np.zeros((n_tok, n_u32), dtype=np.uint32)
        for i in range(8):
            u32 |= (flat[:, i::8].astype(np.uint32) & 0xF) << (4 * i)
        return mx.array(u32.reshape(*batch, n_u32))

    elif bits == 3:
        # 8 × 3-bit per pack → 3 bytes; D*3//32 uint32 words per vector.
        # Vectorised: reshape into [n_tok, n_packs, 8], shift each column by
        # its bit offset, sum → packed_24 [n_tok, n_packs] with 24 bits each.
        n_packs = D // 8
        n_u32 = D * 3 // 32
        flat = indices_np.reshape(-1, D)
        n_tok = flat.shape[0]

        shifts = np.array([0, 3, 6, 9, 12, 15, 18, 21], dtype=np.int64)
        idx_r = flat.reshape(n_tok, n_packs, 8).astype(np.int64)
        packed_24 = np.sum(idx_r << shifts, axis=-1)  # [n_tok, n_packs]

        b0 = (packed_24 & 0xFF).astype(np.uint8)
        b1 = ((packed_24 >> 8) & 0xFF).astype(np.uint8)
        b2 = ((packed_24 >> 16) & 0xFF).astype(np.uint8)
        bytes_arr = np.stack([b0, b1, b2], axis=-1).reshape(n_tok, n_packs * 3)

        u32 = np.frombuffer(bytes_arr.tobytes(), dtype="<u4").reshape(n_tok, n_u32)
        return mx.array(u32.reshape(*batch, n_u32))

    else:
        raise ValueError(f"TurboQuant only supports bits=3 or bits=4, got {bits}")

class TurboQuantKVCache(_BaseCache):
    """KV cache with TurboQuant compression for generation.

    Two-phase behaviour:
    - **Prefill** (Lq > 1 or unsupported head_dim): stores float16,
      returns float16 so the standard SDPA path is used.
    - **Generation** (Lq == 1, head_dim in {64, 128, 256}): compresses new
      tokens and returns (packed, scales) tuples so
      ``mx.fast.quantized_scaled_dot_product_attention`` is used.

    On the first generation step after prefill, all cached float16 tokens
    are re-compressed to TurboQuant format (one-time amortised cost).

    Attributes:
        bits: Quantization bit-width (3 or 4).
        mode: Mode string for mx.fast.quantized_scaled_dot_product_attention.
        group_size: Equals head_dim (set on first call).
    """

    step = 256  # allocation step size

    def __init__(self, bits: int = 3) -> None:
        if bits not in (3, 4):
            raise ValueError(f"TurboQuantKVCache: bits must be 3 or 4, got {bits}")
        self.bits = bits
        self.mode = f"turbo{bits}"

        # Float16 buffer (prefill phase)
        self._keys_f16: Optional[mx.array] = None
        self._values_f16: Optional[mx.array] = None

        # Compressed buffer (generation phase): (packed_uint32, float16_scales)
        self._keys_tq: Optional[tuple[mx.array, mx.array]] = None
        self._values_tq: Optional[tuple[mx.array, mx.array]] = None

        self.offset: int = 0
        self.group_size: Optional[int] = None  # set to head_dim on first call
        self._in_generation: bool = False

    def _codebook(self) -> mx.array:
        return _CODEBOOKS[self.bits]

    def _compress(
        self, keys: mx.array, values: mx.array
    ) -> tuple[
        tuple[mx.array, mx.array],
        tuple[mx.array, mx.array],
    ]:
        cb = self._codebook()
        k_packed, k_scales = _turbo_encode(keys, cb, self.bits, rotate=True)
        v_packed, v_scales = _turbo_encode(values, cb, self.bits, rotate=False)
        return (k_packed, k_scales), (v_packed, v_scales)

    def _init_tq_buffers(
        self,
        B: int,
        H: int,
        D: int,
        dtype: mx.Dtype,
        n_steps: int,
    ) -> None:
        """Allocate pre-sized TurboQuant buffers."""
        n_u32 = D * self.bits // 32
        shape = (B, H, n_steps)
        self._keys_tq = (
            mx.zeros((*shape, n_u32), dtype=mx.uint32),
            mx.zeros((*shape, 1), dtype=dtype),
        )
        self._values_tq = (
            mx.zeros((*shape, n_u32), dtype=mx.uint32),
            mx.zeros((*shape, 1), dtype=dtype),
        )

    def _expand_tq_buffers(self) -> None:
        """Grow TurboQuant buffers using doubling (amortised O(1) reallocations)."""
        B, H, cur_len, n_u32 = self._keys_tq[0].shape
        n_extra = max(self.step, cur_len)  # double: grow by at least current size
        pad_d = mx.zeros((B, H, n_extra, n_u32), dtype=mx.uint32)
        pad_s = mx.zeros((B, H, n_extra, 1), dtype=self._keys_tq[1].dtype)

        self._keys_tq = (
            mx.concatenate([self._keys_tq[0], pad_d], axis=-2),
            mx.concatenate([self._keys_tq[1], pad_s], axis=-2),
        )
        self._values_tq = (
            mx.concatenate([self._values_tq[0], pad_d], axis=-2),
            mx.concatenate([self._values_tq[1], pad_s], axis=-2),
        )

    def _transition_to_generation(self, D: int) -> None:
        """Re-compress all float16 prefill tokens to TurboQuant format."""
        k_f = self._keys_f16[..., : self.offset, :]
        v_f = self._values_f16[..., : self.offset, :]
        B, H, L, _ = k_f.shape
        (k_packed, k_scales), (v_packed, v_scales) = self._compress(k_f, v_f)

        # Allocate with extra room for generation
        n_alloc = (L + self.step - 1) // self.step * self.step + self.step
        self._init_tq_buffers(B, H, D, k_f.dtype, n_alloc)

        self._keys_tq[0][..., :L, :] = k_packed
        self._keys_tq[1][..., :L, :] = k_scales
        self._values_tq[0][..., :L, :] = v_packed
        self._values_tq[1][..., :L, :] = v_scales

        # Free float16 buffers
        self._keys_f16 = None
        self._values_f16 = None
        self._in_generation = True

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple:
        """Append keys/values to the cache and return all cached data.

        During prefill (num_steps > 1) or unsupported head_dim:
            Returns (keys_f16, values_f16).
        During generation (num_steps == 1, head_dim in {64, 128, 256}):
            Returns ((k_packed, k_scales), (v_packed, v_scales)).
        """
        B, H, num_steps, D = keys.shape
        prev = self.offset
        self.offset += num_steps

        use_turbo = num_steps == 1 and D in _SUPPORTED_DIMS

        if not use_turbo:
            new_size = (prev + num_steps + self.step - 1) // self.step * self.step

            if self._keys_f16 is None:
                self._keys_f16 = mx.zeros((B, H, new_size, D), dtype=keys.dtype)
                self._values_f16 = mx.zeros((B, H, new_size, D), dtype=values.dtype)
            elif prev + num_steps > self._keys_f16.shape[-2]:
                pad_k = mx.zeros(
                    (B, H, new_size - self._keys_f16.shape[-2], D), dtype=keys.dtype
                )
                pad_v = mx.zeros(
                    (B, H, new_size - self._values_f16.shape[-2], D),
                    dtype=values.dtype,
                )
                self._keys_f16 = mx.concatenate([self._keys_f16, pad_k], axis=-2)
                self._values_f16 = mx.concatenate([self._values_f16, pad_v], axis=-2)

            self._keys_f16[..., prev : self.offset, :] = keys
            self._values_f16[..., prev : self.offset, :] = values

            return (
                self._keys_f16[..., : self.offset, :],
                self._values_f16[..., : self.offset, :],
            )

        self.group_size = D

        if not self._in_generation:
            # First generation step: re-compress any stored prefill tokens
            if self._keys_f16 is not None:
                self._transition_to_generation(D)
            else:
                # No prefill: jump straight to generation
                self._in_generation = True

        # Grow buffers if needed
        if self._keys_tq is None:
            self._init_tq_buffers(B, H, D, keys.dtype, self.step)
        elif prev + num_steps > self._keys_tq[0].shape[-2]:
            self._expand_tq_buffers()

        # Compress the new token
        (k_packed, k_scales), (v_packed, v_scales) = self._compress(keys, values)

        self._keys_tq[0][..., prev : self.offset, :] = k_packed
        self._keys_tq[1][..., prev : self.offset, :] = k_scales
        self._values_tq[0][..., prev : self.offset, :] = v_packed
        self._values_tq[1][..., prev : self.offset, :] = v_scales

        return (
            (
                self._keys_tq[0][..., : self.offset, :],
                self._keys_tq[1][..., : self.offset, :],
            ),
            (
                self._values_tq[0][..., : self.offset, :],
                self._values_tq[1][..., : self.offset, :],
            ),
        )

    def is_trimmable(self) -> bool:
        return False

    def empty(self) -> bool:
        return self.offset == 0

    def size(self) -> int:
        return self.offset

    @property
    def nbytes(self) -> int:
        total = 0
        if self._keys_f16 is not None:
            total += self._keys_f16.nbytes + self._values_f16.nbytes
        if self._keys_tq is not None:
            for arr in (*self._keys_tq, *self._values_tq):
                total += arr.nbytes
        return total

    @property
    def state(self):
        # Serialisation not supported; included for interface compliance.
        return []

    @state.setter
    def state(self, v):
        if v:
            raise ValueError("TurboQuantKVCache does not support state loading.")

    @property
    def meta_state(self):
        return str(self.bits)

    @meta_state.setter
    def meta_state(self, v):
        self.bits = int(v)
        self.mode = f"turbo{self.bits}"
