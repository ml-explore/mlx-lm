"""Mixed-precision quantized KV cache: K at 8-bit, V at 4-bit.

Same pre-allocation pattern as QuantizedKVCache (step=256, no per-step
concatenation), but K and V use independent bit widths and group sizes.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
from mlx.utils import tree_map


class MixedQuantKVCache:
    step = 256

    def __init__(
        self,
        k_bits: int = 8,
        v_bits: int = 4,
        k_group_size: int = 64,
        v_group_size: int = 64,
    ):
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.k_group_size = k_group_size
        self.v_group_size = v_group_size
        self.offset = 0
        self.keys: Optional[tuple] = None
        self.values: Optional[tuple] = None

    @classmethod
    def from_kvcache(cls, cache, k_bits=8, v_bits=4, k_group_size=64, v_group_size=64):
        """Convert a populated fp16 KVCache to mixed-precision quantized.

        Call this AFTER prefill to avoid reallocations during the
        prefill phase (same pattern as mlx-lm's maybe_quantize_kv_cache).
        """
        obj = cls(k_bits=k_bits, v_bits=v_bits,
                  k_group_size=k_group_size, v_group_size=v_group_size)
        if cache.keys is not None:
            obj.offset = cache.offset
            k_slice = cache.keys[..., :cache.offset, :]
            v_slice = cache.values[..., :cache.offset, :]
            obj.keys = mx.quantize(k_slice, group_size=k_group_size, bits=k_bits)
            obj.values = mx.quantize(v_slice, group_size=v_group_size, bits=v_bits)
        return obj

    def _init_quant(self, B, n_kv_heads, n_steps, dim, group_size, bits, dtype):
        el_per_int = 8 * mx.uint32.size // bits
        shape = (B, n_kv_heads, n_steps)
        return (
            mx.zeros((*shape, dim // el_per_int), dtype=mx.uint32),
            mx.zeros((*shape, dim // group_size), dtype=dtype),
            mx.zeros((*shape, dim // group_size), dtype=dtype),
        )

    def _expand_quant(self, quant_tuple, B, n_kv_heads, new_steps):
        def expand(x):
            new_x = mx.zeros(
                (B, n_kv_heads, new_steps, x.shape[-1]), dtype=x.dtype
            )
            return mx.concatenate([x, new_x], axis=2)
        return tuple(expand(x) for x in quant_tuple)

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        prev = self.offset
        num_steps = keys.shape[2]

        # Expand pre-allocated buffers only when crossing a step boundary.
        # Hot path (decode, num_steps=1) skips all allocation logic.
        need_alloc = self.keys is None or (prev + num_steps) > self.keys[0].shape[2]
        if need_alloc:
            B, n_kv_heads = keys.shape[:2]
            k_dim, v_dim = keys.shape[-1], values.shape[-1]
            n = (self.step + num_steps - 1) // self.step * self.step
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = tuple(x[..., :prev, :] for x in self.keys)
                    self.values = tuple(x[..., :prev, :] for x in self.values)
                self.keys = self._expand_quant(self.keys, B, n_kv_heads, n)
                self.values = self._expand_quant(self.values, B, n_kv_heads, n)
            else:
                self.keys = self._init_quant(
                    B, n_kv_heads, n, k_dim,
                    self.k_group_size, self.k_bits, keys.dtype,
                )
                self.values = self._init_quant(
                    B, n_kv_heads, n, v_dim,
                    self.v_group_size, self.v_bits, values.dtype,
                )

        self.offset = prev + num_steps

        # Quantize + write (hot path: no allocation, just 2 quantize + 6 writes)
        k_q = mx.quantize(keys, group_size=self.k_group_size, bits=self.k_bits)
        v_q = mx.quantize(values, group_size=self.v_group_size, bits=self.v_bits)
        self.keys[0][..., prev:self.offset, :] = k_q[0]
        self.keys[1][..., prev:self.offset, :] = k_q[1]
        self.keys[2][..., prev:self.offset, :] = k_q[2]
        self.values[0][..., prev:self.offset, :] = v_q[0]
        self.values[1][..., prev:self.offset, :] = v_q[1]
        self.values[2][..., prev:self.offset, :] = v_q[2]

        # Return views — no copy, no tuple() overhead
        off = self.offset
        return (
            (self.keys[0][..., :off, :], self.keys[1][..., :off, :], self.keys[2][..., :off, :]),
            (self.values[0][..., :off, :], self.values[1][..., :off, :], self.values[2][..., :off, :]),
        )

    @property
    def state(self):
        if self.keys is None:
            return []
        return [x[..., : self.offset, :] for x in self.keys + self.values]

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        total = 0
        for arrays in (self.keys, self.values):
            for x in arrays:
                total += x[..., : self.offset, :].nbytes
        return total

    def make_mask(self, N, return_array=False, window_size=None):
        from .base import create_causal_mask
        if N == 1:
            return None
        if return_array or (window_size is not None and N > window_size):
            return create_causal_mask(
                N, offset=self.offset, window_size=window_size
            )
        return "causal"

    def empty(self):
        return self.keys is None

    def to_kvcache(self):
        """Dequantize back to a standard KVCache for batch merge compatibility."""
        from .cache import KVCache
        kv = KVCache()
        if self.keys is not None:
            off = self.offset
            k_fp = mx.dequantize(
                *[x[..., :off, :] for x in self.keys],
                group_size=self.k_group_size, bits=self.k_bits,
            )
            v_fp = mx.dequantize(
                *[x[..., :off, :] for x in self.values],
                group_size=self.v_group_size, bits=self.v_bits,
            )
            kv.update_and_fetch(k_fp, v_fp)
        return kv

    def is_trimmable(self):
        return False

    def size(self):
        return self.offset
