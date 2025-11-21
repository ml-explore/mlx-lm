# ABOUTME: Provides per-slot key/value storage for batching runtime.
# ABOUTME: Preallocates fixed-capacity buffers indexed by slot id.

from __future__ import annotations

import threading
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


class SlotKVCache:
    """Fixed-capacity KV storage addressed by slot id.

    Notes:
        This implementation uses numpy arrays for Stage 1 scaffolding. It will
        be swapped to MX tensors when the slot-based runtime is integrated.
    """

    def __init__(
        self,
        max_slots: int,
        capacity_tokens: int,
        kv_heads: int,
        head_dim: int,
        dtype: np.dtype = np.float32,
    ):
        if max_slots <= 0 or capacity_tokens <= 0:
            raise ValueError("max_slots and capacity_tokens must be positive")
        self.max_slots = max_slots
        self.capacity_tokens = capacity_tokens
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.dtype = dtype

        shape = (max_slots, capacity_tokens, kv_heads, head_dim)
        self._keys = np.zeros(shape, dtype=dtype)
        self._values = np.zeros(shape, dtype=dtype)
        self._lengths: List[int] = [0] * max_slots
        self._lock = threading.Lock()

    def reset(self, slot: int) -> None:
        self._validate_slot(slot)
        with self._lock:
            self._lengths[slot] = 0

    def append(self, slot: int, keys: np.ndarray, values: np.ndarray) -> None:
        self._validate_slot(slot)
        if keys.shape != values.shape:
            raise ValueError("keys/values shape mismatch")
        if keys.ndim != 3:
            raise ValueError("expected keys shape [tokens, heads, dim]")
        tokens = keys.shape[0]
        if keys.shape[1] != self.kv_heads or keys.shape[2] != self.head_dim:
            raise ValueError("unexpected key/value head dimensions")

        with self._lock:
            start = self._lengths[slot]
            end = start + tokens
            if end > self.capacity_tokens:
                raise ValueError("slot capacity exceeded")
            self._keys[slot, start:end] = keys
            self._values[slot, start:end] = values
            self._lengths[slot] = end

    def length(self, slot: int) -> int:
        self._validate_slot(slot)
        with self._lock:
            return self._lengths[slot]

    def view(self, slot: int) -> Tuple[np.ndarray, np.ndarray]:
        self._validate_slot(slot)
        with self._lock:
            end = self._lengths[slot]
            return self._keys[slot, :end], self._values[slot, :end]

    def gather(self, slots: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        for s in slots:
            self._validate_slot(s)
        with self._lock:
            max_len = max((self._lengths[s] for s in slots), default=0)
            gathered_k = np.zeros(
                (len(slots), max_len, self.kv_heads, self.head_dim), dtype=self.dtype
            )
            gathered_v = np.zeros_like(gathered_k)
            for i, slot in enumerate(slots):
                length = self._lengths[slot]
                if length:
                    gathered_k[i, :length] = self._keys[slot, :length]
                    gathered_v[i, :length] = self._values[slot, :length]
            return gathered_k, gathered_v

    def _validate_slot(self, slot: int) -> None:
        if slot < 0 or slot >= self.max_slots:
            raise ValueError("slot out of range")


__all__ = ["SlotKVCache"]
