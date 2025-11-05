# ABOUTME: Orchestrates slot-based batch state for decode steps.
# ABOUTME: Provides scaffolding for vectorized decode using slot ids.

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np

from .slot_allocator import SlotAllocator
from .slot_kv_cache import SlotKVCache
from .sampling import select_tokens_argmax


@dataclass
class SlotRecord:
    seq_id: str
    slot_id: int
    finished: bool = False


class SlotBatcher:
    """High-level slot manager for batched decode scaffolding.

    Note: Stage-2 implementation uses numpy arrays to enable unit tests without
    requiring MX kernels. Actual runtime will swap in MX operations.
    """

    def __init__(
        self,
        allocator: SlotAllocator,
        kv_cache: SlotKVCache,
        sampler: Callable[[np.ndarray], np.ndarray] = select_tokens_argmax,
    ):
        self.allocator = allocator
        self.kv_cache = kv_cache
        self.sampler = sampler
        self._records: dict[str, SlotRecord] = {}

    def register(self, seq_id: str) -> int:
        slot = self.allocator.alloc()
        if slot is None:
            raise RuntimeError("No slot capacity available")
        self._records[seq_id] = SlotRecord(seq_id=seq_id, slot_id=slot)
        self.kv_cache.reset(slot)
        return slot

    def release(self, seq_id: str) -> None:
        rec = self._records.pop(seq_id, None)
        if rec is None:
            return
        self.kv_cache.reset(rec.slot_id)
        self.allocator.release(rec.slot_id)

    def active_slots(self, seq_ids: Sequence[str]) -> List[int]:
        slots = []
        for seq in seq_ids:
            rec = self._records.get(seq)
            if rec is None:
                raise KeyError(f"Sequence {seq} not registered")
            slots.append(rec.slot_id)
        return slots

    def decode_with_logits(
        self,
        seq_ids: Sequence[str],
        logits: np.ndarray,
    ) -> np.ndarray:
        """Select tokens for each seq_id given `[B, V]` logits."""
        if logits.ndim != 2 or logits.shape[0] != len(seq_ids):
            raise ValueError("logits shape must be [B, V] aligned with seq_ids")
        return self.sampler(logits)


__all__ = ["SlotBatcher", "SlotRecord"]
