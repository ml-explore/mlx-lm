# ABOUTME: Provides thread-safe slot allocation utilities.
# ABOUTME: Manages reusable slot identifiers for batching resources.

from __future__ import annotations

import threading
from collections import deque
from typing import Optional


class SlotAllocator:
    """Allocates stable slot identifiers up to a fixed capacity."""

    def __init__(self, max_slots: int):
        if max_slots <= 0:
            raise ValueError("max_slots must be positive")
        self._max_slots = max_slots
        self._free: deque[int] = deque(range(max_slots))
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._max_slots

    def available(self) -> int:
        with self._lock:
            return len(self._free)

    def alloc(self) -> Optional[int]:
        """Reserve the next available slot id, or None if exhausted."""
        with self._lock:
            if not self._free:
                return None
            return self._free.popleft()

    def release(self, slot: int) -> None:
        """Return a slot id to the free pool."""
        if slot < 0 or slot >= self._max_slots:
            raise ValueError("slot out of range")
        with self._lock:
            if slot in self._free:
                return
            self._free.append(slot)


__all__ = ["SlotAllocator"]
