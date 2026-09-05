# Copyright © 2026 IndenScale

import copy
import threading
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import mlx.core as mx

from .base import create_causal_mask
from .cache import ArraysCache, _BaseCache, create_attention_mask


class PageAllocationError(RuntimeError):
    """Raised when the page pool cannot satisfy an allocation."""

    def __init__(
        self,
        message: str,
        *,
        requested_pages: Optional[int] = None,
        available_pages: Optional[int] = None,
        pool_generation: Optional[int] = None,
    ):
        super().__init__(message)
        self.requested_pages = requested_pages
        self.available_pages = available_pages
        self.pool_generation = pool_generation

    @property
    def shortfall_pages(self) -> Optional[int]:
        if self.requested_pages is None or self.available_pages is None:
            return None
        return max(0, self.requested_pages - self.available_pages)


class RequestCapacityError(PageAllocationError):
    """Raised when one request cannot fit even in an otherwise empty pool."""


class InvalidPageReferenceError(RuntimeError):
    """Raised when a page reference is invalid or already released."""


class StaleBlockTableError(RuntimeError):
    """Raised when a block table belongs to another pool generation."""


@dataclass(frozen=True)
class PageAllocatorStats:
    capacity_pages: int
    free_pages: int
    live_pages: int
    shared_pages: int
    references: int


class PageAllocator:
    """Allocate fixed-size pages and track their reference counts."""

    def __init__(self, capacity_pages: int):
        if capacity_pages <= 0:
            raise ValueError("capacity_pages must be greater than zero")
        self.capacity_pages = capacity_pages
        self._free = list(range(capacity_pages - 1, -1, -1))
        self._refcounts = [0] * capacity_pages
        self._lock = threading.RLock()

    def _validate_pages(self, page_ids: Sequence[int], *, require_live: bool):
        if len(page_ids) != len(set(page_ids)):
            raise InvalidPageReferenceError("page_ids must be unique")
        for page_id in page_ids:
            if not isinstance(page_id, int) or not 0 <= page_id < self.capacity_pages:
                raise InvalidPageReferenceError(f"invalid page id: {page_id}")
            if require_live and self._refcounts[page_id] == 0:
                raise InvalidPageReferenceError(f"page {page_id} is not live")

    def allocate(self, count: int = 1) -> Tuple[int, ...]:
        if count < 0:
            raise ValueError("count must not be negative")
        if count == 0:
            return ()
        with self._lock:
            if count > len(self._free):
                raise PageAllocationError(
                    f"requested {count} pages with {len(self._free)} available",
                    requested_pages=count,
                    available_pages=len(self._free),
                )
            pages = tuple(self._free.pop() for _ in range(count))
            for page_id in pages:
                self._refcounts[page_id] = 1
            return pages

    def retain(self, page_ids: Sequence[int]):
        page_ids = tuple(page_ids)
        with self._lock:
            self._validate_pages(page_ids, require_live=True)
            for page_id in page_ids:
                self._refcounts[page_id] += 1

    def release(self, page_ids: Sequence[int]):
        page_ids = tuple(page_ids)
        with self._lock:
            self._validate_pages(page_ids, require_live=True)
            for page_id in page_ids:
                self._refcounts[page_id] -= 1
                if self._refcounts[page_id] == 0:
                    self._free.append(page_id)

    def refcount(self, page_id: int) -> int:
        with self._lock:
            self._validate_pages((page_id,), require_live=False)
            return self._refcounts[page_id]

    def stats(self) -> PageAllocatorStats:
        with self._lock:
            free_pages = len(self._free)
            live_pages = self.capacity_pages - free_pages
            shared_pages = sum(refcount > 1 for refcount in self._refcounts)
            references = sum(self._refcounts)
            return PageAllocatorStats(
                capacity_pages=self.capacity_pages,
                free_pages=free_pages,
                live_pages=live_pages,
                shared_pages=shared_pages,
                references=references,
            )


@dataclass(frozen=True)
class BlockTable:
    pool_generation: int
    page_size: int
    page_ids: Tuple[int, ...]
    num_tokens: int


@dataclass(frozen=True)
class PagedKVView:
    block_table: BlockTable
    keys: mx.array
    values: mx.array


@dataclass(frozen=True)
class PagedCacheManagerStats:
    full_attention_layers: int
    capacity_pages: int
    free_pages: int
    live_pages: int
    reserved_pages: int
    shared_pages: int
    references: int
    allocated_bytes: int


class KVBlockPool:
    """Own the physical key and value pages for one attention layer."""

    _next_generation = 1
    _generation_lock = threading.Lock()

    def __init__(
        self,
        *,
        capacity_pages: int,
        page_size: int,
        num_kv_heads: int,
        key_head_dim: int,
        value_head_dim: Optional[int] = None,
        dtype: mx.Dtype = mx.float16,
    ):
        for name, value in (
            ("page_size", page_size),
            ("num_kv_heads", num_kv_heads),
            ("key_head_dim", key_head_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if value_head_dim is None:
            value_head_dim = key_head_dim
        if value_head_dim <= 0:
            raise ValueError("value_head_dim must be greater than zero")

        with self._generation_lock:
            self.generation = self._next_generation
            type(self)._next_generation += 1

        self.page_size = page_size
        self.num_kv_heads = num_kv_heads
        self.key_head_dim = key_head_dim
        self.value_head_dim = value_head_dim
        self.dtype = dtype
        self.allocator = PageAllocator(capacity_pages)
        self._reservations = {}
        self._reservation_lock = threading.RLock()
        self.keys = mx.zeros(
            (capacity_pages, num_kv_heads, page_size, key_head_dim), dtype=dtype
        )
        self.values = mx.zeros(
            (capacity_pages, num_kv_heads, page_size, value_head_dim), dtype=dtype
        )

    @property
    def capacity_pages(self) -> int:
        return self.allocator.capacity_pages

    @property
    def bytes_per_page(self) -> int:
        return (self.keys.nbytes + self.values.nbytes) // self.capacity_pages

    def validate_table(self, table: BlockTable):
        if table.pool_generation != self.generation:
            raise StaleBlockTableError(
                f"block table generation {table.pool_generation} does not match "
                f"pool generation {self.generation}"
            )
        if table.page_size != self.page_size:
            raise StaleBlockTableError(
                f"block table page size {table.page_size} does not match "
                f"pool page size {self.page_size}"
            )
        if table.num_tokens < 0:
            raise InvalidPageReferenceError("block table token count is negative")
        expected_pages = (table.num_tokens + self.page_size - 1) // self.page_size
        if expected_pages != len(table.page_ids):
            raise InvalidPageReferenceError(
                "block table page count does not match its token count"
            )
        if len(table.page_ids) != len(set(table.page_ids)):
            raise InvalidPageReferenceError("block table contains duplicate pages")
        for page_id in table.page_ids:
            if self.allocator.refcount(page_id) == 0:
                raise InvalidPageReferenceError(f"page {page_id} is not live")

    def copy_page(self, source: int, destination: int, num_tokens: int):
        if not 0 <= num_tokens <= self.page_size:
            raise ValueError("num_tokens must fit in one page")
        if self.allocator.refcount(source) == 0:
            raise InvalidPageReferenceError(f"page {source} is not live")
        if self.allocator.refcount(destination) == 0:
            raise InvalidPageReferenceError(f"page {destination} is not live")
        if num_tokens == 0:
            return
        self.keys[destination : destination + 1, :, :num_tokens, :] = self.keys[
            source : source + 1, :, :num_tokens, :
        ]
        self.values[destination : destination + 1, :, :num_tokens, :] = self.values[
            source : source + 1, :, :num_tokens, :
        ]

    def reserve(self, owner, count: int):
        if owner is None:
            raise ValueError("reservation owner must not be None")
        if count < 0:
            raise ValueError("reservation count must not be negative")
        with self._reservation_lock:
            if owner in self._reservations:
                raise ValueError(f"reservation already exists for {owner!r}")
            self._reservations[owner] = list(self._allocate_pages(count))

    def _allocate_pages(self, count: int):
        try:
            return self.allocator.allocate(count)
        except PageAllocationError as error:
            raise PageAllocationError(
                str(error),
                requested_pages=error.requested_pages,
                available_pages=error.available_pages,
                pool_generation=self.generation,
            ) from error

    def allocate_for(self, owner, count: int):
        if owner is None:
            return self._allocate_pages(count), False
        with self._reservation_lock:
            reserved = self._reservations.get(owner)
            if reserved is None:
                return self.allocator.allocate(count), False
            if count > len(reserved):
                raise PageAllocationError(
                    f"reservation {owner!r} has {len(reserved)} pages, "
                    f"requested {count}",
                    requested_pages=count,
                    available_pages=len(reserved),
                    pool_generation=self.generation,
                )
            pages = tuple(reserved[:count])
            del reserved[:count]
            return pages, True

    def restore_allocation(self, owner, page_ids, was_reserved: bool):
        page_ids = tuple(page_ids)
        if not page_ids:
            return
        if not was_reserved:
            self.allocator.release(page_ids)
            return
        with self._reservation_lock:
            reserved = self._reservations.get(owner)
            if reserved is None:
                self.allocator.release(page_ids)
            else:
                reserved[:0] = page_ids

    def cancel_reservation(self, owner) -> int:
        with self._reservation_lock:
            reserved = self._reservations.pop(owner, None)
            if reserved is None:
                return 0
            if reserved:
                self.allocator.release(reserved)
            return len(reserved)

    @property
    def reserved_pages(self) -> int:
        with self._reservation_lock:
            return sum(len(pages) for pages in self._reservations.values())

    def stats(self) -> PageAllocatorStats:
        return self.allocator.stats()


class PagedKVCache(_BaseCache):
    """A single-sequence KV cache backed by fixed-size physical pages."""

    def __init__(
        self,
        pool: KVBlockPool,
        *,
        sequence_id: Optional[str] = None,
        reservation_id=None,
    ):
        self.pool = pool
        self.sequence_id = sequence_id
        self.reservation_id = reservation_id
        self.offset = 0
        self._page_ids = []
        self._page_table_version = 0
        self._device_page_ids = None
        self._device_page_ids_version = -1
        self._released = False

    def _ensure_live(self):
        if self._released:
            raise InvalidPageReferenceError("cache has been released")

    def _validate_update(self, keys: mx.array, values: mx.array):
        self._ensure_live()
        if keys.ndim != 4 or values.ndim != 4:
            raise ValueError("keys and values must have rank 4")
        if keys.shape[:3] != values.shape[:3]:
            raise ValueError("keys and values must share batch, head, and token axes")
        if keys.shape[0] != 1:
            raise ValueError("PagedKVCache supports one sequence")
        if keys.shape[1] != self.pool.num_kv_heads:
            raise ValueError("key head count does not match the pool")
        if keys.shape[3] != self.pool.key_head_dim:
            raise ValueError("key head dimension does not match the pool")
        if values.shape[3] != self.pool.value_head_dim:
            raise ValueError("value head dimension does not match the pool")
        if keys.dtype != self.pool.dtype or values.dtype != self.pool.dtype:
            raise ValueError("key and value dtype must match the pool")

    @property
    def block_table(self) -> BlockTable:
        self._ensure_live()
        return BlockTable(
            pool_generation=self.pool.generation,
            page_size=self.pool.page_size,
            page_ids=tuple(self._page_ids),
            num_tokens=self.offset,
        )

    def device_page_ids(self):
        self._ensure_live()
        if self._device_page_ids_version != self._page_table_version:
            self._device_page_ids = mx.array(self._page_ids, dtype=mx.uint32)
            self._device_page_ids_version = self._page_table_version
        return self._device_page_ids

    def view(self) -> PagedKVView:
        table = self.block_table
        self.pool.validate_table(table)
        return PagedKVView(table, self.pool.keys, self.pool.values)

    def _append(self, keys: mx.array, values: mx.array):
        self._validate_update(keys, values)
        num_tokens = keys.shape[2]
        if num_tokens == 0:
            return

        page_size = self.pool.page_size
        old_num_pages = len(self._page_ids)
        new_num_pages = (self.offset + num_tokens + page_size - 1) // page_size
        tail_size = self.offset % page_size
        copy_tail = (
            tail_size > 0
            and old_num_pages > 0
            and self.pool.allocator.refcount(self._page_ids[-1]) > 1
        )
        allocate_count = new_num_pages - old_num_pages + int(copy_tail)
        if allocate_count:
            allocated_pages, was_reserved = self.pool.allocate_for(
                self.reservation_id, allocate_count
            )
        else:
            allocated_pages, was_reserved = (), False
        allocated = list(allocated_pages)
        new_page_ids = list(self._page_ids)
        old_tail = None
        try:
            if copy_tail:
                old_tail = new_page_ids[-1]
                new_tail = allocated.pop(0)
                self.pool.copy_page(old_tail, new_tail, tail_size)
                new_page_ids[-1] = new_tail
            new_page_ids.extend(allocated)

            source_offset = 0
            while source_offset < num_tokens:
                logical_position = self.offset + source_offset
                page_index, page_offset = divmod(logical_position, page_size)
                write_size = min(page_size - page_offset, num_tokens - source_offset)
                page_id = new_page_ids[page_index]
                self.pool.keys[
                    page_id : page_id + 1,
                    :,
                    page_offset : page_offset + write_size,
                    :,
                ] = keys[..., source_offset : source_offset + write_size, :]
                self.pool.values[
                    page_id : page_id + 1,
                    :,
                    page_offset : page_offset + write_size,
                    :,
                ] = values[..., source_offset : source_offset + write_size, :]
                source_offset += write_size
        except Exception:
            self.pool.restore_allocation(
                self.reservation_id, allocated_pages, was_reserved
            )
            raise

        self._page_ids = new_page_ids
        if allocate_count:
            self._page_table_version += 1
        self.offset += num_tokens
        if old_tail is not None:
            self.pool.allocator.release((old_tail,))

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        self._append(keys, values)
        return self.gather()

    def gather(self):
        self._ensure_live()
        if not self._page_ids:
            return None, None
        page_ids = self.device_page_ids()
        keys = mx.take(self.pool.keys, page_ids, axis=0)
        values = mx.take(self.pool.values, page_ids, axis=0)
        keys = keys.transpose(1, 0, 2, 3).reshape(
            1, self.pool.num_kv_heads, -1, self.pool.key_head_dim
        )[..., : self.offset, :]
        values = values.transpose(1, 0, 2, 3).reshape(
            1, self.pool.num_kv_heads, -1, self.pool.value_head_dim
        )[..., : self.offset, :]
        return keys, values

    def fork(
        self, *, num_tokens: Optional[int] = None, sequence_id: Optional[str] = None
    ):
        self._ensure_live()
        if num_tokens is None:
            num_tokens = self.offset
        if not 0 <= num_tokens <= self.offset:
            raise ValueError("num_tokens must be within the cached sequence")
        num_pages = (num_tokens + self.pool.page_size - 1) // self.pool.page_size
        page_ids = self._page_ids[:num_pages]
        self.pool.allocator.retain(page_ids)
        child = type(self)(
            self.pool,
            sequence_id=sequence_id,
            reservation_id=self.reservation_id,
        )
        child.offset = num_tokens
        child._page_ids = list(page_ids)
        child._page_table_version += 1
        return child

    def __deepcopy__(self, memo):
        cache = self.fork(sequence_id=self.sequence_id)
        memo[id(self)] = cache
        return cache

    @classmethod
    def merge(cls, caches):
        return BatchPagedKVCache.merge(caches)

    def trim(self, num_tokens: int):
        self._ensure_live()
        if num_tokens < 0:
            raise ValueError("num_tokens must not be negative")
        trimmed = min(self.offset, num_tokens)
        self.offset -= trimmed
        num_pages = (self.offset + self.pool.page_size - 1) // self.pool.page_size
        released = self._page_ids[num_pages:]
        if released:
            self.pool.allocator.release(released)
            del self._page_ids[num_pages:]
            self._page_table_version += 1
        return trimmed

    def release(self) -> bool:
        if self._released:
            return False
        if self._page_ids:
            self.pool.allocator.release(self._page_ids)
        self._page_ids = []
        self._page_table_version += 1
        self.offset = 0
        self._released = True
        return True

    close = release

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass

    def size(self):
        return self.offset

    def is_trimmable(self):
        return True

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self):
        return self.offset == 0

    @property
    def nbytes(self):
        return len(self._page_ids) * self.pool.bytes_per_page

    @property
    def state(self):
        return self.gather()

    @property
    def eval_state(self):
        return self.pool.keys, self.pool.values


class BatchPagedKVCache(_BaseCache):
    """Reference batch view over independent paged sequence caches."""

    def __init__(self, caches: Sequence[PagedKVCache]):
        if not caches:
            raise ValueError("caches must not be empty")
        pool = caches[0].pool
        if any(cache.pool is not pool for cache in caches):
            raise ValueError("all paged caches in a batch must share one pool")
        self.pool = pool
        self.caches = list(caches)
        self._block_tables = None
        self._block_table_versions = None
        self._remaining_lengths = None
        self._released = False

    def _ensure_live(self):
        if self._released:
            raise InvalidPageReferenceError("batch cache has been released")

    @property
    def offset(self):
        self._ensure_live()
        return mx.array([cache.offset for cache in self.caches])

    @classmethod
    def merge(cls, caches: Sequence[PagedKVCache]):
        if not caches:
            raise ValueError("caches must not be empty")
        return cls([cache.fork(sequence_id=cache.sequence_id) for cache in caches])

    def __deepcopy__(self, memo):
        self._ensure_live()
        batch = type(self)([copy.deepcopy(cache) for cache in self.caches])
        if self._remaining_lengths is not None:
            batch._remaining_lengths = list(self._remaining_lengths)
        memo[id(self)] = batch
        return batch

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        self._ensure_live()
        if left_padding is not None and any(left_padding):
            raise ValueError("Paged KV stores sequences without physical left padding")
        if lengths is not None:
            if len(lengths) != len(self.caches):
                raise ValueError("lengths must have one entry per sequence")
            if any(length < 0 for length in lengths):
                raise ValueError("lengths must not be negative")
            self._remaining_lengths = list(lengths)

    def finalize(self):
        self._ensure_live()
        if self._remaining_lengths is not None and any(self._remaining_lengths):
            raise ValueError("not all prepared tokens were appended")
        self._remaining_lengths = None

    def _validate_update(self, keys: mx.array, values: mx.array):
        self._ensure_live()
        if keys.ndim != 4 or values.ndim != 4:
            raise ValueError("keys and values must have rank 4")
        if keys.shape[0] != len(self.caches) or values.shape[0] != len(self.caches):
            raise ValueError("batch size does not match the paged cache")
        if keys.shape[1] != values.shape[1]:
            raise ValueError("keys and values must have the same head count")
        if keys.shape[2] != values.shape[2]:
            raise ValueError("keys and values must have the same token count")
        if keys.shape[1] != self.pool.num_kv_heads:
            raise ValueError("key head count does not match the pool")
        if keys.shape[3] != self.pool.key_head_dim:
            raise ValueError("key head dimension does not match the pool")
        if values.shape[3] != self.pool.value_head_dim:
            raise ValueError("value head dimension does not match the pool")
        if keys.dtype != self.pool.dtype or values.dtype != self.pool.dtype:
            raise ValueError("key and value dtype must match the pool")

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        self._validate_update(keys, values)
        # Check the whole batch before any row consumes pages or advances.
        with self.pool._reservation_lock:
            required = {}
            for index, cache in enumerate(self.caches):
                cache._ensure_live()
                count = keys.shape[2]
                if self._remaining_lengths is not None:
                    count = min(count, self._remaining_lengths[index])
                if count == 0:
                    continue
                size = self.pool.page_size
                pages = (cache.offset + count + size - 1) // size - len(cache._page_ids)
                if (
                    cache.offset % size
                    and self.pool.allocator.refcount(cache._page_ids[-1]) > 1
                ):
                    pages += 1
                owner = cache.reservation_id
                if owner not in self.pool._reservations:
                    owner = None
                required[owner] = required.get(owner, 0) + pages
            for owner, count in required.items():
                available = (
                    len(self.pool._reservations[owner])
                    if owner in self.pool._reservations
                    else self.pool.stats().free_pages
                )
                if count > available:
                    raise PageAllocationError(
                        f"batch requires {count} pages with {available} available",
                        requested_pages=count,
                        available_pages=available,
                        pool_generation=self.pool.generation,
                    )
            return self._update_and_fetch(keys, values)

    def _update_and_fetch(self, keys: mx.array, values: mx.array):
        self._validate_update(keys, values)
        step_tokens = keys.shape[2]
        history = self.gather() if self._remaining_lengths is not None else None
        for index, cache in enumerate(self.caches):
            valid_tokens = step_tokens
            if self._remaining_lengths is not None:
                valid_tokens = min(valid_tokens, self._remaining_lengths[index])
                self._remaining_lengths[index] -= valid_tokens
            if valid_tokens > 0:
                cache._append(
                    keys[index : index + 1, :, :valid_tokens, :],
                    values[index : index + 1, :, :valid_tokens, :],
                )
        if history is not None:
            history_keys, history_values = history
            return (
                mx.concatenate([history_keys, keys], axis=2),
                mx.concatenate([history_values, values], axis=2),
            )
        return self.gather()

    def gather(self):
        self._ensure_live()
        if not self.caches:
            return (
                mx.zeros(
                    (0, self.pool.num_kv_heads, 0, self.pool.key_head_dim),
                    dtype=self.pool.dtype,
                ),
                mx.zeros(
                    (0, self.pool.num_kv_heads, 0, self.pool.value_head_dim),
                    dtype=self.pool.dtype,
                ),
            )
        max_length = self.size()
        keys = []
        values = []
        for cache in self.caches:
            row_keys, row_values = cache.gather()
            if row_keys is None:
                row_keys = mx.zeros(
                    (1, self.pool.num_kv_heads, 0, self.pool.key_head_dim),
                    dtype=self.pool.dtype,
                )
                row_values = mx.zeros(
                    (1, self.pool.num_kv_heads, 0, self.pool.value_head_dim),
                    dtype=self.pool.dtype,
                )
            left_padding = max_length - cache.offset
            if left_padding:
                key_padding = [(0, 0), (0, 0), (left_padding, 0), (0, 0)]
                value_padding = [(0, 0), (0, 0), (left_padding, 0), (0, 0)]
                row_keys = mx.pad(row_keys, key_padding)
                row_values = mx.pad(row_values, value_padding)
            keys.append(row_keys)
            values.append(row_values)
        return mx.concatenate(keys, axis=0), mx.concatenate(values, axis=0)

    def filter(self, batch_indices: Sequence[int]):
        self._ensure_live()
        batch_indices = list(batch_indices)
        if len(batch_indices) != len(set(batch_indices)):
            raise ValueError("batch_indices must be unique")
        if any(index < 0 or index >= len(self.caches) for index in batch_indices):
            raise IndexError("batch index out of range")
        keep = set(batch_indices)
        for index, cache in enumerate(self.caches):
            if index not in keep:
                cache.release()
        self.caches = [self.caches[index] for index in batch_indices]
        self._block_table_versions = None
        if self._remaining_lengths is not None:
            self._remaining_lengths = [
                self._remaining_lengths[index] for index in batch_indices
            ]

    def extend(self, other):
        self._ensure_live()
        other._ensure_live()
        if self.pool is not other.pool:
            raise ValueError("batch caches must share one pool")
        if self._remaining_lengths is not None or other._remaining_lengths is not None:
            raise ValueError("cannot extend a prepared paged batch")
        self.caches.extend(other.caches)
        self._block_table_versions = None
        other.caches = []
        other._released = True

    def extract(self, index: int):
        self._ensure_live()
        return self.caches[index].fork(sequence_id=self.caches[index].sequence_id)

    def release(self) -> bool:
        if self._released:
            return False
        for cache in self.caches:
            cache.release()
        self.caches = []
        self._block_tables = None
        self._block_table_versions = None
        self._released = True
        return True

    close = release

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass

    def size(self):
        self._ensure_live()
        return max((cache.offset for cache in self.caches), default=0)

    def make_mask(self, num_queries: int, return_array: bool = False, **kwargs):
        self._ensure_live()
        left_padding = mx.array([self.size() - cache.offset for cache in self.caches])
        return create_causal_mask(
            num_queries,
            offset=self.size(),
            left_padding=left_padding,
            **kwargs,
        )

    def empty(self):
        self._ensure_live()
        return all(cache.empty() for cache in self.caches)

    @property
    def nbytes(self):
        self._ensure_live()
        return sum(cache.nbytes for cache in self.caches)

    @property
    def state(self):
        keys, values = self.gather()
        return keys, values, self.offset

    @property
    def eval_state(self):
        return self.pool.keys, self.pool.values


class QwenHybridPagedKVCacheManager:
    """Create shared paged caches for Qwen3.6 and Qwen3.8 requests."""

    model_types = frozenset(("qwen3_5", "qwen3_5_moe"))
    target_architectures = {
        "qwen3_5_moe": {
            "num_hidden_layers": 40,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "full_attention_interval": 4,
        },
        "qwen3_5": {
            "num_hidden_layers": 64,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "full_attention_interval": 4,
        },
    }

    def __init__(
        self,
        model,
        *,
        capacity_pages: int,
        page_size: int = 64,
        dtype: Optional[mx.Dtype] = None,
        strict_architecture: bool = False,
    ):
        model_type = getattr(model, "model_type", None)
        if model_type not in self.model_types:
            raise ValueError(
                f"unsupported model type {model_type!r}; expected qwen3_5 or "
                "qwen3_5_moe"
            )

        text_model = getattr(model, "language_model", model)
        layers = list(text_model.layers)
        if strict_architecture:
            self._validate_target_architecture(model_type, text_model, layers)
        if dtype is None:
            dtype = layers[0].input_layernorm.weight.dtype
        self.model_type = model_type
        self.capacity_pages = capacity_pages
        self.page_size = page_size
        self.dtype = dtype
        self._pools = []
        self._admitted = set()
        self._admission_metadata = {}
        self._admission_lock = threading.RLock()
        for layer in layers:
            if layer.is_linear:
                self._pools.append(None)
                continue
            attention = layer.self_attn
            self._pools.append(
                KVBlockPool(
                    capacity_pages=capacity_pages,
                    page_size=page_size,
                    num_kv_heads=attention.num_key_value_heads,
                    key_head_dim=attention.head_dim,
                    value_head_dim=attention.head_dim,
                    dtype=dtype,
                )
            )

        if not any(pool is not None for pool in self._pools):
            raise ValueError("model does not contain a full-attention layer")

    @classmethod
    def _validate_target_architecture(cls, model_type, text_model, layers):
        args = getattr(text_model, "args", None)
        if args is None:
            raise ValueError("target Qwen model must expose architecture arguments")
        expected = cls.target_architectures[model_type]
        for field, expected_value in expected.items():
            actual_value = getattr(args, field, None)
            if actual_value != expected_value:
                raise ValueError(
                    f"unsupported {model_type} architecture: {field}={actual_value!r}, "
                    f"expected {expected_value!r}"
                )
        expected_full_layers = tuple(
            range(
                expected["full_attention_interval"] - 1,
                expected["num_hidden_layers"],
                expected["full_attention_interval"],
            )
        )
        actual_full_layers = tuple(
            index for index, layer in enumerate(layers) if not layer.is_linear
        )
        if actual_full_layers != expected_full_layers:
            raise ValueError(
                f"unsupported {model_type} full-attention schedule: "
                f"{actual_full_layers!r}, expected {expected_full_layers!r}"
            )

    @property
    def num_layers(self) -> int:
        return len(self._pools)

    @property
    def num_full_attention_layers(self) -> int:
        return sum(pool is not None for pool in self._pools)

    @property
    def pool_generations(self) -> Tuple[int, ...]:
        return tuple(pool.generation for pool in self._pools if pool is not None)

    def materialize(self):
        """Allocate physical page storage on the manager's owning thread.

        MLX arrays are lazy. Serving runtimes may construct the manager on a
        model-loading thread and run decode on a dedicated worker thread; an
        unevaluated ``mx.zeros`` would otherwise carry the loading thread's
        Metal stream into the first request.
        """
        storage = [
            array
            for pool in self._pools
            if pool is not None
            for array in (pool.keys, pool.values)
        ]
        if storage:
            mx.eval(*storage)

    def make_cache(self, *, sequence_id: Optional[str] = None):
        caches = []
        for layer_index, pool in enumerate(self._pools):
            if pool is None:
                caches.append(ArraysCache(size=2))
            else:
                layer_sequence_id = (
                    f"{sequence_id}:{layer_index}" if sequence_id is not None else None
                )
                caches.append(PagedKVCache(pool, sequence_id=layer_sequence_id))
        return caches

    def admit(
        self,
        uid,
        prompt_tokens: int,
        max_tokens: int,
        cache: Sequence[_BaseCache],
    ):
        segment_lengths = (
            (prompt_tokens - 1, 1) if prompt_tokens > 1 else (prompt_tokens,)
        )
        self.admit_segments(uid, segment_lengths, max_tokens, cache)

    def admit_segments(
        self,
        uid,
        segment_lengths: Sequence[int],
        max_tokens: int,
        cache: Sequence[_BaseCache],
    ):
        segment_lengths = tuple(segment_lengths)
        if not segment_lengths or any(length <= 0 for length in segment_lengths):
            raise ValueError("segment lengths must be positive")
        prompt_tokens = sum(segment_lengths)
        if prompt_tokens <= 0:
            raise ValueError("prompt_tokens must be greater than zero")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")
        self.validate_cache(cache)
        with self._admission_lock:
            if uid in self._admitted:
                raise ValueError(f"request {uid!r} is already admitted")

            reservations = []
            updated_caches = []
            additional_tokens = prompt_tokens + max_tokens
            reservation_plan = []
            for layer_index, (layer_cache, pool) in enumerate(zip(cache, self._pools)):
                if pool is None:
                    continue
                current_pages = len(layer_cache.block_table.page_ids)
                target_pages = (
                    layer_cache.offset + additional_tokens + self.page_size - 1
                ) // self.page_size
                reserve_pages = target_pages - current_pages
                if (
                    additional_tokens > 0
                    and layer_cache.offset % self.page_size
                    and current_pages > 0
                ):
                    reserve_pages += 1
                # Prefix snapshots can share partial tails at segment boundaries.
                # Reserve every possible COW replacement during admission.
                cumulative_tokens = 0
                for segment_length in segment_lengths[:-1]:
                    cumulative_tokens += segment_length
                    boundary_offset = layer_cache.offset + cumulative_tokens
                    if boundary_offset % self.page_size:
                        reserve_pages += 1
                if current_pages + reserve_pages > pool.capacity_pages:
                    raise RequestCapacityError(
                        f"request requires {current_pages + reserve_pages} pages "
                        f"in full-attention layer {layer_index}, pool capacity is "
                        f"{pool.capacity_pages}"
                    )
                reservation_plan.append((layer_index, layer_cache, pool, reserve_pages))
            try:
                for layer_index, layer_cache, pool, reserve_pages in reservation_plan:
                    pool.reserve(uid, reserve_pages)
                    reservations.append(pool)
                    updated_caches.append(
                        (
                            layer_cache,
                            layer_cache.reservation_id,
                            layer_cache.sequence_id,
                        )
                    )
                    layer_cache.reservation_id = uid
                    layer_cache.sequence_id = f"{uid}:{layer_index}"
            except Exception:
                for pool in reservations:
                    pool.cancel_reservation(uid)
                for layer_cache, reservation_id, sequence_id in updated_caches:
                    layer_cache.reservation_id = reservation_id
                    layer_cache.sequence_id = sequence_id
                raise
            self._admitted.add(uid)
            self._admission_metadata[uid] = tuple(
                (reservation_id, sequence_id)
                for _, reservation_id, sequence_id in updated_caches
            )

    def release_admission(self, uid) -> bool:
        with self._admission_lock:
            if uid not in self._admitted:
                return False
            for pool in self._pools:
                if pool is not None:
                    pool.cancel_reservation(uid)
            self._admitted.remove(uid)
            self._admission_metadata.pop(uid, None)
            return True

    def rollback_admission(self, uid, cache: Sequence[_BaseCache]) -> bool:
        self.validate_cache(cache)
        with self._admission_lock:
            if uid not in self._admitted:
                return False
            for pool in self._pools:
                if pool is not None:
                    pool.cancel_reservation(uid)
            metadata = iter(self._admission_metadata.pop(uid))
            for layer_cache, pool in zip(cache, self._pools):
                if pool is None:
                    continue
                layer_cache.reservation_id, layer_cache.sequence_id = next(metadata)
            self._admitted.remove(uid)
            return True

    def validate_cache(self, cache: Sequence[_BaseCache]):
        if len(cache) != self.num_layers:
            raise ValueError("cache layer count does not match the manager")
        for layer_cache, pool in zip(cache, self._pools):
            if pool is None:
                if not isinstance(layer_cache, ArraysCache):
                    raise ValueError("linear-attention cache type does not match")
            elif not isinstance(layer_cache, PagedKVCache):
                raise ValueError("full-attention cache type does not match")
            elif layer_cache.pool is not pool:
                raise ValueError("paged cache belongs to another manager")

    def release(self, cache: Sequence[_BaseCache]):
        self.validate_cache(cache)
        for layer_cache, pool in zip(cache, self._pools):
            if pool is not None:
                layer_cache.release()

    def free_pages_for_pool(self, pool_generation: int) -> Optional[int]:
        for pool in self._pools:
            if pool is not None and pool.generation == pool_generation:
                return pool.stats().free_pages
        return None

    def stats(self) -> PagedCacheManagerStats:
        pools = [pool for pool in self._pools if pool is not None]
        pool_stats = [pool.stats() for pool in pools]
        return PagedCacheManagerStats(
            full_attention_layers=len(pools),
            capacity_pages=sum(stat.capacity_pages for stat in pool_stats),
            free_pages=sum(stat.free_pages for stat in pool_stats),
            live_pages=sum(stat.live_pages for stat in pool_stats),
            reserved_pages=sum(pool.reserved_pages for pool in pools),
            shared_pages=sum(stat.shared_pages for stat in pool_stats),
            references=sum(stat.references for stat in pool_stats),
            allocated_bytes=sum(
                pool.capacity_pages * pool.bytes_per_page for pool in pools
            ),
        )
