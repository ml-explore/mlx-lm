# ABOUTME: Provides paged KV cache utilities backed by KVBlockManager.
# ABOUTME: Offers batch views compatible with mx.fast.paged_attention inputs.

from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import mlx.core as mx

from .safe_eval import safe_eval


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.lower() not in {"", "0", "false", "off", "no"}


_PAGED_TRACE = _env_flag("MLXLM_PAGED_TRACE")
_PAGED_LOG = logging.getLogger("mlx_lm.paged")


def _array_preview(arr, limit: int = 6):
    if arr is None:
        return None
    try:
        safe_eval(arr)
        data = arr.tolist()
    except Exception as exc:  # pragma: no cover - debug helper
        return f"<err:{exc}>"
    if isinstance(data, list) and len(data) > limit:
        preview = list(data[:limit])
        preview.append("...")
        return preview
    return data


def _table_preview(arr, rows: int = 2, cols: int = 4):
    if arr is None:
        return None
    try:
        safe_eval(arr)
        data = arr.tolist()
    except Exception as exc:  # pragma: no cover - debug helper
        return {"shape": tuple(arr.shape), "error": str(exc)}
    preview: List[Any] = []
    for row in data[:rows]:
        if isinstance(row, list) and len(row) > cols:
            truncated = list(row[:cols])
            truncated.append("...")
            preview.append(truncated)
        else:
            preview.append(row)
    if len(data) > rows:
        preview.append(["..."])
    return {"shape": tuple(arr.shape), "preview": preview}


def _seq_preview(values: Sequence[int], limit: int = 6):
    seq = list(values)
    if len(seq) > limit:
        head = list(seq[:limit])
        head.append("...")
        return head
    return seq


@dataclass(frozen=True)
class BackendSignature:
    block_size: int
    vec_width: Optional[int]
    threads_per_head: Optional[int]
    kv_quant_mode: Optional[str]
    model_signature: Optional[str]
    pool_id: int = 0


@dataclass(frozen=True)
class ViewSignature:
    seq_ids: Tuple[int, ...]
    seq_versions: Tuple[int, ...]
    backend: BackendSignature
    kv_mapping_token: Optional[int] = field(default=None)
    manager_epoch: int = 0


class PagedSlotKVCache:
    """Tracks per-request sequence ids backed by a shared KVBlockManager."""

    def __init__(
        self,
        manager,
        *,
        max_active: int,
        backend_signature: Optional[BackendSignature] = None,
    ):
        self.manager = manager
        self.max_active = max_active
        self._request_to_seq: Dict[str, int] = {}
        self._seq_to_request: Dict[int, str] = {}
        self._next_seq_id = 1
        block_size = getattr(manager, "block_size", 16)
        pool_id = manager.pool_id() if hasattr(manager, "pool_id") else 0
        self.backend_signature = backend_signature or BackendSignature(
            block_size=block_size,
            vec_width=None,
            threads_per_head=None,
            kv_quant_mode=None,
            model_signature=None,
            pool_id=pool_id,
        )
        self._seq_versions: Dict[int, int] = {}
        self._seq_lengths: Dict[int, int] = {}

    # ------------------------------------------------------------------#
    # Sequence lifecycle
    # ------------------------------------------------------------------#
    def register(self, request_id: str, prompt_len: int) -> int:
        if request_id in self._request_to_seq:
            raise ValueError(f"request {request_id} already registered")
        seq_id = self._allocate_seq_id()
        self.manager.new_sequence(seq_id, prompt_len)
        self._request_to_seq[request_id] = seq_id
        self._seq_to_request[seq_id] = request_id
        self._seq_versions[seq_id] = 0
        self._seq_lengths[seq_id] = int(prompt_len)
        if _PAGED_TRACE:
            _PAGED_LOG.info(
                "paged.cache.register request=%s seq_id=%s prompt_len=%s max_blocks=%s",
                request_id,
                seq_id,
                prompt_len,
                getattr(self.manager, "max_blocks_per_sequence", None),
            )
        return seq_id

    def sequence_id(self, request_id: str) -> int:
        if request_id not in self._request_to_seq:
            raise KeyError(f"request {request_id} not registered")
        return self._request_to_seq[request_id]

    def release(self, request_id: str) -> None:
        seq_id = self._request_to_seq.pop(request_id, None)
        if seq_id is None:
            return
        self._seq_to_request.pop(seq_id, None)
        self.manager.free(seq_id)
        self._seq_versions.pop(seq_id, None)
        self._seq_lengths.pop(seq_id, None)
        if _PAGED_TRACE:
            _PAGED_LOG.info(
                "paged.cache.release request=%s seq_id=%s", request_id, seq_id
            )

    # ------------------------------------------------------------------#
    # KV writes
    # ------------------------------------------------------------------#
    def write_prefill(
        self,
        seq_id: int,
        layer_idx: int,
        k_chunk: mx.array,
        v_chunk: mx.array,
        start_pos: int,
        *,
        commit: bool = True,
    ) -> None:
        self.manager.write_prefill(
            seq_id, layer_idx, k_chunk, v_chunk, start_pos, commit=commit
        )
        tokens = int(k_chunk.shape[1]) if hasattr(k_chunk, "shape") else 0
        if tokens > 0:
            self._update_seq_length(seq_id, start_pos + tokens)

    def append_decode_token(
        self,
        seq_id: int,
        layer_idx: int,
        k_token: mx.array,
        v_token: mx.array,
    ) -> None:
        self.manager.append_decode_token(seq_id, layer_idx, k_token, v_token)
        self._update_seq_length(seq_id, self._seq_lengths.get(seq_id, 0) + 1)

    # ------------------------------------------------------------------#
    # Batch helpers
    # ------------------------------------------------------------------#
    def make_batch_view(
        self,
        seq_ids: Sequence[int],
        *,
        kv_head_mapping: Optional[mx.array] = None,
        context_override: Optional[Sequence[int]] = None,
    ) -> "PagedBatchView":
        signature = self._build_view_signature(seq_ids, kv_head_mapping)
        return PagedBatchView(
            self.manager,
            seq_ids=tuple(seq_ids),
            kv_head_mapping=kv_head_mapping,
            context_override=context_override,
            signature=signature,
        )

    def begin_prefill(
        self,
        request_id: str,
        chunk_len: int,
        *,
        kv_head_mapping: Optional[mx.array] = None,
    ) -> "_PagedPrefillHandle":
        seq_id = self.sequence_id(request_id)
        base_len, virtual_len = self.manager.prepare_prefill_view(seq_id, chunk_len)
        if _PAGED_TRACE:
            _PAGED_LOG.info(
                "paged.prefill.prepare request=%s seq_id=%s chunk=%s base_len=%s virtual_len=%s",
                request_id,
                seq_id,
                chunk_len,
                base_len,
                virtual_len,
            )
        view = PagedBatchView(
            self.manager,
            seq_ids=(seq_id,),
            kv_head_mapping=kv_head_mapping,
            context_override=[virtual_len],
        )
        view.prefill_base_lens = mx.array([base_len], dtype=mx.int32)
        return _PagedPrefillHandle(seq_id=seq_id, view=view, manager=self.manager)

    def begin_prefill_many(
        self,
        request_ids: Sequence[str],
        chunk_lens: Sequence[int],
        *,
        kv_head_mapping: Optional[mx.array] = None,
    ) -> "_PagedPrefillBatchHandle":
        if len(request_ids) != len(chunk_lens):
            raise ValueError("request_ids and chunk_lens must have the same length")
        if not request_ids:
            raise ValueError("begin_prefill_many expects at least one request id")
        seq_ids: List[int] = []
        base_lens: List[int] = []
        virtual_lens: List[int] = []
        for req_id, chunk in zip(request_ids, chunk_lens):
            seq_id = self.sequence_id(req_id)
            base_len, virtual_len = self.manager.prepare_prefill_view(
                seq_id, int(chunk)
            )
            seq_ids.append(seq_id)
            base_lens.append(base_len)
            virtual_lens.append(virtual_len)
        view = PagedBatchView(
            self.manager,
            seq_ids=seq_ids,
            kv_head_mapping=kv_head_mapping,
            context_override=virtual_lens,
        )
        view.prefill_base_lens = mx.array(base_lens, dtype=mx.int32)
        if _PAGED_TRACE:
            _PAGED_LOG.info(
                "paged.prefill.prepare batch seq_ids=%s chunks=%s base=%s virtual=%s",
                _seq_preview(seq_ids),
                list(chunk_lens),
                base_lens,
                virtual_lens,
            )
        return _PagedPrefillBatchHandle(
            seq_ids=tuple(seq_ids),
            view=view,
            manager=self.manager,
        )

    def batch_tables_from_requests(
        self, request_ids: Sequence[str]
    ) -> Tuple[mx.array, mx.array]:
        seq_ids = [self.sequence_id(req_id) for req_id in request_ids]
        return self.manager.batch_tables(seq_ids)

    # ------------------------------------------------------------------#
    # Internal helpers
    # ------------------------------------------------------------------#
    def _allocate_seq_id(self) -> int:
        seq_id = self._next_seq_id
        self._next_seq_id += 1
        return seq_id

    # ------------------------------------------------------------------#
    # Signature helpers
    # ------------------------------------------------------------------#
    def _build_view_signature(
        self, seq_ids: Sequence[int], kv_head_mapping: Optional[mx.array]
    ) -> Optional[ViewSignature]:
        if not seq_ids:
            return None
        versions = tuple(self._seq_versions.get(seq_id, 0) for seq_id in seq_ids)
        token = self._mapping_token(kv_head_mapping)
        return ViewSignature(
            seq_ids=tuple(seq_ids),
            seq_versions=versions,
            backend=self.backend_signature,
            kv_mapping_token=token,
            manager_epoch=self._manager_epoch(),
        )

    def _mapping_token(self, kv_head_mapping: Optional[mx.array]) -> Optional[int]:
        if kv_head_mapping is None:
            return None
        try:
            safe_eval(kv_head_mapping)
            data = tuple(int(v) for v in kv_head_mapping.tolist())
            return hash(data)
        except Exception:
            return hash(id(kv_head_mapping))

    def _update_seq_length(self, seq_id: int, new_len: int) -> None:
        prev = self._seq_lengths.get(seq_id, 0)
        if new_len <= prev:
            return
        block_size = int(self.backend_signature.block_size or 1)
        prev_block = prev // block_size
        new_block = (new_len - 1) // block_size
        if new_block > prev_block:
            self._seq_versions[seq_id] = self._seq_versions.get(seq_id, 0) + 1
        self._seq_lengths[seq_id] = new_len

    def mark_prefix_reuse(self, seq_id: int, seq_len: int) -> None:
        self._seq_versions[seq_id] = self._seq_versions.get(seq_id, 0) + 1
        self._seq_lengths[seq_id] = seq_len

    def _manager_epoch(self) -> int:
        if hasattr(self.manager, "mapping_epoch"):
            return int(self.manager.mapping_epoch())
        return 0

    def can_bump_view(self, view: "PagedBatchView", deltas: Sequence[int]) -> bool:
        signature = getattr(view, "signature", None)
        if signature is None:
            return False
        if signature.backend != self.backend_signature:
            return False
        if signature.manager_epoch != self._manager_epoch():
            return False
        if tuple(view.seq_ids) != signature.seq_ids:
            return False
        if len(deltas) != len(view.seq_ids):
            return False
        # Verify mapping token has not changed
        if signature.kv_mapping_token != self._mapping_token(view.kv_head_mapping):
            return False
        block_size = int(self.backend_signature.block_size or 1)
        for idx, seq_id in enumerate(view.seq_ids):
            if self._seq_versions.get(seq_id, 0) != signature.seq_versions[idx]:
                return False
            delta = int(deltas[idx])
            if delta < 0:
                return False
            prev_len = self._seq_lengths.get(seq_id, 0)
            new_len = prev_len + delta
            if delta > 0:
                prev_block = prev_len // block_size
                new_block = (new_len - 1) // block_size
                if new_block > prev_block:
                    return False
        return True


class PagedBatchView:
    """Immutable snapshot of block tables/context lengths for a decode batch."""

    def __init__(
        self,
        manager,
        *,
        seq_ids: Sequence[int],
        kv_head_mapping: Optional[mx.array],
        context_override: Optional[Sequence[int]] = None,
        signature: Optional[ViewSignature] = None,
    ):
        self.manager = manager
        self.seq_ids = tuple(seq_ids)
        if self.seq_ids:
            tables, context = manager.batch_tables(
                self.seq_ids, context_override=context_override
            )
        else:
            tables = mx.zeros((0, manager.max_blocks_per_sequence), dtype=mx.int32)
            context = mx.zeros((0,), dtype=mx.int32)
        self.block_tables = tables
        self.context_lens = context
        self.kv_head_mapping = kv_head_mapping
        self.prefill_base_lens = None
        self.signature = signature
        if _PAGED_TRACE:
            _PAGED_LOG.info(
                "paged.batch_view seq_ids=%s context=%s tables=%s override=%s",
                _seq_preview(self.seq_ids),
                _array_preview(self.context_lens),
                _table_preview(self.block_tables),
                context_override,
            )

    def bump_context(self, delta: Sequence[int]) -> None:
        if not delta or not self.seq_ids:
            return
        delta_arr = mx.array(delta, dtype=mx.int32)
        self.context_lens = self.context_lens + delta_arr

    def args_for_layer(
        self,
        layer_idx: int,
    ) -> Tuple[
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        int,
        Optional[mx.array],
        Dict[str, object],
    ]:
        """Return the parameters needed by mx.fast.paged_attention."""
        k_cache = self.manager.k[layer_idx]
        v_cache = self.manager.v[layer_idx]
        quant_kwargs: Dict[str, object] = {}
        helper = getattr(self.manager, "_quant_attention_kwargs", None)
        if helper is not None:
            quant_kwargs = helper(layer_idx) or {}
        return (
            k_cache,
            v_cache,
            self.block_tables,
            self.context_lens,
            layer_idx,
            self.kv_head_mapping,
            quant_kwargs,
        )

    @property
    def size(self) -> int:
        return len(self.seq_ids)


@dataclass
class _PagedPrefillHandle:
    seq_id: int
    view: PagedBatchView
    manager: any

    def commit(self) -> None:
        self.manager.commit_prefill(self.seq_id)
        if _PAGED_TRACE:
            _PAGED_LOG.info("paged.prefill.commit seq_id=%s", self.seq_id)


@dataclass
class _PagedPrefillBatchHandle:
    seq_ids: Tuple[int, ...]
    view: PagedBatchView
    manager: any

    def commit(self) -> None:
        for seq_id in self.seq_ids:
            self.manager.commit_prefill(seq_id)
        if _PAGED_TRACE:
            _PAGED_LOG.info(
                "paged.prefill.commit batch seq_ids=%s",
                _seq_preview(self.seq_ids),
            )


@dataclass
class _PrefixEntry:
    blocks: Tuple[int, ...]
    seq_len: int
    last_used: float


class PrefixCache:
    """Caches prefix block mappings so prefill for repeats can be skipped."""

    def __init__(
        self,
        manager,
        *,
        block_size: int,
        max_entries: int = 512,
        reuse_callback: Optional[Callable[[int, int], None]] = None,
    ):
        self.manager = manager
        self.block_size = block_size
        self.max_entries = max_entries
        self._entries: "OrderedDict[bytes, _PrefixEntry]" = OrderedDict()
        self.lookups = 0
        self.hits = 0
        self.tokens_reused = 0
        self._reuse_callback = reuse_callback

    def try_reuse(self, key: Optional[bytes], seq_id: int, seq_len: int) -> int:
        if key is None or seq_len < self.block_size:
            return 0
        self.lookups += 1
        entry = self._entries.get(key)
        if entry is None or entry.seq_len != seq_len:
            return 0
        self.hits += 1
        self._entries.move_to_end(key, last=True)
        self.manager.reuse_prefix(seq_id, entry.blocks, entry.seq_len)
        if self._reuse_callback is not None:
            try:
                self._reuse_callback(seq_id, entry.seq_len)
            except Exception:  # pragma: no cover - defensive
                logging.debug("prefix reuse callback failed", exc_info=True)
        entry.last_used = time.time()
        self.tokens_reused += entry.seq_len
        return entry.seq_len

    def record(self, key: Optional[bytes], seq_id: int, seq_len: int) -> None:
        if key is None or seq_len < self.block_size:
            return
        blocks = self.manager.snapshot_blocks(seq_id, seq_len)
        if not blocks:
            return
        entry = _PrefixEntry(tuple(blocks), seq_len, time.time())
        self._entries[key] = entry
        self._entries.move_to_end(key, last=True)
        self._evict_if_needed()

    def record_many(
        self, prefixes: Optional[Sequence[Tuple[int, bytes]]], seq_id: int
    ) -> None:
        if not prefixes:
            return
        for seq_len, key in prefixes:
            self.record(key, seq_id, seq_len)

    def stats(self) -> Dict[str, float]:
        return {
            "lookups": float(self.lookups),
            "hits": float(self.hits),
            "hit_rate": float(self.hits) / self.lookups if self.lookups else 0.0,
            "tokens_reused": float(self.tokens_reused),
        }

    def _evict_if_needed(self) -> None:
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)


__all__ = [
    "BackendSignature",
    "ViewSignature",
    "PagedSlotKVCache",
    "PagedBatchView",
    "PrefixCache",
]
