# ABOUTME: Provides helper to run array-based prefill chunks via ArrayDecodeRunner.
# ABOUTME: Streams prompt tokens through the array decode path before flush.

from __future__ import annotations

import logging
import numbers
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import mlx.core as mx

from .array_decode_runner import ArrayDecodeRunner
from .graph_decode import LlamaPrefillGraph
from .paged_arrays import ArrayBatchView


class PrefillArrayRunner:
    """Runs paged prefill chunks through compiled LLaMA graphs."""

    @dataclass
    class PrefillPendingChunk:
        seq_ids: Tuple[int, ...]
        active_rows: int
        chunk_len: int
        block_tables: mx.array
        base_lens: mx.array
        k_tensor: mx.array
        v_tensor: mx.array
        delivered_tokens: int = 0
        written_tokens: int = 0
        ready_time: float = 0.0
        first_delivery_time: Optional[float] = None

        def matches(self, seq_ids: Sequence[int], active: int, chunk_len: int) -> bool:
            if self.chunk_len != chunk_len or self.active_rows != active:
                return False
            return self.seq_ids == tuple(seq_ids[:active])

        @property
        def available_tokens(self) -> int:
            return max(0, self.chunk_len - self.delivered_tokens)

        @property
        def pending_writes(self) -> int:
            return max(0, self.delivered_tokens - self.written_tokens)

        def take_slice(
            self, rows: Sequence[int], max_tokens: int
        ) -> tuple[mx.array, mx.array, mx.array, int]:
            if max_tokens <= 0 or not rows:
                return (
                    mx.array([], dtype=self.k_tensor.dtype),
                    mx.array([], dtype=self.v_tensor.dtype),
                    mx.array([], dtype=mx.int32),
                    0,
                )
            steps = min(self.available_tokens, int(max_tokens))
            if steps <= 0:
                return (
                    mx.array([], dtype=self.k_tensor.dtype),
                    mx.array([], dtype=self.v_tensor.dtype),
                    mx.array([], dtype=mx.int32),
                    0,
                )
            start = self.delivered_tokens
            end = start + steps
            token_slice = slice(start, end)
            k_slice = self.k_tensor[:, token_slice, rows, :, :]
            v_slice = self.v_tensor[:, token_slice, rows, :, :]
            base = mx.array(
                [int(self.base_lens[row].item()) + start for row in rows],
                dtype=mx.int32,
            )
            self.delivered_tokens = end
            return k_slice, v_slice, base, steps

        def slice_for_write(
            self, rows: Sequence[int], max_tokens: int
        ) -> tuple[mx.array, mx.array, mx.array, mx.array, int]:
            if max_tokens <= 0 or not rows:
                return (
                    mx.array([], dtype=self.k_tensor.dtype),
                    mx.array([], dtype=self.v_tensor.dtype),
                    mx.array([], dtype=mx.int32),
                    mx.array([], dtype=mx.int32),
                    0,
                )
            steps = min(self.pending_writes, int(max_tokens))
            if steps <= 0:
                return (
                    mx.array([], dtype=self.k_tensor.dtype),
                    mx.array([], dtype=self.v_tensor.dtype),
                    mx.array([], dtype=mx.int32),
                    mx.array([], dtype=mx.int32),
                    0,
                )
            start = self.written_tokens
            end = start + steps
            token_slice = slice(start, end)
            k_slice = self.k_tensor[:, token_slice, rows, :, :]
            v_slice = self.v_tensor[:, token_slice, rows, :, :]
            base = mx.array(
                [int(self.base_lens[row].item()) + start for row in rows],
                dtype=mx.int32,
            )
            tables = mx.array(self.block_tables[rows], dtype=mx.int32)
            self.written_tokens = end
            return k_slice, v_slice, tables, base, steps

    @dataclass
    class PrefillOverlaySlice:
        chunk: "PrefillArrayRunner.PrefillPendingChunk"
        row: int

    @dataclass
    class PrefillOverlayBatch:
        seq_ids: Tuple[int, ...]
        tokens: int
        base_lens: mx.array
        k_layers: mx.array
        v_layers: mx.array

        def layer_tensors(
            self, layer_idx: int
        ) -> Tuple[mx.array, mx.array, mx.array, int]:
            return (
                self.k_layers[layer_idx],
                self.v_layers[layer_idx],
                self.base_lens,
                self.tokens,
            )

    def __init__(self, decode_runner: ArrayDecodeRunner):
        if not isinstance(decode_runner, ArrayDecodeRunner) and not hasattr(
            decode_runner, "decode"
        ):
            raise TypeError("PrefillArrayRunner requires an ArrayDecodeRunner")
        if not hasattr(mx, "fast"):
            raise RuntimeError("PrefillArrayRunner requires mx.fast support")
        self.decode_runner = decode_runner
        model = getattr(decode_runner.graph, "_outer", None)
        if model is None:
            raise ValueError("ArrayDecodeRunner graph missing original model reference")
        self._prefill_graph = LlamaPrefillGraph(model)
        self._manager = decode_runner.manager
        self._fast = mx.fast
        self._writer_stream = mx.new_stream(mx.default_device())
        self._pending_chunk: Optional[PrefillArrayRunner.PrefillPendingChunk] = None
        self._pending_zero_cache: Dict[Tuple, Tuple[mx.array, mx.array]] = {}
        self._overlay_by_seq: Dict[int, "PrefillArrayRunner.PrefillOverlaySlice"] = {}
        self._metrics = {
            "array_prefill_graph_s": 0.0,
            "array_prefill_writer_s": 0.0,
            "array_prefill_chunk_ms_total": 0.0,
            "array_prefill_chunk_count": 0.0,
            "array_prefill_first_chunk_ms": 0.0,
            "array_prefill_pending_tokens": 0.0,
            "array_prefill_pending_tokens_max": 0.0,
            "array_prefill_attn_s": 0.0,
            "array_prefill_mlp_s": 0.0,
            "array_prefill_overlay_s": 0.0,
            "array_prefill_overlay_wait_s": 0.0,
            "array_prefill_overlay_wait_count": 0.0,
            "array_prefill_waiting_sequences": 0.0,
        }

    def warmup(self, *, batch_size, chunk_len) -> None:
        batches = self._normalize_warm_values(batch_size)
        chunks = self._normalize_warm_values(chunk_len)
        max_blocks = getattr(self._manager, "max_blocks_per_sequence", 1)
        k_cache_shape = tuple(self._manager.k.shape)
        v_cache_shape = tuple(self._manager.v.shape)
        kv_map_shape = None
        dtype = self._manager.k.dtype
        compiled = 0
        for batch in batches:
            block_tables_shape = (batch, max_blocks)
            for chunk in chunks:
                for pending_flag in (0, 1):
                    try:
                        self._prefill_graph.get_compiled(
                            batch_size=batch,
                            chunk_len=chunk,
                            block_tables_shape=block_tables_shape,
                            k_cache_shape=k_cache_shape,
                            v_cache_shape=v_cache_shape,
                            kv_map_shape=kv_map_shape,
                            dtype=dtype,
                            pending_flag=pending_flag,
                        )
                        compiled += 1
                    except Exception:  # pragma: no cover - warmup best effort
                        logging.debug(
                            "PrefillArrayRunner: warmup compile failed (batch=%s chunk=%s pending=%s)",
                            batch,
                            chunk,
                            pending_flag,
                            exc_info=True,
                        )
        if compiled:
            logging.info(
                "PrefillArrayRunner: warmed compiled graphs batches=%s chunks=%s",
                batches,
                chunks,
            )

    def prefill_chunk(
        self,
        token_matrix: mx.array,
        view: ArrayBatchView,
        *,
        chunk_len: int,
    ) -> int:
        chunk_start = time.perf_counter()
        if not isinstance(view, ArrayBatchView):
            raise TypeError("prefill_chunk requires an ArrayBatchView")
        if token_matrix.ndim != 2:
            raise ValueError("prefill_chunk expects [B, chunk_len] token matrix")
        if chunk_len <= 0:
            return 0
        active = min(token_matrix.shape[0], view.active_rows)
        if active <= 0:
            return 0
        if any(view._quant_kwargs):
            return 0
        base_lens = view.prefill_base_lens
        if base_lens is None:
            base_lens = view.context_lens
        base_lens = mx.array(base_lens[:active], dtype=mx.int32)
        block_tables = mx.array(view.block_tables[:active], dtype=mx.int32)
        k_cache = view._k_cache
        v_cache = view._v_cache
        kv_map = view.kv_head_mapping
        tokens = token_matrix[:active]
        seq_ids = tuple(view.seq_ids[:active])
        pending_chunk = self._pending_chunk
        if pending_chunk is not None and not pending_chunk.matches(
            seq_ids, active, chunk_len
        ):
            self._launch_pending_chunk(pending_chunk)
            pending_chunk = None
        pending_flag = 1 if pending_chunk is not None else 0
        pending_k, pending_v = self._pending_inputs(
            pending_chunk,
            layer_count=k_cache.shape[0],
            chunk_len=chunk_len,
            batch_size=active,
            kv_heads=k_cache.shape[1],
            head_dim=k_cache.shape[-1],
            k_dtype=k_cache.dtype,
            v_dtype=v_cache.dtype,
        )
        fn = self._prefill_graph.get_compiled(
            batch_size=active,
            chunk_len=chunk_len,
            block_tables_shape=tuple(block_tables.shape),
            k_cache_shape=tuple(k_cache.shape),
            v_cache_shape=tuple(v_cache.shape),
            kv_map_shape=tuple(kv_map.shape) if kv_map is not None else None,
            dtype=k_cache.dtype,
            pending_flag=pending_flag,
        )
        args = [
            tokens,
            base_lens,
            block_tables,
            k_cache,
            v_cache,
            pending_k,
            pending_v,
        ]
        if kv_map is not None:
            args.append(kv_map)
        graph_start = time.perf_counter()
        _, k_tensor_full, v_tensor_full = fn(*args)
        self._accumulate_prefill_graph_metrics()
        self._metrics["array_prefill_graph_s"] += time.perf_counter() - graph_start
        offset = chunk_len if pending_flag else 0
        k_tensor = k_tensor_full[:, offset : offset + chunk_len, :, :, :]
        v_tensor = v_tensor_full[:, offset : offset + chunk_len, :, :, :]
        if pending_chunk is not None:
            self._launch_pending_chunk(pending_chunk)
        self._pending_chunk = PrefillArrayRunner.PrefillPendingChunk(
            seq_ids=seq_ids,
            active_rows=active,
            chunk_len=chunk_len,
            block_tables=block_tables,
            base_lens=base_lens,
            k_tensor=k_tensor,
            v_tensor=v_tensor,
            ready_time=time.perf_counter(),
        )
        self._register_overlays(self._pending_chunk)
        total_wall_ms = (time.perf_counter() - chunk_start) * 1000.0
        self._metrics["array_prefill_chunk_ms_total"] += total_wall_ms
        self._metrics["array_prefill_chunk_count"] += 1.0
        if self._metrics.get("array_prefill_first_chunk_ms", 0.0) <= 0.0:
            self._metrics["array_prefill_first_chunk_ms"] = total_wall_ms
        if view.seq_ids:
            self._manager.bump_decode_lengths(view.seq_ids[:active], chunk_len)
        return chunk_len

    def flush_pending(self) -> None:
        chunks: list[PrefillArrayRunner.PrefillPendingChunk] = []
        if self._pending_chunk is not None:
            chunks.append(self._pending_chunk)
            self._pending_chunk = None
        for handle in self._overlay_by_seq.values():
            if handle.chunk not in chunks:
                chunks.append(handle.chunk)
        for chunk in chunks:
            remaining = chunk.chunk_len - chunk.written_tokens
            if remaining <= 0:
                continue
            rows = list(range(chunk.active_rows))
            self._write_chunk_slice(chunk, rows, remaining)
        if chunks:
            self._writer_sync()
            self._clear_overlays_for_chunks(chunks)
            self._record_pending_tokens(None)
        self._overlay_by_seq.clear()

    def consume_metrics(self) -> Dict[str, float]:
        stats = dict(self._metrics)
        for key in self._metrics:
            self._metrics[key] = 0.0
        return stats

    def _pending_inputs(
        self,
        chunk: Optional["PrefillArrayRunner.PrefillPendingChunk"],
        *,
        layer_count: int,
        chunk_len: int,
        batch_size: int,
        kv_heads: int,
        head_dim: int,
        k_dtype,
        v_dtype,
    ) -> tuple[mx.array, mx.array]:
        if chunk is not None:
            return chunk.k_tensor, chunk.v_tensor
        key = (
            layer_count,
            chunk_len,
            batch_size,
            kv_heads,
            head_dim,
            str(k_dtype),
            str(v_dtype),
        )
        cached = self._pending_zero_cache.get(key)
        if cached is None:
            shape = (layer_count, chunk_len, batch_size, kv_heads, head_dim)
            cached = (
                mx.zeros(shape, dtype=k_dtype),
                mx.zeros(shape, dtype=v_dtype),
            )
            self._pending_zero_cache[key] = cached
        return cached

    def _launch_pending_chunk(
        self, chunk: Optional["PrefillArrayRunner.PrefillPendingChunk"]
    ) -> None:
        if chunk is None:
            return
        start = time.perf_counter()
        self._fast.paged_kv_write_layers_tokens(
            self._manager.k,
            self._manager.v,
            chunk.block_tables,
            chunk.base_lens,
            chunk.k_tensor,
            chunk.v_tensor,
            stream=self._writer_stream,
        )
        chunk.written_tokens = chunk.chunk_len
        self._writer_sync()
        self._metrics["array_prefill_writer_s"] += time.perf_counter() - start
        self._record_pending_tokens(None)
        self._clear_overlays_for_chunks([chunk])
        self._pending_chunk = None

    def _normalize_warm_values(self, raw) -> list[int]:
        if isinstance(raw, numbers.Integral):
            values = [int(raw)]
        else:
            try:
                iterator = iter(raw)
            except TypeError:
                values = [int(raw)]
            else:
                values = []
                for item in iterator:
                    try:
                        values.append(int(item))
                    except (TypeError, ValueError):
                        continue
        cleaned = sorted({max(1, val) for val in values if val > 0})
        if not cleaned:
            cleaned = [1]
        return cleaned

    def _register_overlays(
        self, chunk: Optional["PrefillArrayRunner.PrefillPendingChunk"]
    ) -> None:
        if chunk is None:
            return
        for idx, seq_id in enumerate(chunk.seq_ids):
            self._overlay_by_seq[seq_id] = PrefillArrayRunner.PrefillOverlaySlice(
                chunk=chunk,
                row=idx,
            )

    def _clear_overlays_for_chunks(
        self, chunks: Sequence["PrefillArrayRunner.PrefillPendingChunk"]
    ) -> None:
        if not chunks:
            return
        active_ids = set()
        for chunk in chunks:
            active_ids.update(chunk.seq_ids)
        for seq_id in list(self._overlay_by_seq.keys()):
            if seq_id in active_ids:
                self._overlay_by_seq.pop(seq_id, None)

    def overlay_seq_ids(self, seq_ids: Sequence[int]) -> Tuple[int, ...]:
        """Return the subset of seq_ids that currently have overlay slices available."""
        if not seq_ids:
            return tuple()
        has_overlay: List[int] = []
        for seq_id in seq_ids:
            if int(seq_id) in self._overlay_by_seq:
                has_overlay.append(int(seq_id))
        return tuple(has_overlay)

    def collect_overlays(
        self, seq_ids: Sequence[int], max_tokens: Optional[int] = None
    ) -> Optional["PrefillArrayRunner.PrefillOverlayBatch"]:
        chunk, handles = self._chunk_handles(seq_ids)
        if chunk is None or not handles:
            return None
        rows = [handle.row for handle in handles]
        max_req = chunk.available_tokens
        if max_req <= 0:
            return None
        if max_tokens is not None:
            max_req = min(max_req, int(max_tokens))
        k_slice, v_slice, base_lens, steps = chunk.take_slice(rows, max_req)
        if steps <= 0:
            return None
        self._record_pending_tokens(chunk)
        if chunk.first_delivery_time is None:
            chunk.first_delivery_time = time.perf_counter()
            wait = chunk.first_delivery_time - chunk.ready_time
            if wait > 0:
                self._metrics["array_prefill_overlay_wait_s"] += wait
                self._metrics["array_prefill_overlay_wait_count"] += 1.0
        return PrefillArrayRunner.PrefillOverlayBatch(
            seq_ids=tuple(seq_ids),
            tokens=steps,
            base_lens=base_lens,
            k_layers=k_slice,
            v_layers=v_slice,
        )

    def release_overlays(self, seq_ids: Sequence[int]) -> None:
        if not seq_ids:
            return
        chunk, handles = self._chunk_handles(seq_ids)
        if chunk is not None:
            rows = [handle.row for handle in handles]
            remaining = chunk.delivered_tokens - chunk.written_tokens
            if remaining > 0:
                self._write_chunk_slice(chunk, rows, remaining)
                self._writer_sync()
            self._clear_overlays_for_chunks([chunk])
        for seq_id in seq_ids:
            self._overlay_by_seq.pop(int(seq_id), None)
        if not self._overlay_by_seq:
            self.flush_pending()
        elif chunk is None:
            self._record_pending_tokens(None)

    def has_active_overlays(self) -> bool:
        return bool(self._overlay_by_seq)

    def advance_and_flush(
        self, seq_ids: Sequence[int], tokens_per_sequence: int
    ) -> int:
        if tokens_per_sequence <= 0 or not seq_ids:
            return 0
        chunk, handles = self._chunk_handles(seq_ids)
        if chunk is None or not handles:
            return 0
        rows = [handle.row for handle in handles]
        max_writes = min(chunk.pending_writes, int(tokens_per_sequence))
        if max_writes <= 0:
            return 0
        k_slice, v_slice, block_tables, base_lens, actual = chunk.slice_for_write(
            rows, max_writes
        )
        if actual <= 0:
            return 0
        start = time.perf_counter()
        self._fast.paged_kv_write_layers_tokens(
            self._manager.k,
            self._manager.v,
            block_tables,
            base_lens,
            k_slice,
            v_slice,
            stream=self._writer_stream,
        )
        self._metrics["array_prefill_writer_s"] += time.perf_counter() - start
        self._writer_sync()
        if chunk.written_tokens >= chunk.chunk_len:
            self._clear_overlays_for_chunks([chunk])
        self._record_pending_tokens(
            chunk if chunk.written_tokens < chunk.chunk_len else None
        )
        return actual

    def _chunk_handles(self, seq_ids: Sequence[int]) -> tuple[
        Optional["PrefillArrayRunner.PrefillPendingChunk"],
        list["PrefillArrayRunner.PrefillOverlaySlice"],
    ]:
        if not seq_ids:
            return None, []
        chunk = None
        handles: list[PrefillArrayRunner.PrefillOverlaySlice] = []
        for seq_id in seq_ids:
            handle = self._overlay_by_seq.get(int(seq_id))
            if handle is None:
                return None, []
            if chunk is None:
                chunk = handle.chunk
            elif chunk is not handle.chunk:
                return None, []
            handles.append(handle)
        return chunk, handles

    def _write_chunk_slice(
        self,
        chunk: "PrefillArrayRunner.PrefillPendingChunk",
        rows: Sequence[int],
        max_tokens: int,
    ) -> int:
        if not rows or max_tokens <= 0:
            return 0
        k_slice, v_slice, block_tables, base_lens, steps = chunk.slice_for_write(
            rows, max_tokens
        )
        if steps <= 0:
            return 0
        self._fast.paged_kv_write_layers_tokens(
            self._manager.k,
            self._manager.v,
            block_tables,
            base_lens,
            k_slice,
            v_slice,
            stream=self._writer_stream,
        )
        self._record_pending_tokens(chunk)
        return steps

    def _writer_sync(self) -> None:
        mx.synchronize(self._writer_stream)

    def _record_pending_tokens(
        self, chunk: Optional["PrefillArrayRunner.PrefillPendingChunk"]
    ) -> None:
        pending = float(chunk.pending_writes) if chunk is not None else 0.0
        self._metrics["array_prefill_pending_tokens"] = pending
        if pending > self._metrics["array_prefill_pending_tokens_max"]:
            self._metrics["array_prefill_pending_tokens_max"] = pending
        self._metrics["array_prefill_waiting_sequences"] = float(
            len(self._overlay_by_seq)
        )

    def _accumulate_prefill_graph_metrics(self) -> None:
        consume = getattr(self._prefill_graph, "consume_metrics", None)
        if consume is None:
            return
        try:
            stats = consume()
        except Exception:
            logging.debug(
                "PrefillArrayRunner: consume_metrics failed",
                exc_info=True,
            )
            return
        if not stats:
            return
        for key in (
            "array_prefill_attn_s",
            "array_prefill_mlp_s",
            "array_prefill_overlay_s",
        ):
            value = float(stats.get(key, 0.0))
            if value:
                self._metrics[key] = self._metrics.get(key, 0.0) + value


__all__ = ["PrefillArrayRunner"]
