# ABOUTME: Tests PrefillArrayRunner pending-chunk overlap behavior.
# ABOUTME: Ensures prefill chunks reuse pending overlays and flush on mismatch.

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple
from unittest import mock

import mlx.core as mx
import numpy as np

from mlx_lm.server_batched.paged_arrays import ArrayBatchView, PagedDecodeArrays
from mlx_lm.server_batched.prefill_array_runner import PrefillArrayRunner
from mlx_lm.server_batched.slot_generator import ArrayDecodeRunner


@dataclass
class _DummyManager:
    layers: int
    kv_heads: int
    max_blocks: int
    block_size: int
    head_dim: int

    def __post_init__(self):
        cache_shape = (
            self.layers,
            self.kv_heads,
            self.max_blocks,
            self.block_size,
            self.head_dim,
        )
        self.k = mx.zeros(cache_shape, dtype=mx.float32)
        self.v = mx.zeros(cache_shape, dtype=mx.float32)
        self._bumps: List[Tuple[Tuple[int, ...], int]] = []

    def bump_decode_lengths(self, seq_ids: Sequence[int], delta: int) -> None:
        self._bumps.append((tuple(int(s) for s in seq_ids), int(delta)))


class _FakeFast:
    def __init__(self):
        self.calls: List[
            Tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]
        ] = []

    def paged_kv_write_layers_tokens(
        self,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        k_tensor,
        v_tensor,
        *,
        stream,
    ):
        self.calls.append((block_tables, context_lens, k_tensor, v_tensor))


class _FakePrefillGraph:
    def __init__(self, layers: int, chunk_len: int, kv_heads: int, head_dim: int):
        self.layers = layers
        self.chunk_len = chunk_len
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.pending_flags: List[int] = []
        self._last_metrics = {}

    def get_compiled(self, *, pending_flag: int, **_unused):
        layers = self.layers
        chunk_len = self.chunk_len
        kv_heads = self.kv_heads
        head_dim = self.head_dim

        def _impl(
            tokens,
            base_lens,
            block_tables,
            k_cache,
            v_cache,
            pending_k,
            pending_v,
            *extras,
        ):
            _ = (
                tokens,
                base_lens,
                block_tables,
                k_cache,
                v_cache,
                pending_v,
                pending_k,
                extras,
            )
            flag = int(pending_flag)
            self.pending_flags.append(flag)
            capacity = chunk_len * 2
            filler_prev = 100 + len(self.pending_flags)
            filler_cur = 200 + len(self.pending_flags)
            shape = (layers, chunk_len, tokens.shape[0], kv_heads, head_dim)
            prev = mx.full(shape, filler_prev, dtype=k_cache.dtype)
            cur = mx.full(shape, filler_cur, dtype=k_cache.dtype)
            stacked = mx.concatenate([prev, cur], axis=1)
            hidden = mx.zeros((tokens.shape[0], chunk_len, 1), dtype=k_cache.dtype)
            self._last_metrics = {
                "array_prefill_attn_s": 0.01,
                "array_prefill_mlp_s": 0.02,
                "array_prefill_overlay_s": 0.03,
            }
            return hidden, stacked, stacked

        return _impl

    def consume_metrics(self):
        metrics = dict(self._last_metrics)
        self._last_metrics.clear()
        return metrics


def _make_view(
    seq_ids: Tuple[int, ...],
    manager: _DummyManager,
    *,
    block_tables: np.ndarray,
    base_lens: np.ndarray,
) -> ArrayBatchView:
    batch = len(seq_ids)
    quant = [{} for _ in range(manager.layers)]
    arrays = PagedDecodeArrays(
        seq_ids=seq_ids,
        block_tables=mx.array(block_tables, dtype=mx.int32),
        context_lens=mx.array(base_lens, dtype=mx.int32),
        kv_head_mapping=None,
        k_cache=manager.k,
        v_cache=manager.v,
        quant_kwargs=quant,
        write_block_ids=mx.zeros((batch,), dtype=mx.int32),
        write_token_offsets=mx.zeros((batch,), dtype=mx.int32),
        prefill_base_lens=mx.array(base_lens, dtype=mx.int32),
        signature=None,
        active_rows=batch,
    )
    return ArrayBatchView(arrays)


def _make_runner(
    *,
    layers: int = 1,
    chunk_len: int = 2,
    kv_heads: int = 1,
    head_dim: int = 1,
):
    manager = _DummyManager(
        layers, kv_heads, max_blocks=4, block_size=4, head_dim=head_dim
    )
    graph = _FakePrefillGraph(layers, chunk_len, kv_heads, head_dim)
    runner = object.__new__(PrefillArrayRunner)
    runner.decode_runner = mock.Mock(spec=ArrayDecodeRunner)
    runner._prefill_graph = graph
    runner._manager = manager
    runner._fast = _FakeFast()
    runner._writer_stream = mx.new_stream(mx.default_device())
    runner._pending_writes = []
    runner._pending_chunk = None
    runner._pending_zero_cache = {}
    runner._overlay_by_seq = {}
    runner._metrics = {
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
    return runner, manager, graph, runner._fast


@mock.patch("mlx_lm.server_batched.prefill_array_runner.mx.synchronize", autospec=True)
def test_prefill_chunk_reuses_pending_overlay(mock_sync):
    chunk_len = 2
    runner, manager, graph, fast = _make_runner(chunk_len=chunk_len)
    seq_ids = (1, 2)
    block_tables = np.zeros((len(seq_ids), manager.max_blocks), dtype=np.int32)
    base_lens = np.zeros((len(seq_ids),), dtype=np.int32)
    view = _make_view(seq_ids, manager, block_tables=block_tables, base_lens=base_lens)
    tokens = mx.zeros((len(seq_ids), chunk_len), dtype=mx.int32)

    produced = runner.prefill_chunk(tokens, view, chunk_len=chunk_len)
    assert produced == chunk_len
    assert not fast.calls
    assert graph.pending_flags == [0]
    assert runner._pending_chunk is not None

    produced = runner.prefill_chunk(tokens, view, chunk_len=chunk_len)
    assert produced == chunk_len
    assert len(graph.pending_flags) == 2 and graph.pending_flags[-1] == 1
    assert len(fast.calls) == 1, "previous chunk should flush when overlap begins"
    assert runner._pending_chunk is not None
    pending_values = runner._pending_chunk.k_tensor
    expected = mx.full(pending_values.shape, 202, dtype=pending_values.dtype)
    assert mx.array_equal(
        pending_values, expected
    ), "pending tensor should store current chunk output slice"

    runner.flush_pending()
    assert len(fast.calls) == 1, "chunk already flushed before flush_pending"
    mock_sync.assert_called()
    metrics = runner.consume_metrics()
    assert metrics.get("array_prefill_chunk_ms_total", 0.0) > 0.0
    assert metrics.get("array_prefill_chunk_count", 0.0) >= 2.0
    assert metrics.get("array_prefill_first_chunk_ms", 0.0) > 0.0
    assert metrics.get("array_prefill_attn_s", 0.0) > 0.0
    assert metrics.get("array_prefill_mlp_s", 0.0) > 0.0
    assert metrics.get("array_prefill_overlay_s", 0.0) > 0.0


@mock.patch("mlx_lm.server_batched.prefill_array_runner.mx.synchronize", autospec=True)
def test_prefill_chunk_flushes_on_mismatch(mock_sync):
    chunk_len = 2
    runner, manager, graph, fast = _make_runner(chunk_len=chunk_len)
    seq_ids = (10, 11)
    block_tables = np.zeros((len(seq_ids), manager.max_blocks), dtype=np.int32)
    base_lens = np.zeros((len(seq_ids),), dtype=np.int32)
    view_a = _make_view(
        seq_ids, manager, block_tables=block_tables, base_lens=base_lens
    )
    view_b = _make_view(
        (20, 21), manager, block_tables=block_tables, base_lens=base_lens
    )
    tokens = mx.zeros((len(seq_ids), chunk_len), dtype=mx.int32)

    runner.prefill_chunk(tokens, view_a, chunk_len=chunk_len)
    assert graph.pending_flags == [0]
    assert runner._pending_chunk is not None

    runner.prefill_chunk(tokens, view_b, chunk_len=chunk_len)
    assert graph.pending_flags[-1] == 0, "pending chunk flushed before mismatch run"
    assert len(fast.calls) == 1, "stale pending chunk flushed eagerly"

    runner.flush_pending()
    assert len(fast.calls) == 1
    mock_sync.assert_called()


def test_overlay_seq_ids_filters_missing_handles():
    runner, manager, graph, _ = _make_runner(chunk_len=2)
    seq_ids = (101, 102)
    block_tables = np.zeros((len(seq_ids), manager.max_blocks), dtype=np.int32)
    base_lens = np.zeros((len(seq_ids),), dtype=np.int32)
    view = _make_view(seq_ids, manager, block_tables=block_tables, base_lens=base_lens)
    tokens = mx.zeros((len(seq_ids), 2), dtype=mx.int32)

    produced = runner.prefill_chunk(tokens, view, chunk_len=2)
    assert produced == 2

    filtered = runner.overlay_seq_ids((102, 999, 101))
    assert filtered == (102, 101)


def test_warmup_compiles_pending_variants():
    runner, manager, graph, fast = _make_runner(chunk_len=2)
    captured: list[int] = []

    def fake_compile(**kwargs):
        captured.append(kwargs.get("pending_flag"))
        return lambda *args, **kw: None

    runner._prefill_graph.get_compiled = fake_compile
    runner.warmup(batch_size=5, chunk_len=4)
    assert captured == [0, 1]


def test_collect_overlays_returns_expected_batch():
    chunk_len = 2
    runner, manager, graph, fast = _make_runner(chunk_len=chunk_len)
    seq_ids = (3, 4)
    block_tables = np.zeros((len(seq_ids), manager.max_blocks), dtype=np.int32)
    base_lens = np.zeros((len(seq_ids),), dtype=np.int32)
    view = _make_view(seq_ids, manager, block_tables=block_tables, base_lens=base_lens)
    tokens = mx.arange(len(seq_ids) * chunk_len, dtype=mx.int32).reshape(
        len(seq_ids), chunk_len
    )
    runner.prefill_chunk(tokens, view, chunk_len=chunk_len)
    overlays = runner.collect_overlays(seq_ids, max_tokens=1)
    assert overlays is not None
    assert overlays.tokens == 1
    assert overlays.k_layers.shape[2] == len(seq_ids)
    assert overlays.k_layers.shape[1] == 1
    # Second slice consumes the remainder.
    overlays_next = runner.collect_overlays(seq_ids, max_tokens=2)
    assert overlays_next is not None
    assert overlays_next.tokens == 1
    runner.release_overlays(seq_ids)
    assert runner.collect_overlays(seq_ids) is None


def test_collect_overlays_drains_chunk_in_windows():
    chunk_len = 6
    runner, manager, graph, fast = _make_runner(chunk_len=chunk_len)
    seq_ids = (1, 2)
    block_tables = np.zeros((len(seq_ids), manager.max_blocks), dtype=np.int32)
    base_lens = np.zeros((len(seq_ids),), dtype=np.int32)
    view = _make_view(seq_ids, manager, block_tables=block_tables, base_lens=base_lens)
    tokens = mx.arange(len(seq_ids) * chunk_len, dtype=mx.int32).reshape(
        len(seq_ids), chunk_len
    )
    runner.prefill_chunk(tokens, view, chunk_len=chunk_len)
    first = runner.collect_overlays(seq_ids, max_tokens=3)
    assert first is not None
    assert first.tokens == 3
    pending = runner._pending_chunk
    assert pending is not None
    assert pending.pending_writes == 3
    flushed = runner.advance_and_flush(seq_ids, tokens_per_sequence=first.tokens)
    assert flushed == 3
    assert pending.pending_writes == 0
    second = runner.collect_overlays(seq_ids, max_tokens=3)
    assert second is not None
    assert second.tokens == 3
    runner.advance_and_flush(seq_ids, tokens_per_sequence=second.tokens)
    metrics = runner.consume_metrics()
    assert metrics.get("array_prefill_pending_tokens_max", 0.0) >= 3.0
    assert metrics.get("array_prefill_pending_tokens", -1.0) == 0.0


@mock.patch("mlx_lm.server_batched.prefill_array_runner.mx.synchronize", autospec=True)
def test_release_overlays_flushes_when_last_sequence_exits(mock_sync):
    chunk_len = 2
    runner, manager, graph, fast = _make_runner(chunk_len=chunk_len)
    seq_ids = (7, 8)
    block_tables = np.zeros((len(seq_ids), manager.max_blocks), dtype=np.int32)
    base_lens = np.zeros((len(seq_ids),), dtype=np.int32)
    view = _make_view(seq_ids, manager, block_tables=block_tables, base_lens=base_lens)
    tokens = mx.zeros((len(seq_ids), chunk_len), dtype=mx.int32)
    runner.prefill_chunk(tokens, view, chunk_len=chunk_len)
    assert runner.has_active_overlays()
    runner.release_overlays((seq_ids[0],))
    assert not runner.has_active_overlays()
    mock_sync.assert_called()


@mock.patch("mlx_lm.server_batched.prefill_array_runner.mx.synchronize", autospec=True)
def test_advance_and_flush_writes_per_slice(mock_sync):
    chunk_len = 4
    runner, manager, graph, fast = _make_runner(chunk_len=chunk_len)
    seq_ids = (5, 6)
    block_tables = np.zeros((len(seq_ids), manager.max_blocks), dtype=np.int32)
    base_lens = np.zeros((len(seq_ids),), dtype=np.int32)
    view = _make_view(seq_ids, manager, block_tables=block_tables, base_lens=base_lens)
    tokens = mx.arange(len(seq_ids) * chunk_len, dtype=mx.int32).reshape(
        len(seq_ids), chunk_len
    )
    runner.prefill_chunk(tokens, view, chunk_len=chunk_len)
    # Deliver two slices (2 tokens) to decode.
    assert runner.collect_overlays(seq_ids, max_tokens=2) is not None
    assert runner.collect_overlays(seq_ids, max_tokens=2) is not None
    # Flush one token per sequence.
    written = runner.advance_and_flush(seq_ids, tokens_per_sequence=1)
    assert written == 1
    mock_sync.assert_called()
