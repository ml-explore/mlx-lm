# ABOUTME: Tests slot-indexed generator vectorized decode path.
# ABOUTME: Ensures prefill and decode operate on stable slot assignments.

from .util import ensure_mlx_stub

ensure_mlx_stub()

import queue
import types
import unittest
from unittest import mock

import mlx.core as mx
import numpy as np

from mlx_lm.models.cache import PagedKVCacheAdapter
from mlx_lm.server_batched.paged_arrays import ArrayBatchView, PagedDecodeArrays
from mlx_lm.server_batched.prefill_array_runner import PrefillArrayRunner
from mlx_lm.server_batched.slot_allocator import SlotAllocator
from mlx_lm.server_batched.slot_generator import (
    PagedBatchCache,
    PrefillSlicer,
    SlotBatchCache,
    SlotGenerator,
)
from mlx_lm.server_batched.state import SequenceContext, SequenceState


class FakeTokenizer:
    bos_token_id = 101
    eos_token_id = 102

    def detokenizer(self):
        detok = types.SimpleNamespace()
        detok.add_token = lambda *_: None
        detok.last_segment = ""
        detok.finalize = lambda: None
        return detok


class _StubPromptCache:
    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0

    @property
    def state(self):
        return self.keys, self.values

    def update_and_fetch(self, keys, values):
        self.keys = keys
        self.values = values
        if hasattr(keys, "shape") and keys.shape and len(keys.shape) >= 3:
            self.offset += int(keys.shape[2])
        return self.keys, self.values

    def make_mask(self, *args, **kwargs):
        return None

    def trim(self, n):
        return 0

    def is_trimmable(self):
        return False


class FakeModel:
    def __init__(self):
        self.calls = []
        self.decode_vocab = 5
        self.layers = [types.SimpleNamespace() for _ in range(2)]

    def __call__(self, tokens, cache=None):
        arr = np.array(tokens)
        self.calls.append({"tokens": arr.copy(), "cache": cache})
        if arr.ndim == 2 and arr.shape[1] > 1:
            # Prefill path, logits unused.
            return np.zeros(
                (arr.shape[0], arr.shape[1], self.decode_vocab), dtype=np.float32
            )
        if arr.ndim == 2:
            batch = arr.shape[0]
            logits = np.stack(
                [
                    np.linspace(i, i + 1, self.decode_vocab, dtype=np.float32)
                    for i in range(batch)
                ],
                axis=0,
            )
        return logits[:, None, :]
        raise AssertionError(f"Unexpected tokens shape {arr.shape}")

    def make_cache(self):
        return [_StubPromptCache() for _ in range(2)]


class _StubPagedManager:
    block_size = 16
    num_layers = 2
    num_kv_heads = 1
    head_dim = 4
    dtype = mx.float16

    def __init__(self):
        self.max_blocks_per_sequence = 1
        self._ctx = {}
        self._prefill_active = set()
        self.k = mx.zeros(
            (self.num_layers, self.num_kv_heads, 1, self.block_size, self.head_dim),
            dtype=self.dtype,
        )
        self.v = mx.zeros_like(self.k)

    def new_sequence(self, seq_id, prompt_len):
        self._ctx[seq_id] = prompt_len
        return None

    def free(self, seq_id):
        self._ctx.pop(seq_id, None)
        self._prefill_active.discard(seq_id)
        return None

    def write_prefill(
        self, seq_id, layer_idx, k_chunk, v_chunk, start_pos, *, commit=True
    ):
        if commit:
            length = start_pos + int(k_chunk.shape[1])
            self._ctx[seq_id] = max(self._ctx.get(seq_id, 0), length)
        else:
            staged = start_pos + int(k_chunk.shape[1])
            self._ctx[seq_id] = max(self._ctx.get(seq_id, 0), staged)
        return None

    def append_decode_token(self, seq_id, layer_idx, k_token, v_token):
        self._ctx[seq_id] = self._ctx.get(seq_id, 0) + 1
        return None

    def table(self, seq_id):
        return mx.zeros((1,), dtype=mx.int32), int(self._ctx.get(seq_id, 0))

    def batch_tables(self, seq_ids, context_override=None):
        tables = mx.zeros((len(seq_ids), 1), dtype=mx.int32)
        context = mx.zeros((len(seq_ids),), dtype=mx.int32)
        for idx, seq_id in enumerate(seq_ids):
            context[idx] = (
                context_override[idx]
                if context_override is not None
                else int(self._ctx.get(seq_id, 0))
            )
        return tables, context

    def prepare_prefill_view(self, seq_id, chunk_len):
        base = self._ctx.get(seq_id, 0)
        virtual = base + chunk_len
        self._prefill_active.add(seq_id)
        self._ctx[seq_id] = virtual
        return base, virtual

    def commit_prefill(self, seq_id):
        self._prefill_active.discard(seq_id)

    def is_prefill_active(self, seq_id):
        return seq_id in self._prefill_active

    def ensure_decode_capacity(self, seq_ids, tokens=1):
        return None

    def decode_write_targets(self, seq_ids):
        return [0 for _ in seq_ids], [0 for _ in seq_ids]

    def _quant_attention_kwargs(self, layer_idx=None):
        return {}

    def bump_decode_lengths(self, seq_ids, delta=1):
        for seq_id in seq_ids:
            self._ctx[seq_id] = self._ctx.get(seq_id, 0) + delta


class _StubPagedCache:
    def __init__(self, manager):
        self._next = 1
        self.manager = manager
        self.last_view = None
        self._request_to_seq = {}
        self.make_batch_view_calls = 0

    def register(self, request_id, prompt_len):
        seq_id = self._next
        self._next += 1
        self.manager.new_sequence(seq_id, prompt_len)
        self._request_to_seq[request_id] = seq_id
        return seq_id

    def release(self, request_id):
        seq_id = self._request_to_seq.pop(request_id, None)
        if seq_id is not None:
            self.manager.free(seq_id)

    def make_batch_view(self, seq_ids, kv_head_mapping=None, context_override=None):
        self.make_batch_view_calls += 1

        class _DummyView:
            def __init__(self, seq_ids, context_override):
                self.seq_ids = tuple(seq_ids)
                self.kv_head_mapping = kv_head_mapping
                shape = (len(seq_ids), 1)
                self.block_tables = mx.zeros(shape, dtype=mx.int32)
                if context_override is None:
                    context_override = [0] * len(seq_ids)
                self.context_lens = mx.array(context_override, dtype=mx.int32)

            def bump_context(self, delta):
                self.context_lens = self.context_lens + mx.array(delta, dtype=mx.int32)

        self.last_view = _DummyView(seq_ids, context_override)
        return self.last_view

    def begin_prefill(self, request_id, chunk_len, kv_head_mapping=None):
        seq_id = self._request_to_seq[request_id]
        base_len, virtual_len = self.manager.prepare_prefill_view(seq_id, chunk_len)
        self.last_view = types.SimpleNamespace(
            seq_ids=(seq_id,),
            kv_head_mapping=kv_head_mapping,
            context_lens=[virtual_len],
            prefill_base_lens=mx.array([base_len], dtype=mx.int32),
        )
        return types.SimpleNamespace(
            view=self.last_view,
            commit=lambda: self.manager.commit_prefill(seq_id),
        )

    def begin_prefill_many(self, request_ids, chunk_lens, kv_head_mapping=None):
        seq_ids = [self._request_to_seq[req_id] for req_id in request_ids]
        base = []
        virtual = []
        for seq_id, chunk in zip(seq_ids, chunk_lens):
            base_len, virtual_len = self.manager.prepare_prefill_view(
                seq_id, int(chunk)
            )
            base.append(base_len)
            virtual.append(virtual_len)
        view = types.SimpleNamespace(
            seq_ids=tuple(seq_ids),
            kv_head_mapping=kv_head_mapping,
            context_lens=virtual,
            prefill_base_lens=mx.array(base, dtype=mx.int32),
        )

        def _commit():
            for seq_id in seq_ids:
                self.manager.commit_prefill(seq_id)

        return types.SimpleNamespace(view=view, commit=_commit)

    def can_bump_view(self, view, deltas):
        return True


class _StubPrefixCache:
    def __init__(self, *, reuse_len=None):
        self.reuse_len = reuse_len
        self.calls = []


class _AppendOnlyCache:
    def __init__(self):
        self.append_calls = 0
        self.update_calls = 0
        self.offset = 0

    def append_token(self, keys, values):
        self.append_calls += 1
        self.offset += 1

    def update_and_fetch(self, keys, values):
        self.update_calls += 1
        return keys, values

    def make_mask(self, *args, **kwargs):
        return None

    def try_reuse(self, key, seq_id, seq_len):
        self.calls.append(seq_len)
        if self.reuse_len is not None and seq_len == self.reuse_len:
            return seq_len
        return 0

    def record(self, key, seq_id, seq_len):
        return None

    def record_many(self, prefixes, seq_id):
        return None

    def stats(self):
        return {
            "lookups": float(len(self.calls)),
            "hits": 1.0 if self.reuse_len in self.calls else 0.0,
            "hit_rate": 0.0,
        }


class SlotGeneratorVectorizedTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.model = FakeModel()
        self.slot_alloc = SlotAllocator(4)
        self.generator = SlotGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            slot_alloc=self.slot_alloc,
            prefill_chunk=2,
        )

    def _make_context(self, request_id, prompt):
        state = SequenceState(
            request_id=request_id,
            prompt_len=len(prompt),
            max_new_tokens=4,
        )
        ctx = SequenceContext(
            state=state,
            prompt_tokens=list(prompt),
            sampler_settings={},
            stopping_settings={},
            tokenizer=self.tokenizer,
        )
        ctx.detokenizer = self.tokenizer.detokenizer()
        ctx.history_tokens = list(prompt)
        ctx.stop_sequences = []
        ctx.events = queue.Queue()
        ctx.uses_default_sampler = True
        return ctx

    def test_prefill_uses_prompt_chunks(self):
        ctx = self._make_context("req-1", [11, 12, 13])
        self.generator.on_admit(ctx)

        self.assertIsNotNone(ctx.state.slot_id)
        self.assertEqual(ctx.state.prompt_pos, 0)

        self.generator.prefill_step([ctx])

        self.assertEqual(ctx.state.prompt_pos, 2)
        self.assertEqual(ctx.last_token_id, 12)
        np.testing.assert_array_equal(
            self.model.calls[0]["tokens"], np.array([[11, 12]], dtype=np.int32)
        )

        self.generator.prefill_step([ctx])
        self.assertEqual(ctx.state.prompt_pos, 3)
        self.assertEqual(ctx.last_token_id, 13)
        np.testing.assert_array_equal(
            self.model.calls[1]["tokens"], np.array([[13]], dtype=np.int32)
        )

    def test_decode_batches_active_sequences(self):
        ctx_a = self._make_context("req-a", [21, 22])
        ctx_b = self._make_context("req-b", [31])
        self.generator.on_admit(ctx_a)
        self.generator.on_admit(ctx_b)

        ctx_a.state.prompt_pos = ctx_a.state.prompt_len
        ctx_b.state.prompt_pos = ctx_b.state.prompt_len
        ctx_a.last_token_id = ctx_a.history_tokens[-1]
        ctx_b.last_token_id = ctx_b.history_tokens[-1]

        logits = self.generator.decode_step([ctx_a, ctx_b])

        if len(self.model.calls) == 1:
            np.testing.assert_array_equal(
                self.model.calls[0]["tokens"], np.array([[22], [31]], dtype=np.int32)
            )
        else:
            np.testing.assert_array_equal(
                self.model.calls[-2]["tokens"], np.array([[22]], dtype=np.int32)
            )
            np.testing.assert_array_equal(
                self.model.calls[-1]["tokens"], np.array([[31]], dtype=np.int32)
            )
        self.assertEqual(logits.shape[0], 2)
        self.assertEqual(logits.shape[1], self.model.decode_vocab)

    def test_prefill_initializes_and_reuses_prompt_cache(self):
        ctx = self._make_context("req-cache", [41, 42, 43])
        self.generator.on_admit(ctx)

        self.assertIsNotNone(ctx.prompt_cache)

        self.generator.prefill_step([ctx])
        first_cache = self.model.calls[0]["cache"]
        self.assertIs(first_cache, ctx.prompt_cache)

        self.generator.prefill_step([ctx])
        second_cache = self.model.calls[1]["cache"]
        self.assertIs(second_cache, ctx.prompt_cache)

    def test_prefill_slicer_splits_large_chunk(self):
        tokenizer = FakeTokenizer()
        model = FakeModel()
        slot_alloc = SlotAllocator(4)
        generator = SlotGenerator(
            model=model,
            tokenizer=tokenizer,
            slot_alloc=slot_alloc,
            prefill_chunk=4,
        )
        generator.prefill_slicer = PrefillSlicer(
            enabled=True,
            target_ms=1.0,
            hard_cap_ms=2.0,
            min_tokens=1,
            max_tokens=1,
        )
        prompt = [11, 12, 13, 14]
        state = SequenceState(
            request_id="req-slice",
            prompt_len=len(prompt),
            max_new_tokens=4,
        )
        ctx = SequenceContext(
            state=state,
            prompt_tokens=list(prompt),
            sampler_settings={},
            stopping_settings={},
            tokenizer=tokenizer,
        )
        ctx.detokenizer = tokenizer.detokenizer()
        ctx.history_tokens = list(prompt)
        ctx.stop_sequences = []
        ctx.events = queue.Queue()
        ctx.uses_default_sampler = True

        generator.on_admit(ctx)

        generator.prefill_step([ctx])

        self.assertGreaterEqual(len(model.calls), 4)
        stats = generator.prefill_slice_stats()
        self.assertIn("prefill_slice_tokens", stats)
        self.assertEqual(stats["prefill_slice_tokens"], 1.0)

    def test_prefill_ramp_limits_initial_chunk(self):
        generator = SlotGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            slot_alloc=self.slot_alloc,
            prefill_chunk=8,
            prefill_ramp_chunk=3,
        )
        ctx = self._make_context("req-ramp", [71, 72, 73, 74, 75])
        generator.on_admit(ctx)
        generator.prefill_step([ctx])
        self.assertEqual(ctx.state.prompt_pos, 3)
        self.assertEqual(ctx.last_token_id, 73)
        generator.prefill_step([ctx])
        self.assertEqual(ctx.state.prompt_pos, 5)

    def test_decode_uses_prompt_cache(self):
        ctx = self._make_context("req-decode-cache", [51, 52])
        self.generator.on_admit(ctx)

        self.generator.prefill_step([ctx])
        self.generator.prefill_step([ctx])

        logits = self.generator.decode_step([ctx])

        cache_arg = self.model.calls[-1]["cache"]
        self.assertIsInstance(cache_arg, list)
        self.assertGreaterEqual(len(cache_arg), 1)
        self.assertIsInstance(cache_arg[0], SlotBatchCache)
        self.assertEqual(logits.shape[-1], self.model.decode_vocab)

    def test_decode_passes_active_slots_to_batch_cache(self):
        ctx_a = self._make_context("req-a", [61, 62])
        ctx_b = self._make_context("req-b", [71, 72, 73])
        self.generator.on_admit(ctx_a)
        self.generator.on_admit(ctx_b)

        for ctx in (ctx_a, ctx_b):
            while ctx.state.prompt_pos < ctx.state.prompt_len:
                self.generator.prefill_step([ctx])

        logits = self.generator.decode_step([ctx_a, ctx_b])
        self.assertEqual(logits.shape[0], 2)

        cache_arg = self.model.calls[-1]["cache"]
        self.assertIsInstance(cache_arg[0], SlotBatchCache)
        layer_view = cache_arg[0]
        self.assertEqual(len(layer_view.slot_ids), 2)
        slot_ids = [ctx.state.slot_id for ctx in (ctx_a, ctx_b)]
        self.assertEqual(layer_view.slot_ids, tuple(slot_ids))

    def test_release_returns_slot(self):
        ctx = self._make_context("req-slot", [5])
        self.generator.on_admit(ctx)
        slot = ctx.state.slot_id
        self.assertIsNotNone(slot)
        available_before = self.slot_alloc.available()
        self.generator.on_release(ctx)
        self.assertIsNone(ctx.state.slot_id)
        self.assertEqual(self.slot_alloc.available(), available_before + 1)

    def test_paged_backend_builds_paged_prompt_cache(self):
        manager = _StubPagedManager()
        paged_cache = _StubPagedCache(manager)
        self.generator.set_paged_backend(
            paged_cache,
            kv_head_mapping=None,
            manager=manager,
            num_layers=len(self.model.layers),
        )
        ctx = self._make_context("req-paged", [81, 82, 83])
        admitted = self.generator.on_admit(ctx)
        self.assertTrue(admitted)
        self.assertIsNotNone(ctx.state.paged_seq_id)
        self.assertTrue(
            all(isinstance(cache, PagedKVCacheAdapter) for cache in ctx.prompt_cache)
        )

    def test_paged_batch_cache_appends_single_decode_tokens(self):
        cache = _AppendOnlyCache()
        ctx = types.SimpleNamespace(
            state=types.SimpleNamespace(finished=False),
            prompt_cache=[cache],
            _kv_lengths=[0],
        )
        batch_cache = PagedBatchCache(layer_idx=0)
        batch_cache.bind([ctx])
        keys = mx.zeros((1, 1, 1, 4), dtype=mx.float32)
        values = mx.zeros_like(keys)

        batch_cache.update_and_fetch(keys, values)

        self.assertEqual(cache.append_calls, 1)
        self.assertEqual(cache.update_calls, 0)
        self.assertEqual(ctx._kv_lengths[0], cache.offset)

    def test_paged_backend_reuses_longest_prefix(self):
        manager = _StubPagedManager()
        manager.block_size = 2
        paged_cache = _StubPagedCache(manager)
        prefix_cache = _StubPrefixCache(reuse_len=4)
        self.generator.set_paged_backend(
            paged_cache,
            kv_head_mapping=None,
            manager=manager,
            num_layers=len(self.model.layers),
            prefix_cache=prefix_cache,
        )
        ctx = self._make_context("req-apc", [1, 2, 3, 4, 5, 6])
        admitted = self.generator.on_admit(ctx)
        self.assertTrue(admitted)
        self.assertIs(self.generator.prefix_cache, prefix_cache)

    def test_prefill_enters_paged_batch_scope(self):
        manager = _StubPagedManager()
        paged_cache = _StubPagedCache(manager)
        self.generator.set_paged_backend(
            paged_cache,
            kv_head_mapping=mx.array([0], dtype=mx.int32),
            manager=manager,
            num_layers=len(self.model.layers),
        )
        ctx = self._make_context("req-scope", [91, 92, 93])
        admitted = self.generator.on_admit(ctx)
        self.assertTrue(admitted)
        seq_id = ctx.state.paged_seq_id
        self.assertIsNotNone(seq_id)

        tokens = ctx.state.remaining_prompt_tokens
        self.assertGreater(tokens, 0)

        consumed = self.generator.prefill_tokens(ctx, tokens)

        self.assertEqual(consumed, 2)
        self.assertIsNotNone(paged_cache.last_view)
        self.assertEqual(paged_cache.last_view.seq_ids, (seq_id,))

    def test_paged_prefill_skips_slot_slab_allocation(self):
        manager = _StubPagedManager()
        paged_cache = _StubPagedCache(manager)
        self.generator.set_paged_backend(
            paged_cache,
            kv_head_mapping=mx.array([0], dtype=mx.int32),
            manager=manager,
            num_layers=len(self.model.layers),
        )
        ctx = self._make_context("req-slab", [101, 102])
        admitted = self.generator.on_admit(ctx)
        self.assertTrue(admitted)
        self.assertIsNone(self.generator._slab)

        self.generator.prefill_step([ctx])

        self.assertIsNone(self.generator._slab)

    def test_paged_decode_uses_paged_batch_cache(self):
        manager = _StubPagedManager()
        paged_cache = _StubPagedCache(manager)
        self.generator.set_paged_backend(
            paged_cache,
            kv_head_mapping=mx.array([0], dtype=mx.int32),
            manager=manager,
            num_layers=len(self.model.layers),
        )
        ctx = self._make_context("req-decode", [5, 6, 7])
        self.generator.on_admit(ctx)
        while ctx.state.remaining_prompt_tokens > 0:
            self.generator.prefill_step([ctx])
        self.assertIsNone(self.generator._slab)

        logits = self.generator.decode_step([ctx])
        self.assertEqual(logits.shape[0], 1)
        self.assertIsNone(self.generator._slab)

        cache_layers = self.model.calls[-1]["cache"]
        self.assertTrue(
            all(isinstance(layer, PagedBatchCache) for layer in cache_layers)
        )

    def test_decode_reuses_batch_cache_instance_when_appending_context(self):
        ctx_a = self._make_context("req-a", [11, 12])
        ctx_b = self._make_context("req-b", [21, 22])
        self.generator.on_admit(ctx_a)
        while ctx_a.state.prompt_pos < ctx_a.state.prompt_len:
            self.generator.prefill_step([ctx_a])

        self.generator.decode_step([ctx_a])
        cache_before = self.model.calls[-1]["cache"][0]

        self.generator.on_admit(ctx_b)
        while ctx_b.state.prompt_pos < ctx_b.state.prompt_len:
            self.generator.prefill_step([ctx_b])

        self.generator.decode_step([ctx_a, ctx_b])
        cache_after = self.model.calls[-1]["cache"][0]

        self.assertIs(cache_before, cache_after)

    def test_decode_state_reuses_paged_view_until_boundary(self):
        manager = _StubPagedManager()
        paged_cache = _StubPagedCache(manager)
        self.generator.set_paged_backend(
            paged_cache,
            kv_head_mapping=mx.array([0], dtype=mx.int32),
            manager=manager,
            num_layers=len(self.model.layers),
        )
        ctx = self._make_context("req-reuse", [1, 2, 3])
        self.generator.on_admit(ctx)
        while ctx.state.remaining_prompt_tokens > 0:
            self.generator.prefill_step([ctx])
        decode_state = {"view": None, "seq_ids": ()}
        active = [ctx]
        self.generator.decode_step(active, decode_state)
        first_calls = paged_cache.make_batch_view_calls
        self.generator.update_decode_state(decode_state, active, safe_bump=True)
        self.generator.decode_step(active, decode_state)
        self.assertEqual(paged_cache.make_batch_view_calls, first_calls)

    def test_decode_state_rebuilds_when_safe_disabled(self):
        manager = _StubPagedManager()
        paged_cache = _StubPagedCache(manager)
        self.generator.set_paged_backend(
            paged_cache,
            kv_head_mapping=mx.array([0], dtype=mx.int32),
            manager=manager,
            num_layers=len(self.model.layers),
        )
        ctx = self._make_context("req-rebuild", [5, 6, 7])
        self.generator.on_admit(ctx)
        while ctx.state.remaining_prompt_tokens > 0:
            self.generator.prefill_step([ctx])
        decode_state = {"view": None, "seq_ids": ()}
        active = [ctx]
        self.generator.decode_step(active, decode_state)
        self.generator.update_decode_state(decode_state, active, safe_bump=False)
        self.generator.decode_step(active, decode_state)
        self.assertGreaterEqual(paged_cache.make_batch_view_calls, 2)


class SlotGeneratorArrayDecodeTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.slot_alloc = SlotAllocator(max_slots=4)
        self.model = FakeModel()
        self._overlay_stub = None
        if not hasattr(mx.fast, "_paged_attention_with_overlay_impl"):
            self._overlay_stub = object()
            mx.fast._paged_attention_with_overlay_impl = lambda *_, **__: mx.zeros(
                (1, 1, 1, 1), dtype=mx.float16
            )

    def tearDown(self):
        if self._overlay_stub is not None:
            try:
                delattr(mx.fast, "_paged_attention_with_overlay_impl")
            except AttributeError:
                pass

    def _make_context(self, request_id, prompt):
        state = SequenceState(
            request_id=request_id,
            prompt_len=len(prompt),
            max_new_tokens=4,
        )
        ctx = SequenceContext(
            state=state,
            prompt_tokens=list(prompt),
            sampler_settings={},
            stopping_settings={},
            tokenizer=self.tokenizer,
        )
        ctx.detokenizer = self.tokenizer.detokenizer()
        ctx.history_tokens = list(prompt)
        ctx.stop_sequences = []
        ctx.events = queue.Queue()
        ctx.uses_default_sampler = True
        return ctx

    def _make_generator(self, decode_engine="dense", **kwargs):
        gen = SlotGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            slot_alloc=self.slot_alloc,
            prefill_chunk=16,
            decode_engine=decode_engine,
            **kwargs,
        )
        gen.kv_block_size = 16
        return gen

    @mock.patch("mlx_lm.server_batched.slot_generator.PrefillArrayRunner")
    @mock.patch("mlx_lm.server_batched.slot_generator.ArrayDecodeRunner")
    def test_set_paged_backend_initializes_array_runner(
        self, mock_runner, mock_prefill
    ):
        mock_runner.SUPPORTED_ENGINES = {"paged-arrays", "paged-arrays+compile"}
        manager = _StubPagedManager()
        paged_cache = _StubPagedCache(manager)
        generator = self._make_generator(
            decode_engine="paged-arrays",
            array_decode_runner_cls=mock_runner,
            prefill_array_runner_cls=mock_prefill,
        )
        generator.set_paged_backend(
            paged_cache,
            kv_head_mapping=None,
            manager=manager,
            num_layers=manager.num_layers,
        )
        self.assertIsNotNone(generator.array_decode_runner)
        mock_runner.assert_called_once()
        mock_prefill.return_value.warmup.assert_called_once()
        warm_args = mock_prefill.return_value.warmup.call_args
        expected_batches = []
        value = 1
        cap = min(self.slot_alloc.capacity, 8)
        while value < cap:
            expected_batches.append(value)
            value <<= 1
        expected_batches.append(cap)
        self.assertEqual(tuple(warm_args.kwargs["batch_size"]), tuple(expected_batches))
        self.assertEqual(
            tuple(warm_args.kwargs["chunk_len"]), (1, 2, 4, 8, generator.prefill_chunk)
        )

    def test_array_prefill_chunks_are_bucketized(self):
        manager = _StubPagedManager()
        paged_cache = _StubPagedCache(manager)
        generator = self._make_generator(
            decode_engine="paged-arrays",
            prefill_ramp_chunk=0,
            prefill_ramp_budget_ms=None,
        )
        generator.paged_cache = paged_cache
        generator._paged_manager = manager
        generator._paged_num_layers = manager.num_layers
        generator.use_array_decode = True
        generator.use_array_prefill = True

        class _FakeArraysProvider:
            def build_view(self, view, seq_ids, decode_steps=1):
                batch = len(seq_ids)
                block_tables = mx.zeros(
                    (batch, manager.max_blocks_per_sequence), dtype=mx.int32
                )
                context_lens = mx.zeros((batch,), dtype=mx.int32)
                arrays = PagedDecodeArrays(
                    seq_ids=tuple(seq_ids),
                    block_tables=block_tables,
                    context_lens=context_lens,
                    kv_head_mapping=None,
                    k_cache=manager.k,
                    v_cache=manager.v,
                    quant_kwargs=[],
                    write_block_ids=mx.zeros((batch,), dtype=mx.int32),
                    write_token_offsets=mx.zeros((batch,), dtype=mx.int32),
                    prefill_base_lens=context_lens,
                    signature=None,
                    active_rows=batch,
                )
                return ArrayBatchView(arrays)

        generator._arrays_provider = _FakeArraysProvider()
        runner = mock.Mock()
        runner.prefill_chunk.return_value = 8
        runner.consume_metrics.return_value = {}
        runner.overlay_seq_ids.return_value = ()
        runner.has_active_overlays.return_value = False
        generator.prefill_array_runner = runner
        ctx = self._make_context("req-bucket", list(range(20)))
        generator.on_admit(ctx)
        counts = generator.prefill_tokens_multi([ctx], chunk_len=12)
        self.assertEqual(counts, [8])
        call = runner.prefill_chunk.call_args
        self.assertEqual(call.kwargs["chunk_len"], 8)

    def test_array_prefill_marks_decode_ready_after_ramp(self):
        manager = _StubPagedManager()
        paged_cache = _StubPagedCache(manager)
        generator = self._make_generator(
            decode_engine="paged-arrays",
            prefill_ramp_chunk=2,
            prefill_ramp_budget_ms=None,
        )
        generator.paged_cache = paged_cache
        generator._paged_manager = manager
        generator._paged_num_layers = manager.num_layers
        generator.use_array_decode = True
        generator.use_array_prefill = True

        class _Provider:
            def build_view(self, view, seq_ids, decode_steps=1):
                batch = len(seq_ids)
                tables = mx.zeros(
                    (batch, manager.max_blocks_per_sequence), dtype=mx.int32
                )
                lens = mx.zeros((batch,), dtype=mx.int32)
                arrays = PagedDecodeArrays(
                    seq_ids=tuple(seq_ids),
                    block_tables=tables,
                    context_lens=lens,
                    kv_head_mapping=None,
                    k_cache=manager.k,
                    v_cache=manager.v,
                    quant_kwargs=[],
                    write_block_ids=mx.zeros((batch,), dtype=mx.int32),
                    write_token_offsets=mx.zeros((batch,), dtype=mx.int32),
                    prefill_base_lens=lens,
                    signature=None,
                    active_rows=batch,
                )
                return ArrayBatchView(arrays)

        generator._arrays_provider = _Provider()
        runner = mock.Mock()
        runner.prefill_chunk.return_value = 2
        runner.consume_metrics.return_value = {}
        runner.overlay_seq_ids.return_value = ()
        runner.has_active_overlays.return_value = False
        generator.prefill_array_runner = runner
        ctx = self._make_context("req-array-ready", [1, 2, 3, 4, 5, 6])
        admitted = generator.on_admit(ctx)
        self.assertTrue(admitted)
        self.assertIsNotNone(ctx.state.paged_seq_id)
        initial_ramp = ctx.state.prefill_ramp_remaining
        self.assertGreater(initial_ramp, 0)

        counts = generator.prefill_tokens_multi([ctx], chunk_len=8)
        self.assertEqual(counts, [2])
        runner.prefill_chunk.assert_called_once()
        self.assertEqual(ctx.state.prefill_ramp_remaining, 0)
        self.assertTrue(ctx.state.prefill_decode_ready)
        self.assertGreater(ctx.state.remaining_prompt_tokens, 0)

    @mock.patch("mlx_lm.server_batched.slot_generator.PrefillArrayRunner")
    @mock.patch("mlx_lm.server_batched.slot_generator.ArrayDecodeRunner")
    def test_call_model_routes_through_array_runner(self, mock_runner, mock_prefill):
        mock_runner.SUPPORTED_ENGINES = {"paged-arrays", "paged-arrays+compile"}
        arrays = PagedDecodeArrays(
            seq_ids=(),
            block_tables=mx.zeros((0, 1), dtype=mx.int32),
            context_lens=mx.zeros((0,), dtype=mx.int32),
            kv_head_mapping=None,
            k_cache=mx.zeros((1, 1, 1, 1, 1), dtype=mx.float16),
            v_cache=mx.zeros((1, 1, 1, 1, 1), dtype=mx.float16),
            quant_kwargs=[],
            write_block_ids=mx.zeros((0,), dtype=mx.int32),
            write_token_offsets=mx.zeros((0,), dtype=mx.int32),
            prefill_base_lens=None,
            signature=None,
            active_rows=0,
        )
        view = ArrayBatchView(arrays)
        generator = self._make_generator(decode_engine="paged-arrays")
        generator.array_decode_runner = mock_runner.return_value
        generator.use_array_decode = True
        inputs = mx.array([[42]], dtype=mx.int32)
        result = generator._call_model(inputs, cache=None, view=view)
        mock_runner.return_value.decode.assert_called_once()
        call_args = mock_runner.return_value.decode.call_args
        self.assertIs(call_args.kwargs.get("pending_state"), None)
        self.assertEqual(result, mock_runner.return_value.decode.return_value)

    @mock.patch("mlx_lm.server_batched.slot_generator.PrefillArrayRunner")
    @mock.patch("mlx_lm.server_batched.slot_generator.ArrayDecodeRunner")
    def test_phase1_compile_stats_increment(self, mock_runner, mock_prefill):
        mock_runner.SUPPORTED_ENGINES = {"paged-arrays", "paged-arrays+compile"}
        mock_runner.return_value.decode.return_value = mx.zeros(
            (1, 1, 1), dtype=mx.float32
        )
        mock_runner.return_value.compile_stats.return_value = {
            "array_phase1_compile_hits": 3.0,
            "array_phase1_compile_misses": 1.0,
            "array_phase1_duration_s": 0.123,
        }
        generator = self._make_generator(
            decode_engine="paged-arrays+compile",
            array_decode_runner_cls=mock_runner,
            prefill_array_runner_cls=mock_prefill,
        )
        manager = _StubPagedManager()
        paged_cache = _StubPagedCache(manager)
        generator.set_paged_backend(
            paged_cache,
            kv_head_mapping=None,
            manager=manager,
            num_layers=manager.num_layers,
        )
        stats = generator.compile_stats()
        self.assertEqual(stats.get("array_phase1_compile_hits"), 3.0)
        self.assertEqual(stats.get("array_phase1_compile_misses"), 1.0)
        self.assertEqual(stats.get("array_phase1_duration_s"), 0.123)

    @mock.patch("mlx_lm.server_batched.slot_generator.PrefillArrayRunner")
    @mock.patch("mlx_lm.server_batched.slot_generator.ArrayDecodeRunner")
    def test_update_decode_state_flushes_pending_on_invalidate(
        self, mock_runner, mock_prefill
    ):
        mock_runner.SUPPORTED_ENGINES = {"paged-arrays", "paged-arrays+compile"}
        generator = self._make_generator(decode_engine="paged-arrays")
        generator.array_decode_runner = mock_runner.return_value
        ctx = self._make_context("req-flush", [9])
        ctx.state.paged_seq_id = "seq-a"
        pending = object()
        decode_state = {
            "view": object(),
            "seq_ids": ("seq-a",),
            "pending": pending,
        }
        generator.update_decode_state(decode_state, [ctx], safe_bump=False)
        mock_runner.return_value.flush_pending.assert_called_once_with(pending)
        self.assertIsNone(decode_state["pending"])

    @mock.patch("mlx_lm.server_batched.slot_generator.PrefillArrayRunner")
    @mock.patch("mlx_lm.server_batched.slot_generator.ArrayDecodeRunner")
    def test_array_prefill_metrics_recorded(self, mock_runner, mock_prefill):
        mock_runner.SUPPORTED_ENGINES = {"paged-arrays", "paged-arrays+compile"}
        generator = self._make_generator(
            decode_engine="paged-arrays",
            array_decode_runner_cls=mock_runner,
            prefill_array_runner_cls=mock_prefill,
        )
        manager = _StubPagedManager()
        paged_cache = _StubPagedCache(manager)
        generator.set_paged_backend(
            paged_cache,
            kv_head_mapping=None,
            manager=manager,
            num_layers=manager.num_layers,
        )

        class DummyPrefillRunner:
            def __init__(self):
                self.tokens = None
                self._metrics = {
                    "array_prefill_graph_s": 0.005,
                    "array_prefill_writer_s": 0.002,
                }

            def prefill_chunk(self, token_matrix, view, chunk_len):
                self.tokens = token_matrix
                return chunk_len

            def consume_metrics(self):
                stats = dict(self._metrics)
                self._metrics = {k: 0.0 for k in self._metrics}
                return stats

            def collect_overlays(self, seq_ids, max_tokens=None):
                return None

            def release_overlays(self, seq_ids):
                return None

            def has_active_overlays(self):
                return False

        generator.prefill_array_runner = DummyPrefillRunner()
        generator.use_array_prefill = True
        ctx = self._make_context("req-metrics", [1, 2, 3])
        self.assertTrue(generator.on_admit(ctx))
        counts = generator.prefill_tokens_multi([ctx], chunk_len=1)
        self.assertEqual(counts[0], 1)
        stats = generator.prefill_slice_stats()
        self.assertGreater(stats.get("array_prefill_graph_s", 0.0), 0.0)
        self.assertGreater(stats.get("array_prefill_writer_s", 0.0), 0.0)

    @mock.patch("mlx_lm.server_batched.slot_generator.PrefillArrayRunner")
    @mock.patch("mlx_lm.server_batched.slot_generator.ArrayDecodeRunner")
    def test_array_prefill_groups_by_effective_chunk_len(
        self, mock_runner, mock_prefill
    ):
        mock_runner.SUPPORTED_ENGINES = {"paged-arrays", "paged-arrays+compile"}
        generator = self._make_generator(
            decode_engine="paged-arrays",
            array_decode_runner_cls=mock_runner,
            prefill_array_runner_cls=mock_prefill,
        )
        manager = _StubPagedManager()
        paged_cache = _StubPagedCache(manager)
        generator.set_paged_backend(
            paged_cache,
            kv_head_mapping=None,
            manager=manager,
            num_layers=manager.num_layers,
        )

        class DummyPrefillRunner:
            def __init__(self):
                self.calls: List[int] = []

            def prefill_chunk(self, token_matrix, view, *, chunk_len):
                self.calls.append(chunk_len)
                return chunk_len

            def consume_metrics(self):
                return {}

            def collect_overlays(self, seq_ids, max_tokens=None):
                return None

            def release_overlays(self, seq_ids):
                return None

            def has_active_overlays(self):
                return False

        dummy_runner = DummyPrefillRunner()
        generator.prefill_array_runner = dummy_runner
        generator.use_array_prefill = True

        ctx_short = self._make_context("req-short", [1])
        ctx_long = self._make_context("req-long", [10, 11, 12, 13])
        for ctx in (ctx_short, ctx_long):
            self.assertTrue(generator.on_admit(ctx))
        counts = generator.prefill_tokens_multi(
            [ctx_short, ctx_long],
            chunk_len=3,
        )
        self.assertEqual(counts, [1, 2])
        self.assertEqual(sorted(dummy_runner.calls), [1, 2])

    @mock.patch("mlx_lm.server_batched.slot_generator.PrefillArrayRunner")
    @mock.patch("mlx_lm.server_batched.slot_generator.ArrayDecodeRunner")
    def test_decode_advances_prefill_overlays(self, mock_runner, mock_prefill):
        mock_runner.SUPPORTED_ENGINES = {"paged-arrays", "paged-arrays+compile"}
        generator = self._make_generator(
            decode_engine="paged-arrays",
            array_decode_runner_cls=mock_runner,
            prefill_array_runner_cls=mock_prefill,
        )
        generator.prefill_ramp_chunk = 3
        manager = _StubPagedManager()
        paged_cache = _StubPagedCache(manager)
        generator.set_paged_backend(
            paged_cache,
            kv_head_mapping=None,
            manager=manager,
            num_layers=manager.num_layers,
        )
        generator.prefill_array_runner = fake_prefill = mock_prefill.return_value
        generator.use_array_prefill = True
        seq_ids = (1,)
        overlay_tokens = 2
        k_layers = mx.zeros(
            (
                manager.num_layers,
                overlay_tokens,
                len(seq_ids),
                manager.num_kv_heads,
                manager.head_dim,
            ),
            dtype=mx.float16,
        )
        v_layers = mx.zeros_like(k_layers)
        overlay_batch = PrefillArrayRunner.PrefillOverlayBatch(
            seq_ids=seq_ids,
            tokens=overlay_tokens,
            base_lens=mx.zeros((len(seq_ids),), dtype=mx.int32),
            k_layers=k_layers,
            v_layers=v_layers,
        )
        fake_prefill.collect_overlays.return_value = overlay_batch
        fake_prefill.overlay_seq_ids.return_value = seq_ids
        ctx = self._make_context("req-advance", [5, 6, 7])
        self.assertTrue(generator.on_admit(ctx))
        generator.prefill_step([ctx])
        generator.decode_step([ctx])
        fake_prefill.collect_overlays.assert_called()
        call_args = fake_prefill.collect_overlays.call_args
        self.assertGreaterEqual(len(call_args.args), 2)
        self.assertEqual(
            call_args.args[1],
            1,
        )
        fake_prefill.advance_and_flush.assert_called_with(mock.ANY, overlay_tokens)

    @mock.patch("mlx_lm.server_batched.slot_generator.PrefillArrayRunner")
    @mock.patch("mlx_lm.server_batched.slot_generator.ArrayDecodeRunner")
    def test_array_overlays_skip_sequences_without_handles(
        self, mock_runner, mock_prefill
    ):
        mock_runner.SUPPORTED_ENGINES = {"paged-arrays", "paged-arrays+compile"}
        generator = self._make_generator(
            decode_engine="paged-arrays",
            array_decode_runner_cls=mock_runner,
            prefill_array_runner_cls=mock_prefill,
        )
        generator.prefill_ramp_chunk = 1
        manager = _StubPagedManager()
        paged_cache = _StubPagedCache(manager)
        generator.set_paged_backend(
            paged_cache,
            kv_head_mapping=None,
            manager=manager,
            num_layers=manager.num_layers,
        )
        generator.prefill_array_runner = fake_prefill = mock_prefill.return_value
        fake_prefill.consume_metrics.return_value = {}
        generator.use_array_prefill = True
        overlay_tokens = 1

        ctx_a = self._make_context("req-overlay-a", [11, 12, 13])
        ctx_b = self._make_context("req-overlay-b", [21, 22, 23])
        self.assertTrue(generator.on_admit(ctx_a))
        self.assertTrue(generator.on_admit(ctx_b))
        seq_a = ctx_a.state.paged_seq_id
        self.assertIsNotNone(seq_a)
        k_layers = mx.zeros(
            (
                manager.num_layers,
                overlay_tokens,
                1,
                manager.num_kv_heads,
                manager.head_dim,
            ),
            dtype=mx.float16,
        )
        v_layers = mx.zeros_like(k_layers)
        overlay_batch = PrefillArrayRunner.PrefillOverlayBatch(
            seq_ids=(seq_a,),
            tokens=overlay_tokens,
            base_lens=mx.zeros((1,), dtype=mx.int32),
            k_layers=k_layers,
            v_layers=v_layers,
        )
        fake_prefill.collect_overlays.return_value = overlay_batch
        fake_prefill.overlay_seq_ids.return_value = (seq_a,)

        for ctx in (ctx_a, ctx_b):
            generator.prefill_step([ctx])
            ctx.state.prompt_pos = ctx.state.prompt_len
            ctx.last_token_id = ctx.history_tokens[-1]

        generator.decode_step([ctx_a, ctx_b])

        fake_prefill.overlay_seq_ids.assert_called()
        call_args = fake_prefill.collect_overlays.call_args
        self.assertEqual(call_args.args[0], (seq_a,))
        fake_prefill.advance_and_flush.assert_called_with((seq_a,), overlay_tokens)

    def test_slot_batch_cache_right_aligns_sequences(self):
        from mlx_lm.server_batched.slot_generator import SlotBatchCache, SlotKVSlab

        n_heads = 1
        head_dim = 2

        class DummyCache:
            def __init__(self, tokens, seed):
                self.offset = tokens
                data = (
                    np.arange(tokens * n_heads * head_dim, dtype=np.float32).reshape(
                        1, n_heads, tokens, head_dim
                    )
                    + seed
                )
                self.keys = mx.array(data)
                self.values = mx.array(data + 100.0)

            def update_and_fetch(self, keys, values):
                return keys, values

            @property
            def state(self):
                return self.keys, self.values

        cache_short = DummyCache(tokens=2, seed=0.0)
        cache_long = DummyCache(tokens=5, seed=10.0)

        slab = SlotKVSlab(max_slots=4)
        ctx_short = types.SimpleNamespace(
            state=types.SimpleNamespace(slot_id=0),
            prompt_cache=[cache_short],
            _kv_lengths=[cache_short.offset],
        )
        ctx_long = types.SimpleNamespace(
            state=types.SimpleNamespace(slot_id=1),
            prompt_cache=[cache_long],
            _kv_lengths=[cache_long.offset],
        )

        slab.append_from_cache(
            0, ctx_short.state.slot_id, cache_short, 0, cache_short.offset
        )
        slab.append_from_cache(
            0, ctx_long.state.slot_id, cache_long, 0, cache_long.offset
        )

        batch_cache = SlotBatchCache(slab, layer_idx=0)
        batch_cache.bind([ctx_short, ctx_long])
        keys, _, offsets, left_padding = batch_cache.state

        self.assertEqual(left_padding.tolist(), [3, 0])
        self.assertEqual(offsets.tolist(), [2 + 3, 5])

        keys_list = keys.tolist()
        # Short sequence should have zeros in the first 3 positions.
        leading = keys_list[0][0][:3]
        self.assertTrue(all(all(abs(v) < 1e-6 for v in vec) for vec in leading))
        trailing = keys_list[0][0][3:]
        self.assertTrue(any(abs(v) > 0 for vec in trailing for v in vec))


class SlotGeneratorHybridDenseTests(unittest.TestCase):
    @mock.patch("mlx_lm.server_batched.slot_generator.PrefillArrayRunner")
    @mock.patch("mlx_lm.server_batched.slot_generator.ArrayDecodeRunner")
    @mock.patch("mlx_lm.server_batched.slot_generator.PagedArraysProvider")
    def test_hybrid_threshold_uses_dense_before_paged(
        self, mock_provider, mock_decode_runner, mock_prefill_runner
    ):
        model = FakeModel()
        tokenizer = FakeTokenizer()
        slot_alloc = SlotAllocator(4)
        mock_decode_runner.SUPPORTED_ENGINES = {"paged-arrays", "paged-arrays+compile"}
        generator = SlotGenerator(
            model=model,
            tokenizer=tokenizer,
            slot_alloc=slot_alloc,
            prefill_chunk=4,
            decode_engine="paged-arrays",
            prefill_ramp_chunk=4,
            prefill_hybrid_threshold=2,
        )
        self.assertEqual(generator.prefill_hybrid_dense_tokens, 2)
        manager = _StubPagedManager()
        paged_cache = _StubPagedCache(manager)
        generator.set_paged_backend(
            paged_cache,
            kv_head_mapping=None,
            manager=manager,
            num_layers=manager.num_layers,
        )
        generator.prefill_array_runner = mock_prefill_runner.return_value
        generator.use_array_prefill = True

        state = SequenceState(
            request_id="req-hybrid",
            prompt_len=4,
            max_new_tokens=1,
        )
        state.hybrid_dense_remaining = generator.prefill_hybrid_dense_tokens
        ctx = SequenceContext(
            state=state,
            prompt_tokens=[1, 2, 3, 4],
            sampler_settings={},
            stopping_settings={},
            tokenizer=tokenizer,
        )
        ctx.prompt_ids_np = np.array(ctx.prompt_tokens, dtype=np.int32)
        ctx.prompt_cache = model.make_cache()
        admitted = generator.on_admit(ctx)
        self.assertTrue(admitted)
        self.assertEqual(
            ctx.state.hybrid_dense_remaining, generator.prefill_hybrid_dense_tokens
        )

        def fake_paged_batch(entries, chunk_len):
            return [min(chunk_len, take) for (_, _, take) in entries]

        with mock.patch.object(
            generator,
            "_prefill_tokens_array_batch",
            return_value=None,
        ), mock.patch.object(
            generator,
            "_prefill_tokens_paged_batch",
            side_effect=fake_paged_batch,
        ) as paged_mock:
            generator.prefill_tokens_multi([ctx], generator.prefill_chunk)
            self.assertEqual(paged_mock.call_count, 0)
            self.assertEqual(ctx.state.hybrid_dense_remaining, 0)
            generator.prefill_tokens_multi([ctx], generator.prefill_chunk)
            self.assertEqual(paged_mock.call_count, 1)


if __name__ == "__main__":
    unittest.main()
