# ABOUTME: Tests slot-indexed generator vectorized decode path.
# ABOUTME: Ensures prefill and decode operate on stable slot assignments.

from .util import ensure_mlx_stub

ensure_mlx_stub()

import queue
import types
import unittest

import numpy as np
import mlx.core as mx

from mlx_lm.server_batched.slot_allocator import SlotAllocator
from mlx_lm.server_batched.state import SequenceContext, SequenceState
from mlx_lm.server_batched.slot_generator import SlotBatchCache, SlotGenerator


class FakeTokenizer:
    bos_token_id = 101
    eos_token_id = 102

    def detokenizer(self):
        detok = types.SimpleNamespace()
        detok.add_token = lambda *_: None
        detok.last_segment = ""
        detok.finalize = lambda: None
        return detok


class FakeModel:
    def __init__(self):
        self.calls = []
        self.decode_vocab = 5

    def __call__(self, tokens, cache=None):
        arr = np.array(tokens)
        self.calls.append({"tokens": arr.copy(), "cache": cache})
        if arr.ndim == 2 and arr.shape[1] > 1:
            # Prefill path, logits unused.
            return np.zeros((arr.shape[0], arr.shape[1], self.decode_vocab), dtype=np.float32)
        if arr.ndim == 2:
            batch = arr.shape[0]
            logits = np.stack(
                [np.linspace(i, i + 1, self.decode_vocab, dtype=np.float32) for i in range(batch)],
                axis=0,
            )
            return logits[:, None, :]
        raise AssertionError(f"Unexpected tokens shape {arr.shape}")




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

    def test_slot_batch_cache_right_aligns_sequences(self):
        from mlx_lm.server_batched.slot_generator import SlotBatchCache, SlotKVSlab

        n_heads = 1
        head_dim = 2

        class DummyCache:
            def __init__(self, tokens, seed):
                self.offset = tokens
                data = np.arange(tokens * n_heads * head_dim, dtype=np.float32).reshape(
                    1, n_heads, tokens, head_dim
                ) + seed
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

        slab.append_from_cache(0, ctx_short.state.slot_id, cache_short, 0, cache_short.offset)
        slab.append_from_cache(0, ctx_long.state.slot_id, cache_long, 0, cache_long.offset)

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


if __name__ == "__main__":
    unittest.main()
