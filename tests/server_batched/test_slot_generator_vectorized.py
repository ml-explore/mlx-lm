# ABOUTME: Tests slot-indexed generator vectorized decode path.
# ABOUTME: Ensures prefill and decode operate on stable slot assignments.

from .util import ensure_mlx_stub

ensure_mlx_stub()

import queue
import types
import unittest

import numpy as np

from mlx_lm.server_batched.slot_allocator import SlotAllocator
from mlx_lm.server_batched.state import SequenceContext, SequenceState
from mlx_lm.server_batched.slot_generator import SlotGenerator


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

    def __call__(self, tokens):
        arr = np.array(tokens)
        self.calls.append(arr.copy())
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
        np.testing.assert_array_equal(self.model.calls[0], np.array([[11, 12]], dtype=np.int32))

        self.generator.prefill_step([ctx])
        self.assertEqual(ctx.state.prompt_pos, 3)
        self.assertEqual(ctx.last_token_id, 13)
        np.testing.assert_array_equal(self.model.calls[1], np.array([[13]], dtype=np.int32))

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

        np.testing.assert_array_equal(self.model.calls[-1], np.array([[22], [31]], dtype=np.int32))
        self.assertEqual(logits.shape[0], 2)
        self.assertEqual(logits.shape[1], self.model.decode_vocab)

    def test_release_returns_slot(self):
        ctx = self._make_context("req-slot", [5])
        self.generator.on_admit(ctx)
        slot = ctx.state.slot_id
        self.assertIsNotNone(slot)
        available_before = self.slot_alloc.available()
        self.generator.on_release(ctx)
        self.assertIsNone(ctx.state.slot_id)
        self.assertEqual(self.slot_alloc.available(), available_before + 1)


if __name__ == "__main__":
    unittest.main()
