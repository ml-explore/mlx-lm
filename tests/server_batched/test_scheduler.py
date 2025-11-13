# ABOUTME: Scheduler unit tests verifying prefill/decode iteration semantics.
# ABOUTME: Ensures continuous batching scheduler admits work between decode steps.

from .util import ensure_mlx_stub

ensure_mlx_stub()

import types
import unittest

from mlx_lm.server_batched.scheduler import Scheduler
from mlx_lm.server_batched.state import SequenceContext, SequenceState


class DummyTokenizer:
    eos_token_id = 0

    @property
    def detokenizer(self):
        detok = types.SimpleNamespace()
        detok.add_token = lambda *_: None
        detok.finalize = lambda: None
        detok.last_segment = ""
        return detok


class FakeRunner:
    def __init__(self, finish_after):
        self.finish_after = dict(finish_after)
        self.prefill_calls = []
        self.decode_calls = []
        self._prefill_count = 0
        self._prefill_tokens = 0

    def begin_step(self):
        self._prefill_count = 0
        self._prefill_tokens = 0

    def prefill_context(self, ctx, tokens):
        self.prefill_calls.append((ctx.state.request_id, tokens))
        ctx.state.prompt_pos += tokens
        self._prefill_count += 1
        self._prefill_tokens += tokens
        return tokens

    def prefill_context_batch(self, contexts, chunk_len):
        outputs = []
        for ctx in contexts:
            outputs.append(self.prefill_context(ctx, chunk_len))
        return outputs

    def decode(self, contexts):
        self.decode_calls.append([ctx.state.request_id for ctx in contexts])
        for ctx in contexts:
            remaining = self.finish_after[ctx.state.request_id] - 1
            self.finish_after[ctx.state.request_id] = remaining
            ctx.state.generated_tokens += 1
            if remaining <= 0:
                ctx.state.finished = True
        return {
            "decode_iterations": 1 if contexts else 0,
            "decode_tokens": len(contexts),
            "decode_duration_s": 0.0,
            "prefill_calls": self._prefill_count,
            "prefill_tokens": self._prefill_tokens,
            "prefill_duration_s": 0.0,
        }

    def collect_step_stats(self):
        return {
            "prefill_calls": self._prefill_count,
            "prefill_tokens": self._prefill_tokens,
            "prefill_duration_s": 0.0,
            "decode_iterations": 0,
            "decode_tokens": 0,
            "decode_duration_s": 0.0,
        }


class LegacyRunner:
    """Mimics prefill-only runners that lack prefill_context_batch."""

    def __init__(self, finish_after):
        self.finish_after = dict(finish_after)
        self.prefill_calls = []
        self.decode_calls = []
        self._prefill_count = 0
        self._prefill_tokens = 0

    def begin_step(self):
        self._prefill_count = 0
        self._prefill_tokens = 0

    def prefill_context(self, ctx, tokens):
        self.prefill_calls.append((ctx.state.request_id, tokens))
        ctx.state.prompt_pos += tokens
        self._prefill_count += 1
        self._prefill_tokens += tokens
        return tokens

    def decode(self, contexts):
        self.decode_calls.append([ctx.state.request_id for ctx in contexts])
        for ctx in contexts:
            remaining = self.finish_after[ctx.state.request_id] - 1
            self.finish_after[ctx.state.request_id] = remaining
            ctx.state.generated_tokens += 1
            if remaining <= 0:
                ctx.state.finished = True
        return {
            "decode_iterations": 1 if contexts else 0,
            "decode_tokens": len(contexts),
            "decode_duration_s": 0.0,
            "prefill_calls": self._prefill_count,
            "prefill_tokens": self._prefill_tokens,
            "prefill_duration_s": 0.0,
        }

    def collect_step_stats(self):
        return {
            "prefill_calls": self._prefill_count,
            "prefill_tokens": self._prefill_tokens,
            "prefill_duration_s": 0.0,
            "decode_iterations": 0,
            "decode_tokens": 0,
            "decode_duration_s": 0.0,
        }


class SchedulerTests(unittest.TestCase):
    def make_context(self, name, prompt_len=3, max_new_tokens=2):
        state = SequenceState(
            request_id=name,
            prompt_len=prompt_len,
            max_new_tokens=max_new_tokens,
        )
        ctx = SequenceContext(
            state=state,
            prompt_tokens=[0] * prompt_len,
            sampler_settings={},
            stopping_settings={},
            tokenizer=DummyTokenizer(),
        )
        ctx.history_tokens = [0] * prompt_len
        return ctx

    def test_prefill_respects_token_budget(self):
        runner = FakeRunner({"a": 1, "b": 1})
        scheduler = Scheduler(
            runner=runner,
            max_num_seqs=2,
            max_tokens_per_step=3,
            prefill_chunk=3,
            prefill_queue_limit=0,
        )

        scheduler.enqueue(self.make_context("a"))
        scheduler.enqueue(self.make_context("b"))

        scheduler.step()
        self.assertEqual(runner.prefill_calls[:2], [("a", 3), ("b", 3)])
        self.assertEqual(runner.decode_calls, [["a", "b"]])
        self.assertEqual(len(scheduler._wait_queue), 0)

        scheduler.step()
        self.assertEqual(runner.decode_calls, [["a", "b"]])
        self.assertFalse(scheduler._wait_queue)
        self.assertFalse(scheduler._active)

    def test_finished_sequences_are_retired_and_new_work_admitted(self):
        runner = FakeRunner({"a": 1, "b": 1})
        scheduler = Scheduler(
            runner=runner,
            max_num_seqs=2,
            max_tokens_per_step=4,
            prefill_chunk=4,
            prefill_queue_limit=0,
        )

        scheduler.enqueue(self.make_context("a"))
        scheduler.enqueue(self.make_context("b"))

        scheduler.step()
        self.assertTrue(runner.prefill_calls)
        self.assertEqual(runner.decode_calls, [["a", "b"]])
        self.assertFalse(scheduler._wait_queue)
        self.assertFalse(scheduler._active)

    def test_bootstrap_decode_runs_when_under_target(self):
        runner = LegacyRunner({"a": 1, "b": 1})
        scheduler = Scheduler(
            runner=runner,
            max_num_seqs=4,
            max_tokens_per_step=2,
            prefill_chunk=4,
            prefill_queue_limit=0,
        )

        scheduler.enqueue(self.make_context("a", prompt_len=2))
        scheduler.enqueue(self.make_context("b", prompt_len=2))

        scheduler.step()
        self.assertTrue(runner.decode_calls)
        metrics = scheduler.metrics
        self.assertGreater(metrics.get("prefill_sequences", 0), 0)
        self.assertGreater(metrics.get("decode_batch_size", 0), 0)

        scheduler.step()
        self.assertIn(("b", 2), runner.prefill_calls)

    def test_ramp_ready_sequence_marked_active_before_prefill_complete(self):
        runner = FakeRunner({"a": 1})
        scheduler = Scheduler(
            runner=runner,
            max_num_seqs=1,
            max_tokens_per_step=2,
            prefill_chunk=2,
            prefill_queue_limit=0,
        )

        ctx = self.make_context("a", prompt_len=6)
        ctx.state.prefill_ramp_remaining = 2
        scheduler.enqueue(ctx)

        scheduler.step()
        self.assertEqual(ctx.state.prompt_pos, 2)
        ctx.state.prefill_ramp_remaining = 0
        ctx.state.prefill_decode_ready = True

        scheduler.step()
        self.assertLess(ctx.state.prompt_pos, ctx.state.prompt_len)
        self.assertEqual(runner.decode_calls, [["a"]])
        self.assertTrue(scheduler._wait_queue)

    def test_prefill_limits_new_sequences_to_capacity(self):
        finish_after = {name: 1 for name in "abcd"}
        runner = FakeRunner(finish_after)
        scheduler = Scheduler(
            runner=runner,
            max_num_seqs=2,
            max_tokens_per_step=16,
            prefill_chunk=4,
            prefill_queue_limit=0,
        )

        for name in "abcd":
            scheduler.enqueue(self.make_context(name))

        scheduler.step()

        processed = [req for req, _ in runner.prefill_calls]
        self.assertEqual(processed[:2], ["a", "b"])
        self.assertEqual(len(processed), 2)
        self.assertEqual(len(scheduler._wait_queue), 2)
        self.assertEqual(
            [ctx.state.request_id for ctx in scheduler._wait_queue], ["c", "d"]
        )

    def test_prefill_slice_cap_limits_tokens_per_batch(self):
        runner = FakeRunner({"a": 1})
        scheduler = Scheduler(
            runner=runner,
            max_num_seqs=1,
            max_tokens_per_step=16,
            prefill_chunk=16,
            prefill_queue_limit=0,
            prefill_slice_cap=4,
        )
        scheduler.enqueue(self.make_context("a", prompt_len=16))
        scheduler.step()
        self.assertEqual(runner.prefill_calls, [("a", 4)])

    def test_prefill_falls_back_when_batch_method_missing(self):
        runner = LegacyRunner({"a": 1})
        scheduler = Scheduler(
            runner=runner,
            max_num_seqs=1,
            max_tokens_per_step=4,
            prefill_chunk=4,
            prefill_queue_limit=0,
        )
        scheduler.enqueue(self.make_context("a", prompt_len=4))
        scheduler.step()
        self.assertEqual(runner.prefill_calls, [("a", 4)])


if __name__ == "__main__":
    unittest.main()
