# ABOUTME: Validates scheduler handles burst admissions within active limits.
# ABOUTME: Ensures decode batches never exceed configured max_num_seqs.

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


class BurstRunner:
    def __init__(self):
        self.decode_batch_sizes = []
        self.prefill_calls = 0
        self._prefill_tokens = 0

    def begin_step(self):
        self.prefill_calls = 0
        self._prefill_tokens = 0

    def prefill_context(self, ctx, tokens):
        self.prefill_calls += 1
        ctx.state.prompt_pos += tokens
        self._prefill_tokens += tokens
        return tokens

    def decode(self, contexts):
        batch_size = len(contexts)
        if batch_size:
            self.decode_batch_sizes.append(batch_size)
        for ctx in contexts:
            ctx.state.generated_tokens += 1
            ctx.state.finished = True
        return {
            "decode_iterations": 1 if contexts else 0,
            "decode_tokens": batch_size,
            "decode_duration_s": 0.0,
            "prefill_calls": self.prefill_calls,
            "prefill_tokens": self._prefill_tokens,
            "prefill_duration_s": 0.0,
        }

    def collect_step_stats(self):
        return {
            "prefill_calls": self.prefill_calls,
            "prefill_tokens": self._prefill_tokens,
            "prefill_duration_s": 0.0,
            "decode_iterations": 0,
            "decode_tokens": 0,
            "decode_duration_s": 0.0,
        }

    def debug_state(self):
        return {
            "generator_active": 0,
            "uid_count": 0,
            "prefill_calls": self.prefill_calls,
            "prefill_tokens": self._prefill_tokens,
            "decode_iterations": 0,
            "decode_tokens": 0,
        }


def _make_context(name, prompt_len=4, max_new_tokens=1):
    state = SequenceState(
        request_id=f"req-{name}",
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


class SchedulerBurstTests(unittest.TestCase):
    def test_burst_never_exceeds_active_limit(self):
        runner = BurstRunner()
        scheduler = Scheduler(
            runner=runner,
            max_num_seqs=16,
            max_tokens_per_step=512,
            prefill_chunk=128,
        )

        for idx in range(64):
            scheduler.enqueue(_make_context(idx))

        step_guard = 0
        while scheduler.has_pending_work and step_guard < 32:
            scheduler.step()
            step_guard += 1

        self.assertFalse(scheduler.has_pending_work)
        self.assertGreater(len(runner.decode_batch_sizes), 0)
        self.assertLessEqual(max(runner.decode_batch_sizes), 16)
        self.assertLess(step_guard, 32)


if __name__ == "__main__":
    unittest.main()
