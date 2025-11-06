# ABOUTME: Ensures runtime shutdown stops worker threads on simulated signal.
# ABOUTME: Verifies scheduler stop clears pending work during shutdown.

from .util import ensure_mlx_stub

ensure_mlx_stub()

import time
import types
import unittest

from mlx_lm.generate import GenerationResponse
from mlx_lm.server_batched.runtime import ContinuousBatchingRuntime
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


class ShutdownRunner:
    def __init__(self):
        self.decode_calls = 0
        self.tokenizer = DummyTokenizer()
        self._prefill_count = 0
        self._prefill_tokens = 0

    def begin_step(self):
        self._prefill_count = 0
        self._prefill_tokens = 0

    def build_context(
        self,
        request_id,
        prompt_tokens,
        *,
        max_new_tokens,
        sampler_settings,
        stopping_settings,
        logit_bias,
        repetition_penalty,
        repetition_context_size,
    ):
        state = SequenceState(
            request_id=request_id,
            prompt_len=len(prompt_tokens),
            max_new_tokens=max_new_tokens,
        )
        ctx = SequenceContext(
            state=state,
            prompt_tokens=list(prompt_tokens),
            sampler_settings=sampler_settings,
            stopping_settings=stopping_settings,
            tokenizer=self.tokenizer,
        )
        ctx.history_tokens = list(prompt_tokens)
        return ctx

    def prefill_context(self, ctx, tokens):
        ctx.state.prompt_pos += tokens
        self._prefill_count += 1
        self._prefill_tokens += tokens
        return tokens

    def decode(self, contexts):
        self.decode_calls += 1
        for ctx in contexts:
            ctx.state.generated_tokens += 1
            ctx.state.finished = True
            ctx.enqueue_event(
                GenerationResponse(
                    text="done",
                    token=1,
                    logprobs=None,
                    from_draft=False,
                    prompt_tokens=ctx.state.prompt_len,
                    prompt_tps=0.0,
                    generation_tokens=ctx.state.generated_tokens,
                    generation_tps=0.0,
                    peak_memory=0.0,
                    finish_reason="stop",
                )
            )
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

    def debug_state(self):
        return {
            "generator_active": 0,
            "uid_count": 0,
            "prefill_calls": self._prefill_count,
            "prefill_tokens": self._prefill_tokens,
            "decode_iterations": 0,
            "decode_tokens": 0,
        }


class ShutdownTests(unittest.TestCase):
    def test_shutdown_clears_worker_thread(self):
        runner = ShutdownRunner()
        runtime = ContinuousBatchingRuntime(
            runner,
            max_num_seqs=2,
            max_tokens_per_step=8,
            prefill_chunk=8,
        )
        _, generator = runtime.submit_request(
            prompt_tokens=[1, 2],
            max_new_tokens=1,
            sampler_settings={},
            stopping_settings={},
            logit_bias=None,
            repetition_penalty=None,
            repetition_context_size=None,
        )

        time_limit = time.time() + 1.0
        while runner.decode_calls == 0 and time.time() < time_limit:
            time.sleep(0.01)

        first = next(generator)
        self.assertEqual(first.text, "done")
        with self.assertRaises(StopIteration):
            next(generator)

        runtime.shutdown()
        self.assertFalse(runtime._worker.is_alive())
        self.assertFalse(runtime.scheduler.has_pending_work)


if __name__ == "__main__":
    unittest.main()
