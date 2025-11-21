# ABOUTME: Tests continuous batching runtime wrapper around scheduler.
# ABOUTME: Ensures requests enqueue and trigger scheduler steps.

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


class FakeRunner:
    def __init__(self):
        self.prefill_requests = []
        self.decode_calls = 0
        self._prefill_count = 0
        self._prefill_tokens = 0

    def begin_step(self):
        self._prefill_count = 0
        self._prefill_tokens = 0

    def build_context(
        self,
        request_id,
        prompt,
        *,
        max_new_tokens,
        sampler_settings,
        stopping_settings,
        logit_bias,
        repetition_penalty,
        repetition_context_size,
    ):
        state = SequenceState(
            request_id=request_id, prompt_len=len(prompt), max_new_tokens=max_new_tokens
        )
        ctx = SequenceContext(
            state=state,
            prompt_tokens=list(prompt),
            sampler_settings=sampler_settings,
            stopping_settings=stopping_settings,
            tokenizer=DummyTokenizer(),
        )
        ctx.history_tokens = list(prompt)
        return ctx

    def prefill_context(self, ctx, tokens):
        self.prefill_requests.append((ctx.state.request_id, tokens))
        ctx.state.prompt_pos += tokens
        self._prefill_count += 1
        self._prefill_tokens += tokens

    def decode(self, contexts):
        self.decode_calls += 1
        for ctx in contexts:
            ctx.state.generated_tokens += 1
            ctx.state.finished = True
            ctx.enqueue_event(
                GenerationResponse(
                    text="hello",
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


class ContinuousBatchingRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.runner = FakeRunner()
        self.runtime = ContinuousBatchingRuntime(
            runner=self.runner,
            max_num_seqs=2,
            max_tokens_per_step=4,
            prefill_chunk=4,
        )

    def tearDown(self):
        self.runtime.shutdown()

    def test_submit_request_returns_generator(self):
        request_id, generator = self.runtime.submit_request(
            prompt_tokens=[1, 2, 3],
            max_new_tokens=1,
            sampler_settings={},
            stopping_settings={},
            logit_bias=None,
            repetition_penalty=None,
            repetition_context_size=None,
        )
        self.assertTrue(request_id)
        events = list(generator)
        self.assertEqual(len(events), 1)
        self.assertEqual(self.runner.prefill_requests[0][0], request_id)

    def test_scheduler_wakes_on_submit(self):
        request_id, generator = self.runtime.submit_request(
            prompt_tokens=[1],
            max_new_tokens=1,
            sampler_settings={},
            stopping_settings={},
            logit_bias=None,
            repetition_penalty=None,
            repetition_context_size=None,
        )
        # Await worker to process
        time_limit = time.time() + 1.0
        while self.runner.decode_calls == 0 and time.time() < time_limit:
            time.sleep(0.01)
        self.assertGreaterEqual(self.runner.decode_calls, 1)
        list(generator)


if __name__ == "__main__":
    unittest.main()
