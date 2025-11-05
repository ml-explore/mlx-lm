# ABOUTME: Validates runtime retires sequences when client disconnects mid-stream.
# ABOUTME: Confirms cancel path emits final event and clears scheduler queues.

from .util import ensure_mlx_stub

ensure_mlx_stub()

import queue
import time
import types
import unittest

from mlx_lm.generate import GenerationResponse
from mlx_lm.server_batched.runtime import ContinuousBatchingRuntime
from mlx_lm.server_batched.state import SequenceContext, SequenceState


class DummyTokenizer:
    eos_token_id = 0
    vocab_size = 4

    @property
    def detokenizer(self):
        detok = types.SimpleNamespace()
        detok.add_token = lambda *_: None
        detok.finalize = lambda: None
        detok.last_segment = ""
        return detok


class DisconnectRunner:
    def __init__(self):
        self.tokenizer = DummyTokenizer()
        self.last_ctx = None
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
            events=queue.Queue(maxsize=1),
        )
        ctx.history_tokens = list(prompt_tokens)
        self.last_ctx = ctx
        return ctx

    def prefill_context(self, ctx, tokens):
        ctx.state.prompt_pos += tokens
        self._prefill_count += 1
        self._prefill_tokens += tokens

    def decode(self, contexts):
        for ctx in contexts:
            ctx.enqueue_event(
                GenerationResponse(
                    text="chunk",
                    token=1,
                    logprobs=None,
                    from_draft=False,
                    prompt_tokens=ctx.state.prompt_len,
                    prompt_tps=0.0,
                    generation_tokens=ctx.state.generated_tokens,
                    generation_tps=0.0,
                    peak_memory=0.0,
                    finish_reason=None,
                )
            )
            # Second event triggers queue.Full -> cancel_requested flag
            ctx.enqueue_event(
                GenerationResponse(
                    text="tail",
                    token=1,
                    logprobs=None,
                    from_draft=False,
                    prompt_tokens=ctx.state.prompt_len,
                    prompt_tps=0.0,
                    generation_tokens=ctx.state.generated_tokens + 1,
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


class ClientDisconnectTests(unittest.TestCase):
    def test_queue_full_marks_cancel_and_clears_runtime(self):
        runner = DisconnectRunner()
        runtime = ContinuousBatchingRuntime(
            runner,
            max_num_seqs=4,
            max_tokens_per_step=8,
            prefill_chunk=8,
        )
        try:
            _, generator = runtime.submit_request(
                prompt_tokens=[1, 2, 3],
                max_new_tokens=2,
                sampler_settings={},
                stopping_settings={"eos_token_id": runner.tokenizer.eos_token_id},
                logit_bias=None,
                repetition_penalty=None,
                repetition_context_size=None,
            )

            time_limit = time.time() + 1.0
            while runner.last_ctx is None and time.time() < time_limit:
                time.sleep(0.01)
            ctx = runner.last_ctx
            self.assertIsNotNone(ctx)

            time_limit = time.time() + 1.0
            while not ctx.state.finished and time.time() < time_limit:
                time.sleep(0.01)

            self.assertTrue(ctx.state.cancel_requested)
            self.assertTrue(ctx.state.finished)

            first_event = next(generator)
            self.assertEqual(first_event.text, "chunk")

            cancel_event = next(generator)
            self.assertEqual(cancel_event.finish_reason, "cancelled")

            self.assertFalse(runtime.scheduler.has_pending_work)
        finally:
            runtime.shutdown()


if __name__ == "__main__":
    unittest.main()
