# ABOUTME: Validates SSE flow emits [DONE] for continuous runtime responses.
# ABOUTME: Uses fake runner to check final streaming payload shape.

import io
import json
import types
import unittest

from .util import ensure_mlx_stub

ensure_mlx_stub()

from mlx_lm.generate import GenerationResponse
from mlx_lm.server_batched.runtime import ContinuousBatchingRuntime
from mlx_lm.server_batched.state import SequenceContext, SequenceState


class DummyTokenizer:
    eos_token_id = 2
    vocab_size = 8

    def encode(self, prompt):
        return [1] * len(prompt)

    def decode(self, ids, skip_special_tokens=False):
        return "X"

    @property
    def detokenizer(self):
        detok = types.SimpleNamespace()
        detok.add_token = lambda *_: None
        detok.finalize = lambda: None
        detok.last_segment = "X"
        return detok


class FakeRunner:
    def __init__(self):
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

    def decode(self, contexts):
        for ctx in contexts:
            ctx.state.generated_tokens += 1
            ctx.state.finished = True
            ctx.enqueue_event(
                GenerationResponse(
                    text="X",
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


class SSEFlowTests(unittest.TestCase):
    def test_generator_emits_done_segment(self):
        runner = FakeRunner()
        runtime = ContinuousBatchingRuntime(
            runner,
            max_num_seqs=2,
            max_tokens_per_step=4,
            prefill_chunk=4,
        )
        try:
            _, generator = runtime.submit_request(
                prompt_tokens=[1, 1, 1],
                max_new_tokens=1,
                sampler_settings={},
                stopping_settings={"eos_token_id": runner.tokenizer.eos_token_id},
                logit_bias=None,
                repetition_penalty=None,
                repetition_context_size=None,
            )
            events = list(generator)
            self.assertTrue(events)
            self.assertEqual(events[-1].finish_reason, "stop")

            buf = io.StringIO()
            for event in events:
                payload = {
                    "choices": [{"delta": {"content": event.text}}],
                }
                buf.write(f"data: {json.dumps(payload)}\n\n")
                if event.finish_reason:
                    buf.write("data: [DONE]\n\n")
                    break

            self.assertIn("data: [DONE]", buf.getvalue())
        finally:
            runtime.shutdown()


if __name__ == "__main__":
    unittest.main()
