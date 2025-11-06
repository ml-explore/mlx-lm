# ABOUTME: Verifies ModelRunner stop conditions honor eos id collections.
# ABOUTME: Ensures list-based eos settings trigger stop completion signals.

from .util import ensure_mlx_stub

ensure_mlx_stub()

import types
import unittest

import mlx.core as mx

from mlx_lm.server_batched.engine import ModelRunner


class _StubTokenizer:
    eos_token_id = [5, 6]
    eos_token_ids = [7, 8]

    @property
    def detokenizer(self):
        detok = types.SimpleNamespace()
        detok.add_token = lambda *_: None
        detok.finalize = lambda: None
        detok.last_segment = ""
        return detok


class _StubModel:
    def __call__(self, tokens):
        if isinstance(tokens, list):
            batch = len(tokens)
        else:
            arr = mx.array(tokens)
            batch = int(arr.shape[0])
        return mx.zeros((batch, 1, 10), dtype=mx.float32)


class ModelRunnerStopTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = _StubTokenizer()
        self.model = _StubModel()

    def test_stop_when_eos_token_in_list_setting(self):
        runner = ModelRunner(
            model=self.model,
            tokenizer=self.tokenizer,
            max_num_seqs=2,
            prefill_chunk=4,
        )
        ctx = runner.build_context(
            "req-list",
            [1, 2, 3],
            max_new_tokens=8,
            sampler_settings={},
            stopping_settings={"eos_token_id": [6]},
            logit_bias=None,
            repetition_penalty=None,
            repetition_context_size=None,
        )
        ctx.history_tokens.append(6)
        result = runner._evaluate_stop_conditions(ctx, 6)
        self.assertEqual(result, "stop")

    def test_stop_when_eos_token_in_tokenizer_ids(self):
        runner = ModelRunner(
            model=self.model,
            tokenizer=self.tokenizer,
            max_num_seqs=2,
            prefill_chunk=4,
        )
        ctx = runner.build_context(
            "req-tokenizer",
            [1, 2, 3],
            max_new_tokens=8,
            sampler_settings={},
            stopping_settings={},
            logit_bias=None,
            repetition_penalty=None,
            repetition_context_size=None,
        )
        ctx.history_tokens.append(7)
        result = runner._evaluate_stop_conditions(ctx, 7)
        self.assertEqual(result, "stop")


if __name__ == "__main__":
    unittest.main()
