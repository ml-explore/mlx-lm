# Copyright © 2024 Apple Inc.

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx

from mlx_lm.evaluate import MLXLM


class TestMLXLM(unittest.TestCase):
    def setUp(self):
        # Mock the load function to avoid loading actual models
        self.mock_model = MagicMock()
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.model_max_length = 2048
        self.mock_tokenizer.chat_template = None
        self.mock_tokenizer.encode = MagicMock(return_value=[1, 2, 3, 4])
        self.mock_tokenizer.has_thinking = False

        with patch("mlx_lm.evaluate.load") as mock_load:
            mock_load.return_value = (self.mock_model, self.mock_tokenizer)
            self.mlx_lm = MLXLM("test_model", max_tokens=128)

    def test_loglikelihood_rolling_processes_all_inputs(self):
        """Test that loglikelihood_rolling processes all inputs correctly when batching."""
        # Create 5 mock requests to test batching with batch_size=2
        mock_requests = [MagicMock(args=(f"text {i}",)) for i in range(5)]

        # Mock inputs
        test_inputs = [(i, i + 1, i + 2) for i in range(5)]
        self.mlx_lm._tokenize = MagicMock(return_value=test_inputs)

        # Mock _score_fn to return different scores for each batch
        def mock_score_fn(batch):
            batch_size = len(batch)
            scores = mx.array([[0.1] * 3 for _ in range(batch_size)])
            lengths = mx.array([3] * batch_size)
            return scores, lengths, None

        self.mlx_lm._score_fn = MagicMock(side_effect=mock_score_fn)
        self.mlx_lm._batch_size = 2

        result = self.mlx_lm.loglikelihood_rolling(mock_requests)

        # Should return 5 results (one per request)
        self.assertEqual(len(result), 5)

        # Should have called _score_fn 3 times (batches of 2, 2, 1)
        self.assertEqual(self.mlx_lm._score_fn.call_count, 3)

        # Verify the batches were correct sizes
        call_args_list = self.mlx_lm._score_fn.call_args_list
        self.assertEqual(len(call_args_list[0][0][0]), 2)  # First batch: 2 items
        self.assertEqual(len(call_args_list[1][0][0]), 2)  # Second batch: 2 items
        self.assertEqual(len(call_args_list[2][0][0]), 1)  # Third batch: 1 item

    def test_loglikelihood_rolling_mixed_lengths_is_independent_of_batch_size(self):
        class Tokenizer:
            chat_template = None

            def encode(self, text, add_special_tokens=True):
                return [int(token) for token in text.split()]

        class Model:
            def make_cache(self):
                return []

            def __call__(self, inputs, cache=None):
                logits = mx.arange(11, dtype=mx.float32)
                return mx.broadcast_to(logits, (*inputs.shape, logits.shape[0]))

        def make_lm(batch_size):
            with patch("mlx_lm.evaluate.load") as mock_load:
                mock_load.return_value = (Model(), Tokenizer())
                return MLXLM("test_model", batch_size=batch_size)

        # The first request is padded only when scored in a mixed-length batch.
        requests = [
            SimpleNamespace(args=("1 2 3 4",)),
            SimpleNamespace(args=("5 6 7 8 9 10",)),
        ]
        batched_scores = make_lm(2).loglikelihood_rolling(requests)

        single_lm = make_lm(1)
        single_scores = [
            single_lm.loglikelihood_rolling([request])[0] for request in requests
        ]

        self.assertAlmostEqual(batched_scores[0], single_scores[0])
        self.assertAlmostEqual(batched_scores[1], single_scores[1])

    @patch("mlx_lm.evaluate.batch_generate")
    def test_generate_strip_until_then_strip_thinking(self, mock_batch_generate):
        self.mock_tokenizer.has_thinking = True
        self.mock_tokenizer.think_end = "</think>"
        mock_batch_generate.return_value.texts = [
            "<think>scratch</think>answer STOP extra"
        ]
        request = MagicMock(args=("prompt", {"until": [" STOP"]}))

        result = self.mlx_lm.generate_until([request])

        self.assertEqual(result, ["answer"])

    def test_loglikelihood_returns_negative_infinity_when_context_is_fully_truncated(
        self,
    ):
        # Make the continuation too long for the scoring window, so all context
        # tokens are truncated and the request is skipped.
        self.mlx_lm._max_tokens = 1
        self.mlx_lm._tokenize = MagicMock(
            side_effect=[
                [[1, 2]],  # Context tokens.
                [[1, 2, 3, 4]],  # Context + continuation tokens.
            ]
        )
        self.mlx_lm._process_prompt = MagicMock()
        self.mlx_lm._score_fn = MagicMock()
        request = MagicMock(args=("context", " continuation"))

        result = self.mlx_lm.loglikelihood([request])

        self.assertEqual(result, [(-float("inf"), False)])
        self.mlx_lm._process_prompt.assert_not_called()
        self.mlx_lm._score_fn.assert_not_called()

    def _make_lm(self, **kwargs):
        with patch("mlx_lm.evaluate.load") as mock_load:
            mock_load.return_value = (self.mock_model, self.mock_tokenizer)
            return MLXLM("test_model", **kwargs)

    def _run_generate_until(self, lm, options):
        requests = [MagicMock(args=(f"prompt {i}", o)) for i, o in enumerate(options)]
        with patch("mlx_lm.evaluate.batch_generate") as mock_generate:
            mock_generate.return_value = MagicMock(texts=[""] * len(requests))
            lm.generate_until(requests)
        return mock_generate.call_args.kwargs

    def test_generate_until_uses_task_max_gen_toks(self):
        """lm-eval passes the per-task generation cap as `max_gen_toks`."""
        lm = self._make_lm(max_tokens=None)
        kwargs = self._run_generate_until(
            lm,
            [
                {"until": ["\n\n"], "max_gen_toks": 10},
                {"until": ["\n\n"], "max_gen_toks": 1024},
            ],
        )
        self.assertEqual(kwargs["max_tokens"], [10, 1024])

    def test_generate_until_max_tokens_overrides_task_default(self):
        """--max-tokens takes precedence over task specific defaults."""
        lm = self._make_lm(max_tokens=128)
        kwargs = self._run_generate_until(lm, [{"until": ["\n\n"], "max_gen_toks": 10}])
        self.assertEqual(kwargs["max_tokens"], [128])

    def test_generate_until_respects_batch_size(self):
        """--batch-size bounds generation, not just loglikelihood scoring."""
        lm = self._make_lm(max_tokens=None, batch_size=4)
        kwargs = self._run_generate_until(lm, [{"until": ["\n\n"]}] * 6)
        self.assertEqual(kwargs["prefill_batch_size"], 4)
        self.assertEqual(kwargs["completion_batch_size"], 4)


if __name__ == "__main__":
    unittest.main()
