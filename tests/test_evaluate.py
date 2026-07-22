# Copyright © 2024 Apple Inc.

import unittest
from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.evaluate import MLXLM


class TestMLXLM(unittest.TestCase):
    def setUp(self):
        # Mock the load function to avoid loading actual models
        self.mock_model = MagicMock()
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.model_max_length = 2048
        self.mock_tokenizer.chat_template = None
        self.mock_tokenizer.encode = MagicMock(return_value=[1, 2, 3, 4])

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


class TestMLXLMLoglikelihood(unittest.TestCase):
    """End-to-end tests for loglikelihood scoring with a real model."""

    @classmethod
    def setUpClass(cls):
        cls.lm = MLXLM("mlx-community/Qwen1.5-0.5B-Chat-4bit")
        base = (
            "The city council met on Tuesday to discuss the new transit "
            "plan, which includes additional bus routes and longer service "
            "hours for the northern districts. "
        )
        ids = cls.lm.tokenizer.encode(base * 60, add_special_tokens=False)[:700]
        cls.long_context = cls.lm.tokenizer.decode(ids)

    def _reference(self, context, continuation):
        """Score a continuation with a single forward pass over the full
        sequence."""
        prefix = self.lm._tokenize([context])[0]
        full = self.lm._tokenize([context + continuation])[0]
        logits = self.lm._model(mx.array(full[:-1])[None])
        logprobs = nn.log_softmax(logits[0].astype(mx.float32), axis=-1)
        score = 0.0
        greedy = True
        for pos in range(len(prefix) - 1, len(full) - 1):
            target = full[pos + 1]
            score += logprobs[pos, target].item()
            greedy &= mx.argmax(logprobs[pos]).item() == target
        return score, greedy

    def test_loglikelihood_matches_full_forward(self):
        cases = [
            ("The capital of France is", " Paris"),
            ("The capital of France is", " the city of Paris, which is known"),
            ("The capital of France is", " Berlin"),
            # A long context exercises the split prefill path
            (self.long_context, " The council approved the plan."),
        ]
        requests = [MagicMock(args=case) for case in cases]
        results = self.lm.loglikelihood(requests)
        self.assertEqual(len(results), len(cases))
        for (score, greedy), case in zip(results, cases):
            ref_score, ref_greedy = self._reference(*case)
            self.assertLess(abs(score - ref_score), 2e-1)
            self.assertEqual(greedy, ref_greedy)

    def test_loglikelihood_continuation_order_invariance(self):
        # Continuations of different lengths after a common long prefix.
        # Scoring them in either order must give the same results, which
        # checks that scoring one continuation does not contaminate the
        # cached prefix state used by the others.
        context = self.long_context
        continuations = [
            " the lazy dog. " * 20,
            " The fence.",
            " A log lies on the other side of the river.",
        ]
        requests = [MagicMock(args=(context, c)) for c in continuations]
        forward = self.lm.loglikelihood(requests)
        backward = self.lm.loglikelihood(list(reversed(requests)))
        for (score_f, greedy_f), (score_b, greedy_b) in zip(
            forward, reversed(backward)
        ):
            self.assertAlmostEqual(score_f, score_b, places=3)
            self.assertEqual(greedy_f, greedy_b)


if __name__ == "__main__":
    unittest.main()
