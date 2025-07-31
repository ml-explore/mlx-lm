# Copyright © 2024 Apple Inc.

import unittest
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

        with patch("mlx_lm.evaluate.load") as mock_load:
            mock_load.return_value = (self.mock_model, self.mock_tokenizer)
            self.mlx_lm = MLXLM("test_model")

    def test_loglikelihood_rolling_bug_fix(self):
        """Test that loglikelihood_rolling uses inputs variable instead of undefined texts variable."""
        # Create mock requests
        mock_requests = [
            MagicMock(args=("test text 1",)),
            MagicMock(args=("test text 2",)),
            MagicMock(args=("test text 3",)),
        ]

        # Mock the _tokenize method to return predictable inputs
        test_inputs = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12)]
        self.mlx_lm._tokenize = MagicMock(return_value=test_inputs)

        # Mock the _score_fn method
        mock_scores = mx.array(
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]]
        )
        mock_lengths = mx.array([4, 4, 4])
        self.mlx_lm._score_fn = MagicMock(
            return_value=(mock_scores, mock_lengths, None)
        )

        # Set batch size to test batching
        self.mlx_lm._batch_size = 2

        # Call loglikelihood_rolling - this should not raise a NameError about 'texts'
        try:
            result = self.mlx_lm.loglikelihood_rolling(mock_requests)
            # The method should complete without error
            self.assertIsInstance(result, list)
        except NameError as e:
            if "texts" in str(e):
                self.fail(
                    "loglikelihood_rolling still references undefined 'texts' variable"
                )
            else:
                raise

        # Verify that _tokenize was called with the correct arguments
        self.mlx_lm._tokenize.assert_called_once_with(
            ["test text 1", "test text 2", "test text 3"]
        )

        # Verify that _score_fn was called the expected number of times (2 batches)
        self.assertEqual(self.mlx_lm._score_fn.call_count, 2)

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


if __name__ == "__main__":
    unittest.main()
