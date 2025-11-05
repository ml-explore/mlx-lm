# ABOUTME: Tests vectorized sampling helpers.
# ABOUTME: Validates argmax selection and selected logprobs.

import unittest
import numpy as np

from mlx_lm.server_batched.sampling import select_tokens_argmax, selected_logprobs


class SamplingTests(unittest.TestCase):
    def test_select_tokens_argmax(self):
        logits = np.array([[0.1, 0.9, -0.4], [2.0, -1.0, 3.5]], dtype=np.float32)
        tokens = select_tokens_argmax(logits)
        np.testing.assert_array_equal(tokens, np.array([1, 2]))

    def test_selected_logprobs(self):
        logits = np.array([[0.0, 2.0], [1.0, 0.0]], dtype=np.float32)
        tokens = np.array([1, 0])
        logps = selected_logprobs(logits, tokens)
        # Compute manually via log-softmax
        def log_softmax(row):
            exps = np.exp(row - row.max())
            logZ = np.log(exps.sum()) + row.max()
            return row - logZ

        expected = np.array([log_softmax(logits[0])[1], log_softmax(logits[1])[0]])
        np.testing.assert_allclose(logps, expected, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
