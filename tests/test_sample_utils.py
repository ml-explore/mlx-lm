import unittest

import mlx.core as mx

from mlx_lm.sample_utils import apply_min_p, apply_top_k, apply_top_p, apply_xtc


class TestSampleUtils(unittest.TestCase):
    def test_apply_top_p(self):
        probs = mx.array([0.9, 0.0, 0.0, 0.1])[None]
        logits = mx.log(probs)

        new_logits = apply_top_p(logits, 0.3)
        actual_probs = mx.softmax(new_logits.squeeze())
        self.assertEqual(actual_probs.tolist(), [1.0, 0.0, 0.0, 0.0])

        new_logits = apply_top_p(logits, 0.95)
        actual_probs = mx.softmax(new_logits.squeeze())
        self.assertTrue(mx.allclose(probs.squeeze(), actual_probs))

        probs = mx.array([0.0, 0.5, 0.4, 0.1])[None]
        logits = mx.log(probs)
        new_logits = apply_top_p(logits, 0.4)
        actual_probs = mx.softmax(new_logits.squeeze())
        self.assertEqual(actual_probs.tolist(), [0.0, 1.0, 0.0, 0.0])

        new_logits = apply_top_p(logits, 0.6)
        actual_probs = mx.softmax(new_logits.squeeze())
        self.assertEqual(
            [round(p, 4) for p in actual_probs.tolist()], [0.0, 0.5556, 0.4444, 0.0]
        )

        new_logits = apply_top_p(logits, 0.95)
        actual_probs = mx.softmax(new_logits.squeeze())
        actual_rounded = [round(p, 4) for p in actual_probs.tolist()]
        expected_rounded = [0.0, 0.5, 0.4, 0.1]
        self.assertEqual(actual_rounded, expected_rounded)
        self.assertAlmostEqual(sum(actual_probs.tolist()), 1.0)

        # Batch mode works
        probs = mx.array([[0.9, 0.0, 0.0, 0.1], [0.0, 0.8, 0.1, 0.1]])
        logits = mx.log(probs)
        new_logits = apply_top_p(logits, 0.5)
        actual_probs = mx.softmax(new_logits, axis=-1)
        self.assertEqual(
            actual_probs.tolist(), [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        )

    def test_apply_min_p(self):
        probs = mx.array([0.9, 0.0, 0.0, 0.1])[None]
        logits = mx.log(probs)
        new_logits = apply_min_p(logits, 0.8)
        actual_probs = mx.softmax(new_logits.squeeze())
        self.assertEqual(actual_probs.tolist(), [1.0, 0.0, 0.0, 0.0])

        probs = mx.array([0.9, 0.0, 0.0, 0.1])[None]
        logits = mx.log(probs)
        new_logits = apply_min_p(logits, 0.05)
        actual_probs = mx.softmax(new_logits.squeeze())
        self.assertTrue(mx.allclose(actual_probs, mx.squeeze(probs)))

        # Batch mode works
        probs = mx.array([[0.9, 0.0, 0.0, 0.1], [0.0, 0.8, 0.0, 0.1]])
        logits = mx.log(probs)
        new_logits = apply_min_p(logits, 0.7)
        actual_probs = mx.softmax(new_logits, axis=-1)
        self.assertEqual(
            actual_probs.tolist(), [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        )

    def test_apply_top_k(self):
        probs = mx.array([0.9, 0.0, 0.0, 0.1])[None]
        logits = mx.log(probs)

        new_logits = apply_top_k(logits, 1)
        actual_probs = mx.softmax(new_logits.squeeze())
        self.assertEqual(actual_probs.tolist(), [1.0, 0.0, 0.0, 0.0])

        probs = mx.array([0.6, 0.0, 0.1, 0.3])[None]
        logits = mx.log(probs)
        new_logits = apply_top_k(logits, 2)
        actual_probs = mx.softmax(new_logits.squeeze())
        self.assertEqual(
            [round(p, 4) for p in actual_probs.tolist()], [0.6667, 0.0, 0.0, 0.3333]
        )

        # Batch mode works
        probs = mx.array([[0.9, 0.0, 0.0, 0.1], [0.0, 0.8, 0.0, 0.1]])
        logits = mx.log(probs)

        new_logits = apply_top_k(logits, 1)
        actual_probs = mx.softmax(new_logits, axis=-1)
        self.assertEqual(
            actual_probs.tolist(), [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        )

    def test_apply_xtc(self):
        # Test the threshold
        probs = mx.array([[0.4, 0.3, 0.15, 0.15]])
        new_probs = mx.softmax(apply_xtc(mx.log(probs), 1, 0.2, []), -1)
        expected = mx.array([[0, 0.5, 0.25, 0.25]])
        self.assertTrue(mx.allclose(new_probs, expected))
        probs = mx.array([[0.4, 0.3, 0.15, 0.15]])
        new_probs = mx.softmax(apply_xtc(mx.log(probs), 1, 0.1, []), -1)
        expected = mx.array([[0, 0.0, 0.5, 0.5]])
        self.assertTrue(mx.allclose(new_probs, expected))

        # Test the special tokens
        probs = mx.array([[0.4, 0.3, 0.15, 0.15]])
        new_probs = mx.softmax(apply_xtc(mx.log(probs), 1, 0.1, [0]), -1)
        expected = mx.array([[4 / 7, 0.0, 1.5 / 7, 1.5 / 7]])
        self.assertTrue(mx.allclose(new_probs, expected))

        # Test that with probability 0 the probs don't change
        probs = mx.array([[0.4, 0.3, 0.15, 0.15]])
        new_probs = mx.softmax(apply_xtc(mx.log(probs), 0, 0.1, [0]), -1)
        self.assertTrue(mx.allclose(new_probs, probs))

    def test_presence_penalty(self):
        from mlx_lm.sample_utils import make_presence_penalty

        # Token appears multiple times - penalty applied once
        tokens = mx.array([0, 0, 0, 1, 1])
        logits = mx.zeros((1, 4))
        processor = make_presence_penalty(0.5, context_size=5)
        result = processor(tokens, logits)
        # Token 0 appears 3 times, token 1 appears 2 times - both penalized once
        self.assertAlmostEqual(result[0, 0].item(), -0.5)
        self.assertAlmostEqual(result[0, 1].item(), -0.5)
        # Tokens not in context not penalized
        self.assertAlmostEqual(result[0, 2].item(), 0.0)
        self.assertAlmostEqual(result[0, 3].item(), 0.0)

    def test_frequency_penalty(self):
        from mlx_lm.sample_utils import make_frequency_penalty

        # Token appears multiple times - penalty applied proportionally
        tokens = mx.array([0, 0, 0, 1, 1])
        logits = mx.zeros((1, 4))
        processor = make_frequency_penalty(0.5, context_size=5)
        result = processor(tokens, logits)
        # Token 0 appears 3 times -> 3 * 0.5 = 1.5 penalty
        self.assertAlmostEqual(result[0, 0].item(), -1.5)
        # Token 1 appears 2 times -> 2 * 0.5 = 1.0 penalty
        self.assertAlmostEqual(result[0, 1].item(), -1.0)
        # Tokens not in context not penalized
        self.assertAlmostEqual(result[0, 2].item(), 0.0)
        self.assertAlmostEqual(result[0, 3].item(), 0.0)

    def test_make_logits_processors(self):
        from mlx_lm.sample_utils import make_logits_processors

        # Create processors with all three penalty types
        tokens = mx.array([0, 0, 0, 1, 1])
        # Use non-zero logits so repetition penalty has effect
        logits = mx.array([[1.0, 0.5, 0.0, -0.5]])
        processors = make_logits_processors(
            repetition_penalty=1.5,
            repetition_context_size=5,
            presence_penalty=0.5,
            presence_context_size=5,
            frequency_penalty=0.25,
            frequency_context_size=5,
        )
        # Apply all processors
        for processor in processors:
            logits = processor(tokens, logits)
        # Token 0 (appears 3x): 1.0/1.5 - 0.5 - 0.75 = -0.5833
        # Token 1 (appears 2x): 0.5/1.5 - 0.5 - 0.5 = -0.6667
        # Token 2 (not in context): 0.0 (no penalty)
        # Token 3 (not in context): -0.5 (no penalty)
        self.assertAlmostEqual(logits[0, 0].item(), -0.5833, places=4)
        self.assertAlmostEqual(logits[0, 1].item(), -0.6667, places=4)
        self.assertAlmostEqual(logits[0, 2].item(), 0.0, places=4)
        self.assertAlmostEqual(logits[0, 3].item(), -0.5, places=4)


if __name__ == "__main__":
    unittest.main()
