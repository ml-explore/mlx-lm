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
        probs = mx.array([0.4, 0.3, 0.15, 0.15])[None]

        # XTC should discard only the first probability
        logprobs = mx.log(probs)
        new_logits = apply_xtc(logprobs, 1, 0.2, [100])
        new_probs = mx.softmax(new_logits.squeeze())
        new_probs_rounded = [round(p, 4) for p in new_probs.tolist()]
        self.assertEqual(new_probs_rounded, [0, 0.5, 0.25, 0.25])

        # All but the two last probs, which are the last ones above the threshold, should be discarded
        new_logits = apply_xtc(logprobs, 1, 0.15, [100])
        new_probs = mx.softmax(new_logits.squeeze())
        new_probs_rounded = [round(p, 4) for p in new_probs.tolist()]
        self.assertEqual(new_probs_rounded, [0.0, 0.0, 0.5, 0.5])

        # If XTC probability = 0, the probs shouldn't change
        new_logits = apply_xtc(logprobs, 0.00, 0.2, [100])
        new_probs = mx.softmax(new_logits.squeeze())
        new_probs_rounded = [round(p, 4) for p in new_probs.tolist()]
        self.assertEqual(new_probs_rounded, [0.4, 0.3, 0.15, 0.15])


if __name__ == "__main__":
    unittest.main()
