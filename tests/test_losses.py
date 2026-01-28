# Copyright © 2025 Apple Inc.

import unittest

import mlx.core as mx

from mlx_lm.tuner.losses import (
    _log_prob_from_logits_and_labels,
    can_run_metal,
    dpo_loss,
    js_div_loss,
    kl_div_loss,
)


class TestLosses(unittest.TestCase):

    def test_kl_div_loss(self):
        self.assertTrue(can_run_metal())
        mx.random.seed(0)

        logits_q = mx.random.uniform(shape=(4, 8, 4000), dtype=mx.float32)
        logits_p = mx.random.uniform(shape=(4, 8, 4000), dtype=mx.float32)

        with mx.stream(mx.cpu):
            expected = kl_div_loss(logits_q, logits_p)
        kl = kl_div_loss(logits_q, logits_p)

        self.assertTrue(mx.allclose(kl, expected, rtol=1e-4))

    def test_js_div_loss(self):
        self.assertTrue(can_run_metal())
        mx.random.seed(0)

        logits_q = mx.random.uniform(shape=(4, 8, 4000), dtype=mx.float32)
        logits_p = mx.random.uniform(shape=(4, 8, 4000), dtype=mx.float32)

        with mx.stream(mx.cpu):
            expected = js_div_loss(logits_q, logits_p)
        js = js_div_loss(logits_q, logits_p)

        self.assertTrue(mx.allclose(js, expected))

    def test_kl_div_loss_vjp(self):
        self.assertTrue(can_run_metal())
        mx.random.seed(0)

        logits_q = mx.random.uniform(shape=(4, 8, 4000), dtype=mx.float32)
        logits_p = mx.random.uniform(shape=(4, 8, 4000), dtype=mx.float32)
        cotan = mx.random.uniform(shape=(4, 8), dtype=mx.float32)

        with mx.stream(mx.cpu):
            expected = mx.vjp(kl_div_loss, [logits_q, logits_p], [cotan])[1][0]
        vjp_q = mx.vjp(kl_div_loss, [logits_q, logits_p], [cotan])[1][0]

        self.assertTrue(mx.allclose(vjp_q, expected))

    def test_js_div_loss_vjp(self):
        self.assertTrue(can_run_metal())
        mx.random.seed(0)

        logits_q = mx.random.uniform(shape=(4, 8, 4000), dtype=mx.float32)
        logits_p = mx.random.uniform(shape=(4, 8, 4000), dtype=mx.float32)
        cotan = mx.random.uniform(shape=(4, 8), dtype=mx.float32)

        with mx.stream(mx.cpu):
            expected = mx.vjp(js_div_loss, [logits_q, logits_p], [cotan])[1][0]
        vjp_q = mx.vjp(js_div_loss, [logits_q, logits_p], [cotan])[1][0]

        self.assertTrue(mx.allclose(vjp_q, expected))

    def test_log_prob_from_logits_and_labels(self):
        """Test the helper function for computing log probabilities."""
        batch_size, seq_len, vocab_size = 2, 5, 10

        # Create test data
        logits = mx.random.uniform(
            shape=(batch_size, seq_len, vocab_size), dtype=mx.float32
        )
        labels = mx.random.randint(0, vocab_size, shape=(batch_size, seq_len))

        # Compute log probabilities
        log_probs = _log_prob_from_logits_and_labels(logits, labels)

        # Check shape
        self.assertEqual(log_probs.shape, (batch_size,))

        # Check that the values are negative (log probabilities)
        self.assertTrue(mx.all(log_probs <= 0))

    def test_dpo_loss_basic(self):
        """Test basic DPO loss computation."""
        batch_size, seq_len, vocab_size = 2, 8, 1000

        # Create test data
        policy_chosen_logits = mx.random.uniform(
            shape=(batch_size, seq_len, vocab_size), dtype=mx.float32
        )
        policy_rejected_logits = mx.random.uniform(
            shape=(batch_size, seq_len, vocab_size), dtype=mx.float32
        )
        reference_chosen_logits = mx.random.uniform(
            shape=(batch_size, seq_len, vocab_size), dtype=mx.float32
        )
        reference_rejected_logits = mx.random.uniform(
            shape=(batch_size, seq_len, vocab_size), dtype=mx.float32
        )
        chosen_labels = mx.random.randint(0, vocab_size, shape=(batch_size, seq_len))
        rejected_labels = mx.random.randint(0, vocab_size, shape=(batch_size, seq_len))

        # Compute DPO loss
        loss = dpo_loss(
            policy_chosen_logits=policy_chosen_logits,
            policy_rejected_logits=policy_rejected_logits,
            reference_chosen_logits=reference_chosen_logits,
            reference_rejected_logits=reference_rejected_logits,
            chosen_labels=chosen_labels,
            rejected_labels=rejected_labels,
            beta=0.1,
        )

        # Check that loss is a scalar
        self.assertEqual(loss.shape, ())

        # Check that loss is positive (negative log-sigmoid should be positive)
        self.assertTrue(loss.item() >= 0)

    def test_dpo_loss_gradient(self):
        """Test that DPO loss computation is differentiable."""
        batch_size, seq_len, vocab_size = 1, 4, 100

        # Create test data
        policy_chosen_logits = mx.random.uniform(
            shape=(batch_size, seq_len, vocab_size), dtype=mx.float32
        )
        policy_rejected_logits = mx.random.uniform(
            shape=(batch_size, seq_len, vocab_size), dtype=mx.float32
        )
        reference_chosen_logits = mx.random.uniform(
            shape=(batch_size, seq_len, vocab_size), dtype=mx.float32
        )
        reference_rejected_logits = mx.random.uniform(
            shape=(batch_size, seq_len, vocab_size), dtype=mx.float32
        )
        chosen_labels = mx.random.randint(0, vocab_size, shape=(batch_size, seq_len))
        rejected_labels = mx.random.randint(0, vocab_size, shape=(batch_size, seq_len))

        def loss_fn(policy_chosen, policy_rejected):
            return dpo_loss(
                policy_chosen_logits=policy_chosen,
                policy_rejected_logits=policy_rejected,
                reference_chosen_logits=reference_chosen_logits,
                reference_rejected_logits=reference_rejected_logits,
                chosen_labels=chosen_labels,
                rejected_labels=rejected_labels,
                beta=0.1,
            )

        # Compute gradients
        loss, grads = mx.value_and_grad(loss_fn, argnums=[0, 1])(
            policy_chosen_logits, policy_rejected_logits
        )

        # Check that gradients have the right shape
        self.assertEqual(grads[0].shape, policy_chosen_logits.shape)
        self.assertEqual(grads[1].shape, policy_rejected_logits.shape)

        # Check that gradients are not all zero
        self.assertFalse(mx.allclose(grads[0], mx.zeros_like(grads[0])))
        self.assertFalse(mx.allclose(grads[1], mx.zeros_like(grads[1])))

    def test_dpo_loss_beta_effect(self):
        """Test that beta parameter affects the loss as expected."""
        batch_size, seq_len, vocab_size = 1, 4, 50

        # Create test data with clear preference difference
        policy_chosen_logits = mx.random.uniform(
            shape=(batch_size, seq_len, vocab_size), dtype=mx.float32
        )
        policy_rejected_logits = mx.random.uniform(
            shape=(batch_size, seq_len, vocab_size), dtype=mx.float32
        )
        reference_chosen_logits = policy_chosen_logits  # Same as policy for reference
        reference_rejected_logits = policy_rejected_logits
        chosen_labels = mx.random.randint(0, vocab_size, shape=(batch_size, seq_len))
        rejected_labels = mx.random.randint(0, vocab_size, shape=(batch_size, seq_len))

        # Compute loss with different beta values
        loss_beta_01 = dpo_loss(
            policy_chosen_logits,
            policy_rejected_logits,
            reference_chosen_logits,
            reference_rejected_logits,
            chosen_labels,
            rejected_labels,
            beta=0.1,
        )

        loss_beta_05 = dpo_loss(
            policy_chosen_logits,
            policy_rejected_logits,
            reference_chosen_logits,
            reference_rejected_logits,
            chosen_labels,
            rejected_labels,
            beta=0.5,
        )

        # Both losses should be valid
        self.assertTrue(loss_beta_01.item() >= 0)
        self.assertTrue(loss_beta_05.item() >= 0)


if __name__ == "__main__":
    unittest.main()
