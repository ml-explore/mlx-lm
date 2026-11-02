# Copyright © 2025 Apple Inc.

import unittest

import mlx.core as mx

from mlx_lm.tuner.losses import can_run_metal, js_div_loss, kl_div_loss


class TestLosses(unittest.TestCase):

    def test_kl_div_loss(self):
        self.assertTrue(can_run_metal())
        mx.random.seed(0)

        logits_q = mx.random.normal((2, 4, 4000))
        logits_p = mx.random.normal((2, 4, 4000))

        with mx.stream(mx.cpu):
            expected = kl_div_loss(logits_q, logits_p)
        kl = kl_div_loss(logits_q, logits_p)

        self.assertTrue(mx.allclose(kl, expected))

    def test_js_div_loss(self):
        self.assertTrue(can_run_metal())
        mx.random.seed(0)

        logits_q = mx.random.normal((2, 4, 4000))
        logits_p = mx.random.normal((2, 4, 4000))

        with mx.stream(mx.cpu):
            expected = js_div_loss(logits_q, logits_p)
        js = js_div_loss(logits_q, logits_p)

        self.assertTrue(mx.allclose(js, expected))

    def test_kl_div_loss_vjp(self):
        self.assertTrue(can_run_metal())
        mx.random.seed(0)

        logits_q = mx.random.normal((2, 4, 4000))
        logits_p = mx.random.normal((2, 4, 4000))
        cotan = mx.random.normal((2, 4))

        with mx.stream(mx.cpu):
            expected = mx.vjp(kl_div_loss, [logits_q, logits_p], [cotan])[1][0]
        vjp_q = mx.vjp(kl_div_loss, [logits_q, logits_p], [cotan])[1][0]

        self.assertTrue(mx.allclose(vjp_q, expected))

    def test_js_div_loss_vjp(self):
        self.assertTrue(can_run_metal())
        mx.random.seed(0)

        logits_q = mx.random.normal((2, 4, 4000))
        logits_p = mx.random.normal((2, 4, 4000))
        cotan = mx.random.normal((2, 4))

        with mx.stream(mx.cpu):
            expected = mx.vjp(js_div_loss, [logits_q, logits_p], [cotan])[1][0]
        vjp_q = mx.vjp(js_div_loss, [logits_q, logits_p], [cotan])[1][0]

        self.assertTrue(mx.allclose(vjp_q, expected))

    def _check_large_input(self, loss_fn):
        self.assertTrue(can_run_metal())
        mx.random.seed(0)

        # Large inputs make a race between simdgroups likely.
        logits_q = mx.random.normal((128, 151936))
        logits_p = mx.random.normal((128, 151936))
        cotan = mx.ones((128,))

        def loss_and_grad():
            (loss,), (grad, _) = mx.vjp(loss_fn, [logits_q, logits_p], [cotan])
            return loss, grad

        with mx.stream(mx.cpu):
            expected_loss, expected_grad = loss_and_grad()
        loss, grad = loss_and_grad()

        self.assertTrue(mx.allclose(loss, expected_loss, rtol=1e-4))
        self.assertTrue(mx.allclose(grad, expected_grad, rtol=1e-4))

        for _ in range(10):
            new_loss, new_grad = loss_and_grad()
            self.assertTrue(mx.array_equal(new_loss, loss))
            self.assertTrue(mx.array_equal(new_grad, grad))

    def test_kl_div_loss_large_input(self):
        self._check_large_input(kl_div_loss)

    def test_js_div_loss_large_input(self):
        self._check_large_input(js_div_loss)


if __name__ == "__main__":
    unittest.main()
