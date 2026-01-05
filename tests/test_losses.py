# Copyright © 2025 Apple Inc.

import unittest

import mlx.core as mx

from mlx_lm.tuner.losses import can_run_metal, js_div_loss, kl_div_loss


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


if __name__ == "__main__":
    unittest.main()
