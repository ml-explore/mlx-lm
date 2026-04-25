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

    def test_kl_div_loss_mlx_implementation(self):
        mx.random.seed(0)

        logits_q = mx.random.normal((2, 4, 128))
        logits_p = mx.random.normal((2, 4, 128))

        kl = kl_div_loss(logits_q, logits_p, implementation="mlx")

        self.assertEqual(kl.shape, logits_q.shape[:-1])
        self.assertTrue(mx.all(mx.isfinite(kl)).item())

    def test_kl_div_loss_auto_large_bfloat16_vocab(self):
        logits_q = mx.zeros((1, 248320), dtype=mx.bfloat16)
        logits_p = mx.zeros((1, 248320), dtype=mx.bfloat16)

        kl = kl_div_loss(logits_q, logits_p, implementation="auto")

        self.assertEqual(kl.shape, logits_q.shape[:-1])
        self.assertTrue(mx.all(mx.isfinite(kl)).item())

    def test_kl_div_loss_rejects_unknown_implementation(self):
        logits = mx.zeros((1, 4))

        with self.assertRaisesRegex(ValueError, "auto.*metal.*mlx"):
            kl_div_loss(logits, logits, implementation="fastest")

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


if __name__ == "__main__":
    unittest.main()
