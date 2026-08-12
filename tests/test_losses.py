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

    def test_kl_div_loss_vjp_large_vocab_deterministic(self):
        self.assertTrue(can_run_metal())
        mx.random.seed(0)

        logits_q = mx.random.normal((64, 100003)) * 6.0
        logits_p = mx.random.normal((64, 100003)) * 6.0
        cotan = mx.ones((64,))

        with mx.stream(mx.cpu):
            expected = mx.vjp(kl_div_loss, [logits_q, logits_p], [cotan])[1][0]

        runs = [
            mx.vjp(kl_div_loss, [logits_q, logits_p], [cotan])[1][0] for _ in range(5)
        ]
        mx.eval(runs, expected)

        for g in runs:
            self.assertTrue(mx.array_equal(g, runs[0]))
            self.assertTrue(mx.allclose(g, expected, rtol=1e-4, atol=1e-5))
            self.assertFalse(bool(mx.any(mx.abs(g) > 1e30)))

    def test_js_div_loss_vjp_large_vocab_deterministic(self):
        self.assertTrue(can_run_metal())
        mx.random.seed(0)

        logits_q = mx.random.normal((64, 100003)) * 6.0
        logits_p = mx.random.normal((64, 100003)) * 6.0
        cotan = mx.ones((64,))

        with mx.stream(mx.cpu):
            expected = mx.vjp(js_div_loss, [logits_q, logits_p], [cotan])[1][0]

        runs = [
            mx.vjp(js_div_loss, [logits_q, logits_p], [cotan])[1][0] for _ in range(5)
        ]
        mx.eval(runs, expected)

        for g in runs:
            self.assertTrue(mx.array_equal(g, runs[0]))
            self.assertTrue(mx.allclose(g, expected, rtol=1e-4, atol=1e-5))
            self.assertFalse(bool(mx.any(mx.abs(g) > 1e30)))

    def test_js_div_loss_large_scale_softplus_finite(self):
        # scale 20 produces log-ratios large enough that the naive
        # log(1+exp(x)) softplus in the fused JS kernels overflowed to
        # inf/NaN before the stable rewrite. Shape matched to a CI cell
        # that stayed finite against the reference path after the fix
        # (float32 scale=20 shape=(3,100): max abs grad diff 1.072e-6).
        self.assertTrue(can_run_metal())
        mx.random.seed(0)

        logits_q = mx.random.normal((8, 100)) * 20.0
        logits_p = mx.random.normal((8, 100)) * 20.0
        cotan = mx.ones((8,))

        with mx.stream(mx.cpu):
            expected_y = js_div_loss(logits_q, logits_p)
            expected_g = mx.vjp(js_div_loss, [logits_q, logits_p], [cotan])[1][0]

        y = js_div_loss(logits_q, logits_p)
        runs = [
            mx.vjp(js_div_loss, [logits_q, logits_p], [cotan])[1][0] for _ in range(5)
        ]
        mx.eval(y, runs, expected_y, expected_g)

        self.assertFalse(bool(mx.any(mx.isnan(y)) | mx.any(mx.isinf(y))))
        self.assertTrue(mx.allclose(y, expected_y, rtol=1e-4, atol=1e-5))
        for g in runs:
            self.assertTrue(mx.array_equal(g, runs[0]))
            self.assertFalse(
                bool(mx.any(mx.isnan(g)) | mx.any(mx.isinf(g)) | (mx.abs(g) > 1e30))
            )
            self.assertTrue(mx.allclose(g, expected_g, rtol=1e-4, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
