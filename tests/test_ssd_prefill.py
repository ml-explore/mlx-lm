# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx

from mlx_lm.models import ssm


def _rel(a, b):
    a = a.astype(mx.float32)
    b = b.astype(mx.float32)
    return (mx.abs(a - b).max() / (mx.abs(b).max() + 1e-9)).item()


def _inputs(T, h, dh, g, ds, dtype, use_state, batch=1, seed=0):
    mx.random.seed(seed)
    x = (mx.random.normal((batch, T, h, dh)) * 0.5).astype(dtype)
    B = (mx.random.normal((batch, T, g, ds)) * 0.5).astype(dtype)
    C = (mx.random.normal((batch, T, g, ds)) * 0.5).astype(dtype)
    dt = mx.random.normal((batch, T, h)).astype(dtype)
    A_log = mx.random.normal((h,)) * 0.5
    D = mx.random.normal((h,))
    dt_bias = mx.random.normal((h,))
    state = None
    if use_state:
        state = (mx.random.normal((batch, h, dh, ds)) * 0.1).astype(mx.float32)
    mx.eval(x, B, C, dt, A_log, D, dt_bias, state)
    return x, A_log, B, C, D, dt, dt_bias, state


@unittest.skipIf(not mx.metal.is_available(), "Metal is not available")
class TestSSDPrefill(unittest.TestCase):
    TOL = 2e-2

    def test_matches_ssm_attn(self):
        for dh, ds in [(64, 128), (128, 128), (64, 256), (32, 128), (64, 64)]:
            for dtype in [mx.bfloat16, mx.float16]:
                for use_state in [False, True]:
                    for T in [200, 1024]:
                        with self.subTest(
                            dh=dh, ds=ds, dtype=dtype, state=use_state, T=T
                        ):
                            args = _inputs(T, 8, dh, 2, ds, dtype, use_state)
                            y_ref, s_ref = ssm.ssm_attn(*args)
                            y_new, s_new = ssm.ssd_prefill(*args)
                            mx.eval(y_ref, s_ref, y_new, s_new)
                            self.assertLess(_rel(y_new, y_ref), self.TOL)
                            self.assertLess(_rel(s_new, s_ref), self.TOL)

    def test_segment_split(self):
        # Force multiple segments without a long sequence.
        segment, ssm.SSD_SEGMENT = ssm.SSD_SEGMENT, 256
        try:
            args = _inputs(700, 8, 64, 2, 128, mx.bfloat16, True, seed=1)
            y_ref, s_ref = ssm.ssm_attn(*args)
            y_new, s_new = ssm.ssd_prefill(*args)
            mx.eval(y_ref, s_ref, y_new, s_new)
            self.assertLess(_rel(y_new, y_ref), self.TOL)
            self.assertLess(_rel(s_new, s_ref), self.TOL)
        finally:
            ssm.SSD_SEGMENT = segment

    def test_carried_state(self):
        # Split pass (reference first half, kernel second half with carried
        # state) matches an unsplit reference run.
        x, A_log, B, C, D, dt, dt_bias, _ = _inputs(
            600, 8, 64, 2, 128, mx.bfloat16, False, seed=2
        )
        y_ref, s_ref = ssm.ssm_attn(x, A_log, B, C, D, dt, dt_bias, None)
        half = 300
        _, s1 = ssm.ssm_attn(
            x[:, :half], A_log, B[:, :half], C[:, :half], D, dt[:, :half], dt_bias, None
        )
        sl = lambda a: mx.contiguous(a[:, half:])
        y2, s2 = ssm.ssd_prefill(sl(x), A_log, sl(B), sl(C), D, sl(dt), dt_bias, s1)
        mx.eval(y_ref, s_ref, y2, s2)
        self.assertLess(_rel(y2, y_ref[:, half:]), self.TOL)
        self.assertLess(_rel(s2, s_ref), self.TOL)

    def test_mask_no_op_semantics(self):
        # A masked token must not change the carried state or the valid
        # outputs: compare against the reference run over only the valid
        # tokens. (ssm_attn's own masked paths return degenerate states, so
        # the valid-slice run is the ground truth here.)
        T, pad = 300, 75
        for side in ["left", "right"]:
            with self.subTest(side=side):
                x, A_log, B, C, D, dt, dt_bias, state = _inputs(
                    T, 8, 64, 2, 128, mx.bfloat16, True, seed=3
                )
                if side == "left":
                    mask = mx.concatenate(
                        [mx.zeros((1, pad)), mx.ones((1, T - pad))], axis=1
                    )
                    valid = slice(pad, None)
                else:
                    mask = mx.concatenate(
                        [mx.ones((1, T - pad)), mx.zeros((1, pad))], axis=1
                    )
                    valid = slice(None, T - pad)
                sl = lambda a: mx.contiguous(a[:, valid])
                y_ref, s_ref = ssm.ssm_attn(
                    sl(x), A_log, sl(B), sl(C), D, sl(dt), dt_bias, state
                )
                y_new, s_new = ssm.ssd_prefill(
                    x, A_log, B, C, D, dt, dt_bias, state, mask=mask
                )
                mx.eval(y_ref, s_ref, y_new, s_new)
                self.assertLess(_rel(y_new[:, valid], y_ref), self.TOL)
                self.assertLess(_rel(s_new, s_ref), self.TOL)

    def test_dispatched_from_ssm_update(self):
        # seq_len >= 32 with supported shapes routes through the kernels and
        # matches ssm_attn; short sequences keep the existing paths.
        args = _inputs(64, 8, 64, 2, 128, mx.bfloat16, True, seed=4)
        y_ref, s_ref = ssm.ssm_attn(*args)
        y_new, s_new = ssm.ssm_update(*args)
        mx.eval(y_ref, s_ref, y_new, s_new)
        self.assertLess(_rel(y_new, y_ref), self.TOL)
        self.assertLess(_rel(s_new, s_ref), self.TOL)


if __name__ == "__main__":
    unittest.main()
