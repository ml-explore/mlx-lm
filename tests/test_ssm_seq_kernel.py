# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx

from mlx_lm.models.ssm import (
    compute_dt,
    ssm_attn,
    ssm_update,
    ssm_update_kernel,
    ssm_update_seq_kernel,
)

TSL = (0.001, 100.0)


def make_inputs(b, S, h, dh, g, ds, dtype=mx.float32, state_dtype=mx.float32):
    x = (mx.random.normal((b, S, h, dh)) * 0.5).astype(dtype)
    B = (mx.random.normal((b, S, g, ds)) * 0.3).astype(dtype)
    C = (mx.random.normal((b, S, g, ds)) * 0.3).astype(dtype)
    dt = (mx.random.normal((b, S, h)) * 0.1).astype(dtype)
    A_log = mx.random.normal((h,)) * 0.5
    D = mx.abs(mx.random.normal((h,)))
    dt_bias = mx.random.normal((h,)) * 0.1
    state = (mx.random.normal((b, h, dh, ds)) * 0.2).astype(state_dtype)
    return x, A_log, B, C, D, dt, dt_bias, state


def exact_ref(x, A_log, B, C, D, dt, dt_bias, state, tsl):
    """Position-at-a-time fp32 recurrence, the same math as the S=1 step
    kernel; the sequential kernel must match it to float rounding."""
    b, S, h, dh = x.shape
    g = B.shape[2]
    dtc = compute_dt(dt, dt_bias, tsl)
    A = -mx.exp(A_log)
    rep = h // g
    ys = []
    st = state
    for s in range(S):
        dA = mx.exp(A * dtc[:, s])
        Bs = mx.repeat(B[:, s], rep, axis=1)
        Cs = mx.repeat(C[:, s], rep, axis=1)
        xs = x[:, s]
        dBx = dtc[:, s][..., None, None] * xs[..., None] * Bs[:, :, None, :]
        st = dA[..., None, None] * st + dBx
        ys.append((st * Cs[:, :, None, :]).sum(-1) + xs * D[None, :, None])
    return mx.stack(ys, axis=1), st


def composed_steps(x, A_log, B, C, D, dt, dt_bias, state, tsl):
    """S sequential calls of the S=1 step kernel — the decode path the
    sequential kernel must agree with bit-for-bit (float32 state)."""
    ys = []
    st = state
    for s in range(x.shape[1]):
        y, st = ssm_update_kernel(
            x[:, s : s + 1],
            A_log,
            B[:, s : s + 1],
            C[:, s : s + 1],
            D,
            dt[:, s : s + 1],
            dt_bias,
            st,
            tsl,
        )
        ys.append(y)
    return mx.concatenate(ys, axis=1), st


@unittest.skipUnless(mx.metal.is_available(), "metal only")
class TestSSMSeqKernel(unittest.TestCase):
    def test_matches_exact_recurrence(self):
        mx.random.seed(0)
        for S in (2, 3, 4, 5, 8):
            args = make_inputs(1, S, 8, 16, 2, 32)
            y_e, s_e = exact_ref(*args, TSL)
            y_k, s_k = ssm_update_seq_kernel(*args, TSL)
            self.assertTrue(
                mx.allclose(y_e, y_k, atol=1e-5).item(), f"y mismatch at S={S}"
            )
            self.assertTrue(
                mx.allclose(s_e, s_k, atol=1e-5).item(), f"state mismatch at S={S}"
            )

    def test_bit_identical_to_step_kernel(self):
        # Verify and decode numerics must agree exactly: for a float32 state
        # (the only state dtype the prefill paths produce), the sequential
        # kernel is bit-identical to composing the S=1 step kernel.
        mx.random.seed(1)
        for dtype in (mx.float32, mx.float16, mx.bfloat16):
            for b in (1, 3):
                for S in (2, 8):
                    # nemotron_h-class dims
                    args = make_inputs(b, S, 128, 64, 8, 128, dtype=dtype)
                    y_k, s_k = ssm_update_seq_kernel(*args, TSL)
                    y_s, s_s = composed_steps(*args, TSL)
                    label = f"dtype={dtype}, b={b}, S={S}"
                    self.assertTrue(
                        mx.array_equal(y_k, y_s).item(), f"y not exact: {label}"
                    )
                    self.assertTrue(
                        mx.array_equal(s_k, s_s).item(), f"state not exact: {label}"
                    )

    def test_dispatch(self):
        mx.random.seed(2)
        # In range: ssm_update must route S in [2, 8] to the sequential
        # kernel (bit-identical outputs; bf16 inputs make an ssm_attn result
        # detectably different).
        args = make_inputs(2, 4, 128, 64, 8, 128, dtype=mx.bfloat16)
        y_u, s_u = ssm_update(*args, TSL)
        y_k, s_k = ssm_update_seq_kernel(*args, TSL)
        self.assertTrue(mx.array_equal(y_u, y_k).item())
        self.assertTrue(mx.array_equal(s_u, s_k).item())

        # S=1 routes to the step kernel.
        args = make_inputs(1, 1, 8, 16, 2, 32)
        y_u, s_u = ssm_update(*args, TSL)
        y_s, s_s = ssm_update_kernel(*args, TSL)
        self.assertTrue(mx.array_equal(y_u, y_s).item())
        self.assertTrue(mx.array_equal(s_u, s_s).item())

        # Above the max length, and for unsupported state dims, ssm_update
        # must fall back to ssm_attn (the sequential kernel requires
        # state_dim >= 32 and state_dim % 32 == 0).
        for b, S, h, dh, g, ds in ((1, 9, 8, 16, 2, 32), (1, 4, 8, 16, 2, 48)):
            args = make_inputs(b, S, h, dh, g, ds)
            y_u, s_u = ssm_update(*args, TSL)
            y_a, s_a = ssm_attn(*args, TSL)
            self.assertTrue(mx.array_equal(y_u, y_a).item(), f"S={S}, ds={ds}")
            self.assertTrue(mx.array_equal(s_u, s_a).item(), f"S={S}, ds={ds}")

        # A mask also falls back.
        x, A_log, B, C, D, dt, dt_bias, state = make_inputs(1, 4, 8, 16, 2, 32)
        mask = mx.ones((1, 4), dtype=mx.bool_)
        y_u, s_u = ssm_update(x, A_log, B, C, D, dt, dt_bias, state, TSL, mask=mask)
        y_a, s_a = ssm_attn(x, A_log, B, C, D, dt, dt_bias, state, TSL, mask=mask)
        self.assertTrue(mx.array_equal(y_u, y_a).item())
        self.assertTrue(mx.array_equal(s_u, s_a).item())


if __name__ == "__main__":
    unittest.main()
