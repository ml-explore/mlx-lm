"""Tests for SSM decode kernel correctness."""
import unittest

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.ssm import (
    compute_dt,
    ssm_attn,
    ssm_update,
    ssm_update_kernel,
)


class TestComputeDt(unittest.TestCase):
    """Test that compute_dt applies lower bound only."""

    def test_lower_bound_enforced(self):
        dt = mx.array([[-5.0, -3.0, 0.0]])
        dt_bias = mx.zeros_like(dt)
        time_step_limit = (0.001, 0.1)
        result = compute_dt(dt, dt_bias, time_step_limit)
        mx.eval(result)
        self.assertTrue((result >= 0.001).all().item())

    def test_no_upper_clamp(self):
        # Large positive input should NOT be clamped to time_step_limit[1]
        dt = mx.array([[10.0, 20.0]])
        dt_bias = mx.zeros_like(dt)
        time_step_limit = (0.001, 0.1)
        result = compute_dt(dt, dt_bias, time_step_limit)
        mx.eval(result)
        # softplus(10) ≈ 10, softplus(20) ≈ 20 — must exceed 0.1
        self.assertTrue((result > 0.1).all().item())


class TestSSMDecodeKernel(unittest.TestCase):
    """Test Metal decode kernel numerical properties."""

    def setUp(self):
        if not mx.metal.is_available():
            self.skipTest("Metal not available")
        # Minimal SSM dimensions: h must be divisible by g,
        # ds must be divisible by 32 (kernel constraint)
        self.b = 1
        self.h = 4        # num_heads
        self.d = 8        # head_dim
        self.ds = 32      # state_dim (minimum for kernel: Ds/32 = 1)
        self.g = 2        # num_groups
        self.time_step_limit = (0.001, 100.0)

    def _make_inputs(self, a_log_scale=1.0):
        mx.random.seed(42)
        x = mx.random.normal((self.b, 1, self.h, self.d)).astype(mx.bfloat16)
        A_log = (mx.random.normal((self.h,)) * a_log_scale).astype(mx.bfloat16)
        B = mx.random.normal((self.b, 1, self.g, self.ds)).astype(mx.bfloat16)
        C = mx.random.normal((self.b, 1, self.g, self.ds)).astype(mx.bfloat16)
        D = mx.random.normal((self.h,)).astype(mx.bfloat16)
        dt = mx.random.normal((self.b, 1, self.h)).astype(mx.bfloat16)
        dt_bias = mx.random.normal((self.h,)).astype(mx.bfloat16)
        state = mx.random.normal(
            (self.b, self.h, self.d, self.ds)
        ).astype(mx.float32)
        return x, A_log, B, C, D, dt, dt_bias, state

    def test_kernel_returns_float32_state(self):
        x, A_log, B, C, D, dt, dt_bias, state = self._make_inputs()
        _, new_state = ssm_update_kernel(
            x, A_log, B, C, D, dt, dt_bias, state, self.time_step_limit,
        )
        mx.eval(new_state)
        self.assertEqual(new_state.dtype, mx.float32)

    def test_kernel_matches_ssm_attn_single_step(self):
        x, A_log, B, C, D, dt, dt_bias, state = self._make_inputs()
        y_attn, s_attn = ssm_attn(
            x, A_log, B, C, D, dt, dt_bias, state, self.time_step_limit,
        )
        y_kern, s_kern = ssm_update_kernel(
            x, A_log, B, C, D, dt, dt_bias, state, self.time_step_limit,
        )
        mx.eval(y_attn, s_attn, y_kern, s_kern)
        y_diff = mx.abs(
            y_attn.astype(mx.float32) - y_kern.astype(mx.float32)
        ).max().item()
        s_diff = mx.abs(s_attn - s_kern).max().item()
        self.assertLess(y_diff, 0.5, f"Output diff too large: {y_diff}")
        self.assertLess(s_diff, 0.5, f"State diff too large: {s_diff}")

    def test_kernel_stable_over_many_steps(self):
        """Verify kernel doesn't diverge from ssm_attn over 50 decode steps."""
        x, A_log, B, C, D, dt, dt_bias, state = self._make_inputs()
        state_attn = state
        state_kern = state
        for _ in range(50):
            _, state_attn = ssm_attn(
                x, A_log, B, C, D, dt, dt_bias, state_attn,
                self.time_step_limit,
            )
            _, state_kern = ssm_update_kernel(
                x, A_log, B, C, D, dt, dt_bias, state_kern,
                self.time_step_limit,
            )
            mx.eval(state_attn, state_kern)
        s_diff = mx.abs(state_attn - state_kern).max().item()
        # Allow larger tolerance for accumulated differences over many steps
        self.assertLess(
            s_diff, 10.0,
            f"State diverged too much over 50 steps: {s_diff}",
        )

    def test_ssm_update_routes_to_kernel(self):
        """ssm_update should route to kernel for seq_len=1 with state."""
        x, A_log, B, C, D, dt, dt_bias, state = self._make_inputs()
        y_update, s_update = ssm_update(
            x, A_log, B, C, D, dt, dt_bias, state, self.time_step_limit,
        )
        y_kern, s_kern = ssm_update_kernel(
            x, A_log, B, C, D, dt, dt_bias, state, self.time_step_limit,
        )
        mx.eval(y_update, s_update, y_kern, s_kern)
        y_diff = mx.abs(
            y_update.astype(mx.float32) - y_kern.astype(mx.float32)
        ).max().item()
        self.assertLess(y_diff, 1e-6, "ssm_update should route to kernel")


if __name__ == "__main__":
    unittest.main()
