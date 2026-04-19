"""Correctness + speedup test for fused compute_g + gated_delta kernel."""

import sys
import time
import mlx.core as mx

from mlx_lm.models.gated_delta import compute_g, gated_delta_kernel
from mlx_lm.models.gated_delta_fused import gated_delta_kernel_fused


def test_correctness():
    mx.random.seed(42)
    B, T, Hk, Hv, Dk, Dv = 1, 1, 16, 32, 128, 128

    q = (mx.random.normal([B, T, Hk, Dk]) * 0.1).astype(mx.bfloat16)
    k = (mx.random.normal([B, T, Hk, Dk]) * 0.1).astype(mx.bfloat16)
    v = (mx.random.normal([B, T, Hv, Dv]) * 0.1).astype(mx.bfloat16)
    a = (mx.random.normal([B, T, Hv]) * 0.1).astype(mx.bfloat16)
    b = (mx.random.normal([B, T, Hv]) * 0.1).astype(mx.bfloat16)
    A_log = (mx.random.normal([Hv]) * 0.01).astype(mx.bfloat16)
    dt_bias = (mx.random.normal([Hv]) * 0.01).astype(mx.bfloat16)
    state = mx.zeros([B, Hv, Dv, Dk], dtype=mx.bfloat16)

    # Expand q/k for Hv heads.
    rf = Hv // Hk
    q_exp = mx.repeat(q, rf, axis=-2)
    k_exp = mx.repeat(k, rf, axis=-2)

    # Reference: compute_g + sigmoid + gated_delta_kernel.
    g = compute_g(A_log, a, dt_bias)
    beta = mx.sigmoid(b)
    y_ref, state_ref = gated_delta_kernel(q_exp, k_exp, v, g, beta, state)
    mx.eval(y_ref, state_ref)

    # Fused: a, b raw to kernel.
    y_fused, state_fused = gated_delta_kernel_fused(
        q_exp, k_exp, v, a, b, A_log, dt_bias, state
    )
    mx.eval(y_fused, state_fused)

    y_diff = float(mx.abs(y_ref.astype(mx.float32) - y_fused.astype(mx.float32)).max())
    s_diff = float(mx.abs(state_ref.astype(mx.float32) - state_fused.astype(mx.float32)).max())
    print(f"y_diff: {y_diff:.6f}, state_diff: {s_diff:.6f}")
    assert y_diff < 5e-3, f"y disagree: {y_diff}"
    assert s_diff < 5e-3, f"state disagree: {s_diff}"
    print("Correctness: PASS")


def bench():
    mx.random.seed(0)
    B, T, Hk, Hv, Dk, Dv = 1, 1, 16, 32, 128, 128
    q = (mx.random.normal([B, T, Hk, Dk]) * 0.1).astype(mx.bfloat16)
    k = (mx.random.normal([B, T, Hk, Dk]) * 0.1).astype(mx.bfloat16)
    v = (mx.random.normal([B, T, Hv, Dv]) * 0.1).astype(mx.bfloat16)
    a = (mx.random.normal([B, T, Hv]) * 0.1).astype(mx.bfloat16)
    b = (mx.random.normal([B, T, Hv]) * 0.1).astype(mx.bfloat16)
    A_log = (mx.random.normal([Hv]) * 0.01).astype(mx.bfloat16)
    dt_bias = (mx.random.normal([Hv]) * 0.01).astype(mx.bfloat16)
    state = mx.zeros([B, Hv, Dv, Dk], dtype=mx.bfloat16)

    rf = Hv // Hk
    q_exp = mx.repeat(q, rf, axis=-2)
    k_exp = mx.repeat(k, rf, axis=-2)

    # Warmup.
    for _ in range(5):
        g = compute_g(A_log, a, dt_bias)
        beta = mx.sigmoid(b)
        y, st = gated_delta_kernel(q_exp, k_exp, v, g, beta, state)
        mx.eval(y, st)
    t0 = time.time()
    for _ in range(1000):
        g = compute_g(A_log, a, dt_bias)
        beta = mx.sigmoid(b)
        y, st = gated_delta_kernel(q_exp, k_exp, v, g, beta, state)
        mx.eval(y, st)
    dt_orig = (time.time() - t0) / 1000

    for _ in range(5):
        y, st = gated_delta_kernel_fused(q_exp, k_exp, v, a, b, A_log, dt_bias, state)
        mx.eval(y, st)
    t0 = time.time()
    for _ in range(1000):
        y, st = gated_delta_kernel_fused(q_exp, k_exp, v, a, b, A_log, dt_bias, state)
        mx.eval(y, st)
    dt_fused = (time.time() - t0) / 1000

    print(f"Original path (compute_g + sigmoid + kernel): {dt_orig*1e6:.1f} μs")
    print(f"Fused kernel:                                 {dt_fused*1e6:.1f} μs")
    print(f"Speedup: {dt_orig/dt_fused:.2f}×")
    print(f"Savings per layer step: {(dt_orig - dt_fused)*1e6:.1f} μs")


def main():
    test_correctness()
    bench()


if __name__ == "__main__":
    sys.exit(main() or 0)
