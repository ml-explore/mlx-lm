"""Correctness + benchmark for T=1 specialized kernel."""

import sys
import time

import mlx.core as mx

from mlx_lm.models.gated_delta import compute_g, gated_delta_kernel
from mlx_lm.models.gated_delta_fused import gated_delta_kernel_fused
from mlx_lm.models.gated_delta_t1 import gated_delta_kernel_t1


def make_inputs():
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
    rf = Hv // Hk
    q_e = mx.repeat(q, rf, axis=-2)
    k_e = mx.repeat(k, rf, axis=-2)
    return q_e, k_e, v, a, b, A_log, dt_bias, state


def test_correctness():
    import mlx.nn as nn

    q_raw, k_raw, v, a, b, A_log, dt_bias, state = make_inputs()
    Dk = q_raw.shape[-1]

    # Reference: compute_g + sigmoid + rms_norm(q) + rms_norm(k) + kernel.
    inv_scale = Dk**-0.5
    q_ref = (inv_scale**2) * mx.fast.rms_norm(q_raw, None, 1e-6)
    k_ref = inv_scale * mx.fast.rms_norm(k_raw, None, 1e-6)
    g = compute_g(A_log, a, dt_bias)
    beta = mx.sigmoid(b)
    y_ref, s_ref = gated_delta_kernel(q_ref, k_ref, v, g, beta, state)
    mx.eval(y_ref, s_ref)

    # T=1 specialized: takes raw q, k.
    y_t1, s_t1 = gated_delta_kernel_t1(q_raw, k_raw, v, a, b, A_log, dt_bias, state)
    mx.eval(y_t1, s_t1)

    y_diff = float(mx.abs(y_ref.astype(mx.float32) - y_t1.astype(mx.float32)).max())
    s_diff = float(mx.abs(s_ref.astype(mx.float32) - s_t1.astype(mx.float32)).max())
    print(f"T=1 kernel y_diff={y_diff:.6f}, s_diff={s_diff:.6f}")
    assert y_diff < 5e-3, f"y disagree: {y_diff}"
    assert s_diff < 5e-3, f"state disagree: {s_diff}"
    print("Correctness: PASS")


def bench():
    q_raw, k_raw, v, a, b, A_log, dt_bias, state = make_inputs()
    Dk = q_raw.shape[-1]
    n = 1000

    # Original path: rms_norm, compute_g, sigmoid, kernel (5 launches).
    inv_scale = Dk**-0.5

    def orig():
        q = (inv_scale**2) * mx.fast.rms_norm(q_raw, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k_raw, None, 1e-6)
        g = compute_g(A_log, a, dt_bias)
        beta = mx.sigmoid(b)
        return gated_delta_kernel(q, k, v, g, beta, state)

    for _ in range(5):
        y, st = orig()
        mx.eval(y, st)
    t0 = time.time()
    for _ in range(n):
        y, st = orig()
        mx.eval(y, st)
    t_orig = (time.time() - t0) / n

    # Fused (compute_g internal, rms_norm external).
    def fused():
        q = (inv_scale**2) * mx.fast.rms_norm(q_raw, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k_raw, None, 1e-6)
        return gated_delta_kernel_fused(q, k, v, a, b, A_log, dt_bias, state)

    for _ in range(5):
        y, st = fused()
        mx.eval(y, st)
    t0 = time.time()
    for _ in range(n):
        y, st = fused()
        mx.eval(y, st)
    t_fused = (time.time() - t0) / n

    # Mega-fused T=1 (everything inside kernel).
    for _ in range(5):
        y, st = gated_delta_kernel_t1(q_raw, k_raw, v, a, b, A_log, dt_bias, state)
        mx.eval(y, st)
    t0 = time.time()
    for _ in range(n):
        y, st = gated_delta_kernel_t1(q_raw, k_raw, v, a, b, A_log, dt_bias, state)
        mx.eval(y, st)
    t_t1 = (time.time() - t0) / n

    print(
        f"Original (rms_norm + compute_g + sigmoid + kernel): {t_orig*1e6:.1f} μs (5 dispatches)"
    )
    print(
        f"Fused compute_g in kernel:                           {t_fused*1e6:.1f} μs ({t_orig/t_fused:.2f}×)"
    )
    print(
        f"T=1 mega-fused kernel (everything inside):           {t_t1*1e6:.1f} μs ({t_orig/t_t1:.2f}×)"
    )


def main():
    test_correctness()
    print()
    bench()


if __name__ == "__main__":
    sys.exit(main() or 0)
