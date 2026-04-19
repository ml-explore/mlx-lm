"""Correctness test: Metal MSL factored kernel vs pure-Python reference."""

import sys

import mlx.core as mx

from mlx_lm.models.gated_delta_factored_kernel import gated_delta_factored_step_metal
from mlx_lm.models.gated_delta_factored_step import factored_step_batched


def test_correctness():
    mx.random.seed(42)
    # Must have Dk divisible by 32 (SIMD width).
    B, Hv, Dv, Dk, R = 1, 4, 32, 64, 4

    U = (mx.random.normal([B, Hv, Dv, R]) * 0.05).astype(mx.float32)
    V = (mx.random.normal([B, Hv, R, Dk]) * 0.05).astype(mx.float32)
    q = (mx.random.normal([B, 1, Hv, Dk]) * 0.1).astype(mx.float32)
    k = (mx.random.normal([B, 1, Hv, Dk]) * 0.1).astype(mx.float32)
    v = (mx.random.normal([B, 1, Hv, Dv]) * 0.1).astype(mx.float32)
    g = mx.array([[[0.9] * Hv]]).astype(mx.float32)
    beta = mx.array([[[0.3] * Hv]]).astype(mx.float32)

    # Metal kernel expects bf16.
    U_b, V_b = U.astype(mx.bfloat16), V.astype(mx.bfloat16)
    q_b, k_b, v_b = q.astype(mx.bfloat16), k.astype(mx.bfloat16), v.astype(mx.bfloat16)
    g_b, beta_b = g.astype(mx.bfloat16), beta.astype(mx.bfloat16)

    y_m, U_new_m, V_new_m = gated_delta_factored_step_metal(
        U_b, V_b, q_b, k_b, v_b, g_b, beta_b
    )
    mx.eval(y_m, U_new_m, V_new_m)

    # Reference (pure MLX ops, fp32 for accuracy).
    y_r, U_new_r, V_new_r = factored_step_batched(
        U, V, q[:, 0], k[:, 0], v[:, 0], g[:, 0], beta[:, 0]
    )
    mx.eval(y_r, U_new_r, V_new_r)

    # Compare.
    y_m_fp32 = y_m[:, 0].astype(mx.float32)
    y_diff = float(mx.abs(y_m_fp32 - y_r).max().item())
    U_diff = float(mx.abs(U_new_m.astype(mx.float32) - U_new_r).max().item())
    V_diff = float(mx.abs(V_new_m.astype(mx.float32) - V_new_r).max().item())

    print(f"Max |y_diff|: {y_diff:.6f}")
    print(f"Max |U_diff|: {U_diff:.6f}")
    print(f"Max |V_diff|: {V_diff:.6f}")
    # bf16 precision ~1e-3 acceptable.
    assert y_diff < 5e-3, f"y disagree too much: {y_diff}"
    assert U_diff < 5e-3, f"U disagree: {U_diff}"
    assert V_diff < 5e-3, f"V disagree: {V_diff}"
    print("PASS")
    return True


def main():
    test_correctness()


if __name__ == "__main__":
    sys.exit(main() or 0)
