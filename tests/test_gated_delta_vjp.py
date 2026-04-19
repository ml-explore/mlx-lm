"""Numerical gradient check for ``gated_delta_update_vjp``.

Toy: B=1, T=8, Hk=2, Hv=4, Dk=16, Dv=8, fp32.
Central finite-differences with eps=1e-3, threshold rel.err < 1e-3.

Run: caffeinate -i .venv-llm/bin/python llm/test_deltanet_vjp.py
"""

import sys
import traceback
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.gated_delta_vjp import gated_delta_update_vjp
from mlx_lm.models.gated_delta_vjp_metal import gated_delta_update_vjp_metal

B, T, Hk, Hv, Dk, Dv = 1, 8, 2, 4, 16, 8
SEED = 42
EPS = 1e-3
TOL_FP32 = 1e-3  # rel.err
ATOL_FP32 = 1e-4


def make_inputs(seed: int = SEED, vectorized_g: bool = False, dtype=mx.float32):
    """Generate deterministic random toy inputs."""
    mx.random.seed(seed)
    q = mx.random.normal((B, T, Hk, Dk)).astype(dtype) * 0.3
    k = mx.random.normal((B, T, Hk, Dk)).astype(dtype) * 0.3
    v = mx.random.normal((B, T, Hv, Dv)).astype(dtype) * 0.3
    a = mx.random.normal((B, T, Hv)).astype(dtype) * 0.3
    b = mx.random.normal((B, T, Hv)).astype(dtype) * 0.3
    A_log = mx.log(mx.random.uniform(0.5, 4.0, (Hv,)).astype(dtype))
    dt_bias = mx.ones((Hv,)).astype(dtype)
    state = mx.zeros((B, Hv, Dv, Dk), dtype=dtype)
    return q, k, v, a, b, A_log, dt_bias, state


def loss_fn(q, k, v, a, b, A_log, dt_bias, state, cot_y, cot_s):
    """Scalar loss used to drive analytical and numerical gradients."""
    y, s = gated_delta_update_vjp(q, k, v, a, b, A_log, dt_bias, state)
    return (y * cot_y).sum() + (s * cot_s).sum()


def fd_grad_elem(loss_callable, x, idx, eps=EPS):
    """Central finite difference for a single scalar element x.flat[idx]."""
    flat = x.flatten()
    n = flat.shape[0]
    e_plus = mx.zeros((n,), dtype=x.dtype)
    e_minus = mx.zeros((n,), dtype=x.dtype)
    # MLX has no scatter-update by index; build the perturbation via concat.
    e = mx.concatenate(
        [
            mx.zeros((idx,), dtype=x.dtype),
            mx.array([eps], dtype=x.dtype),
            mx.zeros((n - idx - 1,), dtype=x.dtype),
        ]
    )
    e = e.reshape(x.shape)
    l_plus = loss_callable(x + e)
    l_minus = loss_callable(x - e)
    return float((l_plus - l_minus) / (2 * eps))


def check_grad_for_arg(
    arg_name: str,
    arg_value: mx.array,
    loss_with_one_arg,
    analytical_grad: mx.array,
    n_samples: int = 8,
):
    """Compare analytical gradient against FD on n_samples indices."""
    flat_size = int(mx.prod(mx.array(list(arg_value.shape))).item())
    n_check = min(n_samples, flat_size)
    # Deterministic subset of indices for reproducibility.
    step = max(1, flat_size // n_check)
    indices = list(range(0, flat_size, step))[:n_check]
    analytical_flat = analytical_grad.flatten()

    max_rel = 0.0
    failures = []
    for idx in indices:
        fd = fd_grad_elem(loss_with_one_arg, arg_value, idx)
        an = float(analytical_flat[idx].item())
        denom = max(abs(fd), abs(an), 1e-8)
        rel = abs(fd - an) / denom
        if rel > TOL_FP32 and abs(fd - an) > ATOL_FP32:
            failures.append((idx, fd, an, rel))
        max_rel = max(max_rel, rel)

    status = "PASS" if not failures else "FAIL"
    print(f"  {arg_name:10s} max_rel={max_rel:.3e}  [{status}]")
    if failures:
        for idx, fd, an, rel in failures[:3]:
            print(f"    idx={idx} fd={fd:+.6e} an={an:+.6e} rel={rel:.3e}")
    return not failures


def run_one_config(label: str, mask: Optional[mx.array] = None):
    print(f"\n--- {label} ---")
    q0, k0, v0, a0, b0, A0, dt0, S0 = make_inputs()
    cot_y = mx.random.normal((B, T, Hv, Dv)) * 0.5
    cot_s = mx.random.normal((B, Hv, Dv, Dk)) * 0.5

    def loss_full(q, k, v, a, b, A_log, dt_bias, S):
        y, s = gated_delta_update_vjp(q, k, v, a, b, A_log, dt_bias, S, mask)
        return (y * cot_y).sum() + (s * cot_s).sum()

    grad_fn = mx.value_and_grad(loss_full, argnums=(0, 1, 2, 3, 4, 5, 6, 7))
    (_, grads) = grad_fn(q0, k0, v0, a0, b0, A0, dt0, S0)
    names = ["q", "k", "v", "a", "b", "A_log", "dt_bias", "S0"]
    args = [q0, k0, v0, a0, b0, A0, dt0, S0]

    ok_all = True
    for i, (name, val, gr) in enumerate(zip(names, args, grads)):

        def lo(x, i=i):
            a = list(args)
            a[i] = x
            return loss_full(*a)

        ok = check_grad_for_arg(name, val, lo, gr)
        ok_all = ok_all and ok
    return ok_all


def run_mask_equivalence():
    """A mask of all-True must give identical output to the unmasked path."""
    print("\n--- Masked equivalence (mask=all True) ---")
    q, k, v, a, b, A_log, dt_bias, S0 = make_inputs()
    y_u, s_u = gated_delta_update_vjp(q, k, v, a, b, A_log, dt_bias, S0)
    mask = mx.ones((B, T), dtype=mx.bool_)
    y_m, s_m = gated_delta_update_vjp(q, k, v, a, b, A_log, dt_bias, S0, mask)
    diff_y = float(mx.abs(y_u - y_m).max().item())
    diff_s = float(mx.abs(s_u - s_m).max().item())
    ok = diff_y < 1e-5 and diff_s < 1e-5
    print(
        f"  max|y_diff|={diff_y:.3e}  max|S_diff|={diff_s:.3e}  "
        f"[{'PASS' if ok else 'FAIL'}]"
    )
    return ok


def run_metal_forward_equivalence():
    """Metal forward must match the Python VJP forward to floating noise."""
    print("\n--- Metal forward equivalence vs Python VJP ---")
    # Metal kernel requires Dk % 32 == 0; use Dk=64 (toy-size multiple of 32).
    B_m, T_m, Hk_m, Hv_m, Dk_m, Dv_m = 1, 16, 2, 4, 64, 16
    mx.random.seed(SEED)
    q = mx.random.normal((B_m, T_m, Hk_m, Dk_m)) * 0.3
    k = mx.random.normal((B_m, T_m, Hk_m, Dk_m)) * 0.3
    v = mx.random.normal((B_m, T_m, Hv_m, Dv_m)) * 0.3
    a = mx.random.normal((B_m, T_m, Hv_m)) * 0.3
    b = mx.random.normal((B_m, T_m, Hv_m)) * 0.3
    A_log = mx.log(mx.random.uniform(0.5, 4.0, (Hv_m,)))
    dt_bias = mx.ones((Hv_m,))
    S0 = mx.zeros((B_m, Hv_m, Dv_m, Dk_m))
    y_py, s_py = gated_delta_update_vjp(q, k, v, a, b, A_log, dt_bias, S0)
    y_mt, s_mt = gated_delta_update_vjp_metal(q, k, v, a, b, A_log, dt_bias, S0)
    diff_y = float(mx.abs(y_py - y_mt).max().item())
    diff_s = float(mx.abs(s_py - s_mt).max().item())
    ok = diff_y < 1e-5 and diff_s < 1e-5
    print(
        f"  max|y_diff|={diff_y:.3e}  max|S_diff|={diff_s:.3e}  "
        f"[{'PASS' if ok else 'FAIL'}]"
    )
    return ok


def run_metal_gradient_equivalence():
    """Metal-VJP gradients must match Python-VJP gradients to floating noise."""
    print("\n--- Metal gradient equivalence vs Python VJP ---")
    B_m, T_m, Hk_m, Hv_m, Dk_m, Dv_m = 1, 16, 2, 4, 64, 16
    mx.random.seed(SEED)
    q = mx.random.normal((B_m, T_m, Hk_m, Dk_m)) * 0.3
    k = mx.random.normal((B_m, T_m, Hk_m, Dk_m)) * 0.3
    v = mx.random.normal((B_m, T_m, Hv_m, Dv_m)) * 0.3
    a = mx.random.normal((B_m, T_m, Hv_m)) * 0.3
    b = mx.random.normal((B_m, T_m, Hv_m)) * 0.3
    A_log = mx.log(mx.random.uniform(0.5, 4.0, (Hv_m,)))
    dt_bias = mx.ones((Hv_m,))
    S0 = mx.zeros((B_m, Hv_m, Dv_m, Dk_m))
    cot_y = mx.random.normal((B_m, T_m, Hv_m, Dv_m)) * 0.5
    cot_s = mx.random.normal((B_m, Hv_m, Dv_m, Dk_m)) * 0.5

    def loss(fn, *args):
        y, s = fn(*args)
        return (y * cot_y).sum() + (s * cot_s).sum()

    grad_py = mx.grad(
        lambda *xs: loss(gated_delta_update_vjp, *xs), argnums=(0, 1, 2, 3, 4)
    )(q, k, v, a, b, A_log, dt_bias, S0)
    grad_mt = mx.grad(
        lambda *xs: loss(gated_delta_update_vjp_metal, *xs), argnums=(0, 1, 2, 3, 4)
    )(q, k, v, a, b, A_log, dt_bias, S0)

    ok_all = True
    for name, gp, gm in zip(["q", "k", "v", "a", "b"], grad_py, grad_mt):
        diff = float(mx.abs(gp - gm).max().item())
        scale = float(mx.abs(gp).max().item()) + 1e-8
        rel = diff / scale
        ok = rel < 1e-5
        ok_all = ok_all and ok
        print(
            f"  {name:3s}: max|diff|={diff:.3e}  rel={rel:.3e}  "
            f"[{'PASS' if ok else 'FAIL'}]"
        )
    return ok_all


def run_forward_equivalence():
    """Forward output must match the reference ``gated_delta_update`` ops path.

    ``use_kernel=False`` pins the reference to the pure-Python implementation,
    which has the same semantics as our chunked forward (kernel has a slightly
    different mask handling for padded positions).
    """
    from mlx_lm.models.gated_delta import gated_delta_update as ref_fn

    print("\n--- Forward equivalence vs gated_delta_update (use_kernel=False) ---")
    q, k, v, a, b, A_log, dt_bias, S0 = make_inputs()
    # Reference works with repeat-head-expanded q/k like our wrapper does.
    y_ref, s_ref = ref_fn(q, k, v, a, b, A_log, dt_bias, S0, use_kernel=False)
    y_vjp, s_vjp = gated_delta_update_vjp(q, k, v, a, b, A_log, dt_bias, S0)
    diff_y = float(mx.abs(y_ref - y_vjp).max().item())
    diff_s = float(mx.abs(s_ref - s_vjp).max().item())
    ok = diff_y < 1e-4 and diff_s < 1e-4
    print(
        f"  max|y_diff|={diff_y:.3e}  max|S_diff|={diff_s:.3e}  "
        f"[{'PASS' if ok else 'FAIL'}]"
    )
    return ok


def run_edge_case_lengths():
    """T=1, T=CHUNK_SIZE, T=CHUNK_SIZE+1 must all give finite results."""
    from mlx_lm.models.gated_delta import gated_delta_update as ref_fn
    from mlx_lm.models.gated_delta_vjp import CHUNK_SIZE

    print("\n--- Edge case sequence lengths ---")
    ok_all = True
    for T_edge in (1, CHUNK_SIZE, CHUNK_SIZE + 1):
        mx.random.seed(SEED)
        q = mx.random.normal((B, T_edge, Hk, Dk)) * 0.3
        k = mx.random.normal((B, T_edge, Hk, Dk)) * 0.3
        v = mx.random.normal((B, T_edge, Hv, Dv)) * 0.3
        a = mx.random.normal((B, T_edge, Hv)) * 0.3
        b = mx.random.normal((B, T_edge, Hv)) * 0.3
        A_log = mx.log(mx.random.uniform(0.5, 4.0, (Hv,)))
        dt_bias = mx.ones((Hv,))
        S0 = mx.zeros((B, Hv, Dv, Dk))
        y_ref, s_ref = ref_fn(q, k, v, a, b, A_log, dt_bias, S0, use_kernel=False)
        y_vjp, s_vjp = gated_delta_update_vjp(q, k, v, a, b, A_log, dt_bias, S0)
        diff_y = float(mx.abs(y_ref - y_vjp).max().item())
        diff_s = float(mx.abs(s_ref - s_vjp).max().item())
        any_nan = any(bool(mx.isnan(t).any().item()) for t in (y_vjp, s_vjp))
        ok = not any_nan and diff_y < 1e-4 and diff_s < 1e-4
        ok_all = ok_all and ok
        print(
            f"  T={T_edge:3d}  max|y|={diff_y:.2e}  max|S|={diff_s:.2e}  "
            f"nan={any_nan}  [{'PASS' if ok else 'FAIL'}]"
        )
    return ok_all


def run_extreme_clamp():
    """Extreme A_log + softplus arg must not produce NaN thanks to the clamp."""
    print("\n--- Numerical clamp under extreme A_log and a ---")
    mx.random.seed(SEED)
    q = mx.random.normal((B, T, Hk, Dk)) * 0.3
    k = mx.random.normal((B, T, Hk, Dk)) * 0.3
    v = mx.random.normal((B, T, Hv, Dv)) * 0.3
    a = mx.full((B, T, Hv), 15.0)  # softplus(16) ~ 16
    b = mx.zeros((B, T, Hv))
    A_log = mx.full((Hv,), 5.0)  # exp(5) ≈ 148, combined arg ≈ -2370
    dt_bias = mx.ones((Hv,))
    S0 = mx.zeros((B, Hv, Dv, Dk))
    y, s = gated_delta_update_vjp(q, k, v, a, b, A_log, dt_bias, S0)
    any_nan = bool(mx.isnan(y).any().item()) or bool(mx.isnan(s).any().item())
    any_inf = bool(mx.isinf(y).any().item()) or bool(mx.isinf(s).any().item())
    ok = not any_nan and not any_inf
    print(f"  nan={any_nan}  inf={any_inf}  [{'PASS' if ok else 'FAIL'}]")
    return ok


def run_mask_state_carryover():
    """mask=[1,1,1,1,0,0,0,0]: final state must equal the state after only
    the first four unmasked steps."""
    print("\n--- Masked state-carryover (first half True, second half False) ---")
    q, k, v, a, b, A_log, dt_bias, S0 = make_inputs()
    mask_half = mx.concatenate(
        [
            mx.ones((B, T // 2), dtype=mx.bool_),
            mx.zeros((B, T - T // 2), dtype=mx.bool_),
        ],
        axis=1,
    )
    _, s_full = gated_delta_update_vjp(q, k, v, a, b, A_log, dt_bias, S0, mask_half)
    # Reference: run only the first T//2 steps unmasked.
    half = T // 2
    _, s_ref = gated_delta_update_vjp(
        q[:, :half],
        k[:, :half],
        v[:, :half],
        a[:, :half],
        b[:, :half],
        A_log,
        dt_bias,
        S0,
    )
    diff = float(mx.abs(s_full - s_ref).max().item())
    ok = diff < 1e-5
    print(f"  max|S_diff|={diff:.3e}  [{'PASS' if ok else 'FAIL'}]")
    return ok


def main():
    print("=" * 60)
    print("Numerical gradient check for gated_delta_update_vjp")
    print(f"Toy: B={B} T={T} Hk={Hk} Hv={Hv} Dk={Dk} Dv={Dv}")
    print(f"Eps={EPS}, tol_rel={TOL_FP32}, tol_abs={ATOL_FP32}")
    print("=" * 60)

    results = {}
    try:
        results["scalar_g"] = run_one_config("Scalar gating (production path)")
    except Exception:
        traceback.print_exc()
        results["scalar_g"] = False

    try:
        results["fwd_equivalence"] = run_forward_equivalence()
    except Exception:
        traceback.print_exc()
        results["fwd_equivalence"] = False

    try:
        results["edge_lengths"] = run_edge_case_lengths()
    except Exception:
        traceback.print_exc()
        results["edge_lengths"] = False

    try:
        results["extreme_clamp"] = run_extreme_clamp()
    except Exception:
        traceback.print_exc()
        results["extreme_clamp"] = False

    try:
        results["mask_equivalence"] = run_mask_equivalence()
    except Exception:
        traceback.print_exc()
        results["mask_equivalence"] = False

    try:
        results["mask_carryover"] = run_mask_state_carryover()
    except Exception:
        traceback.print_exc()
        results["mask_carryover"] = False

    try:
        # FD grads through masked path (half-mask, non-trivial).
        half_mask = mx.concatenate(
            [
                mx.ones((B, T // 2), dtype=mx.bool_),
                mx.zeros((B, T - T // 2), dtype=mx.bool_),
            ],
            axis=1,
        )
        results["masked_fd"] = run_one_config(
            "Masked gating FD (half True / half False)", mask=half_mask
        )
    except Exception:
        traceback.print_exc()
        results["masked_fd"] = False

    try:
        results["metal_equivalence"] = run_metal_gradient_equivalence()
    except Exception:
        traceback.print_exc()
        results["metal_equivalence"] = False

    try:
        results["metal_forward"] = run_metal_forward_equivalence()
    except Exception:
        traceback.print_exc()
        results["metal_forward"] = False

    print("\n" + "=" * 60)
    for k, v in results.items():
        print(f"  {k:20s}: {'PASS' if v else 'FAIL'}")
    print("=" * 60)
    return all(results.values())


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
