"""Reproduce the absorbed-vs-materialized deepseek_v2 A/B (PR reproducibility).

Method: this script imports whatever `mlx_lm.models.deepseek_v2` is on the path,
so run it once on this branch (absorbed decode) and once on upstream/main
(materialized), then compare. It reports decode tok/s + peak GB at several
context lengths, and the last-position logit correlation between two saved logit
dumps (to show the two implementations are numerically equivalent, not bitwise).

  # on this branch:
  python bench_absorbed_mla.py --model mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx --logits abs.npy
  # on a pristine upstream/main checkout:
  python bench_absorbed_mla.py --model mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx --logits mat.npy
  # then:
  python bench_absorbed_mla.py --corr abs.npy mat.npy
"""

import argparse
import time

import mlx.core as mx
import numpy as np

from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache


def decode_bench(model, tok, ctx, ndecode=64):
    base = tok.encode("The history of computing is long and detailed. " * 400)
    ids = (base * ((ctx // len(base)) + 1))[:ctx]
    model(mx.array([ids[:8]]), cache=make_prompt_cache(model))
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    cache = make_prompt_cache(model)
    logits = model(mx.array([ids]), cache=cache)
    mx.eval(logits, [c.state for c in cache])
    t = int(mx.argmax(logits[0, -1]).item())
    t0 = time.perf_counter()
    for _ in range(ndecode):
        logits = model(mx.array([[t]]), cache=cache)
        mx.eval(logits)
        t = int(mx.argmax(logits[0, -1]).item())
    dt = time.perf_counter() - t0
    peak = (mx.get_peak_memory() if hasattr(mx, "get_peak_memory") else 0) / 1e9
    return ndecode / dt, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument(
        "--contexts", type=int, nargs="+", default=[4096, 8192, 16384, 32768]
    )
    ap.add_argument(
        "--logits", help="save last-prompt-position prefill logits to this .npy"
    )
    ap.add_argument("--corr", nargs=2, help="two .npy logit dumps to correlate")
    args = ap.parse_args()

    if not args.corr and not args.model:
        ap.error("--model is required unless running --corr")

    if args.corr:
        a, b = np.load(args.corr[0]), np.load(args.corr[1])
        d = np.abs(a - b)
        print(
            f"logit corr={np.corrcoef(a, b)[0, 1]:.6f} "
            f"mean|d|={d.mean():.4f} max|d|={d.max():.4f} "
            f"top1_agree={a.argmax() == b.argmax()}"
        )
        return

    model, tok = load(args.model)
    mx.eval(model.parameters())
    if args.logits:
        ids = tok.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": "Write a short Python function that returns the n-th Fibonacci number.",
                }
            ],
            add_generation_prompt=True,
        )
        logits = model(mx.array([ids]), cache=make_prompt_cache(model))
        mx.eval(logits)
        np.save(args.logits, np.array(logits[0, -1], dtype=np.float32))
        print("saved logits to", args.logits)
    for ctx in args.contexts:
        tps, peak = decode_bench(model, tok, ctx)
        print(f"ctx={ctx} decode_tps={tps:.1f} peak_gb={peak:.2f}")


if __name__ == "__main__":
    main()
