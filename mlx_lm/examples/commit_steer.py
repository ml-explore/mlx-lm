# Copyright © 2026 Apple Inc.

"""End-to-end example: extract a commit direction, then steer generation.

Step 1 (offline, once per model) records a handful of greedy reasoning traces on
checkable problems and extracts the commit direction by difference-of-means.
Step 2 (runtime, content-blind) generates with the two-mode CommitSteerer and
prints the token counts with and without it.

    python -m mlx_lm.examples.commit_steer --model mlx-community/Qwen3-8B-4bit
"""

import argparse

import mlx.core as mx
import numpy as np

from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.steer import CommitSteerer, extract_commit_direction

# A few checkable problems to elicit reasoning traces for extraction.
PROBLEMS = [
    "What is 23 times 4? Show your reasoning, then give the final answer.",
    "A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. "
    "How much is the ball?",
    "What is 37 times 43? Reason step by step, then answer.",
    "Find the smallest positive integer x with x = 2 (mod 3) and x = 3 (mod 5).",
]


def greedy_trace(model, tok, prompt, max_tokens=2048):
    ids = mx.array(
        tok.encode(
            tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    )
    out = []
    for t, _ in generate_step(ids, model, max_tokens=max_tokens):
        t = int(t)
        out.append(t)
        if t == tok.eos_token_id:
            break
    return out


def run(model, tok, prompt, steerer=None, max_tokens=2048):
    ids = mx.array(
        tok.encode(
            tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    )
    close_id = tok.encode("</think>", add_special_tokens=False)
    close_id = close_id[0] if len(close_id) == 1 else -1
    procs = [steerer.logits_processor] if steerer else None
    ctx = steerer if steerer else _null()
    n = think = 0
    closed = False
    with ctx:
        for t, _ in generate_step(
            ids, model, max_tokens=max_tokens, logits_processors=procs
        ):
            t = int(t)
            n += 1
            if t == close_id:
                closed = True
            elif not closed:
                think += 1
            if t == tok.eos_token_id:
                break
    return n, think


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=None, help="commit locus (default ~57%)")
    ap.add_argument("--alpha-bias", type=float, default=0.2)
    ap.add_argument("--alpha-hammer", type=float, default=0.8)
    ap.add_argument("--hammer-budget", type=int, default=900)
    ap.add_argument("--test-prompt", default="Is 91 a prime number? Explain, then answer.")
    args = ap.parse_args()

    model, tok = load(args.model)
    n_layers = len(model.model.layers)
    layer = args.layer if args.layer is not None else round(0.57 * n_layers)
    span = sorted({max(2, min(n_layers - 1, round(f * n_layers)))
                   for f in (0.43, 0.5, 0.57, 0.64)})

    print(f"[1/2] extracting commit direction on {len(PROBLEMS)} traces ...")
    traces = [greedy_trace(model, tok, p) for p in PROBLEMS]
    vecs = extract_commit_direction(model, tok, traces, span)
    v_hat, rms = vecs[layer]
    print(f"      layer {layer}/{n_layers}  ||v||={np.linalg.norm(v_hat):.3f}  rms={rms:.1f}")

    steerer = CommitSteerer(
        model, tok, v_hat, rms, layer=layer,
        alpha_bias=args.alpha_bias, alpha_hammer=args.alpha_hammer,
        hammer_budget=args.hammer_budget,
    )
    print(f"[2/2] generating '{args.test_prompt}'")
    n_off, think_off = run(model, tok, args.test_prompt, steerer=None)
    n_on, think_on = run(model, tok, args.test_prompt, steerer=steerer)
    cut = (1 - think_on / max(think_off, 1)) * 100
    print(f"      OFF: {n_off} tokens ({think_off} thinking)")
    print(f"      ON : {n_on} tokens ({think_on} thinking)  -> {cut:.0f}% fewer think tokens")


if __name__ == "__main__":
    main()
