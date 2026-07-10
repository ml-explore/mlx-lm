"""Benchmark the prompt-cache prefix-anchor-stride API (upstream PR).

Exercises the NEW public API in ``mlx_lm.models.cache``:

  * ``PrefixAnchorCapture(cache, stride)`` -- a
    ``prompt_progress_callback(processed, total)``-compatible callable wired into
    ``generate_step``. It deep-copies the live cache at stride boundaries during
    the prefill the server is doing anyway, recording ``PrefixAnchor`` snapshots.
  * ``LRUPromptCache.insert_anchors(model, tokens, anchors)`` -- stores each
    snapshot as an ordinary *shorter* trie entry. Retrieval is unchanged:
    ``fetch_nearest_cache`` finds the longest stored shorter prefix + tail.

The scenario that distinguishes the anchor API from plain upstream
``LRUPromptCache``: a long SHARED prefix followed by DIVERGENT tails --
``req_a = P + tail_a``, ``req_b = P + tail_b``. Upstream's ``fetch_nearest_cache``
can reuse ``P`` for ``req_b`` only by *trimming* the stored ``req_a`` cache back
to ``|P|``; for a NON-TRIMMABLE cache (recurrent/hybrid models whose caches
report ``can_trim_prompt_cache == False``, e.g. ``ArraysCache``) that path is
skipped, so ``req_b`` re-prefills from scratch (reused == 0). Anchor-stride
pre-saved an exact shorter snapshot at each stride boundary of ``req_a``, so
``req_b`` reuses up to the nearest anchor ``<= |P|``.

Two modes:

  synthetic (default, CPU, no download)
      Builds a genuinely tiny NON-TRIMMABLE recurrent nn model whose
      ``make_prompt_cache`` yields ``ArraysCache`` per layer, and drives a real
      prefill through ``mlx_lm.generate.generate_step`` with a
      ``PrefixAnchorCapture`` wired as ``prompt_progress_callback`` (the intended
      usage). Proves: upstream reuses 0 across the divergent tail; anchor mode
      reuses the nearest-anchor prefix; and the anchor-reused continuation is
      next-token-IDENTICAL to a cold full prefill.

  --model PATH (optional, real recurrent/hybrid model)
      Mirrors ``bench_prefix_cache_vs_upstream.py``: cold full prefill of req_b
      vs upstream ``LRUPromptCache`` reuse vs anchor-stride reuse, reporting
      reused-tokens and prefill/TTFT ms per arm, each gated on next-token
      equivalence vs cold. The anchor arm captures during req_a's prefill (not a
      separate manual re-prefill). Prefill/TTFT is reported separately from
      decode throughput; no decode-token/s win is claimed.

  # synthetic (default)
  /Users/pierrelamy/Desktop/mlx-uag/.venv-mlxmain/bin/python \
      benchmarks/bench_prompt_cache_anchor_stride.py --stride 128 --target 600

  # real recurrent/hybrid model
  .venv/bin/python benchmarks/bench_prompt_cache_anchor_stride.py \
      --model ~/.cache/huggingface/hub/models--.../snapshots/* \
      --stride 128 --target 4096
"""

import argparse
import copy
import json
import os
import random
import sys
import time

# Prefer THIS worktree's mlx_lm (which carries the anchor API) over any mlx_lm
# installed in the interpreter's site-packages.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

from mlx_lm.generate import generate_step  # noqa: E402
from mlx_lm.models.cache import (  # noqa: E402
    ArraysCache,
    LRUPromptCache,
    PrefixAnchorCapture,
    can_trim_prompt_cache,
    make_prompt_cache,
)

# One string key stands in for the model in the trie (the module object is
# unhashable); upstream's server keys the same way.
MODEL_KEY = "bench-anchor-model"


# --------------------------------------------------------------------------- #
# Tiny non-trimmable recurrent model (synthetic mode)
# --------------------------------------------------------------------------- #
class _RecurrentCell(nn.Module):
    """A minimal per-token linear recurrence: h_t = tanh(Wx x_t + Wh h_{t-1}).

    The final hidden state depends on the ENTIRE processed prefix and is computed
    identically per token regardless of how prefill is chunked -- exactly the
    property that makes reuse-from-a-shorter-prefix bitwise-equivalent to a cold
    full prefill.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.Wx = nn.Linear(dim, dim, bias=False)
        self.Wh = nn.Linear(dim, dim, bias=False)

    def __call__(self, x, cache):
        B, T, D = x.shape
        h = None if cache is None else cache[0]
        if h is None:
            h = mx.zeros((B, D), dtype=x.dtype)
        outs = []
        for t in range(T):
            h = mx.tanh(self.Wx(x[:, t]) + self.Wh(h))
            outs.append(h)
        if cache is not None:
            cache[0] = h  # store the final hidden state (the recurrent state)
        return mx.stack(outs, axis=1)


class TinyRecurrentModel(nn.Module):
    """A genuinely tiny recurrent LM whose cache is a non-trimmable ArraysCache.

    ``make_cache`` returns one single-slot ``ArraysCache`` per layer.
    ``can_trim_prompt_cache`` is False for it, so this reproduces the
    recurrent/hybrid situation the anchor API targets, on the CPU, with no
    download.
    """

    def __init__(self, vocab: int, dim: int, n_layers: int):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.layers = [_RecurrentCell(dim) for _ in range(n_layers)]
        self.out = nn.Linear(dim, vocab, bias=False)

    def make_cache(self):
        return [ArraysCache(1) for _ in self.layers]

    def __call__(self, inputs, cache=None):
        x = self.embed(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        for layer, c in zip(self.layers, cache):
            x = layer(x, c)
        return self.out(x)


def build_tiny_model(vocab: int, dim: int, n_layers: int, seed: int):
    mx.random.seed(seed)
    model = TinyRecurrentModel(vocab, dim, n_layers)
    mx.eval(model.parameters())
    return model


def build_synthetic_tokens(vocab: int, target: int, tail: int, seed: int):
    """Deterministic shared prefix P (len ``target``) + two divergent tails.

    Returns ``(req_a, req_b, shared_prefix_len)``; the tails are guaranteed to
    differ at their first token so the shared prefix is exactly ``target``.
    """
    rng = random.Random(seed)
    prefix = [rng.randrange(vocab) for _ in range(target)]
    tail_a = [rng.randrange(vocab) for _ in range(tail)]
    tail_b = [rng.randrange(vocab) for _ in range(tail)]
    if tail_b[0] == tail_a[0]:
        tail_b[0] = (tail_a[0] + 1) % vocab
    return prefix + tail_a, prefix + tail_b, target


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def common_prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def next_tok(model, cache, ids):
    """Argmax next token from ``ids`` fed through ``cache`` (may be pre-filled)."""
    logits = model(mx.array([ids]), cache=cache)
    mx.eval(logits)
    return int(mx.argmax(logits[0, -1]).item())


def timed_next(model, cache, ids):
    t0 = time.perf_counter()
    tk = next_tok(model, cache, ids)
    return tk, time.perf_counter() - t0


def cache_classes(cache):
    return [type(c).__name__ for c in cache]


def serve_and_capture(model, tokens, stride):
    """Serve ``tokens`` via a real prefill, capturing anchors as a byproduct.

    Wires ``PrefixAnchorCapture`` as ``generate_step``'s
    ``prompt_progress_callback`` and sets ``prefill_step_size == stride`` so the
    callback fires at each stride boundary. Consuming one token forces the full
    prefill. Returns ``(full_cache, capture)``.
    """
    cache = make_prompt_cache(model)
    capture = PrefixAnchorCapture(cache, stride)
    gen = generate_step(
        mx.array(tokens),
        model,
        max_tokens=1,
        prompt_cache=cache,
        prompt_progress_callback=capture,
        prefill_step_size=stride,
    )
    next(gen)  # drive prefill + final callback
    mx.eval([c.state for c in cache])
    return cache, capture


# --------------------------------------------------------------------------- #
# Core case (shared by both modes)
# --------------------------------------------------------------------------- #
def run_case(model, req_a, req_b, shared_pref, stride):
    if shared_pref <= stride:
        raise RuntimeError(f"shared prefix {shared_pref} <= stride {stride}")

    # Reference: cold full prefill of req_b (before any reuse).
    cold_cache = make_prompt_cache(model)
    trimmable = can_trim_prompt_cache(cold_cache)
    classes = cache_classes(cold_cache)
    cold_tok, cold_s = timed_next(model, cold_cache, req_b)

    # Serve req_a once, capturing shorter-prefix anchors during its prefill.
    cache_a, capture = serve_and_capture(model, req_a, stride)
    anchor_lengths = [a.length for a in capture.anchors]

    # ---- UPSTREAM arm: real LRUPromptCache, NO anchors ----
    store_up = LRUPromptCache(max_size=64)
    store_up.insert_cache(
        MODEL_KEY, req_a, copy.deepcopy(cache_a), cache_type="assistant"
    )
    t0 = time.perf_counter()
    up_cache, up_rest = store_up.fetch_nearest_cache(MODEL_KEY, req_b)
    up_fetch_s = time.perf_counter() - t0
    if up_cache is None:
        up_cache, up_rest = make_prompt_cache(model), req_b
    up_reused = len(req_b) - len(up_rest)
    up_tok, up_prefill_s = timed_next(model, up_cache, up_rest)
    up_ttft = up_fetch_s + up_prefill_s

    # ---- ANCHOR arm: same store + insert_anchors ----
    store_an = LRUPromptCache(max_size=64)
    store_an.insert_cache(
        MODEL_KEY, req_a, copy.deepcopy(cache_a), cache_type="assistant"
    )
    store_an.insert_anchors(MODEL_KEY, req_a, capture.anchors, cache_type="assistant")
    t0 = time.perf_counter()
    an_cache, an_rest = store_an.fetch_nearest_cache(MODEL_KEY, req_b)
    an_fetch_s = time.perf_counter() - t0
    if an_cache is None:
        an_cache, an_rest = make_prompt_cache(model), req_b
    an_reused = len(req_b) - len(an_rest)
    an_tok, an_prefill_s = timed_next(model, an_cache, an_rest)
    an_ttft = an_fetch_s + an_prefill_s

    expected_anchor = (shared_pref // stride) * stride

    return {
        "capture_enabled": bool(capture.enabled),
        "trimmable": bool(trimmable),
        "cache_classes": classes,
        "prompt_tokens": len(req_b),
        "shared_prefix_tokens": shared_pref,
        "stride": stride,
        "anchor_lengths": anchor_lengths,
        "expected_anchor_len": expected_anchor,
        # reuse capability (the headline)
        "upstream_reused_tokens": up_reused,
        "anchor_reused_tokens": an_reused,
        # TTFT (ms) -- prefill only, decode excluded by construction
        "cold_ttft_ms": round(cold_s * 1e3, 3),
        "upstream_ttft_ms": round(up_ttft * 1e3, 3),
        "anchor_ttft_ms": round(an_ttft * 1e3, 3),
        "anchor_vs_upstream": round(up_ttft / an_ttft, 3) if an_ttft else None,
        "anchor_vs_cold": round(cold_s / an_ttft, 3) if an_ttft else None,
        # correctness gates (must all hold)
        "cold_next_token": cold_tok,
        "upstream_next_equiv": up_tok == cold_tok,
        "anchor_next_equiv": an_tok == cold_tok,
    }


# --------------------------------------------------------------------------- #
# Real-model mode (--model)
# --------------------------------------------------------------------------- #
_SHARED_SEED = (
    "You are a senior engineer reviewing a large local codebase. Preserve "
    "existing behavior, cite exact files and line numbers, and never propose a "
    "broad rewrite. The stable shared project context follows and is identical "
    "across every request in this session.\n\n"
)
_FILLER = (
    "Context note: request-level prompt reuse should avoid recomputing the "
    "stable system, chat-history, retrieved-document, and few-shot prefixes on "
    "every call. The correctness gate is cold-vs-warm next-token equivalence; "
    "prefill/TTFT is reported separately from decode throughput.\n"
)
_TAIL_A = "List the first two acceptance gates for prefix-cache reuse."
_TAIL_B = "Explain why this is a prefill win and not a decode-token/s win, in detail."


def _to_ids(tok, prompt):
    return list(tok.encode(prompt))


def _render(tok, shared, tail):
    msgs = [{"role": "system", "content": shared}, {"role": "user", "content": tail}]
    try:
        p = tok.apply_chat_template(msgs, add_generation_prompt=True)
        return [int(t) for t in p]
    except Exception:
        return _to_ids(tok, shared + "\n\nUser: " + tail + "\nAssistant:")


def _build_shared(tok, target):
    base = len(_render(tok, _SHARED_SEED, _TAIL_A))
    one = len(_render(tok, _SHARED_SEED + _FILLER, _TAIL_A))
    per = max(1, one - base)
    n = max(0, (target - base) // per)
    shared = _SHARED_SEED + _FILLER * n
    while len(_render(tok, shared, _TAIL_A)) < target:
        shared += _FILLER
    return shared


def run_real(model_path, target, stride):
    from mlx_lm import load

    model, tok = load(model_path)
    mx.eval(model.parameters())
    # warm up compile / first-call so timings are steady-state
    serve_and_capture(model, _to_ids(tok, "warmup prompt for compile"), stride)

    shared = _build_shared(tok, target)
    req_a = _render(tok, shared, _TAIL_A)
    req_b = _render(tok, shared, _TAIL_B)
    shared_pref = common_prefix_len(req_a, req_b)

    # warm the req_b prefill shape so the cold reference is steady-state
    serve_and_capture(model, req_b, stride)

    row = run_case(model, req_a, req_b, shared_pref, stride)
    row["mode"] = "real"
    row["model"] = model_path
    return [row]


# --------------------------------------------------------------------------- #
# Synthetic mode
# --------------------------------------------------------------------------- #
def run_synthetic(target, stride, vocab, dim, n_layers, tail, seed):
    model = build_tiny_model(vocab, dim, n_layers, seed)
    req_a, req_b, shared_pref = build_synthetic_tokens(vocab, target, tail, seed)
    row = run_case(model, req_a, req_b, shared_pref, stride)
    row["mode"] = "synthetic"
    row["model"] = f"TinyRecurrentModel(vocab={vocab},dim={dim},layers={n_layers})"
    return [row]


# --------------------------------------------------------------------------- #
# Gates + reporting
# --------------------------------------------------------------------------- #
def aggregate_failures(rows):
    """Every case must: gate both next-token equivs true, upstream reuse == 0,
    and anchor reuse == the nearest stride boundary <= shared prefix."""
    failures = []
    for row in rows:
        for gate in ("upstream_next_equiv", "anchor_next_equiv"):
            if row.get(gate) is not True:
                failures.append({"check": gate, "value": row.get(gate)})
        if row.get("upstream_reused_tokens") != 0:
            failures.append(
                {
                    "check": "upstream_reuses_zero",
                    "value": row["upstream_reused_tokens"],
                }
            )
        if row.get("anchor_reused_tokens") != row.get("expected_anchor_len"):
            failures.append(
                {
                    "check": "anchor_reuses_nearest_anchor",
                    "value": row.get("anchor_reused_tokens"),
                    "expected": row.get("expected_anchor_len"),
                }
            )
    return failures


def print_summary(rows, failures):
    print("\n=== prompt-cache anchor-stride benchmark ===")
    for row in rows:
        print(f"\nmode={row['mode']}  model={row['model']}")
        print(f"  cache_classes        : {row['cache_classes']}")
        print(f"  trimmable            : {row['trimmable']}")
        print(f"  capture_enabled      : {row['capture_enabled']}")
        print(
            f"  prompt/shared/stride : "
            f"{row['prompt_tokens']}/{row['shared_prefix_tokens']}/{row['stride']}"
        )
        print(f"  anchors captured     : {row['anchor_lengths']}")
        print(
            f"  reused tokens        : "
            f"upstream={row['upstream_reused_tokens']}  "
            f"anchor={row['anchor_reused_tokens']}  "
            f"(nearest-anchor={row['expected_anchor_len']})"
        )
        print(
            f"  TTFT ms              : "
            f"cold={row['cold_ttft_ms']}  upstream={row['upstream_ttft_ms']}  "
            f"anchor={row['anchor_ttft_ms']}"
        )
        print(
            f"  next-token equiv     : "
            f"upstream={row['upstream_next_equiv']}  anchor={row['anchor_next_equiv']}  "
            f"(cold_tok={row['cold_next_token']})"
        )
    verdict = "PASS" if not failures else "FAIL"
    print(f"\nverdict: {verdict}")
    if failures:
        print(f"failures: {json.dumps(failures)}")
    else:
        print(
            "  upstream reuses 0 across the divergent tail (non-trimmable cache);\n"
            "  anchor mode reuses the nearest-anchor prefix;\n"
            "  the anchor-reused continuation is next-token-identical to cold."
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stride", type=int, default=128, help="anchor stride (tokens)")
    ap.add_argument(
        "--target", type=int, default=600, help="shared-prefix length (tokens)"
    )
    ap.add_argument("--model", default=None, help="real model path (optional)")
    ap.add_argument("--output", default=None, help="JSONL output path (append)")
    # synthetic-model knobs (kept small so it runs instantly on CPU)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--tail", type=int, default=64, help="divergent tail length")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.stride <= 0:
        print(json.dumps({"error": f"--stride must be positive, got {args.stride}"}))
        raise SystemExit(2)
    if args.target <= args.stride:
        print(
            json.dumps(
                {
                    "error": f"--target ({args.target}) must exceed --stride ({args.stride})"
                }
            )
        )
        raise SystemExit(2)

    if args.model:
        rows = run_real(args.model, args.target, args.stride)
    else:
        rows = run_synthetic(
            args.target,
            args.stride,
            args.vocab,
            args.dim,
            args.layers,
            args.tail,
            args.seed,
        )

    for row in rows:
        print(json.dumps(row, sort_keys=True), flush=True)
        if args.output:
            with open(args.output, "a") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")

    failures = aggregate_failures(rows)
    print_summary(rows, failures)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
