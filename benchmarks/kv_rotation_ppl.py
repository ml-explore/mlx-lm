"""Perplexity: plain-affine vs Hadamard-rotated quantized KV cache.

Demonstrates the failure mode that `--kv-bits` (PR #1476) exposes to server
users — plain affine low-bit KV degrades sharply — and the fix: a self-inverse
orthonormal Hadamard rotation of K (in the cache) + Q (at attention), which
keeps every q·k inner product identical while making the stored keys quantize
cleanly. The effect grows with head_dim (more quant groups) and at lower bits.

Run on Apple silicon:

    python benchmarks/kv_rotation_ppl.py \
        --model mlx-community/Qwen3-1.7B-bf16 --bits 8 4 3 2
"""

import argparse
import math

import mlx.core as mx

from mlx_lm import load
from mlx_lm.models.cache import QuantizedKVCache, make_prompt_cache

# A few hundred tokens of varied, non-repetitive prose so the KV cache is
# genuinely exercised over a real context (repetition would flatter ppl).
TEXT = (
    "The gongfu tea ceremony is a choreographed ritual of preparing and serving "
    "tea, valued for attention, restraint, and the quiet reading of small signs. "
    "A small clay pot is warmed, rinsed, and filled with tightly rolled oolong. "
    "Hot water is poured in a high thin stream to agitate the leaves, the lid is "
    "set back, and the first brief steeping is discarded to awaken the leaf. "
    "Later infusions grow longer, and the host judges each by colour and aroma "
    "before decanting into a pitcher so every cup is of equal strength. When a "
    "guest wants more hot water, the lid is tipped ajar and rested on the rim — "
    "a silent signal read across a crowded room without a word.\n\n"
    "Cities encode similar signals in their streets. A raised hand at the kerb, "
    "a folded newspaper on a cafe table, a lantern left burning past midnight: "
    "each is a small protocol, learned rather than written, that lets strangers "
    "coordinate without speaking. Markets run on such conventions too. A trader's "
    "open palm, a nod across a pit, a chalk mark on a crate — these once moved "
    "fortunes faster than any ledger, and the ledger only caught up afterward.\n\n"
    "Machines are late to this game. A camera can be taught to notice the tipped "
    "lid, the raised hand, the lantern — but only if it attends to the right "
    "region at the right moment and ignores the thousand irrelevant motions of a "
    "busy room. Attention, in that sense, is the whole problem: deciding which "
    "of the many things in view actually carry a message, and which are noise to "
    "be discarded before they crowd out the signal that matters. The art is old; "
    "the apparatus is new; the difficulty is exactly the same as it ever was."
)


def perplexity(model, ids, cache):
    logits = model(ids[None], cache=cache)[0, :-1].astype(mx.float32)
    targets = ids[1:]
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    nll = -mx.take_along_axis(logp, targets[:, None], axis=-1)[:, 0]
    return math.exp(mx.mean(nll).item())


def quant_caches(model, bits, group_size, rotate):
    n = len(make_prompt_cache(model))
    return [
        QuantizedKVCache(group_size=group_size, bits=bits, rotate=rotate)
        for _ in range(n)
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-1.7B-bf16")
    ap.add_argument("--bits", type=int, nargs="+", default=[8, 4, 3, 2])
    ap.add_argument("--group-size", type=int, default=64)
    args = ap.parse_args()

    model, tok = load(args.model)
    ids = mx.array(tok.encode(TEXT))
    fp16 = perplexity(model, ids, make_prompt_cache(model))
    print(f"model={args.model}  tokens={ids.size}  group_size={args.group_size}")
    print(f"  fp16 KV                       ppl = {fp16:10.2f}")
    print(f"  {'bits':>4}  {'affine':>12}  {'rotated':>12}  {'gap recovered':>14}")
    for b in args.bits:
        affine = perplexity(model, ids, quant_caches(model, b, args.group_size, False))
        rotated = perplexity(model, ids, quant_caches(model, b, args.group_size, True))
        gap = max(affine - fp16, 1e-9)
        pct = (affine - rotated) / gap * 100
        print(f"  {b:>4}  {affine:>12.2f}  {rotated:>12.2f}  {pct:>13.0f}%")


if __name__ == "__main__":
    main()
