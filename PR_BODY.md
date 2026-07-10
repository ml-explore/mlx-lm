# deepseek_v2: absorbed-MLA decode (cache the latent, like deepseek_v3)

## What

Make `deepseek_v2`'s attention cache the **MLA latent** (`kv_lora_rank` +
`qk_rope_head_dim`) instead of the decompressed per-head K/V, folding `kv_b_proj`
into `embed_q`/`unembed_out` so decode runs in latent space. This is the **same
absorbed-MLA pattern `deepseek_v3` and `glm4_moe_lite` already use upstream**;
`deepseek_v2` is the flagship MLA family still materializing K/V (`minicpm3` and
`youtu_llm` also materialize — the same recipe applies there if this lands).

Uses the existing `mla.MultiLinear` helper (already imported by `deepseek_v3`).
Prefill stays materialized (a fully-fused absorbed prefill is slower); only the
`L == 1` decode path runs absorbed — the same approach as
`deepseek_v3.Attention`, with one deliberate difference: the forward
transparently handles a `QuantizedKVCache` latent, **preserving `deepseek_v2`'s
existing `--kv-bits` support** (which works today on the materialized path, and
which the absorbed `deepseek_v3` currently lacks — it errors with a quantized
cache). That is why this file keeps two attention call sites instead of v3's
one; the quantized branches are inert on a normal cache.

## Why it matters / when you benefit

Decode is memory-bandwidth-bound; the cached latent is ~8.9× smaller per token
than the materialized per-head K/V on V2-Lite (16 heads: 5120 vs 576 elements
per token per layer) — and ~71× on full DeepSeek-V2 (128 heads) — so the dominant
KV read shrinks. The win is a **long-context** one and grows with context length.

## Measured (DeepSeek-V2-Lite-Chat-4bit, isolated A/B — only `deepseek_v2.py` changed)

| context | materialized | absorbed | decode speedup | peak mem saved |
|---|---|---|---|---|
| 4k  | 112.1 tok/s / 12.4 GB | 111.0 / 11.0 | 0.99× | 1.4 GB |
| 8k  | 79.7 / 15.4 | 100.4 / 14.1 | 1.26× | 1.3 GB |
| 16k | 53.9 / 25.9 | 87.8 / 20.6 | **1.63×** | 5.3 GB |
| 32k | 31.5 / 59.6 | 75.2 / 49.8 | **2.39×** | 9.8 GB |

So: a tie at short context, **~2.4× decode and ~10 GB less peak memory at 32k**,
growing with length. Short-context users are unaffected (the split keeps prefill
materialized).

## Correctness — read this carefully

Absorbed MLA is *mathematically* equivalent to materialized, but **not
bitwise-identical on quantized weights**, and this PR does not claim otherwise.
On the 4-bit model, greedy output diverges from the materialized path after a few
tokens (both remain correct and coherent) because the fused `embed_q`/`unembed_out`
projections are **requantized** and the matmuls are reordered. Measured at the
first position: **logit correlation 0.9998**, top-5 token ids identical, only a
top-2 near-tie swaps. This is the **same numerical tradeoff `deepseek_v3` already
ships** (it is absorbed too); on bf16 weights the two are much closer. If exact
bit-parity with the current materialized path matters to you, this is the
expected, documented difference.

## Provenance

The MLA absorption identity is from the DeepSeek-V2 paper (arXiv:2405.04434 §2.1)
and the official DeepSeek inference code; the implementation mirrors the absorbed
`deepseek_v3` already in this repo. Faithful MLX port, not new research.

## Tests

- The existing offline `tests/test_models.py::test_deepseek_v2` (tiny synthetic
  config, no weights) now exercises the absorbed `embed_q`/`unembed_out` path at
  both prefill (`L>1`) and decode (`L==1`) and passes — no new test needed.
- `bench_absorbed_mla.py` reproduces the real-weight A/B (decode tok/s + peak GB
  at {4,8,16,32}k, and the logit correlation) by running it on this branch vs a
  pristine upstream checkout.

One migration note: prompt-cache snapshots saved with the old materialized layout
are shape-incompatible with the latent cache (the same one-time cutover
`deepseek_v3` went through).
