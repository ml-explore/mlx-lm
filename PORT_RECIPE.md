# PR A — deepseek_v2 absorbed-MLA decode (port recipe)

**Status: staged, GPU-verification pending** (the 48/48 bitwise gate needs
DeepSeek-V2-Lite loaded; a GPU session is currently in use).

## The finding that defines this PR

- Upstream **`deepseek_v3`** (deepseek_v3.py:145) and **`glm4_moe_lite`**
  (glm4_moe_lite.py:151) **already cache the latent** — `cache.update_and_fetch(
  kv_latent, k_pe)` — i.e. they are already absorbed-MLA.
- Upstream **`deepseek_v2`** (deepseek_v2.py:219–237) is the **only** MLA model
  left that decompresses via `kv_b_proj` and caches full per-head K/V.

So this PR is a **consistency fix**: bring `deepseek_v2` in line with the
absorbed pattern upstream already ships in `deepseek_v3`. The technique is already
accepted upstream (it's in v3), so review friction is low; the payoff is the
lossless long-context win v2 currently leaves on the table.

## The port (mirror deepseek_v3's `Attention.__call__`)

Template = `deepseek_v3.py:118–170` (captured). Apply the same structure to
`deepseek_v2.py::Attention`:

1. **Projections:** add the absorbed projections v3 uses — `embed_q` (folds
   `W_UK` into the query path) and `unembed_out` (folds `W_UV` into output).
   v2 keeps `kv_b_proj` weights on disk; absorb them at load into these (see how
   v3 builds `embed_q`/`unembed_out`; v2 has no `q_lora_rank`, so the q path is
   just `q_proj`).
2. **Cache the latent:** replace the decompress-then-cache block with
   `kv_latent = self.kv_a_layernorm(compressed_kv)` → `cache.update_and_fetch(
   kv_latent, k_pe)` (cache 512-dim latent + rope key, not per-head K/V).
3. **Decode fast path (`L == 1`):** `q_nope = self.embed_q(q_nope)`,
   `k = v = kv_latent`, run SDPA in latent space, then `output =
   self.unembed_out(output)`. This is the absorbed decode path — the win.
4. **Prefill (`L > 1`):** materialize `k = embed_q(kv_latent, transpose=False)`,
   `v = unembed_out(kv_latent)` exactly as v3 — keeps prefill cheap (do NOT run
   the fully-fused absorbed prefill; it is 3.3× slower, per our measurements).
5. **`sanitize`:** keep the `kv_b_proj` → `embed_q`/`unembed_out` absorption in
   the model's weight loading, mirroring v3.

Net: v2's `Attention` becomes v3's `Attention` modulo the q-lora difference.
Do NOT add quantized-latent KV here (memory-only, measured slower — separate).

## Verification (GPU, pending)

`bench_absorbed_mla_v2.py` (to write): load
`mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx`, greedy(48) on the absorbed port
vs a materialized reference, assert **48/48 bitwise-identical**, and record the
decode-tok/s + KV-GB at {8k,32k,64k} for the PR table (expected ~3–7× decode,
~2.5× KV vs cold, per experiments/deepseek-v2-lite-results.md).

## Test (offline, to add)

`tests/test_models.py`: a tiny synthetic deepseek_v2 config forward-shape check
(absorbed path produces correct shapes at L==1 and L>1) — CPU, no weights.
