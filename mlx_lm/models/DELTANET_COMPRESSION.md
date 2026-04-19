# GatedDeltaNet training compression (theorem-guided)

Compression-aware training utilities for GatedDeltaNet-style linear
attention (Qwen3.5, Qwen3-Next, Kimi-Linear).

## Quick start

```python
from mlx_lm import load
from mlx_lm.compress import estimate_rank, estimate_rank_per_layer
import json, os, subprocess

model, tokenizer = load("mlx-community/Qwen3.5-9B-MLX-4bit")
model.eval()

# Option A — uniform rank across all layers.
r = estimate_rank(model, tokenizer, safety_buffer=2, probe_state=True)
os.environ["MLX_DELTANET_COMPRESS_RANK"] = str(r)

# Option B — per-layer ranks (optimal, ~56% memory savings vs uniform).
per_layer = estimate_rank_per_layer(model, tokenizer, safety_buffer=2)
with open("ranks.json", "w") as f:
    json.dump({str(k): v for k, v in per_layer.items()}, f)
os.environ["MLX_DELTANET_COMPRESS_RANK_PER_LAYER"] = "ranks.json"

# Release probe model before training starts.
del model, tokenizer

# Launch normal LoRA trainer — the env var activates compression.
subprocess.run(["mlx_lm.lora", "-c", "your_config.yaml"])
```

## Why this works

Empirical finding: the recurrent state ``S_t ∈ ℝ^{D_v × D_k}`` of a
trained GatedDeltaNet has stable rank O(1) — on Qwen3.5-9B, at most
~2 out of up to 128 possible dimensions are used.

Replicated on:
- Qwen3.5 at 4B, 9B, 27B, 35B-A3B scales (GatedDeltaNet)
- Mamba-2-370M (different diagonal-SSM recurrence)
- RWKV-7-1.5B (different WKV recurrence)

Formal theorem: under bounded decay ``g_t ≤ g < 1``, unit keys,
bounded values, and a smooth recent-window key stream
(``r_k := stable_rank([k_{t-W+1}, …, k_t]) ≤ r*``, empirically
verified over 576 measurements on Qwen3.5-9B — max ``r* = 8.34``):

    stable_rank(S_T) ≤ r* · 1/(1 - g²)  ≈ 92 for Qwen3.5-9B

independent of sequence length. A compression rank of
``ceil(r*) + safety_buffer`` (typically 8-16) therefore preserves
essentially all state information while cutting the boundary
activation memory used in backward passes by ~5×.

## Environment variables

All optional — only ``MLX_DELTANET_COMPRESS_RANK`` or
``MLX_DELTANET_COMPRESS_RANK_PER_LAYER`` is required to activate
compression.

- ``MLX_DELTANET_VJP`` — backend: "metal" (default), "python",
  "lowrank", "compress"
- ``MLX_DELTANET_COMPRESS_RANK`` — int, uniform rank (enables
  compression if > 0)
- ``MLX_DELTANET_COMPRESS_RANK_PER_LAYER`` — path to JSON
  ``{"layer_idx": rank}``; overrides uniform
- ``MLX_DELTANET_COMPRESS_ITERS`` — power-iteration steps
  (default 6; rarely needs change)

## Performance

Benchmark on Qwen3.5-9B DeltaNet shape (Hk=16, Hv=64, Dk=192, Dv=128,
bf16, 3-repeat median):

| T     | Metal VJP (ms) | Python VJP (ms) | speedup | Metal mem (GB) | Python mem (GB) |
|-------|----------------|-----------------|---------|----------------|-----------------|
|  256  |  13.7          | 146.4           | 10.7×   | 1.77           | 2.78            |
|  512  |  28.1          | 290.2           | 10.3×   | 2.99           | 4.57            |
| 1024  |  62.4          | 587.8           |  9.4×   | 4.69           | 8.47            |
| 2048  | 149.2          | 1221.5          |  8.2×   | 8.10           | 15.41           |

Metal backend fuses forward-with-save + backward in a single chunked
dispatch (CHUNK_SIZE=64). Additional token-level fusion (single MSL
source combining both passes) is a follow-up of ~1.5× further
speedup; the chunked implementation is already the practical win.

## Reference

Full derivation of the O(1) stable rank theorem and the per-layer
rank choice will appear in a companion arXiv preprint (in preparation).
