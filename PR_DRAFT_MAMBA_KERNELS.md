## Summary
Integrate optimized Mamba/SSM kernels for Apple Silicon in `mlx-lm` with correctness-preserving fallbacks.

## What Changed
- Added optimized Mamba-1 prefill path using selective-scan kernel.
- Added optimized Mamba-1 decode path using fused selective state-update kernel.
- Added optimized Mamba-2 prefill path using SSD chunk kernels.
- Kept fallback paths for unsupported shapes/devices/runtime conditions.
- Added model-local compatibility handling for `mamba2-2.7b` config parsing in `mlx_lm/models/mamba2.py` (derive `intermediate_size` from `hidden_size * expand`).

## Why
These kernels reduce prefill/decode overhead on Metal and improve throughput while preserving generation behavior.

## Scope
- [x] Decode path (`seq_len == 1`) optimization for Mamba-1
- [x] Prefill path (`seq_len > 1`) optimization for Mamba-1 and Mamba-2
- [x] No public API changes

## Correctness
Reference: existing `mlx-lm` implementation.

Validation included:
- [x] End-to-end generation functional checks across Mamba family models
- [x] Cache/state update path exercised in decode and prefill
- [x] Fallback path retained when kernel conditions are not met

Tolerance used:
- `rtol`: 1e-3
- `atol`: 1e-3

## Benchmarks
Environment:
- Device: Apple M1 Max (64 GB)
- Python: 3.12.13
- mlx: 0.31.1
- mlx-metal: 0.31.1
- Baseline: clean `main` (`d9c63ff`)
- Optimized: `feat/mamba-mlx-kernels`

Workload settings:
- Prompt tokens: 1024
- Generation tokens: 128
- Batch size: 1
- Trials: 3 for `mamba-370m`; 1 for larger models

Results (before -> after):

- `mlx-community/mamba-370m-hf-f16` (Mamba-1)
	- prompt_tps: `291.996 -> 5630.994` (`19.28x`)
	- generation_tps: `161.584 -> 167.765` (`1.04x`)
	- peak_memory (GB): `1.027 -> 2.172`

- `mlx-community/mamba2-2.7b` (Mamba-2)
	- baseline (`main`) status: model load fails (`ModelArgs` missing `intermediate_size`)
	- after model-local compatibility fix:
		- prompt_tps: `1110.885`
		- generation_tps: `50.252`
		- peak_memory (GB): `7.776`

- `mlx-community/Mamba-Codestral-7B-v0.1` (Mamba-2)
	- prompt_tps: `390.578 -> 468.743` (`1.20x`)
	- generation_tps: `21.932 -> 21.942` (`1.00x`)
	- peak_memory (GB): `22.935 -> 18.797`

- `mlx-community/Falcon3-Mamba-7B-Instruct` (Mamba-1)
	- prompt_tps: `69.419 -> 520.845` (`7.50x`)
	- generation_tps: `20.868 -> 21.370` (`1.02x`)
	- peak_memory (GB): `15.952 -> 16.623`

- `mlx-community/mamba-1.4b-hf-f16` (Mamba-1)
	- prompt_tps: `295.790 -> 2174.908` (`7.35x`)
	- generation_tps: `80.601 -> 84.746` (`1.05x`)
	- peak_memory (GB): `3.312 -> 4.281`

## Fallback Behavior
When assumptions are not met, execution falls back to the reference implementation.

Fallback conditions include:
- Non-GPU/default non-Metal execution.
- Unsupported dtypes/shapes or state-size constraints.
- Missing cache/state for decode kernel path.

## Testing
Commands run:
- `PYTHONPATH=. python -m mlx_lm.benchmark --model mlx-community/mamba-370m-hf-f16 --prompt-tokens 1024 --generation-tokens 128 --num-trials 3`
- `PYTHONPATH=. python -m mlx_lm.benchmark --model mlx-community/mamba2-2.7b --prompt-tokens 1024 --generation-tokens 128 --num-trials 1`
- `PYTHONPATH=. python -m mlx_lm.benchmark --model mlx-community/Mamba-Codestral-7B-v0.1 --prompt-tokens 1024 --generation-tokens 128 --num-trials 1`
- `PYTHONPATH=. python -m mlx_lm.benchmark --model mlx-community/Falcon3-Mamba-7B-Instruct --prompt-tokens 1024 --generation-tokens 128 --num-trials 1`
- `PYTHONPATH=. python -m mlx_lm.benchmark --model mlx-community/mamba-1.4b-hf-f16 --prompt-tokens 1024 --generation-tokens 128 --num-trials 1`

## Notes for Reviewers
- Start with `mlx_lm/models/mamba.py` for Mamba-1 prefill/decode split.
- Then inspect `mlx_lm/models/ssm.py` and `mlx_lm/models/ssd_chunk.py` for Mamba-2 prefill dispatch.
- Kernel modules:
	- `mlx_lm/models/ssm_selective_scan.py`
	- `mlx_lm/models/mamba_selective_state_update.py`
	- `mlx_lm/models/ssd_chunk_state.py` ← **NEW: Metal kernel for 12.99x speedup**
	- `mlx_lm/models/ssd_state_passing.py`
	- `mlx_lm/models/ssd_chunk_scan.py`

## Latest Improvements (April 15, 2026)
- **ssd_chunk_state Metal kernel**: Replaced slow einsum-only reference with GPU Metal kernel
  - **Speedup**: 12.99x faster (23ms → 1.8ms for small test)
  - **Accuracy**: Excellent (relative error: 3.61e-07 on float32)
  - **Impact**: Addresses Mamba2 prefill bottleneck; expected to significantly improve Mamba2 throughput
  - Fallback to reference einsum implementation if Metal kernel unavailable
  
- **Adaptive chunk_size**: Dynamic chunking (128-512) based on sequence length
  - Reduces padding overhead for short sequences
  - Improves cache utilization for medium sequences
  - Reduces kernel dispatch overhead for long sequences
  
These optimizations target the root bottleneck identified during analysis: Mamba2 prefill chain uses 3 separate kernels (chunk_state→state_passing→chunk_scan), causing only 1.20x speedup vs Mamba1's 7-19x. The Metal kernel for chunk_state removes the slowest operation in this chain.

