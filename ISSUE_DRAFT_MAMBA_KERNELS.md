## Title
Optimize Mamba/SSM decode and prefill kernels on Apple Silicon

## Summary
I have an existing MLX Mamba kernel implementation with strong performance results and want to upstream the relevant parts into `mlx-lm`.

This issue proposes integrating kernel-level optimizations for SSM/Mamba paths used in inference, while preserving current behavior and adding safe fallbacks.

Type: enhancement/performance proposal (not a bug report).

## Motivation
Current Mamba/SSM paths are correct and portable, but there is still meaningful room for Apple Silicon-specific speedups in decode and/or prefill.

## Proposed Scope
- Optimize SSM update path for decode (`seq_len == 1`) with a faster Metal kernel path.
- Evaluate prefill improvements for chunked scan/selective scan where applicable.
- Keep existing numerics and output contract intact.
- Keep current path as fallback when constraints are not met.

## Non-Goals
- No model architecture changes.
- No API changes for users.
- No mandatory dependency changes.

## Correctness Plan
- Compare kernel outputs against current implementation with tolerance checks.
- Validate across relevant dtypes used in MLX-LM inference.
- Validate cache/state updates match existing behavior.

## Benchmark Plan
I will report:
- Prefill tokens/s and latency
- Decode tokens/s and latency
- End-to-end generation latency on representative prompts

## Reproducible Baseline
These commands reproduce current baseline behavior in `main` and establish a comparison point for the PR:

```bash
PYTHONPATH=. python -m mlx_lm.benchmark --model mlx-community/mamba-370m-hf-f16 --prompt-tokens 1024 --generation-tokens 128
PYTHONPATH=. python -m mlx_lm.benchmark --model mlx-community/Mamba-Codestral-7B-v0.1 --prompt-tokens 1024 --generation-tokens 128
PYTHONPATH=. python -m mlx_lm.benchmark --model mlx-community/Falcon3-Mamba-7B-Instruct --prompt-tokens 1024 --generation-tokens 128
```

For `mlx-community/mamba2-2.7b`, baseline currently fails to load due to missing `intermediate_size`; this will be documented explicitly in the PR with compatibility handling and before/after numbers where applicable.

Hardware/software matrix (to be filled):
- Device: M-series
- macOS:
- mlx-lm commit:
- mlx version:

## Risks and Mitigations
- Risk: shape/dtype-specific kernel assumptions.
  - Mitigation: explicit guards + fallback to existing path.
- Risk: numerical drift.
  - Mitigation: parity tests + tolerances documented in PR.

## PR
I will open a PR linked to this issue with:
- Code changes
- Tests
- Benchmarks
- Notes on fallback behavior
- Explicit `pre-commit` output summary and commands used
