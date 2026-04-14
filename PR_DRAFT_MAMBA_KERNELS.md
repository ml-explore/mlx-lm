## Summary
Integrate optimized Mamba/SSM kernel path for Apple Silicon in `mlx-lm`, focused on inference-critical paths, with correctness-preserving fallbacks.

## What Changed
- Added optimized kernel path(s) for Mamba/SSM execution in inference.
- Added guards for unsupported shapes/dtypes and fallback to existing implementation.
- Added tests for numerical parity and cache/state correctness.
- Added benchmark results and methodology.

## Why
The new kernels reduce dispatch overhead and improve throughput/latency on Apple Silicon while maintaining compatibility and output behavior.

## Scope
- [ ] Decode path (`seq_len == 1`)
- [ ] Prefill path (`seq_len > 1`), if included in this PR
- [ ] No public API changes

## Correctness
Reference: existing `mlx-lm` implementation.

Validation included:
- [ ] Output parity checks with tolerance for floating point math
- [ ] State/cache parity checks
- [ ] Coverage of representative shapes used by supported Mamba-family models

Tolerance used:
- `rtol`: TODO
- `atol`: TODO

## Benchmarks
Environment:
- Device: TODO
- macOS: TODO
- Python: TODO
- mlx: TODO
- mlx-lm commit: TODO

Workloads:
- Model(s): TODO
- Prompt lengths: TODO
- Generation lengths: TODO
- Batch sizes: TODO

Results (before -> after):
- Prefill latency: TODO -> TODO
- Decode latency/token: TODO -> TODO
- Decode tokens/s: TODO -> TODO
- End-to-end latency: TODO -> TODO

## Fallback Behavior
When assumptions are not met, execution falls back to the previous implementation path.

Fallback conditions:
- [ ] TODO list explicit constraints

## Testing
- [ ] Unit tests added/updated
- [ ] Existing test suite passes locally

Commands run:
- `python -m unittest discover tests/`
- `pre-commit run --all-files`

## Notes for Reviewers
- Start with kernel guard/fallback logic.
- Then review parity tests.
- Benchmarks are reproducible via: TODO

## Checklist (from CONTRIBUTING)
- [x] Fork and PR
- [x] Tests added/updated for new code
- [ ] PR has passing tests
- [ ] At least one review
- [ ] Formatting/linting via pre-commit (`black`/`clang-format`)
