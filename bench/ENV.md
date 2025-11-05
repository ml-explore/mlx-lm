ABOUTME: Captures hardware and software environment for continuous batching bench.
ABOUTME: Ensures reviewers can reproduce PR-1 benchmark numbers.

- Date: 2025-11-05
- `mlx-lm` commit: `663b822de51d29c5321ddb3dcc8ad5116ebc37eb`
- `mlx` commit: `761f901a411f4ff4df15d0077665c88f633ed029`
- Branch: `feat/continuous-batching-pr1`
- Python (venv): `Python 3.9.6`
- macOS: `15.6.1 (24G90)`
- Hardware: Apple M4 Max (16 cores: 12 performance / 4 efficiency), 128 GB unified memory
- Key env vars impacting bench: `PATH=/usr/local/bin:…:/Users/sohailmo/.lmstudio/bin`
- Notes: Benchmarks executed under `.venv` with MLX Metal backend enabled; no network access in sandbox.
