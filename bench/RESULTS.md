ABOUTME: Captures benchmark attempts for continuous batching PR-1 acceptance.
ABOUTME: Documents tokens/sec comparison and outstanding blockers.

| Model | Concurrency | Prompt Length | Max New Tokens | Static TPS | Continuous TPS | Δ | Notes |
| ----- | ----------- | ------------- | -------------- | ---------- | -------------- | --- | ----- |
| mlx-community/Llama-3.2-3B-Instruct-4bit | 8 | 64 | 64 | N/A | N/A | — | Run blocked: `mlx_lm` import fails because compiled MLX package unavailable in sandbox. |
| mlx-community/Llama-3.2-3B-Instruct-4bit | 4 | 1024 | 64 | N/A | N/A | — | Same as above. |

Attempts captured in `bench/bench_run_conc8.txt` and `bench/bench_run_conc4.txt`. Both runs abort before download due to missing local MLX bindings (network access restricted, `mlx` wheels not installed). Continuous batching runtime otherwise validated via unit tests and synthetic logging (`bench/TRACE.log`).
