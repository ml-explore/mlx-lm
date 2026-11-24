# Benchmarks

## Commands 

The command for evaluating on MMLU Pro:

```
mlx_lm.evaluate --model model/repo --task mmlu_pro
```
 
The command for efficiency benchmarks:

```
mlx_lm.benchmark --model model/repo -p 2048 -g 128
```

To get the package versions run:

```
python -m mlx --version && python -m mlx_lm --version
```

## Python API

The `run_benchmark()` function allows developers to run LLM
benchmarks directly from Python and access performance measurements
as structured results.

### Benchmark using a text prompt

```python
from mlx_lm.benchmark import run_benchmark

result = run_benchmark(
    model="mlx-community/SmolLM-135M-Instruct-4bit",
    prompt="Write a quicksort in Python",
    generation_tokens=8,
    num_trials=2,
)

print("Generation speed:", result.generation_tps)
print("Prompt processing speed:", result.prompt_tps)
print("Peak memory (GB):", result.peak_memory)
```

The text is converted into token IDs using the model tokenizer.
The function runs a warmup and then measures two inference trials, because (num_trials = 2).
The function performs one warmup, followed by num_trials measured inference runs. In this example, num_trials=2, so it performs two measured trials.

### Benchmark using synthetic input

```python
result = run_benchmark(
    model="mlx-community/SmolLM-135M-Instruct-4bit",
    prompt_tokens=100,
    generation_tokens=50,
    num_trials=3,
)
```

When no text prompt is provided, the function generates random
input token IDs of the requested length.

If both `prompt` and `prompt_tokens` are provided, the text
prompt takes priority and the actual tokenized length is used.

### Parameter sweeps

The same function can be called repeatedly with different settings.

```python
from mlx_lm.benchmark import run_benchmark

for length in [16, 32, 64]:
    result = run_benchmark(
        model="mlx-community/SmolLM-135M-Instruct-4bit",
        prompt_tokens=length,
        generation_tokens=8,
        num_trials=2,
    )

    print(length, result.generation_tps)
```

Developers can also vary compatible model paths, generation
lengths, batch sizes, and supported quantization options.

### Quantization options

Developers can benchmark compatible quantized models,
such as 4-bit model variants, to compare inference
speed and memory usage.

The `quantize_activations` option requests lower-precision
intermediate calculations where supported by the model.

This option does not enable or disable activations.
It is only supported for compatible quantization modes.

### Returned measurements

`run_benchmark()` returns a `BenchmarkResult` object containing:

- `prompt_tps`: Average prompt-processing tokens per second.
- `generation_tps`: Average generation tokens per second.
- `peak_memory`: Average reported peak memory in GB.
- `prompt_tokens`: Number of input tokens used.
- `generation_tokens`: Requested generation-token limit.
- `batch_size`: Number of input sequences processed together.
- `trials`: Individual `TrialMetrics` objects.
- `model`: Name or path of the model being benchmarked.

Each trial also contains `total_time` in seconds.

The function returns benchmarking measurements rather than
the generated response text.

For text input, `batch_size > 1` repeats the supplied prompt
across the batch.

### CLI compatibility

The Python benchmarking API preserves the existing command-line
interface. The CLI continues to use synthetic input tokens and
calls the same `run_benchmark()` function internally.

Existing commands remain supported:

```bash
python -m mlx_lm.benchmark \
  --model mlx-community/SmolLM-135M-Instruct-4bit \
  -p 16 -g 8 -n 1


## Models

<details>

 <summary> Qwen/Qwen3-4B-Instruct-2507 </summary>

Precision | MMLU Pro | Prompt (2048) tok/sec | Generation (128) tok/sec | Memory GB | Repo
--------- | -------- | ------------------- | ------------------------ | --------- | ----
bf16      | 64.05    | 1780.63             | 52.47                    | 9.02    | Qwen/Qwen3-4B-Instruct-2507
q8 | 63.85 | 1606.573| 86.907 | 5.254 | mlx-community/Qwen3-4B-Instruct-2507-8bit
q6 | 63.53 | 1576.73 | 104.68 | 4.25 | mlx-community/Qwen3-4B-Instruct-2507-6bit
q5 g32 | 63.16 | 1570.80 | 110.29 | 4.00 | mlx-community/Qwen3-4B-Instruct-2507-5bit-g32
q5 | 62.38 | 1584.33 | 116.39 | 3.86 | mlx-community/Qwen3-4B-Instruct-2507-5bit
q4 g32 | 61.46 | 1610.03 | 126.00 | 3.603 | mlx-community/Qwen3-4B-Instruct-2507-4bit-g32
q4 | 60.72 | 1622.27 | 134.52 | 3.35 | mlx-community/Qwen3-4B-Instruct-2507-4bit

- Performance benchmark on 64GB M4 Max
- mlx 0.29.2.dev20251008+85a8824a8
- mlx-lm 0.28.2
- macOS 26.1
 
</details>

<details>
<summary> Qwen/Qwen3-30B-A3B-Instruct-2507 </summary>

Precision | MMLU Pro | Prompt (2048) tok/sec | Generation (128) tok/sec | Memory GB | Repo
--------- | -------- | ------------------- | ------------------------ | --------- | ----
bf16 | 72.62 | :skull: | :skull: | :skull: | Qwen/Qwen3-30B-A3B-Instruct-2507
q8 | 72.46 | 1719.47 | 83.16 | 33.46 | mlx-community/Qwen3-30B-A3B-Instruct-2507-8bit 
q6 | 72.41 | 1667.45 | 94.14 | 25.82 | mlx-community/Qwen3-30B-A3B-Instruct-2507-6bit
q5 | 71.97 | 1664.24 | 101.00 |22.01 | mlx-community/Qwen3-30B-A3B-Instruct-2507-5bit
q4 | 70.71 | 1753.90 | 113.33 |18.20 | mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit

 
- Performance benchmarks on 64GB M4 Max
- mlx 0.29.2.dev20251008+85a8824a8
- mlx-lm 0.28.2
- macOS 26.1

</details>
