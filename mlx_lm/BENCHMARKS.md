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
