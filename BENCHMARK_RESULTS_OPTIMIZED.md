# Benchmark Results with Metal Optimization

## Test Date: April 15, 2026

### Mamba-370m (Mamba-1)
- Model: mlx-community/mamba-370m-hf-f16
- Settings: 512 prompt tokens, 128 generation tokens
- **Results:**
  - prompt_tps: **4814.10** (5  trials, avg)
  - generation_tps: **169.13**
  - peak_memory: **1.688 GB**

### Mamba2-2.7b (Mamba-2)
- Model: mlx-community/mamba2-2.7b
- Settings: 1024 prompt tokens, 128 generation tokens
- **Results:**
  - prompt_tps: **159.62** (5 trials, avg)
  - generation_tps: **49.44**
  - peak_memory: **7.49 GB**

### Codestral-7B (Mamba-2)
- Model: mlx-community/Mamba-Codestral-7B-v0.1
- Settings: 1024 prompt tokens, 128 generation tokens  
- **Results:**
  - prompt_tps: **85.21** (5 trials, avg)
  - generation_tps: **21.41**
  - peak_memory: **18.31 GB**

## Observations

1. **Mamba-370m performs well** - 4814 prompt_tps is a solid throughput
2. **Mamba2 models show lower throughput** - 159.62 for 2.7b and 85.21 for Codestral
   - Generation throughput is similar to baseline (49.44 vs 50.252 for 2.7b)
   - **Prompt throughput unexpectedly low** compared to baseline metrics
3. **Metal kernel diagnostic passes** - kernel compiles and executes successfully

## Next Steps

Need to investigate if:
- The baseline metrics (1110.885 for mamba2-2.7b) were anomalies or correctly measured
- The adaptive chunk size is affecting performance negatively
- Metal kernel is actually being used during Mamba2 prefill
- There's a regression in the code
