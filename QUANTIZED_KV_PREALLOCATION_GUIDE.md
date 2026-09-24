# Quantized KV Cache Preallocation & Long-Context Serving in `mlx-lm`

This guide documents the design, implementation, debugging journey, architecture, and production usage of **Quantized KV Cache Preallocation** for `mlx-lm`. This enables serving 27B+ parameter models at **64k context** with rock-solid stability on memory-constrained hardware (e.g., 24GB Unified Memory Apple Silicon) without kernel panics or Out-Of-Memory (OOM) crashes.

---

## 1. Quick Start / How to Run

### Environment Setup
Make sure you are in the repository virtual environment:
```bash
cd /Users/pranavshinde/Developer/mlx-lm
source .venv/bin/activate
```

### Launch the Server
To serve `Qwen3.8-27B-4bit` at 64k context on a 24GB Mac:

```bash
mlx_lm.server \
  --model mlx-community/Qwen3.8-27B-4bit \
  --port 8080 \
  --kv-bits 4 \
  --kv-group-size 64 \
  --kv-preallocate-size 65536 \
  --prefill-step-size 256
```

### Explanation of Key Flags

| Flag | Value | Purpose |
| :--- | :--- | :--- |
| `--model` | `mlx-community/Qwen3.8-27B-4bit` | The quantized 4-bit model weights (~15.1 GB active RAM). |
| `--port` | `8080` | Local port for the OpenAI-compatible HTTP API. |
| `--kv-bits` | `4` | Quantizes full-attention Key/Value cache elements to 4-bit precision (~71 MB per full-attention layer at 64k tokens). |
| `--kv-group-size` | `64` | Quantization group size for scale and bias calculation. |
| `--kv-preallocate-size` | `65536` | Preallocates the full 64k token buffer up front. Prevents `step=256` reallocation churn and wired-memory spikes that previously caused macOS kernel panics. |
| `--prefill-step-size` | `256` | Chunks prompt evaluation into 256-token slices during prefill. Crucial for keeping intermediate MLX layer activation memory bounded on 24GB RAM. |

---

## 2. OpenCode / Web UI Configuration

To use this model in [OpenCode](https://opencode.ai) or any OpenAI-compatible client, add this provider block to your `config.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "mlx/mlx-community/Qwen3.8-27B-4bit",
  "provider": {
    "mlx": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "MLX-LM Server (Port 8080)",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1"
      },
      "models": {
        "mlx-community/Qwen3.8-27B-4bit": {
          "name": "Qwen3.8-27B-4bit-MLX",
          "modalities": {
            "input": ["text"],
            "output": ["text"]
          },
          "limit": {
            "context": 65536,
            "output": 32768
          }
        }
      }
    }
  }
}
```

---

## 3. Background: The Problems Encountered & Solved

### A. The Kernel Panic Trigger (Unpatched `step=256` Dynamic Growth)
In standard `mlx-lm`, `KVCache` and `QuantizedKVCache` grow dynamically in chunks of `step=256` tokens. As context expands (e.g., 10k &rarr; 30k &rarr; 60k tokens), every step executes:
1. `mx.zeros` allocation for the expanded shape.
2. `mx.concatenate` copying previous tokens to the new buffer.
3. Rapid Apple Silicon unified memory page churn.

Under heavy context length, rapid buffer allocations and deallocations under memory pressure triggered **hard macOS kernel panics / system reboots** (`iogpu` driver crash).

### B. The Activation Memory Spike
When processing prompts with `--prefill-step-size 2048`, MLX's deferred evaluation graph for a 64-layer model with large `intermediate_size` (`17,408`) accumulated several gigabytes of intermediate activation memory during the forward pass. Combined with model weights (~15.1 GB) on 24GB RAM, the Metal command buffer exceeded the working set and raised an OOM exception at token 0.

### C. The Multi-Turn "Double KV Cache" OOM Bug
When serving consecutive requests:
1. In unpatched `server.py`, `LRUPromptCache` stored the 1.36 GB cache from turn $N-1$ in `_trie`.
2. When turn $N$ arrived:
   - On a cache miss: `make_prealloc_prompt_cache` allocated a *second* 1.36 GB cache while the old one was still in `_trie`.
   - On a cache hit: `fetch_nearest_cache` executed `copy.deepcopy(...)`, creating a duplicate working copy while leaving the original in `_trie`.
3. Holding **2 × 1.36 GB = 2.72 GB** of KV buffers simultaneously alongside 15.5 GB model weights and prefill activations pushed total usage past 24 GB, triggering `[METAL] Insufficient Memory` after 5–7 turns.

---

## 4. How the Solutions Work Under the Hood

### Architecture & Memory Math (24GB Unified RAM)

Qwen 3.5 / 3.8 27B uses a hybrid architecture across 64 layers:
* **48 Linear-Attention Layers**: Fixed-size recurrent state (`ArraysCache`), state size remains constant regardless of context length (~0.22 GB total).
* **16 Full-Attention Layers**: Standard self-attention layers requiring full history storage in `QuantizedKVCache` at 4-bit precision.

#### 1.36 GB Cache Instance Breakdown (65,536 Tokens)
* **16 Full-Attention Layers (`QuantizedKVCache` at 64k tokens, 4-bit, group size 64):**
  * `keys`: $16 \times (4 \times 65,536 \times 256 \times 0.5\text{ B}) = 536.87\text{ MB}$
  * `values`: $16 \times (4 \times 65,536 \times 256 \times 0.5\text{ B}) = 536.87\text{ MB}$
  * `scales + biases` (FP16): $16 \times 2 \times (4 \times 65,536 \times 4 \times 2\text{ B}) = 67.11\text{ MB}$
  * **Subtotal (Full Attention):** **`~1.14 GB`**
* **48 Linear-Attention Layers (`ArraysCache`):** **`~0.22 GB`**
* **Total per Single Cache Instance:** **`1.36 GB`**

#### Total System Memory Budget
```
Total Hardware Memory:                     24.00 GB
---------------------------------------------------
Model Weights (Qwen 27B 4-bit):           ~15.13 GB
Single 4-bit KV Cache (65,536 tok):       ~ 1.36 GB
Prefill Activations (step_size=256):       ~ 0.80 GB
macOS System & Display Buffer:             ~ 3.50 GB
---------------------------------------------------
Total Working Footprint:                   ~20.79 GB (Comfortably under 24GB)
```

---

## 5. Summary of Code Changes

### 1. [`mlx_lm/models/cache.py`](file:///Users/pranavshinde/Developer/mlx-lm/mlx_lm/models/cache.py)
* **Preallocated Buffers**: Added `max_size` parameter to `KVCache` and `QuantizedKVCache`. Allocates the full buffer upfront (`mx.zeros` + `mx.eval`) on initial step without dynamic `step=256` reallocations.
* **Hybrid Model Cache Factory**: Updated `make_prealloc_prompt_cache` to preserve `ArraysCache` from `model.make_cache()` for linear-attention layers while swapping only `KVCache` for preallocated `QuantizedKVCache`.
* **Ownership Transfer on Cache Hit (`pop=True`)**: Updated `fetch_nearest_cache(model, tokens, pop=True)` to pop and take direct ownership of the matched cache from `_trie` instead of creating a `copy.deepcopy()`.
* **Safe nbytes Calculation**: Updated `QuantizedKVCache.nbytes` to safely handle uninitialized `keys`/`values`.
* **Cache Purge**: Added `LRUPromptCache.clear()` (`trim_to(n_sequences=0)`).

### 2. [`mlx_lm/generate.py`](file:///Users/pranavshinde/Developer/mlx-lm/mlx_lm/generate.py)
* Imported `make_prealloc_prompt_cache`.
* Updated `BatchGenerator.__init__` and `insert_segments` to accept and pass `kv_bits`, `kv_group_size`, and `kv_preallocate_size`.

### 3. [`mlx_lm/server.py`](file:///Users/pranavshinde/Developer/mlx-lm/mlx_lm/server.py)
* **CLI Options**: Added `--kv-bits`, `--kv-group-size`, `--quantized-kv-start`, and `--kv-preallocate-size`.
* **Pre-Allocation Eviction on Cache Miss**: In `_serve_single`, if `cache is None`, explicitly calls `self.prompt_cache.clear()` and `mx.clear_cache()` *before* invoking `make_prealloc_prompt_cache()` so old and new caches never coexist.
* **Single-Instance Enforcement**: Sets `pop = self.cli_args.kv_preallocate_size is not None` when calling `fetch_nearest_cache`.
* **Post-Request Garbage Collection**: Added `finally: mx.clear_cache()` in `_serve_single` to reclaim Metal memory pool allocations between requests.
* **Routing**: Forces `_is_batchable` to `False` when `kv_bits` is set, routing single-client quantized requests safely through `_serve_single`.

---

## 6. Empirical Verification & Benchmarks

All tests executed on Apple Silicon (24GB Unified Memory) running `mlx-community/Qwen3.8-27B-4bit`:

### Single-Prompt Context Scaling

| Preallocation Size | Prefill Step Size | Prompt Tokens | Completion Tokens | Result | Speed (Tokens/sec) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `65,536` | `2048` | ~40,000 | — | **Metal OOM** (Token 0) | Activation spike |
| `32,768` | `2048` | ~20,000 | — | **Metal OOM** (Token 6,144) | Activation accumulation |
| `16,384` | `1024` | `10,062` | `30` | **PASSED** | ~400 tok/s |
| `16,384` | `1024` | `16,062` | `30` | **PASSED** | ~380 tok/s |
| **`65,536`** | **`256`** | **`40,062`** | **`30`** | **PASSED** | **~380 tok/s** |
| **`65,536`** | **`256`** | **`60,062`** | **`30`** | **PASSED** | **~340 tok/s** |

### Multi-Turn Consecutive Requests Stress Test (`--prompt-cache-size 1`)

Ran 10 sequential requests in the same server session:
* **Requests 1 through 10**: All returned `200 OK` with `Prompt Cache: 1 sequences, 1.36 GB` strictly flat memory.
* **Result**: **10/10 passed with zero memory leaks, zero OOMs, and zero kernel panics.**

### Decode Speed Comparison vs `llama-server` (GGUF)

| Metric | `llama-server` (UD-Q3_K_XL / Q8 KV) | `mlx_lm.server` (4-bit / 4-bit KV) | MLX Advantage |
| :--- | :--- | :--- | :--- |
| **Prefill Speed (40k–60k ctx)** | ~110 – 160 tok/s | **~340 – 380 tok/s** | **~2.4x – 3x faster prefill** |
| **Decode Speed (Generation)** | ~10.4 – 12.9 tok/s | **~14.0 – 16.8 tok/s** | **~30% – 60% faster decode** |
| **Max KV Context on 24GB** | Reaches ~28k before slowdown | Full **65,536 tokens** | Full 64k without OOM |
