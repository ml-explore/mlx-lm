# feat: Add batching support for ArraysCache/MambaCache with prompt caches

## Summary

This PR enables `batch_generate` to work with prompt caches for hybrid models that use `MambaCache` (like Qwen3-Next, Falcon-H1, and other models with linear attention or SSM layers).

Previously, attempting to use `prompt_caches` with `batch_generate` on these models would fail with:
```
ValueError: <class 'mlx_lm.models.cache.MambaCache'> does not yet support batching with history
```

## Changes

### `mlx_lm/models/cache.py`

Added methods to `ArraysCache` (parent class of `MambaCache`):

- **`merge(cls, caches)`** - Class method to merge multiple ArraysCache instances into a single batched cache by concatenating arrays along the batch dimension
- **`extract(self, idx)`** - Extract a single cache entry from a batched cache
- **`prepare(self, *, left_padding=None, ...)`** - Prepare cache for batch processing with padding info
- **`finalize(self)`** - Finalize cache after batch processing (no-op for ArraysCache)

Added methods to `CacheList`:

- **`merge(cls, cache_lists)`** - Merge multiple CacheList instances by recursively merging sub-caches
- **`extract(self, idx)`** - Extract from batched CacheList

### `mlx_lm/generate.py`

Updated `_merge_caches()` to handle `ArraysCache` and `CacheList`:

```python
elif isinstance(caches[0][i], ArraysCache):
    cache = type(caches[0][i]).merge([c[i] for c in caches])
elif isinstance(caches[0][i], CacheList):
    cache = CacheList.merge([c[i] for c in caches])
```

## Testing

### Unit Tests (15 tests, all passing)

- `TestArraysCacheMerge` - Basic merge, edge cases (None entries, empty list)
- `TestArraysCacheExtract` - Basic extraction, edge cases
- `TestCacheListMerge` - Merging CacheLists with MambaCache sub-caches
- `TestCacheListExtract` - Extracting from batched CacheList
- `TestMergeRoundTrip` - Verifies merge→extract preserves data
- `TestArraysCachePrepareFinalize` - Tests prepare() and finalize() methods

### Integration Test with Qwen3-Next-80B

```
Model: lmstudio-community/Qwen3-Next-80B-A3B-Instruct-MLX-4bit
Cache types (first 5 layers): ['MambaCache', 'MambaCache', 'MambaCache', 'KVCache', 'MambaCache']

Test 1: Single Generation (baseline) - PASSED
Test 2: Batch Generation (no cache) - PASSED
Test 3: Batch Generation WITH cache creation - PASSED
Test 4: Batch Generation REUSING caches - PASSED

ALL TESTS PASSED!
```

## Usage Example

```python
from mlx_lm import batch_generate, load

# Load a hybrid model (e.g., Qwen3-Next with MambaCache + KVCache)
model, tokenizer = load("qwen/Qwen3-Next-xxx")

# First batch generation with cache creation
prompts = [
    tokenizer.encode("Tell me about Python"),
    tokenizer.encode("What is MLX?"),
]
result1 = batch_generate(
    model, tokenizer, prompts,
    max_tokens=50,
    return_prompt_caches=True,  # Now works with MambaCache!
)

# Follow-up generation reusing cached state
followups = [
    tokenizer.encode("Give me an example"),
    tokenizer.encode("Show me the code"),
]
result2 = batch_generate(
    model, tokenizer, followups,
    max_tokens=50,
    prompt_caches=result1.caches,  # Reuse MambaCache state!
)
```
