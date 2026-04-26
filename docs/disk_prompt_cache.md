# Disk-Backed Prompt Cache

## Motivation

`mlx_lm.server` keeps an in-memory `LRUPromptCache` (token-trie + per-leaf full
KV state) so repeat prompts skip prefill. The cache is bounded by
`--prompt-cache-bytes` and is **lost on every process exit**.

For long-lived `mlx_lm.server` deployments (Claude-Code-style agentic CLIs,
OpenAI-compatible API gateways, RAG pipelines), this has two visible failure
modes:

1. **Reboot wipes the cache.** A scheduled OS update, kernel panic, or power
   blip restarts the host. The first request after restart re-prefills from
   scratch — for a ~26K-token agentic prompt on Apple Silicon, that's ~200s of
   latency.
2. **Runtime LRU eviction also wipes.** Within a single server lifetime, when
   the cache hits its byte cap and an entry is evicted, future requests for
   that prefix re-prefill — even though the underlying compute happened
   minutes ago.

The disk-backed cache adds an **opt-in L2 disk tier** behind `LRUPromptCache`.
Every entry the in-memory cache holds is also persisted to disk via
write-through async writes. A reboot loses the in-memory cache but preserves
the disk cache; subsequent requests load from disk (~1-3s for ~5GB on Apple
Silicon NVMe) instead of recomputing.

The feature is **off by default**: behavior is byte-identical to the pre-PR
baseline unless `--prompt-cache-disk-dir` is set.

## When to use it

Set `--prompt-cache-disk-dir` when:
- The server is long-lived and survives multiple process restarts
- Prompts are large (system prompt + tools, RAG context) and prefill is slow
- You're willing to dedicate disk for cached KV state (typically ~2-5 GB per
  unique prefix on quantized models)

Don't set it when:
- The server is short-lived or one-off
- You're memory-bound and prefill is fast anyway
- You don't want any persistent state on disk

## Architecture

```
mlx_lm/disk_prompt_cache.py     # NEW: DiskPromptCache class, writer thread,
                                #      atomic write protocol, eviction
mlx_lm/cache_admin.py           # NEW: stats/list/prune/verify/remove CLI
mlx_lm/models/cache.py          # MODIFIED: LRUPromptCache.__init__ accepts
                                #           optional disk= param; insert_cache
                                #           and fetch_nearest_cache hooked
mlx_lm/server.py                # MODIFIED: 6 new CLI flags, conditional
                                #           DiskPromptCache instantiation,
                                #           SIGTERM/SIGINT handler
```

The hot path: RAM trie → in-memory disk-trie (built at startup) → if disk
match wins by quality+length, materialize via `load_prompt_cache(...)` →
promote into RAM → trim if needed → serve.

## On-disk schema

```
${disk-dir}/                                ← --prompt-cache-disk-dir
  .lock                                     ← advisory flock, one writer
  format-version                            ← "1\n"
  models/
    <model-id-16hex>/                       ← sha256 of weights manifest +
                                            tokenizer + chat template +
                                            mlx_lm major version
      info.json                             ← human-readable metadata
      entries/
        <token-hash-16hex>.safetensors      ← KV state + metadata
```

Each leaf is one safetensors file containing the full KV state. Metadata in
the safetensors header carries the token sequence, layer cache types,
trimmable flag, model_id, and format version.

Atomic writes: write `.tmp.safetensors` → optional fsync → `os.rename` to
`.safetensors`. A crash never leaves a half-written file readable; partial
`.tmp.safetensors` files are removed at next startup.

## Read path

```
fetch_nearest_cache(model, tokens):
    ram_result = ram._trie.search(model, tokens)
    if disk is None:
        return _serve_ram_match(ram_result)             # byte-identical to baseline
    disk_result = disk.search(tokens)
    pick the better of (ram_quality, ram_length) vs (disk_quality, disk_length)
    if disk wins:
        load via in-flight Future dedup    # one disk read per hash, even under N
                                           # concurrent requests for the same hash
        promote into RAM
        touch disk mtime
        serve via RAM path
    else:
        serve via RAM path
        if matched-tokens has a disk file: touch disk mtime
```

The mtime touch on every fetch (whether RAM or disk hit) keeps disk-side LRU
synchronized: hot-in-RAM entries don't lose their disk-eviction priority.

## Write path

```
insert_cache(...):
    do the existing in-memory trie + LRU update
    if disk is not None:
        eval mx.array state in this thread (writer thread has no GPU stream)
        compute parents_to_evict via disk.find_dominated_prefixes
        enqueue WriteJob on bounded queue (default size 4)
```

A single background writer thread:
1. Pop WriteJob
2. write_entry_atomic (`.tmp.safetensors` → fsync? → rename)
3. Update in-memory disk-trie under `_trie_lock`
4. Delete dominated parent files (path-deepest dedup)
5. If `total_bytes > max_bytes`: run eviction pass

Path-deepest dominator replacement: when inserting a deeper trimmable leaf,
shorter leaves on the same trie path are deleted because the deeper file
serves their queries via existing `trim_prompt_cache`.

## Configuration

| Flag | Default | Purpose |
|---|---|---|
| `--prompt-cache-disk-dir <path>` | None | Enables the feature |
| `--prompt-cache-disk-bytes <int>` | 100GB | Per-model cap (accepts `100GB`/`4TB`/etc.) |
| `--prompt-cache-disk-fsync` | off | fsync each entry; ~5-10x slower writes |
| `--prompt-cache-disk-write-queue-size <int>` | 4 | In-flight async writes |
| `--prompt-cache-disk-warm <mode>` | `lazy` | `lazy` or `eager-top-N` |
| `--prompt-cache-disk-eviction-headroom <float>` | 0.95 | Evict to this fraction of cap |

LRU eviction is mtime-based — `os.utime` on every fetch, sort-by-mtime to
evict. APFS / ext4 nanosecond mtime is reliable; we never `stat()` during
eviction (everything is in-memory).

## Operations

```
python -m mlx_lm.cache_admin stats <disk-dir>
python -m mlx_lm.cache_admin list <disk-dir> [--model <id>]
python -m mlx_lm.cache_admin prune <disk-dir> --older-than 30d
python -m mlx_lm.cache_admin verify <disk-dir>
python -m mlx_lm.cache_admin remove <disk-dir> --model <id>
```

`verify` walks every entry, attempts to load metadata, and reports any
unreadable / format-version-mismatched / orphan-`.tmp` files.

## Backwards compatibility

- `LRUPromptCache(max_size, max_bytes)` constructor — unchanged signature;
  `disk=None` is the default keyword-only addition.
- `LRUPromptCache.fetch_nearest_cache`, `insert_cache`, `nbytes`, `__len__` —
  unchanged behavior when `disk=None` (byte-identical fast path verified by
  unit tests).
- Existing tests under `tests/` pass unchanged; new tests are additive.

## Testing

- Unit tests (`tests/test_disk_prompt_cache.py`, `tests/test_cache_admin.py`):
  no model load, 64 tests, runs in <10s.
- Integration tests (`tests/test_disk_prompt_cache_e2e.py`): real
  `mlx_lm.server` subprocess + tiny model. Skipped under
  `MLX_LM_E2E_SKIP=1`.
- Performance benchmark (`benchmarks/disk_cache_overhead.py`): asserts
  no-miss-path overhead vs disk=None is <5%.
