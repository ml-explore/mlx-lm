# Changelog

## [Unreleased]

### Added
- Opt-in disk-backed prompt cache (`--prompt-cache-disk-dir`) for `mlx_lm.server`.
  Persists in-memory cache entries to disk via write-through async writes;
  survives reboots and rescues runtime LRU evictions. Off by default; zero
  behavior change when not enabled.
- `python -m mlx_lm.cache_admin` CLI for cache inspection (`stats`, `list`,
  `verify`), pruning (`prune --older-than`), and removal (`remove --model`).
