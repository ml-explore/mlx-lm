# Copyright © 2024 Apple Inc.

import unittest

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.cache import (
    KVCache,
    PositionPreservingKVCache,
    SnapKVAttentionCapture,
    compact_prompt_cache,
    evict_prompt_cache,
    snapkv_keep_indices,
    trim_prompt_cache,
)


class TinyAttnModel(nn.Module):
    """A tiny causal-attention LM that routes through mx.fast SDPA (so the
    SnapKV capture hook fires) and caches with KVCache."""

    def __init__(self, vocab=64, dim=32, n_layers=2, n_heads=4):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim**-0.5
        self.qkv = [nn.Linear(dim, 3 * dim, bias=False) for _ in range(n_layers)]
        self.o = [nn.Linear(dim, dim, bias=False) for _ in range(n_layers)]
        self.out = nn.Linear(dim, vocab, bias=False)
        self.n_layers = n_layers

    def make_cache(self):
        return [KVCache() for _ in range(self.n_layers)]

    def __call__(self, inputs, cache=None):
        B, T = inputs.shape
        x = self.embed(inputs)
        if cache is None:
            cache = [None] * self.n_layers
        for qkv, o, c in zip(self.qkv, self.o, cache):
            H, d = self.n_heads, self.head_dim
            q, k, v = mx.split(qkv(x), 3, axis=-1)
            q = q.reshape(B, T, H, d).transpose(0, 2, 1, 3)
            k = k.reshape(B, T, H, d).transpose(0, 2, 1, 3)
            v = v.reshape(B, T, H, d).transpose(0, 2, 1, 3)
            if c is not None:
                k, v = c.update_and_fetch(k, v)
            key_len = k.shape[2]
            if T > 1:
                qpos = key_len - T + mx.arange(T)
                mask = (mx.arange(key_len)[None, :] <= qpos[:, None]).astype(x.dtype)
                mask = mx.where(mask > 0, 0.0, -1e9)
            else:
                mask = None
            att = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=self.scale, mask=mask
            )
            att = att.transpose(0, 2, 1, 3).reshape(B, T, H * d)
            x = x + o(att)
        return self.out(x)


class TestSnapKVKeepIndices(unittest.TestCase):
    def test_keeps_sinks_recent_and_top_scored(self):
        seq_len = 200
        scores = [0.0] * seq_len
        scores[100] = 9.0  # a strong middle token
        scores[150] = 8.0
        keep = snapkv_keep_indices(
            seq_len, budget=64, scores=scores, sink_tokens=4, min_tokens=128
        )
        self.assertEqual(len(keep), 64)
        self.assertEqual(keep, tuple(sorted(keep)))
        for s in range(4):  # sinks (budget//8 == 8 here, so all 4 fit)
            self.assertIn(s, keep)
        self.assertIn(seq_len - 1, keep)  # recent window
        self.assertIn(100, keep)  # top-scored middle tokens survive
        self.assertIn(150, keep)

    def test_sink_guard_survives_tight_budget_and_distractors(self):
        # #1552-style adversary: a tight budget plus middle rows scored far
        # higher than the sinks. The sink guard must still keep all N sinks and
        # the most-recent row — eviction can never drop them.
        seq_len = 300
        budget = 16  # budget // 8 == 2 < sink_tokens, the old cap's weak spot
        scores = [0.0] * seq_len
        for i in range(4, seq_len - 1):  # every middle row screams to be kept
            scores[i] = 1e9
        keep = snapkv_keep_indices(
            seq_len, budget, scores, sink_tokens=4, min_tokens=128
        )
        self.assertEqual(len(keep), budget)
        for s in range(4):  # all four sinks survive despite tiny budget
            self.assertIn(s, keep)
        self.assertIn(seq_len - 1, keep)  # recency window survives too

    def test_guarantee_sinks_flag_toggles_floor(self):
        # With the guard off, the budget//8 cap keeps fewer than N sinks; with
        # it on (default) all N survive.
        scores = [0.0] * 300
        capped = snapkv_keep_indices(
            300, 16, scores, sink_tokens=4, min_tokens=128, guarantee_sinks=False
        )
        self.assertNotIn(3, capped)  # 4th sink dropped by the budget//8 cap
        guarded = snapkv_keep_indices(
            300, 16, scores, sink_tokens=4, min_tokens=128, guarantee_sinks=True
        )
        for s in range(4):
            self.assertIn(s, guarded)

    def test_noop_paths(self):
        # budget covers the prompt, or prompt <= min_tokens -> keep everything.
        self.assertEqual(
            snapkv_keep_indices(50, 999, [0.0] * 50, min_tokens=128),
            tuple(range(50)),
        )
        self.assertEqual(
            snapkv_keep_indices(80, 16, [0.0] * 80, min_tokens=128),
            tuple(range(80)),
        )

    def test_validation(self):
        with self.assertRaises(ValueError):
            snapkv_keep_indices(10, 0, [0.0] * 10)
        with self.assertRaises(ValueError):
            snapkv_keep_indices(10, 4, [0.0] * 3)
        with self.assertRaises(ValueError):
            snapkv_keep_indices(-1, 4, [])


class TestPositionPreservingKVCache(unittest.TestCase):
    def _fill(self, cache, n):
        x = mx.random.uniform(shape=(1, 2, n, 4))
        return cache.update_and_fetch(x, x)

    def test_offset_and_stored_diverge_after_eviction(self):
        # Build from a sparse set of positions: stored rows < true offset.
        keys = mx.random.uniform(shape=(1, 2, 3, 4))
        c = PositionPreservingKVCache(keys, keys, offset=100, positions=(0, 1, 99))
        self.assertEqual(c.size(), 3)
        self.assertEqual(c.offset, 100)
        self.assertEqual(c.positions, (0, 1, 99))
        self.assertTrue(c.is_trimmable())

    def test_state_meta_state_roundtrip(self):
        keys = mx.random.uniform(shape=(1, 2, 3, 4))
        c = PositionPreservingKVCache(keys, keys, offset=100, positions=(0, 1, 99))
        restored = PositionPreservingKVCache.from_state(c.state, c.meta_state)
        self.assertEqual(restored.offset, 100)
        self.assertEqual(restored.size(), 3)
        self.assertEqual(restored.positions, (0, 1, 99))

    def test_logical_prefix_trim_drops_out_of_range_positions(self):
        keys = mx.random.uniform(shape=(1, 2, 3, 4))
        c = PositionPreservingKVCache(keys, keys, offset=100, positions=(0, 40, 99))
        trimmed = c.trim(70)  # new offset 30 -> keep only position 0
        self.assertEqual(trimmed, 70)
        self.assertEqual(c.offset, 30)
        self.assertEqual(c.positions, (0,))
        self.assertEqual(c.size(), 1)

    def test_speculative_suffix_trim(self):
        keys = mx.random.uniform(shape=(1, 2, 5, 4))
        c = PositionPreservingKVCache(keys, keys, offset=5, positions=tuple(range(5)))
        c.start_speculation()
        self._fill(c, 3)  # append 3 speculative rows
        self.assertEqual(c.offset, 8)
        c.trim(2)  # roll back 2 of the appended rows
        self.assertEqual(c.offset, 6)
        self.assertEqual(c.size(), 6)
        c.stop_speculation()

    def test_growth_and_nbytes(self):
        c = PositionPreservingKVCache()
        self._fill(c, 300)  # crosses the step=256 growth boundary
        self.assertEqual(c.size(), 300)
        self.assertEqual(c.offset, 300)
        self.assertGreater(c.nbytes, 0)


class TestEvictPromptCache(unittest.TestCase):
    def test_evicts_kv_layers_preserving_offset(self):
        cache = [KVCache(), KVCache()]
        x = mx.random.uniform(shape=(1, 2, 300, 4))
        for c in cache:
            c.update_and_fetch(x, x)
        scores = [0.0] * 300
        scores[123] = 5.0
        keep = snapkv_keep_indices(300, 32, scores, sink_tokens=4, min_tokens=128)
        result = evict_prompt_cache(cache, keep, true_offset=300)
        self.assertTrue(result.evicted)
        self.assertEqual(result.retained_tokens, len(keep))
        self.assertEqual(result.kv_layers, 2)
        for c in result.cache:
            self.assertIsInstance(c, PositionPreservingKVCache)
            self.assertEqual(c.offset, 300)  # true position preserved
            self.assertEqual(c.size(), len(keep))
        self.assertLess(result.compact_cache_nbytes, x.nbytes * 2)

    def test_non_kv_layers_untouched(self):
        from mlx_lm.models.cache import ArraysCache

        cache = [ArraysCache(size=1)]
        cache[0][0] = mx.zeros((1, 4))
        result = evict_prompt_cache(cache, (0,), true_offset=1)
        self.assertIs(result.cache[0], cache[0])
        self.assertEqual(result.kv_layers, 0)


class TestCompactPromptCacheEndToEnd(unittest.TestCase):
    def test_compaction_scores_and_decodes(self):
        mx.random.seed(0)
        model = TinyAttnModel(vocab=64, dim=32, n_layers=2, n_heads=4)
        mx.eval(model.parameters())
        prompt = [int(t) for t in mx.random.randint(0, 64, shape=(260,)).tolist()]

        result = compact_prompt_cache(model, prompt, budget=48, min_tokens=128)
        self.assertTrue(result.evicted)
        self.assertEqual(result.retained_tokens, 48)
        for c in result.cache:
            self.assertEqual(c.offset, len(prompt))  # RoPE offset preserved
            self.assertEqual(c.size(), 48)

        # The compacted cache decodes: append the next token, offset advances.
        logits = model(mx.array([[prompt[-1]]]), cache=result.cache)
        mx.eval(logits)
        self.assertEqual(logits.shape, (1, 1, 64))
        for c in result.cache:
            self.assertEqual(c.offset, len(prompt) + 1)

    def test_capture_produces_nontrivial_scores(self):
        mx.random.seed(1)
        model = TinyAttnModel(vocab=64, dim=32, n_layers=2, n_heads=4)
        mx.eval(model.parameters())
        prompt = [int(t) for t in mx.random.randint(0, 64, shape=(200,)).tolist()]
        cache = model.make_cache()
        with SnapKVAttentionCapture(window=24) as cap:
            model(mx.array([prompt]), cache=cache)
        scores = cap.snap_scores(len(prompt))
        self.assertEqual(len(scores), len(prompt))
        self.assertGreater(max(scores), 0.0)  # the hook actually fired

    def test_short_prompt_is_noop(self):
        mx.random.seed(2)
        model = TinyAttnModel()
        mx.eval(model.parameters())
        prompt = [int(t) for t in mx.random.randint(0, 64, shape=(64,)).tolist()]
        result = compact_prompt_cache(model, prompt, budget=16, min_tokens=128)
        self.assertFalse(result.evicted)
        self.assertEqual(result.retained_tokens, 64)


if __name__ == "__main__":
    unittest.main()
