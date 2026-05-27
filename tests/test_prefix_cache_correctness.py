# Copyright © 2024 Apple Inc.

import unittest

import mlx.core as mx

from mlx_lm.models.cache import (
    ArraysCache,
    CacheList,
    KVCache,
    LRUPromptCache,
    RotatingKVCache,
)


def make_kv_cache(length):
    cache = KVCache()
    if length > 0:
        x = mx.arange(length, dtype=mx.float32).reshape(1, 1, length, 1)
        cache.update_and_fetch(x, x)
    return cache


def make_rotating_cache(length, max_size=4):
    cache = RotatingKVCache(max_size=max_size)
    for i in range(length):
        x = mx.array([i], dtype=mx.float32).reshape(1, 1, 1, 1)
        cache.update_and_fetch(x, x)
    return cache


def make_prefill_rotating_cache(length, max_size=4):
    cache = RotatingKVCache(max_size=max_size)
    if length > 0:
        x = mx.arange(length, dtype=mx.float32).reshape(1, 1, length, 1)
        cache.update_and_fetch(x, x)
    return cache


def make_arrays_cache(value):
    cache = ArraysCache(size=1)
    cache[0] = mx.array([[value]], dtype=mx.float32)
    return cache


class TestHybridPrefixCacheCorrectness(unittest.TestCase):
    def test_exact_hit_reuses_non_trimmable_arrays_cache(self):
        lru = LRUPromptCache(max_size=4)
        model = ("toy",)
        prompt = [1, 2, 3]
        lru.insert_cache(model, prompt, [make_arrays_cache(11), make_kv_cache(3)])

        cache, rest = lru.fetch_nearest_cache(model, prompt)

        self.assertEqual(rest, [])
        self.assertIsNotNone(cache)
        self.assertTrue(mx.array_equal(cache[0][0], mx.array([[11]], dtype=mx.float32)))

    def test_shorter_checkpoint_reuses_non_trimmable_arrays_cache(self):
        lru = LRUPromptCache(max_size=4)
        model = ("toy",)
        lru.insert_cache(model, [1, 2], [make_arrays_cache(22), make_kv_cache(2)])
        lru.insert_cache(model, [1, 2, 3, 4], [make_arrays_cache(44), make_kv_cache(4)])

        cache, rest = lru.fetch_nearest_cache(model, [1, 2, 9])

        self.assertEqual(rest, [9])
        self.assertIsNotNone(cache)
        self.assertTrue(mx.array_equal(cache[0][0], mx.array([[22]], dtype=mx.float32)))

    def test_longer_non_trimmable_cache_is_not_used_without_checkpoint(self):
        lru = LRUPromptCache(max_size=4)
        model = ("toy",)
        lru.insert_cache(model, [1, 2, 3, 4], [make_arrays_cache(44), make_kv_cache(4)])

        cache, rest = lru.fetch_nearest_cache(model, [1, 2, 9])

        self.assertIsNone(cache)
        self.assertEqual(rest, [1, 2, 9])

    def test_nested_cache_list_preserves_non_trimmable_boundary(self):
        lru = LRUPromptCache(max_size=4)
        model = ("toy",)
        nested = CacheList(make_arrays_cache(5), make_kv_cache(4))
        lru.insert_cache(model, [1, 2, 3, 4], [nested])

        cache, rest = lru.fetch_nearest_cache(model, [1, 2, 9])

        self.assertIsNone(cache)
        self.assertEqual(rest, [1, 2, 9])

    def test_saturated_rotating_cache_documents_current_longer_prefix_miss(self):
        lru = LRUPromptCache(max_size=4)
        model = ("toy",)
        lru.insert_cache(model, [1, 2, 3, 4, 5, 6], [make_rotating_cache(6, max_size=4)])

        cache, rest = lru.fetch_nearest_cache(model, [1, 2, 3, 9])

        self.assertIsNone(cache)
        self.assertEqual(rest, [1, 2, 3, 9])

    def test_kv_cache_can_restore_to_shorter_prefix(self):
        cache = make_kv_cache(5)

        self.assertTrue(cache.can_restore_to(3))
        restored = cache.restore_to(3)

        self.assertTrue(restored.restored)
        self.assertEqual(cache.offset, 3)

    def test_saturated_rotating_cache_reports_retained_logical_range(self):
        cache = make_rotating_cache(8, max_size=4)

        retained = cache.retained_range()

        self.assertEqual(retained.logical_start, 4)
        self.assertEqual(retained.logical_end, 8)

    def test_saturated_rotating_cache_rejects_restore_requiring_evicted_context(self):
        cache = make_rotating_cache(8, max_size=4)

        self.assertFalse(cache.can_restore_to(7))

    def test_saturated_rotating_cache_accepts_current_boundary_noop_restore(self):
        cache = make_rotating_cache(8, max_size=4)

        self.assertTrue(cache.can_restore_to(8))

    def test_unsaturated_rotating_cache_restore_slices_temporal_state(self):
        cache = make_rotating_cache(3, max_size=4)

        result = cache.restore_to(2)

        self.assertTrue(result.restored)
        self.assertEqual(cache.offset, 2)
        self.assertEqual(cache.keys.shape[2], 2)

    def test_saturated_rotating_cache_restore_to_current_boundary_is_noop(self):
        cache = make_rotating_cache(8, max_size=4)

        result = cache.restore_to(8)

        self.assertTrue(result.restored)
        self.assertEqual(cache.offset, 8)
        self.assertEqual(cache.keys.shape[2], 4)

    def test_saturated_rotating_cache_restore_requiring_evicted_context_does_not_mutate(self):
        cache = make_rotating_cache(8, max_size=4)
        before_offset = cache.offset
        before_shape = cache.keys.shape

        result = cache.restore_to(7)

        self.assertFalse(result.restored)
        self.assertEqual(cache.offset, before_offset)
        self.assertEqual(cache.keys.shape, before_shape)

    def test_lru_uses_restorable_longer_prefill_rotating_cache(self):
        lru = LRUPromptCache(max_size=4)
        model = ("toy",)
        lru.insert_cache(
            model,
            [1, 2, 3, 4, 5, 6, 7, 8],
            [make_prefill_rotating_cache(8, max_size=4)],
        )

        cache, rest = lru.fetch_nearest_cache(model, [1, 2, 3, 4, 5, 6, 7, 99])

        self.assertIsNotNone(cache)
        self.assertEqual(rest, [99])
        self.assertEqual(cache[0].offset, 7)

    def test_lru_prefers_shorter_safe_rotating_checkpoint_over_unrestorable_longer_hit(self):
        lru = LRUPromptCache(max_size=4)
        model = ("toy",)
        lru.insert_cache(model, [1, 2, 3, 4, 5, 6, 7], [make_rotating_cache(7, max_size=4)])
        lru.insert_cache(model, [1, 2, 3, 4, 5, 6, 7, 8], [make_rotating_cache(8, max_size=4)])

        cache, rest = lru.fetch_nearest_cache(model, [1, 2, 3, 4, 5, 6, 7, 99])

        self.assertIsNotNone(cache)
        self.assertEqual(rest, [99])
        self.assertEqual(cache[0].offset, 7)

    def test_cache_list_restores_when_all_children_restore(self):
        cache = CacheList(make_kv_cache(8), make_prefill_rotating_cache(8, max_size=4))

        self.assertTrue(cache.can_restore_to(7))
        result = cache.restore_to(7)

        self.assertTrue(result.restored)
        self.assertEqual(cache[0].offset, 7)
        self.assertEqual(cache[1].offset, 7)

    def test_cache_list_rejects_when_any_child_cannot_restore(self):
        cache = CacheList(make_arrays_cache(1), make_kv_cache(5))

        self.assertFalse(cache.can_restore_to(3))

    def test_recurrent_cache_prefers_shorter_checkpoint_over_unrestorable_longer_hit(self):
        lru = LRUPromptCache(max_size=8)
        model = ("toy",)
        lru.insert_cache(model, [1, 2], [make_arrays_cache(20), make_kv_cache(2)])
        lru.insert_cache(model, [1, 2, 3, 4, 5], [make_arrays_cache(50), make_kv_cache(5)])

        cache, rest = lru.fetch_nearest_cache(model, [1, 2, 9, 10])

        self.assertIsNotNone(cache)
        self.assertEqual(rest, [9, 10])
        self.assertTrue(mx.array_equal(cache[0][0], mx.array([[20]], dtype=mx.float32)))

    def test_arrays_cache_cannot_restore_to_shorter_prefix_by_default(self):
        cache = make_arrays_cache(7)

        self.assertFalse(cache.can_restore_to(1))
        result = cache.restore_to(1)
        self.assertFalse(result.restored)
        self.assertIn("exact checkpoint", result.reason)
        self.assertTrue(mx.array_equal(cache[0], mx.array([[7]], dtype=mx.float32)))

    def test_arrays_cache_can_be_reused_for_exact_entry_only_through_lru(self):
        lru = LRUPromptCache(max_size=4)
        model = ("toy",)
        lru.insert_cache(model, [1, 2, 3], [make_arrays_cache(7)])

        cache, rest = lru.fetch_nearest_cache(model, [1, 2, 3])

        self.assertEqual(rest, [])
        self.assertIsNotNone(cache)


class TestPrefixCacheSession(unittest.TestCase):
    def test_session_returns_rest_and_cached_count(self):
        from mlx_lm.generate import PrefixCacheSession

        session = PrefixCacheSession(max_size=4)
        model = ("toy",)
        session.insert(model, [1, 2, 3], [make_kv_cache(3)])

        hit = session.lookup(model, [1, 2, 3, 4])

        self.assertEqual(hit.cached_tokens, 3)
        self.assertEqual(hit.tokens_to_process, [4])
        self.assertIsNotNone(hit.prompt_cache)


if __name__ == "__main__":
    unittest.main()
