# Copyright © 2024 Apple Inc.

import unittest

from mlx_lm.server import LRUPromptCache
from tests.prompt_cache_test_utils import RewindRecorderLayer


class MockCache:
    def __init__(self, value):
        self.value = value

    @property
    def nbytes(self):
        return len(self.value)

    def __eq__(self, other):
        return other.value == self.value


class TestLRUPromptCacheBehavior(unittest.TestCase):
    def test_regular_refcounted_hit_refreshes_regular_lru_recency(self):
        cache = LRUPromptCache(max_size=2)
        model = ("regular-hit-refresh", None, None)

        cache.insert_cache(model, [1], [MockCache("test1")])
        cache.insert_cache(model, [1], [MockCache("test1")])
        cache.insert_cache(model, [2], [MockCache("test2")])

        c, t = cache.fetch_nearest_cache(model, [1])
        self.assertEqual(c, [MockCache("test1")])
        self.assertEqual(t, [])

        cache.insert_cache(model, [3], [MockCache("test3")])

        c, t = cache.fetch_nearest_cache(model, [1])
        self.assertEqual(c, [MockCache("test1")])
        self.assertEqual(t, [])
        c, t = cache.fetch_nearest_cache(model, [2])
        self.assertIsNone(c)
        self.assertEqual(t, [2])
        c, t = cache.fetch_nearest_cache(model, [3])
        self.assertEqual(c, [MockCache("test3")])
        self.assertEqual(t, [])

    def test_checkpoint_hit_refreshes_checkpoint_lru_recency(self):
        cache = LRUPromptCache(max_size=2)
        model = ("checkpoint-hit-refresh", None, None)

        cache.insert_cache(model, [1], [MockCache("test1")], checkpoint=True)
        cache.insert_cache(model, [2], [MockCache("test2")], checkpoint=True)

        c, t = cache.fetch_nearest_cache(model, [1, 99])
        self.assertEqual(c, [MockCache("test1")])
        self.assertEqual(t, [99])

        cache.insert_cache(model, [3], [MockCache("test3")], checkpoint=True)

        c, t = cache.fetch_nearest_cache(model, [1, 98])
        self.assertEqual(c, [MockCache("test1")])
        self.assertEqual(t, [98])
        c, t = cache.fetch_nearest_cache(model, [2, 77])
        self.assertIsNone(c)
        self.assertEqual(t, [2, 77])
        c, t = cache.fetch_nearest_cache(model, [3, 55])
        self.assertEqual(c, [MockCache("test3")])
        self.assertEqual(t, [55])

    def test_farther_rewindable_prefix_outranks_nearer_non_rewindable_longer_prefix(
        self,
    ):
        lru = LRUPromptCache(max_size=10)
        model = ("farther-rewindable-prefix", None, None)
        requested_tokens = [1, 2, 5]

        shorter_cache = RewindRecorderLayer(max_rewind=0, offset=1)
        nearer_longer_cache = RewindRecorderLayer(max_rewind=0, offset=3)
        farther_longer_cache = RewindRecorderLayer(max_rewind=2, offset=4)
        lru.insert_cache(model, [1], [shorter_cache])
        lru.insert_cache(model, [1, 2, 9], [nearer_longer_cache])
        lru.insert_cache(model, [1, 2, 3, 4], [farther_longer_cache])

        reused_cache, remaining = lru.fetch_nearest_cache(model, requested_tokens)
        self.assertIsNotNone(reused_cache)
        self.assertEqual(remaining, [5])
        self.assertEqual(reused_cache[0].offset, 2)
        self.assertEqual(farther_longer_cache.can_rewind_calls, [2])
        self.assertEqual(farther_longer_cache.rewind_calls, [2])

    def test_longer_rewindable_prefix_outranks_shorter_stored_prefix(self):
        lru = LRUPromptCache(max_size=10)
        model = ("longer-rewindable-prefix", None, None)
        shorter_tokens = [1]
        longer_tokens = [1, 2, 9]
        requested_tokens = [1, 2, 5]

        shorter_cache = RewindRecorderLayer(max_rewind=0, offset=1)
        longer_cache = RewindRecorderLayer(max_rewind=1, offset=3)
        lru.insert_cache(model, shorter_tokens, [shorter_cache])
        lru.insert_cache(model, longer_tokens, [longer_cache])

        reused_cache, remaining = lru.fetch_nearest_cache(model, requested_tokens)
        self.assertIsNotNone(reused_cache)
        self.assertEqual(remaining, [5])
        self.assertEqual(reused_cache[0].offset, 2)
        self.assertEqual(longer_cache.can_rewind_calls, [1])
        self.assertEqual(longer_cache.rewind_calls, [1])

        exact_cache, exact_remaining = lru.fetch_nearest_cache(model, longer_tokens)
        self.assertIsNotNone(exact_cache)
        self.assertEqual(exact_remaining, [])
        self.assertEqual(exact_cache[0].offset, 3)

    def test_longer_path_reuse_refreshes_recency_for_regular_and_checkpoint_entries(
        self,
    ):
        for checkpoint in (False, True):
            with self.subTest(checkpoint=checkpoint):
                lru = LRUPromptCache(max_size=2)
                model = ("longer-path-refresh", checkpoint, None)
                reused_tokens = [1, 2, 9]
                sibling_tokens = [1, 3, 9]
                fresh_tokens = [4, 5, 6]

                reused_entry = RewindRecorderLayer(max_rewind=1, offset=3)
                sibling_entry = RewindRecorderLayer(max_rewind=1, offset=3)
                lru.insert_cache(
                    model, reused_tokens, [reused_entry], checkpoint=checkpoint
                )
                lru.insert_cache(
                    model, sibling_tokens, [sibling_entry], checkpoint=checkpoint
                )

                reused_cache, remaining = lru.fetch_nearest_cache(model, [1, 2, 5])
                self.assertIsNotNone(reused_cache)
                self.assertEqual(remaining, [5])
                self.assertEqual(reused_cache[0].offset, 2)

                lru.insert_cache(
                    model,
                    fresh_tokens,
                    [MockCache("fresh")],
                    checkpoint=checkpoint,
                )

                cache_entry, rest = lru.fetch_nearest_cache(model, reused_tokens)
                self.assertIsNotNone(cache_entry)
                self.assertEqual(rest, [])
                cache_entry, rest = lru.fetch_nearest_cache(model, sibling_tokens)
                self.assertIsNone(cache_entry)
                self.assertEqual(rest, sibling_tokens)


if __name__ == "__main__":
    unittest.main()
