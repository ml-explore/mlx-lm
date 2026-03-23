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

    # -- Extraction and count semantics --

    def test_checkpoint_extract_persists_through_multiple_fetches(self):
        """Checkpoint entries are persistent: extraction always deepcopies and
        never removes the entry."""
        cache = LRUPromptCache(max_size=10)
        model = ("checkpoint-persist", None, None)

        cache.insert_cache(model, [1, 2], [MockCache("ckpt")], checkpoint=True)

        c1, t1 = cache.fetch_nearest_cache(model, [1, 2])
        self.assertIsNotNone(c1)
        self.assertEqual(t1, [])

        c2, t2 = cache.fetch_nearest_cache(model, [1, 2])
        self.assertIsNotNone(c2)
        self.assertEqual(t2, [])

        # Entry is still alive.
        result = cache._search(model, [1, 2])
        self.assertIsNotNone(result.exact)
        self.assertEqual(len(cache), 1)

    def test_regular_entry_promoted_to_checkpoint_becomes_persistent(self):
        """A regular entry promoted to checkpoint via a subsequent checkpoint
        insert becomes persistent (extract no longer consumes it)."""
        cache = LRUPromptCache(max_size=10)
        model = ("promote-checkpoint", None, None)

        cache.insert_cache(model, [1, 2], [MockCache("reg")])
        cache.insert_cache(model, [1, 2], [MockCache("reg")], checkpoint=True)

        c1, _ = cache.fetch_nearest_cache(model, [1, 2])
        self.assertIsNotNone(c1)
        c2, _ = cache.fetch_nearest_cache(model, [1, 2])
        self.assertIsNotNone(c2)
        self.assertEqual(len(cache), 1)

    def test_insert_existing_key_keeps_original_cache(self):
        """Re-inserting the same token key increments count but keeps the
        original prompt_cache list (the new one is silently dropped)."""
        cache = LRUPromptCache(max_size=10)
        model = ("reinsert-keeps-original", None, None)

        cache.insert_cache(model, [1, 2], [MockCache("original")])
        cache.insert_cache(model, [1, 2], [MockCache("different")])

        # First extract: deepcopy (count 2 → 1), returns original value.
        c, t = cache.fetch_nearest_cache(model, [1, 2])
        self.assertEqual(t, [])
        self.assertEqual(c, [MockCache("original")])

        # Second extract: count==1 ownership transfer, still original.
        c2, t2 = cache.fetch_nearest_cache(model, [1, 2])
        self.assertEqual(t2, [])
        self.assertEqual(c2, [MockCache("original")])

        # Now fully consumed.
        c3, t3 = cache.fetch_nearest_cache(model, [1, 2])
        self.assertIsNone(c3)
        self.assertEqual(t3, [1, 2])

    def test_deepcopy_failure_on_refcounted_entry_does_not_decrement_count(self):
        """When deepcopy fails on a refcounted (count > 1) non-checkpoint entry,
        _extract returns None without decrementing the count."""

        class FailDeepCopy:
            @property
            def nbytes(self):
                return 1

            def __deepcopy__(self, memo):
                raise RuntimeError("deepcopy fails")

        cache = LRUPromptCache(max_size=10)
        model = ("deepcopy-fail-refcount", None, None)

        cache.insert_cache(model, [1, 2], [FailDeepCopy()])
        cache.insert_cache(model, [1, 2], [FailDeepCopy()])  # count -> 2

        c, t = cache.fetch_nearest_cache(model, [1, 2])
        self.assertIsNone(c)
        self.assertEqual(t, [1, 2])

        # Entry still alive — count was not decremented.
        result = cache._search(model, [1, 2])
        self.assertIsNotNone(result.exact)
        self.assertEqual(len(cache), 1)

    # -- Rewind safety --

    def test_partial_rewind_on_longer_hit_discards_copy_preserves_original(self):
        """When rewind succeeds on some layers but fails on others in a longer
        cache hit, the corrupted deepcopy is discarded and the original entry
        is unmodified."""
        cache = LRUPromptCache(max_size=10)
        model = ("partial-rewind-discard", None, None)

        good_layer = RewindRecorderLayer(max_rewind=2, offset=4)
        bad_layer = RewindRecorderLayer(max_rewind=2, offset=4, rewind_result=False)
        cache.insert_cache(model, [1, 2, 3, 4], [good_layer, bad_layer])

        # Request [1, 2, 5] — longer candidate needs to rewind 2.
        c, t = cache.fetch_nearest_cache(model, [1, 2, 5])

        # Falls through to no match.
        self.assertIsNone(c)
        self.assertEqual(t, [1, 2, 5])

        # Original entry intact.
        result = cache._search(model, [1, 2, 3, 4])
        self.assertIsNotNone(result.exact)
        original = cache._get(model, [1, 2, 3, 4])
        self.assertEqual(original.prompt_cache[0].offset, 4)
        self.assertEqual(original.prompt_cache[1].offset, 4)

    # -- Search behavior changes from upstream --

    def test_single_token_shorter_match_is_valid(self):
        """A single-token prefix cache is returned as a valid shorter match."""
        cache = LRUPromptCache(max_size=10)
        model = ("single-token-shorter", None, None)

        cache.insert_cache(model, [1], [MockCache("one")])

        c, t = cache.fetch_nearest_cache(model, [1, 2, 3])
        self.assertEqual(c, [MockCache("one")])
        self.assertEqual(t, [2, 3])

    def test_shorter_prefix_not_evicted_by_longer_insert(self):
        """Inserting a longer token sequence does not evict a shorter prefix
        entry."""
        cache = LRUPromptCache(max_size=10)
        model = ("no-prefix-eviction", None, None)

        cache.insert_cache(model, [1, 2], [MockCache("short")])
        cache.insert_cache(model, [1, 2, 3, 4], [MockCache("long")])

        self.assertEqual(len(cache), 2)

        c, t = cache.fetch_nearest_cache(model, [1, 2])
        self.assertEqual(c, [MockCache("short")])
        self.assertEqual(t, [])


if __name__ == "__main__":
    unittest.main()
