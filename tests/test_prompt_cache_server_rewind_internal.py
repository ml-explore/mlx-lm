# Copyright © 2024 Apple Inc.

import unittest

import mlx.core as mx

from mlx_lm.models.cache import BatchRotatingKVCache, _BaseCache
from mlx_lm.server import LRUPromptCache


class TestLRUPromptCacheRewindInternals(unittest.TestCase):
    def test_batch_rotating_rewind_after_rotation_restores_pre_step_behavior(self):
        prompt_kv = mx.zeros((2, 1, 4, 1), dtype=mx.float32)
        decode_kv = mx.array([[[[11.0]]], [[[22.0]]]], dtype=mx.float32)

        cache = BatchRotatingKVCache(max_size=4, left_padding=[2, 0])
        cache.update_and_fetch(prompt_kv, prompt_kv)
        cache.update_and_fetch(decode_kv, decode_kv)
        mx.eval(cache.keys, cache.values)

        self.assertTrue(cache.rotated)
        self.assertEqual(cache._offset, 5)
        self.assertEqual(cache._idx, 1)

        pre_rewind_offsets = mx.array(cache.offset)
        self.assertTrue(cache.rewind(1))
        self.assertEqual(cache._offset, 4)
        self.assertEqual(cache._idx, 0)
        self.assertTrue(mx.array_equal(cache.offset, pre_rewind_offsets - 1))
        self.assertFalse(cache.can_rewind(1))

        rewind_mask = cache.make_mask(1)
        mx.eval(rewind_mask)
        self.assertEqual(rewind_mask.shape, (2, 1, 1, 4))
        self.assertEqual(rewind_mask[0, 0, 0].tolist(), [True, False, True, True])
        self.assertEqual(rewind_mask[1, 0, 0].tolist(), [True, True, True, True])


class TestHasRewindImpl(unittest.TestCase):
    def test_base_cache_has_no_rewind_impl(self):
        """_BaseCache itself should not report a rewind implementation."""
        base = _BaseCache.__new__(_BaseCache)
        self.assertFalse(base._has_rewind_impl())

    def test_subclass_without_rewind_override_has_no_impl(self):
        """A _BaseCache subclass that does not override rewind() should not
        report a rewind implementation."""

        class NoRewind(_BaseCache):
            pass

        cache = NoRewind.__new__(NoRewind)
        self.assertFalse(cache._has_rewind_impl())

    def test_subclass_with_rewind_override_has_impl(self):
        """A _BaseCache subclass that overrides rewind() should be recognized
        as having a rewind implementation — this is the contract that lets
        third-party caches participate without an explicit opt-in flag."""

        class CustomRewind(_BaseCache):
            def rewind(self, num_to_trim):
                return True

        cache = CustomRewind.__new__(CustomRewind)
        self.assertTrue(cache._has_rewind_impl())

    def test_batch_rotating_has_rewind_impl(self):
        """BatchRotatingKVCache should be recognized as having rewind."""
        cache = BatchRotatingKVCache(max_size=4, left_padding=[0])
        self.assertTrue(cache._has_rewind_impl())


class _LegacyTrimCache:
    """Non-_BaseCache cache with only the legacy is_trimmable/trim contract.
    No can_rewind, no rewind. Exercises the legacy fallback in both
    _can_rewind_layer_cache and _rewind_layer_cache."""

    def __init__(self, offset=10):
        self._offset = offset

    @property
    def offset(self):
        return self._offset

    def is_trimmable(self):
        return True

    def trim(self, num_to_trim):
        if num_to_trim > self._offset:
            return 0
        self._offset -= num_to_trim
        return num_to_trim


class _BaseCacheWithTrim(_BaseCache):
    """_BaseCache subclass with is_trimmable/trim but no rewind override.
    This is the shape of KVCache / RotatingKVCache — inherits the base-class
    rewind() stub. The rewind path must NOT be attempted; the legacy trim
    fallback must be used instead."""

    def __init__(self, offset=10):
        self._offset = offset

    @property
    def offset(self):
        return self._offset

    def is_trimmable(self):
        return True

    def trim(self, num_to_trim):
        if num_to_trim > self._offset:
            return 0
        self._offset -= num_to_trim
        return num_to_trim

    @property
    def state(self):
        raise NotImplementedError

    def is_empty(self):
        return self._offset == 0


class TestCanRewindAndRewindAgreement(unittest.TestCase):
    """Verify that _can_rewind_layer_cache and _rewind_layer_cache agree:
    if _can_rewind says yes, _rewind must succeed (not waste a deepcopy)."""

    def setUp(self):
        self.lru = LRUPromptCache(max_size=10)

    def test_legacy_non_basecache_can_rewind(self):
        """Legacy cache (no can_rewind) should be rewindable via the
        is_trimmable/trim fallback."""
        cache = _LegacyTrimCache(offset=10)
        self.assertTrue(self.lru._can_rewind_layer_cache(cache, 3))

    def test_legacy_non_basecache_rewind_succeeds(self):
        """If _can_rewind says yes, _rewind must actually succeed."""
        cache = _LegacyTrimCache(offset=10)
        can = self.lru._can_rewind_layer_cache(cache, 3)
        did = self.lru._rewind_layer_cache(cache, 3)
        self.assertTrue(can)
        self.assertTrue(did)

    def test_basecache_with_trim_no_rewind_override_can_rewind(self):
        """_BaseCache subclass with trim but no rewind override should still
        be rewindable via legacy fallback — not via the base-class stub."""
        cache = _BaseCacheWithTrim(offset=10)
        self.assertTrue(self.lru._can_rewind_layer_cache(cache, 3))

    def test_basecache_with_trim_no_rewind_override_rewind_succeeds(self):
        """The critical regression: _rewind must use the trim fallback, not
        call the base-class rewind() stub that raises NotImplementedError."""
        cache = _BaseCacheWithTrim(offset=10)
        can = self.lru._can_rewind_layer_cache(cache, 3)
        did = self.lru._rewind_layer_cache(cache, 3)
        self.assertTrue(can)
        self.assertTrue(did)
        self.assertEqual(cache.offset, 7)

    def test_basecache_with_trim_rewind_beyond_offset_fails(self):
        """Rewinding more than available offset should fail gracefully."""
        cache = _BaseCacheWithTrim(offset=2)
        self.assertFalse(self.lru._can_rewind_layer_cache(cache, 5))
        self.assertFalse(self.lru._rewind_layer_cache(cache, 5))

    def test_can_rewind_and_rewind_agree_for_batch_rotating(self):
        """BatchRotatingKVCache uses the real rewind path — sanity check
        that the agreement holds here too."""
        kv = mx.zeros((1, 1, 4, 1), dtype=mx.float32)
        decode = mx.array([[[[1.0]]]], dtype=mx.float32)
        cache = BatchRotatingKVCache(max_size=4, left_padding=[0])
        cache.update_and_fetch(kv, kv)
        cache.update_and_fetch(decode, decode)
        mx.eval(cache.keys, cache.values)

        can = self.lru._can_rewind_layer_cache(cache, 1)
        did = self.lru._rewind_layer_cache(cache, 1)
        self.assertTrue(can)
        self.assertTrue(did)


if __name__ == "__main__":
    unittest.main()
