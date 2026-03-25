# Copyright © 2024 Apple Inc.

import unittest

import mlx.core as mx

from mlx_lm.models.cache import BatchRotatingKVCache, _BaseCache


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


if __name__ == "__main__":
    unittest.main()
