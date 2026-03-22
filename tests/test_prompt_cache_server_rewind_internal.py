# Copyright © 2024 Apple Inc.

import unittest

import mlx.core as mx

from mlx_lm.models.cache import BatchRotatingKVCache


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


if __name__ == "__main__":
    unittest.main()
