# ABOUTME: Tests slot-indexed KV cache scaffolding.
# ABOUTME: Validates append, reset, and gather semantics.

import unittest

import numpy as np

from mlx_lm.server_batched.slot_kv_cache import SlotKVCache


class SlotKVCacheTests(unittest.TestCase):
    def setUp(self):
        self.cache = SlotKVCache(max_slots=3, capacity_tokens=8, kv_heads=2, head_dim=4)

    def _kv(self, tokens, base=0):
        keys = np.arange(base, base + tokens * 8, dtype=np.float32).reshape(
            tokens, 2, 4
        )
        values = keys + 100
        return keys, values

    def test_append_and_view(self):
        k, v = self._kv(3)
        self.cache.append(0, k, v)
        view_k, view_v = self.cache.view(0)
        np.testing.assert_array_equal(view_k, k)
        np.testing.assert_array_equal(view_v, v)
        self.assertEqual(self.cache.length(0), 3)

    def test_reset_clears_length(self):
        k, v = self._kv(2)
        self.cache.append(1, k, v)
        self.cache.reset(1)
        self.assertEqual(self.cache.length(1), 0)
        view_k, _ = self.cache.view(1)
        self.assertEqual(view_k.shape[0], 0)

    def test_gather_returns_padded_arrays(self):
        k0, v0 = self._kv(2, base=0)
        k1, v1 = self._kv(4, base=50)
        self.cache.append(0, k0, v0)
        self.cache.append(2, k1, v1)

        gathered_k, gathered_v = self.cache.gather([0, 2])
        self.assertEqual(gathered_k.shape, (2, 4, 2, 4))
        np.testing.assert_array_equal(gathered_k[0, :2], k0)
        np.testing.assert_array_equal(gathered_k[1, :4], k1)
        np.testing.assert_array_equal(gathered_v[1, :4], v1)

    def test_capacity_overflow_raises(self):
        k, v = self._kv(9)
        with self.assertRaises(ValueError):
            self.cache.append(0, k, v)


if __name__ == "__main__":
    unittest.main()
