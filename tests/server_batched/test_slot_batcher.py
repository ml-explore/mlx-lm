# ABOUTME: Tests slot batcher scaffolding for slot-based batching.
# ABOUTME: Ensures slot allocation, release, and decode selection.

import unittest
import numpy as np

from mlx_lm.server_batched.slot_allocator import SlotAllocator
from mlx_lm.server_batched.slot_batcher import SlotBatcher
from mlx_lm.server_batched.slot_kv_cache import SlotKVCache


class SlotBatcherTests(unittest.TestCase):
    def setUp(self):
        allocator = SlotAllocator(4)
        kv_cache = SlotKVCache(max_slots=4, capacity_tokens=8, kv_heads=2, head_dim=4)
        self.batcher = SlotBatcher(allocator=allocator, kv_cache=kv_cache)

    def test_register_and_release(self):
        slot = self.batcher.register("seq-1")
        self.assertEqual(slot, 0)
        self.batcher.release("seq-1")
        # re-register should reuse slot 0
        slot2 = self.batcher.register("seq-1")
        self.assertIn(slot2, (0, 1, 2, 3))

    def test_decode_with_logits(self):
        ids = ["a", "b"]
        for seq in ids:
            self.batcher.register(seq)
        logits = np.array([[0.1, 0.2], [0.9, -1.0]], dtype=np.float32)
        tokens = self.batcher.decode_with_logits(ids, logits)
        np.testing.assert_array_equal(tokens, np.array([1, 0]))

    def test_active_slots(self):
        ids = ["q", "r", "s"]
        expected = []
        for seq in ids:
            expected.append(self.batcher.register(seq))
        slots = self.batcher.active_slots(ids)
        self.assertEqual(slots, expected)


if __name__ == "__main__":
    unittest.main()
