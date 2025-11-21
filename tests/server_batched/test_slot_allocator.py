# ABOUTME: Tests slot allocator behaviour for batching runtime.
# ABOUTME: Ensures slots reuse deterministically under contention.

import unittest

from mlx_lm.server_batched.slot_allocator import SlotAllocator


class SlotAllocatorTests(unittest.TestCase):
    def test_alloc_and_release(self):
        alloc = SlotAllocator(2)
        first = alloc.alloc()
        second = alloc.alloc()
        self.assertIn(first, (0, 1))
        self.assertIn(second, (0, 1))
        self.assertNotEqual(first, second)
        self.assertIsNone(alloc.alloc())

        alloc.release(first)
        self.assertEqual(alloc.available(), 1)
        third = alloc.alloc()
        self.assertEqual(third, first)

        # Double release is a no-op
        alloc.release(third)
        alloc.release(second)
        self.assertEqual(alloc.available(), 2)

    def test_invalid_release_raises(self):
        alloc = SlotAllocator(1)
        with self.assertRaises(ValueError):
            alloc.release(-1)
        with self.assertRaises(ValueError):
            alloc.release(5)


if __name__ == "__main__":
    unittest.main()
