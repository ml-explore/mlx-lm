"""
Tests for MambaCache/ArraysCache batching support.

This module tests the merge() and extract() methods added to ArraysCache
and CacheList to enable batch generation with prompt caches for hybrid
models like Qwen3-Next.
"""

import unittest
import mlx.core as mx
from mlx_lm.models.cache import (
    ArraysCache,
    MambaCache,
    CacheList,
    KVCache,
    BatchKVCache,
)


class TestArraysCacheMerge(unittest.TestCase):
    """Test ArraysCache.merge() functionality."""

    def test_merge_basic(self):
        """Test basic merge of two MambaCache instances."""
        cache1 = MambaCache()
        cache1[0] = mx.ones((1, 3, 8))  # conv state
        cache1[1] = mx.ones((1, 4, 16, 32)) * 2  # ssm state

        cache2 = MambaCache()
        cache2[0] = mx.ones((1, 3, 8)) * 3
        cache2[1] = mx.ones((1, 4, 16, 32)) * 4

        merged = MambaCache.merge([cache1, cache2])

        # Verify shapes
        self.assertEqual(merged.cache[0].shape, (2, 3, 8))
        self.assertEqual(merged.cache[1].shape, (2, 4, 16, 32))

        # Verify data integrity
        self.assertTrue(mx.allclose(merged.cache[0][0:1], cache1.cache[0]))
        self.assertTrue(mx.allclose(merged.cache[0][1:2], cache2.cache[0]))
        self.assertTrue(mx.allclose(merged.cache[1][0:1], cache1.cache[1]))
        self.assertTrue(mx.allclose(merged.cache[1][1:2], cache2.cache[1]))

        # Verify left_padding
        self.assertEqual(merged.left_padding.tolist(), [0, 0])

    def test_merge_three_caches(self):
        """Test merging three MambaCache instances."""
        caches = []
        for i in range(3):
            cache = MambaCache()
            cache[0] = mx.ones((1, 3, 8)) * (i + 1)
            cache[1] = mx.ones((1, 4, 16, 32)) * (i + 1)
            caches.append(cache)

        merged = MambaCache.merge(caches)

        self.assertEqual(merged.cache[0].shape, (3, 3, 8))
        self.assertEqual(merged.cache[1].shape, (3, 4, 16, 32))

    def test_merge_empty_list_raises(self):
        """Test that merging empty list raises ValueError."""
        with self.assertRaises(ValueError):
            MambaCache.merge([])

    def test_merge_with_none_entries(self):
        """Test merging when some cache entries are None."""
        cache1 = MambaCache()
        cache1[0] = mx.ones((1, 3, 8))
        cache1[1] = None  # Not initialized

        cache2 = MambaCache()
        cache2[0] = mx.ones((1, 3, 8)) * 2
        cache2[1] = mx.ones((1, 4, 16, 32))

        merged = MambaCache.merge([cache1, cache2])

        self.assertEqual(merged.cache[0].shape, (2, 3, 8))
        self.assertEqual(merged.cache[1].shape, (2, 4, 16, 32))
        # First entry should be zeros (initialized from None)
        self.assertTrue(mx.allclose(merged.cache[1][0:1], mx.zeros((1, 4, 16, 32))))

    def test_merge_all_none(self):
        """Test merging when all entries for a slot are None."""
        cache1 = MambaCache()
        cache1[0] = mx.ones((1, 3, 8))
        cache1[1] = None

        cache2 = MambaCache()
        cache2[0] = mx.ones((1, 3, 8))
        cache2[1] = None

        merged = MambaCache.merge([cache1, cache2])

        self.assertEqual(merged.cache[0].shape, (2, 3, 8))
        self.assertIsNone(merged.cache[1])


class TestArraysCacheExtract(unittest.TestCase):
    """Test ArraysCache.extract() functionality."""

    def test_extract_basic(self):
        """Test extracting a single entry from batched cache."""
        cache = MambaCache()
        cache.cache[0] = mx.arange(24).reshape((3, 4, 2)).astype(mx.float32)
        cache.cache[1] = mx.arange(48).reshape((3, 2, 4, 2)).astype(mx.float32)
        cache.left_padding = mx.array([1, 0, 2])

        extracted = cache.extract(1)

        self.assertEqual(extracted.cache[0].shape, (1, 4, 2))
        self.assertEqual(extracted.cache[1].shape, (1, 2, 4, 2))
        self.assertTrue(mx.allclose(extracted.cache[0], cache.cache[0][1:2]))
        self.assertTrue(mx.allclose(extracted.cache[1], cache.cache[1][1:2]))

    def test_extract_with_none(self):
        """Test extracting when some entries are None."""
        cache = MambaCache()
        cache.cache[0] = mx.ones((3, 4, 2))
        cache.cache[1] = None

        extracted = cache.extract(0)

        self.assertEqual(extracted.cache[0].shape, (1, 4, 2))
        self.assertIsNone(extracted.cache[1])


class TestCacheListMerge(unittest.TestCase):
    """Test CacheList.merge() functionality."""

    def test_merge_mamba_only(self):
        """Test merging CacheLists containing only MambaCache."""
        cl1 = CacheList(MambaCache(), MambaCache())
        cl1[0][0] = mx.ones((1, 3, 8))
        cl1[0][1] = mx.ones((1, 4, 16, 32))
        cl1[1][0] = mx.ones((1, 3, 8)) * 2
        cl1[1][1] = mx.ones((1, 4, 16, 32)) * 2

        cl2 = CacheList(MambaCache(), MambaCache())
        cl2[0][0] = mx.ones((1, 3, 8)) * 3
        cl2[0][1] = mx.ones((1, 4, 16, 32)) * 3
        cl2[1][0] = mx.ones((1, 3, 8)) * 4
        cl2[1][1] = mx.ones((1, 4, 16, 32)) * 4

        merged = CacheList.merge([cl1, cl2])

        # Verify both sub-caches were merged
        self.assertEqual(merged[0].cache[0].shape, (2, 3, 8))
        self.assertEqual(merged[1].cache[0].shape, (2, 3, 8))

    def test_merge_empty_list_raises(self):
        """Test that merging empty list raises ValueError."""
        with self.assertRaises(ValueError):
            CacheList.merge([])


class TestCacheListExtract(unittest.TestCase):
    """Test CacheList.extract() functionality."""

    def test_extract_basic(self):
        """Test extracting from batched CacheList."""
        cl = CacheList(MambaCache(), MambaCache())
        cl[0].cache[0] = mx.ones((3, 4, 2))
        cl[0].cache[1] = mx.ones((3, 2, 4, 2))
        cl[1].cache[0] = mx.ones((3, 4, 2)) * 2
        cl[1].cache[1] = mx.ones((3, 2, 4, 2)) * 2

        extracted = cl.extract(1)

        self.assertEqual(extracted[0].cache[0].shape, (1, 4, 2))
        self.assertEqual(extracted[1].cache[0].shape, (1, 4, 2))


class TestMergeRoundTrip(unittest.TestCase):
    """Test merge -> extract round trip preserves data."""

    def test_roundtrip_mamba(self):
        """Test that merge followed by extract preserves data."""
        original_caches = []
        for i in range(4):
            cache = MambaCache()
            cache[0] = mx.ones((1, 3, 8)) * (i + 1)
            cache[1] = mx.ones((1, 4, 16, 32)) * (i + 1)
            original_caches.append(cache)

        # Merge all
        merged = MambaCache.merge(original_caches)

        # Extract each and verify
        for i, original in enumerate(original_caches):
            extracted = merged.extract(i)
            self.assertTrue(mx.allclose(extracted.cache[0], original.cache[0]))
            self.assertTrue(mx.allclose(extracted.cache[1], original.cache[1]))


class TestArraysCachePrepareFinalize(unittest.TestCase):
    """Test prepare() and finalize() methods for batch processing."""

    def test_prepare_with_left_padding(self):
        """Test prepare() updates left_padding correctly."""
        cache = MambaCache()
        cache.cache[0] = mx.ones((3, 4, 8))
        cache.cache[1] = mx.ones((3, 2, 16, 32))
        cache.left_padding = mx.array([0, 0, 0])

        cache.prepare(left_padding=[1, 2, 0])

        self.assertEqual(cache.left_padding.tolist(), [1, 2, 0])

    def test_prepare_accumulates_padding(self):
        """Test prepare() accumulates padding when called multiple times."""
        cache = MambaCache()
        cache.left_padding = mx.array([1, 1, 1])

        cache.prepare(left_padding=[2, 3, 4])

        self.assertEqual(cache.left_padding.tolist(), [3, 4, 5])

    def test_prepare_with_none_padding(self):
        """Test prepare() initializes padding when None."""
        cache = MambaCache()
        cache.left_padding = None

        cache.prepare(left_padding=[1, 2, 3])

        self.assertEqual(cache.left_padding.tolist(), [1, 2, 3])

    def test_finalize_is_noop(self):
        """Test finalize() doesn't modify the cache."""
        cache = MambaCache()
        cache.cache[0] = mx.ones((2, 3, 8))
        cache.cache[1] = mx.ones((2, 4, 16, 32))
        original_0 = cache.cache[0].tolist()
        original_1 = cache.cache[1].tolist()

        cache.finalize()

        self.assertEqual(cache.cache[0].tolist(), original_0)
        self.assertEqual(cache.cache[1].tolist(), original_1)

    def test_prepare_with_lengths(self):
        """Test prepare() sets _lengths for right padding."""
        cache = MambaCache()
        cache._lengths = None

        cache.prepare(lengths=[2, 3, 4])

        self.assertEqual(cache._lengths.tolist(), [2, 3, 4])


class TestArraysCacheMakeMask(unittest.TestCase):
    """Test ArraysCache.make_mask() with left padding and lengths."""

    def test_mask_left_padding_only(self):
        """Test mask generation with left padding only."""
        from mlx_lm.models.cache import ArraysCache

        left_padding = [1, 2]
        cache = ArraysCache(size=2, left_padding=left_padding)

        mask = cache.make_mask(4)
        # Expected:
        # Row 0: [F, T, T, T] (pad 1)
        # Row 1: [F, F, T, T] (pad 2)

        self.assertIsNotNone(mask)
        self.assertTrue(mx.array_equal(mask[0], mx.array([False, True, True, True])))
        self.assertTrue(mx.array_equal(mask[1], mx.array([False, False, True, True])))

    def test_mask_with_lengths(self):
        """Test mask generation with lengths (right padding)."""
        from mlx_lm.models.cache import ArraysCache

        cache = ArraysCache(size=2, left_padding=[0, 0])
        cache.prepare(lengths=[2, 1])
        # N=3.
        # Row 0: lengths 2. Valid 0, 1. Mask idx < 2.
        # Row 1: lengths 1. Valid 0. Mask idx < 1.

        mask = cache.make_mask(3)
        # Expected:
        # Row 0: [T, T, F]
        # Row 1: [T, F, F]

        self.assertIsNotNone(mask)
        self.assertTrue(mx.array_equal(mask[0], mx.array([True, True, False])))
        self.assertTrue(mx.array_equal(mask[1], mx.array([True, False, False])))

    def test_mask_mixed_padding_and_lengths(self):
        """Test mask with both left padding and lengths."""
        from mlx_lm.models.cache import ArraysCache

        cache = ArraysCache(size=2, left_padding=[1])
        cache.prepare(lengths=[2])

        mask = cache.make_mask(3)
        # left_padding=1: idx >= 1 → [F, T, T]
        # lengths=2: idx < 2 → [T, T, F]
        # Combined: [F, T, F]
        self.assertTrue(mx.array_equal(mask[0], mx.array([False, True, False])))

    def test_mask_no_left_padding(self):
        """Test that mask returns None when no left_padding set."""
        from mlx_lm.models.cache import ArraysCache

        cache = ArraysCache(size=2)
        mask = cache.make_mask(4)

        self.assertIsNone(mask)


if __name__ == "__main__":
    unittest.main()
