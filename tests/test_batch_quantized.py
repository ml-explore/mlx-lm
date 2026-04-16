"""Tests for BatchQuantizedKVCache and memory_config."""

import unittest

import mlx.core as mx
import numpy as np

from mlx_lm.models.cache import (
    BatchKVCache,
    BatchQuantizedKVCache,
    KVCache,
    QuantizedKVCache,
)
from mlx_lm.memory_config import (
    InferenceConfig,
    _kv_bytes_per_token,
    _model_bytes,
    auto_configure,
    describe_config,
)


B, H, D = 2, 8, 128  # batch, heads, head_dim


def rand_kv(b=1, h=H, s=10, d=D):
    k = mx.random.normal((b, h, s, d))
    v = mx.random.normal((b, h, s, d))
    mx.eval(k, v)
    return k, v


class TestBatchQuantizedKVCache(unittest.TestCase):
    """Core correctness and edge case tests for BatchQuantizedKVCache."""

    # ---- Correctness ----

    def test_update_and_fetch_returns_quantized_tuples(self):
        cache = BatchQuantizedKVCache([0, 0], group_size=64, bits=8)
        k, v = rand_kv(b=B, s=10)
        result = cache.update_and_fetch(k, v)
        q_keys, q_values = result

        # Must be 3-tuples (data, scales, biases)
        self.assertEqual(len(q_keys), 3)
        self.assertEqual(len(q_values), 3)
        self.assertEqual(q_keys[0].dtype, mx.uint32)
        self.assertEqual(q_keys[0].shape[0], B)
        self.assertEqual(q_keys[0].shape[1], H)
        self.assertEqual(q_keys[0].shape[2], 10)

    def test_incremental_update(self):
        cache = BatchQuantizedKVCache([0, 0], group_size=64, bits=8)
        # Prefill
        k1, v1 = rand_kv(b=B, s=20)
        cache.update_and_fetch(k1, v1)
        self.assertEqual(cache._idx, 20)

        # Decode
        k2, v2 = rand_kv(b=B, s=1)
        result = cache.update_and_fetch(k2, v2)
        self.assertEqual(cache._idx, 21)
        self.assertEqual(result[0][0].shape[2], 21)

    def test_batch_quantized_matches_single_quantized(self):
        """Merged batch cache should match individual quantized caches."""
        lengths = [5, 10, 3]
        caches = []
        for l in lengths:
            c = QuantizedKVCache(group_size=64, bits=8)
            k, v = rand_kv(s=l)
            c.update_and_fetch(k, v)
            caches.append(c)

        batched = QuantizedKVCache.merge(caches)
        self.assertIsInstance(batched, BatchQuantizedKVCache)

        # Extract and compare
        for i, orig in enumerate(caches):
            extracted = batched.extract(i)
            for j in range(3):
                self.assertTrue(
                    mx.array_equal(
                        orig.keys[j][..., : orig.offset, :],
                        extracted.keys[j],
                    ),
                    f"keys[{j}] mismatch for cache {i}",
                )

    # ---- Round-trip ----

    def test_merge_then_extract_roundtrip(self):
        c1 = QuantizedKVCache(group_size=64, bits=8)
        c2 = QuantizedKVCache(group_size=64, bits=8)
        k1, v1 = rand_kv(s=7)
        k2, v2 = rand_kv(s=3)
        c1.update_and_fetch(k1, v1)
        c2.update_and_fetch(k2, v2)

        batched = QuantizedKVCache.merge([c1, c2])
        e1 = batched.extract(0)
        e2 = batched.extract(1)

        self.assertEqual(e1.offset, 7)
        self.assertEqual(e2.offset, 3)
        self.assertEqual(e1.keys[0].shape[2], 7)
        self.assertEqual(e2.keys[0].shape[2], 3)

    def test_merge_with_empty_cache(self):
        c1 = QuantizedKVCache(group_size=64, bits=8)
        c2 = QuantizedKVCache(group_size=64, bits=8)
        k1, v1 = rand_kv(s=5)
        c1.update_and_fetch(k1, v1)
        # c2 is empty

        batched = QuantizedKVCache.merge([c1, c2])
        self.assertEqual(batched._idx, 5)
        extracted = batched.extract(0)
        self.assertEqual(extracted.offset, 5)

    # ---- Batch Operations ----

    def test_filter(self):
        cache = BatchQuantizedKVCache([0, 0, 0], group_size=64, bits=8)
        k, v = rand_kv(b=3, s=10)
        cache.update_and_fetch(k, v)

        cache.filter([0, 2])
        self.assertEqual(cache.keys[0].shape[0], 2)
        self.assertEqual(cache._offset.shape[0], 2)

    def test_trim(self):
        cache = BatchQuantizedKVCache([0, 0], group_size=64, bits=8)
        k, v = rand_kv(b=B, s=10)
        cache.update_and_fetch(k, v)

        trimmed = cache.trim(3)
        self.assertEqual(trimmed, 3)
        self.assertEqual(cache._idx, 7)

    def test_prepare_left_padding(self):
        cache = BatchQuantizedKVCache([0, 0], group_size=64, bits=8)
        cache.prepare(left_padding=[3, 1])
        self.assertEqual(cache.left_padding.tolist(), [3, 1])

    def test_prepare_on_nonempty_raises(self):
        cache = BatchQuantizedKVCache([0, 0], group_size=64, bits=8)
        k, v = rand_kv(b=B, s=5)
        cache.update_and_fetch(k, v)
        with self.assertRaises(ValueError):
            cache.prepare(left_padding=[1, 1])

    def test_state_roundtrip(self):
        cache = BatchQuantizedKVCache([0, 0], group_size=64, bits=8)
        k, v = rand_kv(b=B, s=10)
        cache.update_and_fetch(k, v)

        state = cache.state
        cache2 = BatchQuantizedKVCache([0, 0], group_size=64, bits=8)
        cache2.state = state
        self.assertEqual(cache2._idx, 10)
        # State getter trims to _idx, so compare trimmed data
        s1 = cache.state
        s2 = cache2.state
        self.assertTrue(mx.array_equal(s1[0][0], s2[0][0]))

    def test_make_mask(self):
        cache = BatchQuantizedKVCache([3, 0], group_size=64, bits=8)
        k, v = rand_kv(b=B, s=5)
        cache.update_and_fetch(k, v)
        mask = cache.make_mask(1)
        self.assertIsNotNone(mask)

    def test_nbytes(self):
        cache = BatchQuantizedKVCache([0, 0], group_size=64, bits=8)
        k, v = rand_kv(b=B, s=100)
        cache.update_and_fetch(k, v)
        nbytes = cache.nbytes
        self.assertGreater(nbytes, 0)

    # ---- Edge Cases ----

    def test_single_element_batch(self):
        cache = BatchQuantizedKVCache([0], group_size=64, bits=8)
        k, v = rand_kv(b=1, s=5)
        result = cache.update_and_fetch(k, v)
        self.assertEqual(result[0][0].shape[0], 1)

    def test_bits_4(self):
        cache = BatchQuantizedKVCache([0, 0], group_size=64, bits=4)
        k, v = rand_kv(b=B, s=10)
        result = cache.update_and_fetch(k, v)
        # 4-bit: 8 elements per uint32, so packed dim = 128/8 = 16
        self.assertEqual(result[0][0].shape[3], D // 8)

    def test_bits_8(self):
        cache = BatchQuantizedKVCache([0, 0], group_size=64, bits=8)
        k, v = rand_kv(b=B, s=10)
        result = cache.update_and_fetch(k, v)
        # 8-bit: 4 elements per uint32, so packed dim = 128/4 = 32
        self.assertEqual(result[0][0].shape[3], D // 4)

    def test_large_prefill_exceeds_step(self):
        """Prefill larger than step=256 should work."""
        cache = BatchQuantizedKVCache([0], group_size=64, bits=8)
        k, v = rand_kv(b=1, s=300)
        cache.update_and_fetch(k, v)
        self.assertEqual(cache._idx, 300)

        # Then decode
        k2, v2 = rand_kv(b=1, s=1)
        cache.update_and_fetch(k2, v2)
        self.assertEqual(cache._idx, 301)

    def test_multiple_buffer_expansions(self):
        cache = BatchQuantizedKVCache([0], group_size=64, bits=8)
        for _ in range(5):
            k, v = rand_kv(b=1, s=200)
            cache.update_and_fetch(k, v)
        self.assertEqual(cache._idx, 1000)

    def test_sdpa_routing(self):
        """BatchQuantizedKVCache should trigger quantized SDPA path."""
        cache = BatchQuantizedKVCache([0], group_size=64, bits=8)
        self.assertTrue(hasattr(cache, "bits"))
        self.assertEqual(cache.bits, 8)
        self.assertEqual(cache.group_size, 64)

    # ---- Memory Comparison ----

    def test_quantized_uses_less_memory(self):
        """4-bit quantized cache should use ~4x less memory than FP16."""
        fp16_cache = BatchKVCache([0, 0])
        q4_cache = BatchQuantizedKVCache([0, 0], group_size=64, bits=4)

        k, v = rand_kv(b=B, s=500)
        fp16_cache.update_and_fetch(k, v)
        q4_cache.update_and_fetch(k, v)

        fp16_bytes = fp16_cache.nbytes
        q4_bytes = q4_cache.nbytes
        ratio = fp16_bytes / q4_bytes
        # 4-bit should be roughly 3-4x smaller (accounting for scales/biases)
        self.assertGreater(ratio, 2.5, f"Expected >2.5x savings, got {ratio:.1f}x")


class TestMemoryConfig(unittest.TestCase):
    """Tests for memory-aware auto-configuration."""

    @classmethod
    def setUpClass(cls):
        from mlx_lm import load

        cls.model, cls.tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")

    def test_model_bytes_positive(self):
        size = _model_bytes(self.model)
        self.assertGreater(size, 0)
        # 0.5B 4-bit model should be ~300MB
        self.assertGreater(size, 100 * 1024 * 1024)
        self.assertLess(size, 2 * 1024 * 1024 * 1024)

    def test_kv_bytes_ordering(self):
        fp16 = _kv_bytes_per_token(self.model, None)
        q8 = _kv_bytes_per_token(self.model, 8)
        q4 = _kv_bytes_per_token(self.model, 4)

        self.assertGreater(fp16, 0)
        self.assertGreater(fp16, q8)
        self.assertGreater(q8, q4)
        self.assertGreater(q4, 0)

    def test_kv_bytes_fp16_manual(self):
        """Verify FP16 formula matches manual computation."""
        args = self.model.args
        num_kv_heads = getattr(args, "num_key_value_heads", 2)
        head_dim = getattr(args, "hidden_size", 896) // getattr(
            args, "num_attention_heads", 14
        )
        num_layers = len(self.model.layers)

        expected = 2 * num_kv_heads * head_dim * 2 * num_layers
        actual = _kv_bytes_per_token(self.model, None)
        self.assertEqual(actual, expected)

    def test_auto_configure_returns_config(self):
        config = auto_configure(self.model)
        self.assertIsInstance(config, InferenceConfig)
        self.assertGreater(config.estimated_max_context, 0)
        self.assertIn(config.kv_bits, [None, 4, 8])

    def test_auto_configure_small_target(self):
        config = auto_configure(self.model, target_context=512)
        self.assertGreaterEqual(config.estimated_max_context, 512)
        # Small model + small target → should pick FP16
        self.assertIsNone(config.kv_bits)

    def test_auto_configure_impossible_target(self):
        config = auto_configure(self.model, target_context=100_000_000)
        self.assertGreater(len(config.warnings), 0)

    def test_auto_configure_budget_fraction(self):
        loose = auto_configure(self.model, memory_budget_fraction=0.9)
        tight = auto_configure(self.model, memory_budget_fraction=0.1)
        self.assertGreater(
            loose.estimated_max_context, tight.estimated_max_context
        )

    def test_describe_config(self):
        config = auto_configure(self.model)
        desc = describe_config(config)
        self.assertIsInstance(desc, str)
        self.assertIn("max context", desc.lower())


if __name__ == "__main__":
    unittest.main()
