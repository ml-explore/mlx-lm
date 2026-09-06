# Copyright © 2024 Apple Inc.

import os
import tempfile
import unittest

import mlx.core as mx

from mlx_lm.models.cache import (
    CompressedKVCache,
    KVCache,
    compress_prompt_cache,
    load_prompt_cache,
    make_prompt_cache,
    save_prompt_cache,
)


class TestCompressedKVCache(unittest.TestCase):

    def _make_cache_with_data(
        self,
        compression_ratio=0.5,
        n_sink=4,
        seq_len=100,
        batch=1,
        n_heads=8,
        head_dim=16,
    ):
        cache = CompressedKVCache(compression_ratio=compression_ratio, n_sink=n_sink)
        keys = mx.random.normal(shape=(batch, n_heads, seq_len, head_dim))
        values = mx.random.normal(shape=(batch, n_heads, seq_len, head_dim))
        cache.update_and_fetch(keys, values)
        mx.eval(cache.keys, cache.values)
        return cache, keys, values

    def test_compression_reduces_sequence_length(self):
        cache, _, _ = self._make_cache_with_data(compression_ratio=0.5, seq_len=100)
        self.assertEqual(cache.offset, 100)

        cache.compress()
        mx.eval(cache.keys, cache.values)

        expected_kept = max(int(100 * 0.5), 4 + 1)
        self.assertEqual(cache.offset, expected_kept)
        self.assertEqual(cache.keys.shape[2], expected_kept)
        self.assertEqual(cache.values.shape[2], expected_kept)

    def test_output_shape_matches_expected(self):
        B, H, D = 2, 4, 32
        cache, _, _ = self._make_cache_with_data(
            compression_ratio=0.4,
            batch=B,
            n_heads=H,
            head_dim=D,
            seq_len=200,
        )
        cache.compress()
        mx.eval(cache.keys, cache.values)

        n_kept = max(int(200 * 0.6), 4 + 1)
        self.assertEqual(cache.keys.shape, (B, H, n_kept, D))
        self.assertEqual(cache.values.shape, (B, H, n_kept, D))

    def test_compression_is_idempotent(self):
        cache, _, _ = self._make_cache_with_data(compression_ratio=0.5, seq_len=100)
        cache.compress()
        mx.eval(cache.keys, cache.values)
        first_keys = cache.keys
        first_offset = cache.offset

        cache.compress()
        mx.eval(cache.keys, cache.values)
        self.assertEqual(cache.offset, first_offset)
        self.assertTrue(mx.array_equal(cache.keys, first_keys))

    def test_sink_tokens_always_retained(self):
        n_sink = 4
        cache, original_keys, original_values = self._make_cache_with_data(
            compression_ratio=0.7,
            n_sink=n_sink,
            seq_len=100,
        )
        cache.compress()
        mx.eval(cache.keys, cache.values)

        compressed_sink_keys = cache.keys[..., :n_sink, :]
        original_sink_keys = original_keys[..., :n_sink, :]
        self.assertTrue(
            mx.allclose(compressed_sink_keys, original_sink_keys, atol=1e-5),
        )

    def test_sequence_order_preserved(self):
        cache = CompressedKVCache(compression_ratio=0.5, n_sink=2)
        seq_len = 50
        keys = mx.zeros((1, 1, seq_len, 4))
        values = mx.zeros((1, 1, seq_len, 4))
        for i in range(seq_len):
            keys = keys.at[0, 0, i, 0].add(float(i))
            values = values.at[0, 0, i, 0].add(float(i))
        cache.update_and_fetch(keys, values)
        mx.eval(cache.keys, cache.values)

        cache.compress()
        mx.eval(cache.keys, cache.values)

        position_markers = cache.keys[0, 0, :, 0].tolist()
        for i in range(len(position_markers) - 1):
            self.assertLess(position_markers[i], position_markers[i + 1])

    def test_zero_compression_ratio_is_noop(self):
        cache, _, _ = self._make_cache_with_data(compression_ratio=0.0, seq_len=50)
        cache.compress()
        mx.eval(cache.keys)
        self.assertEqual(cache.offset, 50)

    def test_very_short_sequence_no_crash(self):
        cache = CompressedKVCache(compression_ratio=0.5, n_sink=4)
        keys = mx.random.normal(shape=(1, 4, 3, 16))
        values = mx.random.normal(shape=(1, 4, 3, 16))
        cache.update_and_fetch(keys, values)
        mx.eval(cache.keys, cache.values)

        cache.compress()
        mx.eval(cache.keys, cache.values)
        self.assertEqual(cache.offset, 3)

    def test_empty_cache_compress_is_noop(self):
        cache = CompressedKVCache(compression_ratio=0.5)
        cache.compress()
        self.assertTrue(cache.empty())

    def test_compression_ratio_one_keeps_minimum(self):
        n_sink = 4
        cache, _, _ = self._make_cache_with_data(
            compression_ratio=1.0,
            n_sink=n_sink,
            seq_len=100,
        )
        cache.compress()
        mx.eval(cache.keys, cache.values)
        self.assertEqual(cache.offset, n_sink + 1)

    def test_update_after_compression(self):
        cache, _, _ = self._make_cache_with_data(compression_ratio=0.5, seq_len=100)
        cache.compress()
        mx.eval(cache.keys, cache.values)
        compressed_offset = cache.offset

        for _ in range(10):
            new_k = mx.random.normal(shape=(1, 8, 1, 16))
            new_v = mx.random.normal(shape=(1, 8, 1, 16))
            k_out, v_out = cache.update_and_fetch(new_k, new_v)
            mx.eval(k_out, v_out)

        self.assertEqual(cache.offset, compressed_offset + 10)
        self.assertEqual(k_out.shape[2], compressed_offset + 10)

    def test_trim_after_compression(self):
        cache, _, _ = self._make_cache_with_data(compression_ratio=0.5, seq_len=100)
        cache.compress()
        mx.eval(cache.keys, cache.values)
        pre_trim = cache.offset

        trimmed = cache.trim(5)
        self.assertEqual(trimmed, 5)
        self.assertEqual(cache.offset, pre_trim - 5)

    def test_save_load_compressed_cache(self):
        cache, _, _ = self._make_cache_with_data(compression_ratio=0.5, seq_len=100)
        cache.compress()
        mx.eval(cache.keys, cache.values)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "compressed_cache.safetensors")
            save_prompt_cache(path, [cache])
            loaded = load_prompt_cache(path)

        lc = loaded[0]
        self.assertIsInstance(lc, CompressedKVCache)
        self.assertEqual(lc.offset, cache.offset)
        self.assertAlmostEqual(lc.compression_ratio, cache.compression_ratio)
        self.assertEqual(lc.n_sink, cache.n_sink)
        self.assertEqual(lc._compressed, cache._compressed)
        self.assertTrue(mx.array_equal(lc.keys, cache.keys))
        self.assertTrue(mx.array_equal(lc.values, cache.values))

    def test_compress_prompt_cache_utility(self):
        caches = []
        for _ in range(4):
            c = CompressedKVCache(compression_ratio=0.5)
            keys = mx.random.normal(shape=(1, 4, 80, 16))
            values = mx.random.normal(shape=(1, 4, 80, 16))
            c.update_and_fetch(keys, values)
            mx.eval(c.keys, c.values)
            caches.append(c)

        compress_prompt_cache(caches)
        mx.eval([c.state for c in caches])

        for c in caches:
            expected = max(int(80 * 0.5), 4 + 1)
            self.assertEqual(c.offset, expected)

    def test_compress_prompt_cache_ignores_regular_kvcache(self):
        kv = KVCache()
        keys = mx.random.normal(shape=(1, 4, 80, 16))
        values = mx.random.normal(shape=(1, 4, 80, 16))
        kv.update_and_fetch(keys, values)
        mx.eval(kv.keys, kv.values)

        compress_prompt_cache([kv])
        self.assertEqual(kv.offset, 80)

    def test_make_prompt_cache_with_compression_ratio(self):

        class FakeModel:
            def __init__(self, n_layers):
                self.layers = [None] * n_layers

        model = FakeModel(4)
        cache = make_prompt_cache(model, compression_ratio=0.3)
        self.assertEqual(len(cache), 4)
        for c in cache:
            self.assertIsInstance(c, CompressedKVCache)
            self.assertAlmostEqual(c.compression_ratio, 0.3)

    def test_make_prompt_cache_without_compression_returns_kvcache(self):

        class FakeModel:
            def __init__(self, n_layers):
                self.layers = [None] * n_layers

        model = FakeModel(4)
        cache = make_prompt_cache(model)
        for c in cache:
            self.assertIsInstance(c, KVCache)
            self.assertNotIsInstance(c, CompressedKVCache)


if __name__ == "__main__":
    unittest.main()
