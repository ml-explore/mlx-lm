# Copyright © 2024 Apple Inc.

import os
import tempfile
import unittest

import mlx.core as mx
from test_snapkv_cache import TinyAttnModel

from mlx_lm.models.cache import (
    KVCache,
    PositionPreservingKVCache,
    evict_prompt_cache,
    evict_prompt_cache_by_head,
    load_prompt_cache,
    save_prompt_cache,
    snapkv_keep_indices,
)


def _prefilled_kv_layers(n_layers, n_heads, tokens):
    cache = [KVCache() for _ in range(n_layers)]
    x = mx.random.uniform(shape=(1, n_heads, tokens, 4))
    for c in cache:
        c.update_and_fetch(x, x)
    return cache


class TestSnapKVSerialization(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mktemp(suffix=".safetensors")

    def tearDown(self):
        if os.path.exists(self.tmp):
            os.remove(self.tmp)

    def test_position_preserving_roundtrip(self):
        cache = _prefilled_kv_layers(2, 2, 300)
        scores = [0.0] * 300
        scores[123] = 5.0
        keep = snapkv_keep_indices(300, 32, scores, min_tokens=128)
        result = evict_prompt_cache(cache, keep, true_offset=300)

        save_prompt_cache(self.tmp, result.cache)
        loaded = load_prompt_cache(self.tmp)

        for orig, got in zip(result.cache, loaded):
            self.assertIsInstance(got, PositionPreservingKVCache)
            self.assertEqual(got.offset, orig.offset)
            self.assertEqual(got.size(), orig.size())
            self.assertEqual(got.positions, orig.positions)
            self.assertTrue(mx.array_equal(got.state[0], orig.state[0]))

    def test_head_partitioned_roundtrip(self):
        cache = _prefilled_kv_layers(1, 4, 300)
        head_keep = [
            tuple(range(300)) if h == 0 else (0, 1, 2, 3, 298, 299) for h in range(4)
        ]
        result = evict_prompt_cache_by_head(
            cache, head_keep, true_offset=300, query_heads=4
        )
        save_prompt_cache(self.tmp, result.cache)
        loaded = load_prompt_cache(self.tmp)

        orig, got = result.cache[0], loaded[0]
        self.assertEqual(type(got).__name__, "HeadPartitionedKVCache")
        self.assertEqual(got.offset, orig.offset)
        self.assertEqual(got.size(), orig.size())
        self.assertEqual(got.positions, orig.positions)
        self.assertEqual(got.head_positions, orig.head_positions)

    def test_metadata_and_mixed_list_roundtrip(self):
        cache = _prefilled_kv_layers(2, 2, 300)
        keep = snapkv_keep_indices(300, 32, [0.0] * 300, min_tokens=128)
        # Compact only the first layer -> a mixed [PPKV, KVCache] list.
        result = evict_prompt_cache([cache[0]], keep, true_offset=300)
        mixed = [result.cache[0], cache[1]]
        save_prompt_cache(self.tmp, mixed, metadata={"model": "tiny"})
        loaded, meta = load_prompt_cache(self.tmp, return_metadata=True)
        self.assertIsInstance(loaded[0], PositionPreservingKVCache)
        self.assertIsInstance(loaded[1], KVCache)
        self.assertEqual(meta["model"], "tiny")

    def test_decode_after_load(self):
        mx.random.seed(0)
        model = TinyAttnModel(vocab=64, dim=32, n_layers=2, n_heads=4)
        mx.eval(model.parameters())
        prompt = [int(t) for t in mx.random.randint(0, 64, shape=(260,)).tolist()]

        cache = model.make_cache()
        model(mx.array([prompt[:-1]]), cache=cache)
        keep = snapkv_keep_indices(len(prompt) - 1, 48, [0.0] * (len(prompt) - 1))
        result = evict_prompt_cache(cache, keep, true_offset=len(prompt) - 1)

        save_prompt_cache(self.tmp, result.cache)
        loaded = load_prompt_cache(self.tmp)

        # The reloaded compact cache decodes and advances the true offset.
        logits = model(mx.array([prompt[-1:]]), cache=loaded)
        mx.eval(logits)
        self.assertEqual(logits.shape, (1, 1, 64))
        for c in loaded:
            self.assertEqual(c.offset, len(prompt))


if __name__ == "__main__":
    unittest.main()
