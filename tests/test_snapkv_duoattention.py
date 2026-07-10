# Copyright © 2024 Apple Inc.

import unittest

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.cache import (
    HeadPartitionedKVCache,
    KVCache,
    evict_prompt_cache_by_head,
)


class TinyAttnModel(nn.Module):
    """A tiny causal-attention LM that routes through mx.fast SDPA and caches
    with KVCache (mirrors tests/test_snapkv_cache.py)."""

    def __init__(self, vocab=64, dim=32, n_layers=2, n_heads=4):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim**-0.5
        self.qkv = [nn.Linear(dim, 3 * dim, bias=False) for _ in range(n_layers)]
        self.o = [nn.Linear(dim, dim, bias=False) for _ in range(n_layers)]
        self.out = nn.Linear(dim, vocab, bias=False)
        self.n_layers = n_layers

    def make_cache(self):
        return [KVCache() for _ in range(self.n_layers)]

    def __call__(self, inputs, cache=None):
        B, T = inputs.shape
        x = self.embed(inputs)
        if cache is None:
            cache = [None] * self.n_layers
        for qkv, o, c in zip(self.qkv, self.o, cache):
            H, d = self.n_heads, self.head_dim
            q, k, v = mx.split(qkv(x), 3, axis=-1)
            q = q.reshape(B, T, H, d).transpose(0, 2, 1, 3)
            k = k.reshape(B, T, H, d).transpose(0, 2, 1, 3)
            v = v.reshape(B, T, H, d).transpose(0, 2, 1, 3)
            if c is not None:
                k, v = c.update_and_fetch(k, v)
            key_len = k.shape[2]
            if T > 1:
                qpos = key_len - T + mx.arange(T)
                mask = (mx.arange(key_len)[None, :] <= qpos[:, None]).astype(x.dtype)
                mask = mx.where(mask > 0, 0.0, -1e9)
            else:
                mask = None
            att = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=self.scale, mask=mask
            )
            att = att.transpose(0, 2, 1, 3).reshape(B, T, H * d)
            x = x + o(att)
        return self.out(x)


class TestHeadPartitionedKVCache(unittest.TestCase):
    def _build(self):
        # Two KV heads, four stored rows (union of head-0/head-1 keeps).
        keys = mx.random.uniform(shape=(1, 2, 4, 4))
        # union positions = {0, 1, 98, 99}; head 0 keeps all, head 1 keeps
        # only the sink (0) and recent (99).
        return HeadPartitionedKVCache(
            keys,
            keys,
            offset=100,
            positions=(0, 1, 98, 99),
            head_positions=((0, 1, 98, 99), (0, 99)),
            query_heads=4,
        )

    def test_construction_and_accessors(self):
        c = self._build()
        self.assertEqual(c.size(), 4)
        self.assertEqual(c.offset, 100)
        self.assertEqual(c.positions, (0, 1, 98, 99))
        self.assertEqual(c.head_positions, ((0, 1, 98, 99), (0, 99)))
        self.assertTrue(c.is_trimmable())
        self.assertGreater(c.nbytes, 0)
        self.assertFalse(c.empty())

    def test_state_meta_state_roundtrip(self):
        c = self._build()
        restored = HeadPartitionedKVCache.from_state(c.state, c.meta_state)
        self.assertEqual(restored.offset, 100)
        self.assertEqual(restored.size(), 4)
        self.assertEqual(restored.positions, (0, 1, 98, 99))
        self.assertEqual(restored.head_positions, ((0, 1, 98, 99), (0, 99)))
        self.assertEqual(restored.query_heads, 4)

    def test_make_mask_per_head_semantics(self):
        c = self._build()
        # No query_heads expansion: one row per KV head.
        mask = c.make_mask(1, query_heads=2)
        self.assertEqual(mask.shape, (1, 2, 1, 4 + 1))
        prefix = mask[0, :, 0, :4]  # (kv_heads, stored)
        # head 0 attends to all four retained rows; head 1 only to {0, 99}.
        self.assertEqual(prefix[0].tolist(), [True, True, True, True])
        self.assertEqual(prefix[1].tolist(), [True, False, False, True])
        # The trailing block is causal over the new tokens.
        self.assertEqual(mask[0, 0, 0, 4].tolist(), True)

    def test_make_mask_query_head_expansion(self):
        c = self._build()
        mask = c.make_mask(1, query_heads=4)  # 4 query heads over 2 KV heads
        self.assertEqual(mask.shape, (1, 4, 1, 4 + 1))
        prefix = mask[0, :, 0, :4]
        # query heads 0,1 map to KV head 0; 2,3 map to KV head 1.
        self.assertEqual(prefix[0].tolist(), [True, True, True, True])
        self.assertEqual(prefix[1].tolist(), [True, True, True, True])
        self.assertEqual(prefix[2].tolist(), [True, False, False, True])
        self.assertEqual(prefix[3].tolist(), [True, False, False, True])

    def test_trim_drops_out_of_range_positions(self):
        c = self._build()
        trimmed = c.trim(50)  # new offset 50 -> keep only positions {0, 1}
        self.assertEqual(trimmed, 50)
        self.assertEqual(c.offset, 50)
        self.assertEqual(c.positions, (0, 1))
        self.assertEqual(c.head_positions, ((0, 1), (0,)))
        self.assertEqual(c.size(), 2)

    def test_update_and_fetch_appends_and_advances(self):
        c = self._build()
        new = mx.random.uniform(shape=(1, 2, 1, 4))
        keys, values = c.update_and_fetch(new, new)
        self.assertEqual(c.offset, 101)
        self.assertEqual(c.size(), 5)
        self.assertEqual(keys.shape[2], 5)
        # The appended row (position 100) is visible to every head.
        self.assertEqual(c.head_positions[0][-1], 100)
        self.assertEqual(c.head_positions[1][-1], 100)


class TestEvictPromptCacheByHead(unittest.TestCase):
    def test_asymmetric_per_head_eviction(self):
        cache = [KVCache(), KVCache()]
        x = mx.random.uniform(shape=(1, 2, 300, 4))
        for c in cache:
            c.update_and_fetch(x, x)
        original = sum(c.nbytes for c in cache)  # dense keys+values, all rows
        # head 0 = retrieval: keeps a long window (sinks + first 200 + recent).
        # head 1 = streaming: keeps only sinks (0..3) + recent (296..299).
        # The union drops the 200..295 middle band, so storage shrinks.
        head0 = tuple(range(200)) + tuple(range(296, 300))
        head1 = tuple(range(4)) + tuple(range(296, 300))
        head_keep = (head0, head1)
        result = evict_prompt_cache_by_head(
            cache, head_keep, true_offset=300, query_heads=4
        )
        self.assertTrue(result.evicted)
        self.assertEqual(result.kv_layers, 2)
        self.assertEqual(result.per_head_retained_tokens, (204, 8))
        # union = head 0's retained rows (head 1's are a subset of them).
        self.assertEqual(result.union_retained_tokens, 204)
        for c in result.cache:
            self.assertIsInstance(c, HeadPartitionedKVCache)
            self.assertEqual(c.offset, 300)
            self.assertEqual(c.head_positions, head_keep)
        self.assertLess(result.compact_cache_nbytes, original)

    def test_non_kv_layers_untouched(self):
        from mlx_lm.models.cache import ArraysCache

        cache = [ArraysCache(size=1)]
        cache[0][0] = mx.zeros((1, 4))
        result = evict_prompt_cache_by_head(cache, ((0,), (0,)), true_offset=1)
        self.assertIs(result.cache[0], cache[0])
        self.assertEqual(result.kv_layers, 0)


class TestHeadPartitionedEndToEnd(unittest.TestCase):
    def test_prefill_evict_by_head_then_decode(self):
        mx.random.seed(0)
        model = TinyAttnModel(vocab=64, dim=32, n_layers=2, n_heads=4)
        mx.eval(model.parameters())
        prompt = [int(t) for t in mx.random.randint(0, 64, shape=(200,)).tolist()]

        cache = model.make_cache()
        logits = model(mx.array([prompt]), cache=cache)
        mx.eval(logits, [c.state for c in cache])

        seq_len = len(prompt)
        # The model has 4 KV heads: heads 0,1 are retrieval (keep the full
        # context), heads 2,3 are streaming (sinks + recent window only).
        retrieval = tuple(range(seq_len))
        streaming = tuple(range(4)) + tuple(range(seq_len - 8, seq_len))
        head_keep = (retrieval, retrieval, streaming, streaming)
        result = evict_prompt_cache_by_head(
            cache, head_keep, true_offset=seq_len, query_heads=model.n_heads
        )
        self.assertTrue(result.evicted)
        for c in result.cache:
            self.assertIsInstance(c, HeadPartitionedKVCache)
            self.assertEqual(c.offset, seq_len)

        # Decode one token from the head-partitioned cache.
        out = model(mx.array([[prompt[-1]]]), cache=result.cache)
        mx.eval(out)
        self.assertEqual(out.shape, (1, 1, 64))
        for c in result.cache:
            self.assertEqual(c.offset, seq_len + 1)


if __name__ == "__main__":
    unittest.main()
