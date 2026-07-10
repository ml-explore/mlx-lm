# Copyright © 2024 Apple Inc.

import unittest

import mlx.core as mx
from test_snapkv_cache import TinyAttnModel

from mlx_lm.models.cache import (
    HeadClassification,
    HeadPartitionedKVCache,
    SnapKVAttentionCapture,
    classify_retrieval_heads,
    duoattention_head_keep_indices,
    evict_prompt_cache_by_head,
)


class TestPerHeadScoring(unittest.TestCase):
    def test_snap_scores_by_head_fires(self):
        mx.random.seed(0)
        model = TinyAttnModel(vocab=64, dim=32, n_layers=2, n_heads=4)
        mx.eval(model.parameters())
        seq_len = 40
        prompt = [int(t) for t in mx.random.randint(0, 64, shape=(seq_len,)).tolist()]

        cache = model.make_cache()
        with SnapKVAttentionCapture(window=24) as cap:
            logits = model(mx.array([prompt]), cache=cache)
            mx.eval(logits, [c.state for c in cache])

        by_head = cap.snap_scores_by_head(seq_len)
        # One row per query head, each of length seq_len.
        self.assertEqual(len(by_head), model.n_heads)
        for row in by_head:
            self.assertEqual(len(row), seq_len)
            self.assertGreater(sum(row), 0.0)  # nonzero attention mass

        # The pooled score stays available and unchanged in shape.
        pooled = cap.snap_scores(seq_len)
        self.assertEqual(len(pooled), seq_len)
        self.assertGreater(sum(pooled), 0.0)


class TestClassifyRetrievalHeads(unittest.TestCase):
    def _synthetic_scores(self):
        seq_len = 200
        sink, recent = 4, 64
        # Heads 0 and 1 dump most mass on distant keys (retrieval); heads 2 and
        # 3 concentrate on sinks + the recent window (streaming).
        retrieval_row = [0.0] * seq_len
        for i in range(sink, seq_len - recent):
            retrieval_row[i] = 1.0
        streaming_row = [0.0] * seq_len
        for i in range(sink):
            streaming_row[i] = 1.0
        for i in range(seq_len - recent, seq_len):
            streaming_row[i] = 1.0
        return [list(retrieval_row), list(retrieval_row), streaming_row, streaming_row]

    def test_distant_heads_are_retrieval(self):
        by_head = self._synthetic_scores()
        result = classify_retrieval_heads(
            by_head, sink_tokens=4, recent_tokens=64, retrieval_fraction=0.5
        )
        self.assertIsInstance(result, HeadClassification)
        self.assertEqual(result.retrieval_heads, (0, 1))
        self.assertEqual(result.streaming_heads, (2, 3))
        # Distant-mass heads have high distant ratio; streaming heads ~0.
        self.assertGreater(result.distant_ratios[0], 0.9)
        self.assertEqual(result.distant_ratios[2], 0.0)

    def test_retrieval_fraction_respected(self):
        by_head = self._synthetic_scores()
        # Only 25% of 4 heads -> 1 retrieval head.
        result = classify_retrieval_heads(by_head, retrieval_fraction=0.25)
        self.assertEqual(len(result.retrieval_heads), 1)
        self.assertEqual(len(result.streaming_heads), 3)
        self.assertIn(result.retrieval_heads[0], (0, 1))

    def test_min_distant_ratio_gates(self):
        by_head = self._synthetic_scores()
        # An impossibly high bar leaves no head clearing it.
        result = classify_retrieval_heads(by_head, min_distant_ratio=2.0)
        self.assertEqual(result.retrieval_heads, ())
        self.assertEqual(result.streaming_heads, (0, 1, 2, 3))

    def test_arg_validation(self):
        by_head = self._synthetic_scores()
        with self.assertRaises(ValueError):
            classify_retrieval_heads([])
        with self.assertRaises(ValueError):
            classify_retrieval_heads(by_head, retrieval_fraction=0.0)
        with self.assertRaises(ValueError):
            classify_retrieval_heads(by_head, retrieval_fraction=1.5)
        with self.assertRaises(ValueError):
            classify_retrieval_heads(by_head, sink_tokens=-1)
        with self.assertRaises(ValueError):
            classify_retrieval_heads(by_head, recent_tokens=-1)


class TestDuoAttentionHeadKeepIndices(unittest.TestCase):
    def _synthetic_scores(self):
        seq_len = 200
        sink, recent = 4, 16
        retrieval_row = [0.0] * seq_len
        for i in range(sink, seq_len - recent):
            retrieval_row[i] = 1.0
        streaming_row = [0.0] * seq_len
        for i in range(sink):
            streaming_row[i] = 1.0
        for i in range(seq_len - recent, seq_len):
            streaming_row[i] = 1.0
        return seq_len, [
            list(retrieval_row),
            list(retrieval_row),
            streaming_row,
            streaming_row,
        ]

    def test_build_valid_head_keep(self):
        seq_len, by_head = self._synthetic_scores()
        budget = 120
        head_keep = duoattention_head_keep_indices(
            by_head,
            seq_len,
            budget=budget,
            sink_tokens=4,
            recent_tokens=16,
            retrieval_fraction=0.5,
            min_tokens=64,
        )
        self.assertEqual(len(head_keep), 4)  # one per head
        for row in head_keep:
            self.assertEqual(tuple(sorted(row)), row)  # sorted
            self.assertTrue(all(0 <= p < seq_len for p in row))  # in range

        # Heads 0,1 retrieval (SnapKV keep-set = budget); heads 2,3 streaming
        # (sinks + recent only, strictly fewer positions).
        retrieval_len = len(head_keep[0])
        streaming_len = len(head_keep[2])
        self.assertEqual(retrieval_len, budget)
        self.assertLess(streaming_len, retrieval_len)
        self.assertEqual(streaming_len, 4 + 16)

    def test_accepted_by_evict_prompt_cache_by_head(self):
        from mlx_lm.models.cache import KVCache

        seq_len, by_head = self._synthetic_scores()
        cache = [KVCache(), KVCache()]
        x = mx.random.uniform(shape=(1, 4, seq_len, 4))
        for c in cache:
            c.update_and_fetch(x, x)
        head_keep = duoattention_head_keep_indices(
            by_head,
            seq_len,
            budget=120,
            sink_tokens=4,
            recent_tokens=16,
            retrieval_fraction=0.5,
            min_tokens=64,
        )
        result = evict_prompt_cache_by_head(
            cache, head_keep, true_offset=seq_len, query_heads=4
        )
        self.assertTrue(result.evicted)
        self.assertEqual(result.kv_layers, 2)
        for c in result.cache:
            self.assertIsInstance(c, HeadPartitionedKVCache)
            self.assertEqual(c.head_positions, head_keep)


class TestEndToEnd(unittest.TestCase):
    def test_prefill_capture_evict_decode(self):
        mx.random.seed(0)
        model = TinyAttnModel(vocab=64, dim=32, n_layers=2, n_heads=4)
        mx.eval(model.parameters())
        seq_len = 200
        prompt = [int(t) for t in mx.random.randint(0, 64, shape=(seq_len,)).tolist()]

        cache = model.make_cache()
        with SnapKVAttentionCapture(window=24) as cap:
            logits = model(mx.array([prompt]), cache=cache)
            mx.eval(logits, [c.state for c in cache])

        by_head = cap.snap_scores_by_head(seq_len)
        self.assertEqual(len(by_head), model.n_heads)

        head_keep = duoattention_head_keep_indices(
            by_head,
            seq_len,
            budget=120,
            sink_tokens=4,
            recent_tokens=16,
            retrieval_fraction=0.5,
            min_tokens=64,
        )
        result = evict_prompt_cache_by_head(
            cache, head_keep, true_offset=seq_len, query_heads=model.n_heads
        )
        self.assertTrue(result.evicted)
        for c in result.cache:
            self.assertIsInstance(c, HeadPartitionedKVCache)
            self.assertEqual(c.offset, seq_len)

        # Decode one token from the head-partitioned cache without error.
        out = model(mx.array([[prompt[-1]]]), cache=result.cache)
        mx.eval(out)
        self.assertEqual(out.shape, (1, 1, 64))
        for c in result.cache:
            self.assertEqual(c.offset, seq_len + 1)


if __name__ == "__main__":
    unittest.main()
