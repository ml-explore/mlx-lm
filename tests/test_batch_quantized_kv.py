# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx

from mlx_lm.generate import BatchGenerator, _batch_layers, stream_generate
from mlx_lm.models.cache import (
    BatchKVCache,
    BatchQuantizedKVCache,
    KVCache,
    QuantizedKVCache,
    QuantizedRotatingKVCache,
    RotatingKVCache,
    make_prompt_cache,
)
from mlx_lm.utils import load


def _kv(n, H=2, D=64, seed=0):
    mx.random.seed(seed)
    c = KVCache()
    if n:
        c.update_and_fetch(mx.random.normal((1, H, n, D)), mx.random.normal((1, H, n, D)))
    return c


class TestBatchQuantizedKVCache(unittest.TestCase):
    def assertQuantEqual(self, a, b):
        for x, y in zip(a, b):
            self.assertEqual(x.shape, y.shape)
            self.assertTrue(mx.array_equal(x, y))

    def test_merge_extract_round_trip(self):
        caches = [_kv(n, seed=n).to_quantized(group_size=64, bits=8) for n in (5, 0, 12)]
        batch = BatchQuantizedKVCache.merge(caches)
        self.assertEqual(batch.size(), 12)
        self.assertEqual(batch.left_padding.tolist(), [7, 12, 0])
        for i, c in enumerate(caches):
            out = batch.extract(i)
            self.assertIsInstance(out, QuantizedKVCache)
            self.assertEqual(out.offset, c.offset)
            if c.offset:
                self.assertQuantEqual(out.keys_and_values()[0], c.keys_and_values()[0])
                self.assertQuantEqual(out.keys_and_values()[1], c.keys_and_values()[1])

    def test_update_matches_per_sequence_quantization(self):
        batch = BatchQuantizedKVCache([2, 0], group_size=64, bits=4)
        mx.random.seed(3)
        k = mx.random.normal((2, 2, 3, 64))
        v = mx.random.normal((2, 2, 3, 64))
        keys, values = batch.update_and_fetch(k, v)
        self.assertEqual(keys[0].shape[2], 3)
        expect = mx.quantize(k, group_size=64, bits=4)
        self.assertQuantEqual(keys, expect)
        # The dequantized batch is close to the input.
        back = mx.dequantize(*values, group_size=64, bits=4)
        self.assertLess(mx.abs(back - v).max().item(), 0.5)

    def test_filter_and_extend_keep_each_sequence(self):
        a = BatchQuantizedKVCache.merge([_kv(n, seed=n).to_quantized(bits=8) for n in (4, 9)])
        b = BatchQuantizedKVCache.merge([_kv(6, seed=6).to_quantized(bits=8)])
        a.extend(b)
        self.assertEqual(a.offset.tolist(), [4, 9, 6])
        ref = _kv(6, seed=6).to_quantized(bits=8)
        self.assertQuantEqual(a.extract(2).keys_and_values()[0], ref.keys_and_values()[0])
        a.filter([2])
        self.assertEqual(a.offset.tolist(), [6])
        self.assertEqual(a.left_padding.tolist(), [0])
        self.assertQuantEqual(a.extract(0).keys_and_values()[0], ref.keys_and_values()[0])
        with self.assertRaises(ValueError):
            a.extend(BatchQuantizedKVCache([0], bits=4))

    def test_layers_made_ready_for_a_batch(self):
        rot = RotatingKVCache(max_size=8)
        mx.random.seed(1)
        rot.update_and_fetch(mx.random.normal((1, 2, 5, 64)), mx.random.normal((1, 2, 5, 64)))
        qrot = rot.to_quantized(group_size=64, bits=8)
        layers = _batch_layers([_kv(3), _kv(3).to_quantized(bits=4), qrot], kv_bits=8, kv_group_size=64)
        self.assertIsInstance(layers[0], QuantizedKVCache)
        self.assertEqual((layers[1].bits, layers[1].offset), (8, 3), "requantized to the batch's bits")
        self.assertIsInstance(layers[2], RotatingKVCache)
        self.assertEqual((layers[2].offset, layers[2]._idx), (5, 5))
        self.assertLess(mx.abs(layers[2].keys[..., :5, :] - rot.keys[..., :5, :]).max().item(), 0.1)
        plain = _batch_layers([_kv(3).to_quantized(bits=8)], kv_bits=None, kv_group_size=64)
        self.assertIsInstance(plain[0], KVCache)
        self.assertEqual(plain[0].offset, 3)

    def test_batch_kv_to_quantized(self):
        b = BatchKVCache.merge([_kv(n, seed=n) for n in (3, 7)])
        q = b.to_quantized(group_size=64, bits=8)
        self.assertEqual(q.left_padding.tolist(), [4, 0])
        self.assertEqual(q.size(), 7)


class TestBatchedGenerationWithQuantizedKV(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.tokenizer = load("mlx-community/Qwen1.5-0.5B-Chat-4bit")

    def _prompts(self, texts):
        return [
            self.tokenizer.apply_chat_template([{"role": "user", "content": t}], tokenize=True, add_generation_prompt=True)
            for t in texts
        ]

    def _batched(self, prompts, n, caches=None):
        gen = BatchGenerator(self.model, stop_tokens=[], max_tokens=n, kv_bits=8, kv_group_size=64)
        uids = gen.insert(prompts, caches=caches)
        out = {u: [] for u in uids}
        while responses := gen.next_generated():
            for r in responses:
                out[r.uid].append(r.token)
        return [out[u] for u in uids]

    def _single(self, prompt, n):
        return [
            r.token
            for r in stream_generate(self.model, self.tokenizer, prompt, max_tokens=n, kv_bits=8, kv_group_size=64,
                                     quantized_kv_start=0)
        ]

    def test_matches_the_unbatched_path(self):
        prompts = self._prompts(["Write a story about Einstein", "Hi", "How tall is Mt Everest?"])
        batched = self._batched(prompts, 8)
        for prompt, tokens in zip(prompts, batched):
            single = self._single(prompt, 8)
            # Greedy: the same first tokens (quantized attention, padded or not).
            self.assertEqual(tokens[:4], single[:4])

    def test_history_from_the_unbatched_path_joins_a_batch(self):
        prompts = self._prompts(["Name three colors.", "Count to five."])
        # A cache the unbatched path left quantized (as the server stores it).
        cache = make_prompt_cache(self.model)
        for _ in stream_generate(self.model, self.tokenizer, prompts[0][:-1], max_tokens=1, prompt_cache=cache,
                                 kv_bits=8, kv_group_size=64, quantized_kv_start=0):
            pass
        self.assertTrue(any(isinstance(c, QuantizedKVCache) for c in cache))
        for c in cache:
            c.trim(c.offset - (len(prompts[0]) - 1))
        tokens = self._batched([prompts[0][-1:], prompts[1]], 4, caches=[cache, None])
        self.assertEqual(len(tokens[0]), 4)
        self.assertEqual(len(tokens[1]), 4)


if __name__ == "__main__":
    unittest.main()
