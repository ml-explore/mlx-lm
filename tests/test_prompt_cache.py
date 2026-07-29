# Copyright © 2024 Apple Inc.

import copy
import os
import pickle
import tempfile
import threading
import unittest

import mlx.core as mx
import numpy as np

from mlx_lm.generate import generate_step
from mlx_lm.models.base import create_attention_mask, create_causal_mask
from mlx_lm.models.cache import (
    ArraysCache,
    BatchKVCache,
    BatchRotatingKVCache,
    CacheList,
    ChunkedKVCache,
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
    load_prompt_cache,
    make_prompt_cache,
    save_prompt_cache,
    trim_prompt_cache,
)
from mlx_lm.utils import load

HF_MODEL_PATH = "mlx-community/Qwen1.5-0.5B-Chat-4bit"


class TestPromptCache(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir_fid = tempfile.TemporaryDirectory()
        cls.test_dir = cls.test_dir_fid.name
        cls.model, cls.tokenizer = load(HF_MODEL_PATH)

    @classmethod
    def tearDownClass(cls):
        cls.test_dir_fid.cleanup()

    def test_save_load(self):
        cache = [KVCache() for _ in range(4)]
        for c in cache:
            x = mx.random.uniform(shape=(1, 8, 10, 4))
            c.update_and_fetch(x, x)
        cache_file = os.path.join(self.test_dir, "prompt_cache.safetensors")
        save_prompt_cache(cache_file, cache)
        loaded_cache = load_prompt_cache(cache_file)
        self.assertTrue(len(cache), len(loaded_cache))
        for c, lc in zip(cache, loaded_cache):
            self.assertEqual(c.offset, lc.offset)
            self.assertTrue(mx.array_equal(c.state[0], lc.state[0]))
            self.assertTrue(mx.array_equal(c.state[1], lc.state[1]))

        # Test with metadata
        cache_file = os.path.join(self.test_dir, "prompt_cache.safetensors")
        metadata = {"a": "b", "c": "d"}
        save_prompt_cache(cache_file, cache, metadata)
        _, loaded_metadata = load_prompt_cache(cache_file, return_metadata=True)
        self.assertEqual(metadata, loaded_metadata)

    def test_save_load_rotating_cache(self):
        cache_file = os.path.join(self.test_dir, "prompt_cache.safetensors")

        # Test with rotating cache
        cache = [RotatingKVCache(max_size=8, keep=2) for _ in range(4)]
        for c in cache:
            x = mx.random.uniform(shape=(1, 8, 10, 4))
            c.update_and_fetch(x, x)

        save_prompt_cache(cache_file, cache)
        loaded_cache = load_prompt_cache(cache_file)
        self.assertTrue(len(cache), len(loaded_cache))
        for c, lc in zip(cache, loaded_cache):
            self.assertEqual(c.offset, lc.offset)
            self.assertEqual(c.keep, lc.keep)
            self.assertEqual(c.max_size, lc.max_size)
            self.assertEqual(c.step, lc.step)
            self.assertTrue(mx.array_equal(c.state[0], lc.state[0]))
            self.assertTrue(mx.array_equal(c.state[1], lc.state[1]))

        # Do a couple single token updates to get a rotation
        for _ in range(2):
            for c in cache:
                x = mx.random.uniform(shape=(1, 8, 1, 4))
                c.update_and_fetch(x, x)

        save_prompt_cache(cache_file, cache)
        loaded_cache = load_prompt_cache(cache_file)

        for c, lc in zip(cache, loaded_cache):
            x = mx.random.uniform(shape=(1, 8, 1, 4))
            k, v = c.update_and_fetch(x, x)
            lk, lv = lc.update_and_fetch(x, x)
            self.assertEqual(c.offset, lc.offset)
            self.assertTrue(mx.array_equal(k, lk))
            self.assertTrue(mx.array_equal(v, lv))

    def test_save_load_mixed_cache(self):
        cache_file = os.path.join(self.test_dir, "prompt_cache.safetensors")

        cache = [
            ArraysCache(size=2),
            KVCache(),
            RotatingKVCache(8),
            ArraysCache(size=2),
            ChunkedKVCache(256),
        ]
        for c in cache:
            if isinstance(c, ArraysCache):
                c[0] = mx.random.uniform(shape=(4, 4, 4))
                c[1] = mx.random.uniform(shape=(4, 4, 4))
            else:
                x = mx.random.uniform(shape=(4, 4, 7, 4))
                y = mx.random.uniform(shape=(4, 4, 7, 4))
                c.update_and_fetch(x, y)

        save_prompt_cache(cache_file, cache)
        loaded_cache = load_prompt_cache(cache_file)
        for c, lc in zip(cache, loaded_cache):
            if isinstance(c, ArraysCache):
                self.assertTrue(mx.array_equal(c[0], lc[0]))
                self.assertTrue(mx.array_equal(c[1], lc[1]))
            else:
                x = mx.random.uniform(shape=(4, 4, 1, 4))
                y = mx.random.uniform(shape=(4, 4, 1, 4))
                k, v = c.update_and_fetch(x, y)
                lk, lv = lc.update_and_fetch(x, y)
                self.assertEqual(c.offset, lc.offset)
                self.assertTrue(mx.array_equal(k, lk))
                self.assertTrue(mx.array_equal(v, lv))

    def test_save_load_cache_list(self):
        cache_file = os.path.join(self.test_dir, "prompt_cache.safetensors")

        cache = [
            ArraysCache(size=2),
            KVCache(),
            RotatingKVCache(8),
            ArraysCache(size=2),
            ChunkedKVCache(256),
        ]
        for c in cache:
            if isinstance(c, ArraysCache):
                c[0] = mx.random.uniform(shape=(4, 4, 4))
                c[1] = mx.random.uniform(shape=(4, 4, 4))
            else:
                x = mx.random.uniform(shape=(4, 4, 7, 4))
                y = mx.random.uniform(shape=(4, 4, 7, 4))
                c.update_and_fetch(x, y)
        cache = [CacheList(*cache)]

        save_prompt_cache(cache_file, cache)
        loaded_cache = load_prompt_cache(cache_file)
        for c, lc in zip(cache[0].caches, loaded_cache[0].caches):
            if isinstance(c, ArraysCache):
                self.assertTrue(mx.array_equal(c[0], lc[0]))
                self.assertTrue(mx.array_equal(c[1], lc[1]))
            else:
                x = mx.random.uniform(shape=(4, 4, 1, 4))
                y = mx.random.uniform(shape=(4, 4, 1, 4))
                k, v = c.update_and_fetch(x, y)
                lk, lv = lc.update_and_fetch(x, y)
                self.assertEqual(c.offset, lc.offset)
                self.assertTrue(mx.array_equal(k, lk))
                self.assertTrue(mx.array_equal(v, lv))

    def test_save_load_arrays_cache(self):
        cache_file = os.path.join(self.test_dir, "prompt_cache.safetensors")

        cache = [ArraysCache(size=2)]
        cache[0][0] = mx.zeros((1, 4, 4))
        cache[0][1] = mx.zeros((1, 4, 4))

        save_prompt_cache(cache_file, cache)
        loaded = load_prompt_cache(cache_file)

        # Try to make a mask
        mask = loaded[0].make_mask(4)

    def test_cache_with_generate(self):
        model, tokenizer = self.model, self.tokenizer
        prompt = tokenizer.encode("this is a prompt", return_tensors="mlx")[0]
        results = list(generate_step(prompt, model, max_tokens=4))
        toks, all_logits = zip(*results)

        prompt_cache = make_prompt_cache(model)
        i = 0
        for tok, logits in generate_step(
            prompt, model, prompt_cache=prompt_cache, max_tokens=2
        ):
            self.assertEqual(tok, toks[i])
            self.assertTrue(mx.allclose(logits, all_logits[i]))
            i += 1

        for tok, logits in generate_step(
            mx.array([toks[i]]), model, prompt_cache=prompt_cache, max_tokens=1
        ):
            i += 1
            self.assertEqual(tok, toks[i])
            self.assertTrue(mx.allclose(logits, all_logits[i]))

    def test_trim_cache(self):
        cache = [KVCache() for _ in range(2)]
        for c in cache:
            x = mx.random.uniform(shape=(1, 8, 10, 4))
            c.update_and_fetch(x, x)

        # Trim
        num_trimmed = trim_prompt_cache(cache, 7)
        self.assertEqual(num_trimmed, 7)

        # Trim more tokens than remain
        num_trimmed = trim_prompt_cache(cache, 4)
        self.assertEqual(num_trimmed, 3)

        # Can't trim arrays cache
        cache = [ArraysCache(size=2) for _ in range(2)]
        for c in cache:
            c[0] = mx.zeros((5, 5))
            c[1] = mx.zeros((5, 5))
        num_trimmed = trim_prompt_cache(cache, 7)
        self.assertEqual(num_trimmed, 0)

        # All cache's have to be trimmable
        cache = [ArraysCache(size=2), KVCache()]
        cache[0][0] = mx.zeros((5, 5))
        cache[0][1] = mx.zeros((5, 5))
        x = mx.random.uniform(shape=(1, 8, 10, 4))
        cache[1].update_and_fetch(x, x)
        num_trimmed = trim_prompt_cache(cache, 1)
        self.assertEqual(num_trimmed, 0)

        cache = [RotatingKVCache(max_size=6) for _ in range(2)]
        for c in cache:
            x = mx.random.uniform(shape=(1, 8, 5, 4))
            c.update_and_fetch(x, x)

        num_trimmed = trim_prompt_cache(cache, 4)
        self.assertEqual(num_trimmed, 4)

        # Can't trim fixed-size KV cache after processing
        # more than max_kv_size tokens
        for c in cache:
            x = mx.random.uniform(shape=(1, 8, 10, 4))
            c.update_and_fetch(x, x)

        num_trimmed = trim_prompt_cache(cache, 4)
        self.assertEqual(num_trimmed, 0)

        cache = [QuantizedKVCache() for _ in range(2)]
        for c in cache:
            x = mx.random.uniform(shape=(1, 8, 10, 64))
            c.update_and_fetch(x, x)

        num_trimmed = trim_prompt_cache(cache, 7)
        self.assertEqual(num_trimmed, 7)

        # Trim more tokens than remain
        num_trimmed = trim_prompt_cache(cache, 4)
        self.assertEqual(num_trimmed, 3)

    def test_trim_cache_with_generate(self):
        model, tokenizer = self.model, self.tokenizer
        prompt = tokenizer.encode("this is a prompt", return_tensors="mlx")[0]

        prompt_cache = make_prompt_cache(model)

        # Generate one token so we process the full prompt
        last_tok, _ = next(generate_step(prompt, model, prompt_cache=prompt_cache))
        last_tok = mx.array([last_tok])

        # Generate two more tokens
        results = zip(
            range(2), generate_step(last_tok, model, prompt_cache=prompt_cache)
        )
        toks, all_logits = zip(*(r[1] for r in results))

        # To get back to the cache just after processing the prompt,
        # trim by 3 tokens
        trim_prompt_cache(prompt_cache, 3)

        # Generate the same thing again
        results = zip(
            range(2), generate_step(last_tok, model, prompt_cache=prompt_cache)
        )
        second_toks, second_all_logits = zip(*(r[1] for r in results))
        self.assertEqual(toks, second_toks)
        self.assertTrue(
            all(mx.allclose(l, l2) for l, l2 in zip(all_logits, second_all_logits))
        )

    def test_cache_copying(self):
        cache = [KVCache()]

        x = mx.random.uniform(shape=(1, 8, 10, 4))
        cache[0].update_and_fetch(x, x)

        y = mx.random.uniform(shape=(1, 8, 1, 4))
        cache[0].update_and_fetch(y, y)

        old_cache = copy.deepcopy(cache)

        trim_prompt_cache(cache, 1)

        self.assertTrue(old_cache[0].offset, 11)
        self.assertTrue(cache[0].offset, 10)

        z = mx.random.uniform(shape=(1, 8, 1, 4))
        cache[0].update_and_fetch(z, z)

        self.assertTrue(mx.allclose(old_cache[0].keys[..., 10:11, :], y))
        self.assertTrue(mx.allclose(cache[0].keys[..., 10:11, :], z))

    def test_save_load_quantized_cache(self):
        cache = [QuantizedKVCache(bits=4, group_size=32) for _ in range(4)]
        for c in cache:
            x = mx.random.uniform(shape=(1, 8, 10, 32))
            c.update_and_fetch(x, x)
        cache_file = os.path.join(self.test_dir, "prompt_cache.safetensors")
        save_prompt_cache(cache_file, cache)
        loaded_cache = load_prompt_cache(cache_file)
        self.assertTrue(loaded_cache[0].bits == cache[0].bits)
        self.assertTrue(loaded_cache[0].group_size == cache[0].group_size)
        self.assertTrue(len(cache), len(loaded_cache))
        for c, lc in zip(cache, loaded_cache):
            self.assertEqual(c.offset, lc.offset)
            # Loop over quantized tuple
            for i in range(3):
                self.assertTrue(mx.array_equal(c.state[0][i], lc.state[0][i]))
                self.assertTrue(mx.array_equal(c.state[1][i], lc.state[1][i]))

        # Test with metadata
        cache_file = os.path.join(self.test_dir, "prompt_cache.safetensors")
        metadata = {"a": "b", "c": "d"}
        save_prompt_cache(cache_file, cache, metadata)
        _, loaded_metadata = load_prompt_cache(cache_file, return_metadata=True)
        self.assertEqual(metadata, loaded_metadata)

    def test_cache_to_quantized(self):
        model, tokenizer = self.model, self.tokenizer
        prompt = tokenizer.encode("this is a prompt", return_tensors="mlx")[0]
        results = zip(range(4), generate_step(prompt, model))
        toks, all_logits = zip(*(r[1] for r in results))

        prompt_cache = make_prompt_cache(model)
        i = 0
        for _, (tok, logits) in zip(
            range(2), generate_step(prompt, model, prompt_cache=prompt_cache)
        ):
            self.assertEqual(tok, toks[i])
            self.assertTrue(mx.allclose(logits, all_logits[i]))
            i += 1

        prompt_cache = [c.to_quantized(bits=8, group_size=32) for c in prompt_cache]

        for _, (tok, logits) in zip(
            range(1),
            generate_step(mx.array([toks[i]]), model, prompt_cache=prompt_cache),
        ):
            i += 1
            self.assertEqual(tok, toks[i])
            self.assertTrue(mx.allclose(logits, all_logits[i], rtol=4e-2))

    def test_cache_list(self):
        c = CacheList(KVCache(), KVCache())
        self.assertTrue(c.is_trimmable())
        k = mx.zeros((1, 2, 8, 8))
        v = mx.zeros((1, 2, 8, 8))
        c[0].update_and_fetch(k, v)
        c[1].update_and_fetch(k, v)
        m = c.trim(5)
        self.assertEqual(m, 5)

        c = CacheList(ArraysCache(size=2), KVCache())
        self.assertFalse(c.is_trimmable())

        c1 = CacheList(ArraysCache(size=1), KVCache())
        c1[0][0] = mx.random.normal(shape=(1, 2, 4, 4))
        c1[1].update_and_fetch(
            mx.random.normal(shape=(1, 2, 5, 4)), mx.random.normal(shape=(1, 2, 5, 4))
        )

        c2 = CacheList(ArraysCache(size=1), KVCache())
        c2[0][0] = mx.random.normal(shape=(1, 2, 4, 4))
        c2[1].update_and_fetch(
            mx.random.normal(shape=(1, 2, 7, 4)), mx.random.normal(shape=(1, 2, 7, 4))
        )

        merged_cache = CacheList.merge((c1, c2))
        c1_ex = merged_cache.extract(0)
        self.assertTrue(mx.array_equal(c1_ex[0][0], c1[0][0]))
        self.assertTrue(mx.array_equal(c1_ex[1].state[0], c1[1].state[0]))
        c2_ex = merged_cache.extract(1)
        self.assertTrue(mx.array_equal(c2_ex[0][0], c2[0][0]))
        self.assertTrue(mx.array_equal(c2_ex[1].state[0], c2[1].state[0]))

    def test_make_mask_with_cache(self):
        # For 1 time step with no cache, don't need a mask
        mask = create_attention_mask(mx.zeros((1, 1)), cache=None, return_array=False)
        self.assertEqual(mask, None)

        mask = create_attention_mask(mx.zeros((1, 1)), cache=None, return_array=True)
        self.assertEqual(mask, None)

        # Regular causal mask
        mask = create_attention_mask(mx.zeros((1, 4)), cache=None, return_array=False)
        self.assertEqual(mask, "causal")

        mask = create_attention_mask(mx.zeros((1, 4)), cache=None, return_array=True)
        self.assertTrue(mx.array_equal(mask, create_causal_mask(4)))

        # With a window size
        mask = create_attention_mask(
            mx.zeros((1, 4)), cache=None, window_size=4, return_array=False
        )
        self.assertEqual(mask, "causal")

        mask = create_attention_mask(
            mx.zeros((1, 4)), cache=None, window_size=3, return_array=False
        )
        self.assertTrue(mx.array_equal(mask, create_causal_mask(4, window_size=3)))

        # With a regular KV cache
        cache = KVCache()
        mask = create_attention_mask(mx.zeros((1, 4)), cache=cache, return_array=False)
        self.assertEqual(mask, "causal")

        mask = create_attention_mask(mx.zeros((1, 4)), cache=cache, return_array=True)
        self.assertTrue(mx.array_equal(mask, create_causal_mask(4)))

        k = v = mx.zeros((1, 2, 16, 8))
        cache.update_and_fetch(k, v)
        mask = create_attention_mask(mx.zeros((1, 4)), cache=cache, return_array=True)
        self.assertEqual(mask.shape, (4, 20))

    def test_rotating_cache_mask(self):
        cache = RotatingKVCache(max_size=8)

        mask = cache.make_mask(4, window_size=5)
        self.assertEqual(mask, "causal")
        mask = create_attention_mask(mx.zeros((1, 4, 32)), cache, window_size=5)
        self.assertEqual(mask, "causal")
        mask = create_attention_mask(
            mx.zeros((1, 4, 32)), cache, window_size=5, return_array=True
        )
        self.assertEqual(mask.dtype, mx.bool_)
        self.assertEqual(mask.shape, (4, 4))

        mask = cache.make_mask(6, window_size=5)
        self.assertEqual(mask.dtype, mx.bool_)
        self.assertEqual(mask.sum(axis=-1).max(), 5)
        cmask = create_attention_mask(mx.zeros((1, 6, 32)), cache, window_size=5)
        self.assertTrue(mx.array_equal(cmask, mask))

        mask = cache.make_mask(1, window_size=5)
        self.assertEqual(mask, None)
        mask = create_attention_mask(mx.zeros((1, 1, 32)), cache, window_size=5)
        self.assertEqual(mask, None)

        kv = mx.zeros((1, 1, 10, 32))
        cache.update_and_fetch(kv, kv)
        mask = cache.make_mask(3, window_size=5)
        self.assertEqual(mask.shape, (3, 10))
        self.assertTrue(mx.all(mask.sum(axis=-1) == 5))
        for i in range(3):
            s = 11 - 3 + i
            self.assertTrue(mx.all(mask[s - 5 : s]))
        cmask = create_attention_mask(mx.zeros((1, 3, 32)), cache, window_size=5)
        self.assertTrue(mx.array_equal(cmask, mask))

        mask = cache.make_mask(1)
        self.assertEqual(mask, None)
        mask = create_attention_mask(mx.zeros((1, 1, 32)), cache)
        self.assertEqual(mask, None)

        mask = cache.make_mask(1, window_size=5)
        self.assertEqual(mask.tolist(), [True] + [False] * 3 + [True] * 4)
        cmask = create_attention_mask(mx.zeros((1, 1, 32)), cache, window_size=5)
        self.assertTrue(mx.array_equal(cmask, mask))

        kv = mx.zeros((1, 1, 1, 32))
        cache.update_and_fetch(kv, kv)

        mask = cache.make_mask(1, window_size=5)
        self.assertEqual(mask.tolist(), [True] * 2 + [False] * 3 + [True] * 3)
        cmask = create_attention_mask(mx.zeros((1, 1, 32)), cache, window_size=5)
        self.assertTrue(mx.array_equal(cmask, mask))

    def test_batch_kv_cache(self):
        cache = BatchKVCache(left_padding=[2, 3, 4])
        k, v = mx.zeros((3, 1, 4, 8)), mx.zeros((3, 1, 4, 8))
        # Update works
        k, v = cache.update_and_fetch(k, v)
        self.assertEqual(k.shape, (3, 1, 4, 8))

        # State can be evaluated
        mx.eval(cache.state)

        # State can be set
        cache.state = cache.state

        # Test filtering
        cache.filter([0, 1])

        # In this case filtering left shifts the cache so it has zero padding
        self.assertEqual(cache.state[0].shape, (2, 1, 2, 8))

        mask = cache.make_mask(1)
        self.assertEqual(mask[0].squeeze().tolist(), [True, True, True])
        self.assertEqual(mask[1].squeeze().tolist(), [False, True, True])

        # Test extension
        cache_a = BatchKVCache(left_padding=[2, 1, 2])
        cache_b = BatchKVCache(left_padding=[3, 0])

        k = mx.zeros((3, 1, 8, 1))
        v = mx.zeros((3, 1, 8, 1))
        cache_a.update_and_fetch(k, v)

        k = mx.zeros((2, 1, 4, 1))
        v = mx.zeros((2, 1, 4, 1))
        cache_b.update_and_fetch(k, v)

        cache_a.extend(cache_b)
        self.assertEqual(cache_a.keys.shape[0], 5)
        self.assertEqual(cache_a.values.shape[0], 5)
        self.assertEqual(cache_a.offset.tolist(), [6, 7, 6, 1, 4])
        self.assertEqual(cache_a.left_padding.tolist(), [2, 1, 2, 7, 4])

    def test_batch_rotating_kv_cache(self):
        cache = BatchRotatingKVCache(max_size=4, left_padding=[2, 0])
        mask = cache.make_mask(4)
        self.assertFalse(mx.any(mask[0, 0, 0, :]))
        self.assertTrue(
            mx.array_equal(mask[1, 0, 0, :], mx.array([True, False, False, False]))
        )

        # Batch update works
        k, v = mx.zeros((2, 1, 4, 8)), mx.zeros((2, 1, 4, 8))
        k, v = cache.update_and_fetch(k, v)

        mask = cache.make_mask(4)
        k, v = mx.zeros((2, 1, 4, 8)), mx.zeros((2, 1, 4, 8))
        k, v = cache.update_and_fetch(k, v)
        self.assertEqual(mask.shape[-2:], (4, k.shape[2]))
        self.assertEqual(
            mask[0, 0, 0, :].tolist(), [False, True, True, True, False, False, False]
        )

        # Single query update works
        cache = BatchRotatingKVCache(max_size=4, left_padding=[2, 0])
        k, v = mx.zeros((2, 1, 4, 8)), mx.zeros((2, 1, 4, 8))
        k, v = cache.update_and_fetch(k, v)

        mask = cache.make_mask(1)
        k, v = mx.zeros((2, 1, 1, 8)), mx.zeros((2, 1, 1, 8))

        k, v = cache.update_and_fetch(k, v)
        self.assertEqual(mask.shape[-2:], (1, k.shape[2]))
        self.assertEqual(mask[0, 0, 0].tolist(), [True, False, True, True])
        self.assertEqual(mask[1, 0, 0].tolist(), [True, True, True, True])

        # Check filtering
        cache = BatchRotatingKVCache(max_size=4, left_padding=[2, 0, 3])
        k, v = mx.zeros((3, 1, 3, 8)), mx.zeros((3, 1, 3, 8))
        cache.update_and_fetch(k, v)
        cache.filter(mx.array([1]))
        self.assertEqual(cache.keys.shape, (1, 1, 3, 8))

        # Check extend
        cache = BatchRotatingKVCache(max_size=4, left_padding=[2, 1])
        other = BatchRotatingKVCache(max_size=4, left_padding=[2, 2])
        k, v = mx.zeros((2, 1, 5, 8)), mx.zeros((2, 1, 5, 8))
        cache.update_and_fetch(k, v)
        other.update_and_fetch(k, v)
        k, v = mx.zeros((2, 1, 1, 8)), mx.zeros((2, 1, 1, 8))
        cache.update_and_fetch(k, v)
        cache.extend(other)

        # Check mask when going from prompt -> extend -> prompt
        cache = BatchRotatingKVCache(max_size=8, left_padding=[4])
        k, v = mx.zeros((1, 1, 8, 8)), mx.zeros((1, 1, 8, 8))
        cache.update_and_fetch(k, v)

        mask = cache.make_mask(1)
        self.assertEqual(
            mask.squeeze().tolist(), [True, False, False, False, True, True, True, True]
        )

        k, v = mx.zeros((1, 1, 1, 8)), mx.zeros((1, 1, 1, 8))
        cache.update_and_fetch(k, v)

        mask = cache.make_mask(2)
        expected = mx.array(
            [
                [False, False, False, True, True, True, True, True, False],
                [False, False, False, True, True, True, True, True, True],
            ]
        )
        self.assertTrue(mx.array_equal(mask.squeeze(), expected))

    def test_save_load_batch_caches(self):
        cache_file = os.path.join(self.test_dir, "prompt_cache.safetensors")

        cache = [
            ArraysCache(size=2, left_padding=[1, 2]),
            BatchKVCache(left_padding=[1, 2]),
            BatchRotatingKVCache(max_size=10, left_padding=[1, 2]),
        ]
        for c in cache:
            if isinstance(c, ArraysCache):
                c[0] = mx.random.uniform(shape=(4, 4, 4))
                c[1] = mx.random.uniform(shape=(4, 4, 4))
            else:
                x = mx.random.uniform(shape=(4, 4, 7, 4))
                y = mx.random.uniform(shape=(4, 4, 7, 4))
                c.update_and_fetch(x, y)

        save_prompt_cache(cache_file, cache)
        loaded_cache = load_prompt_cache(cache_file)
        left_padding = mx.array([1, 2])
        for c, lc in zip(cache, loaded_cache):
            self.assertTrue(mx.array_equal(c.left_padding, left_padding))
            # the loaded cache, not just the source, must carry the padding
            self.assertTrue(mx.array_equal(lc.left_padding, left_padding))

    def test_rotating_cache_updates(self):
        cache = RotatingKVCache(max_size=8)
        k = v = mx.zeros((1, 1, 10, 1))
        cache.update_and_fetch(k, v)

        for _ in range(3):
            k = v = mx.zeros((1, 1, 1, 1))
            cache.update_and_fetch(k, v)

        k = v = mx.zeros((1, 1, 3, 1))
        k, v = cache.update_and_fetch(k, v)
        self.assertEqual(k.shape[2], 10)
        self.assertEqual(v.shape[2], 10)

    def test_merge_with_empty_caches(self):
        c1 = ArraysCache(2)
        c2 = ArraysCache(2)
        c2[0] = mx.zeros((1, 4))
        c2[1] = mx.zeros((1, 4))
        c_out = ArraysCache.merge((c1, c2))
        self.assertEqual(c_out[0].shape, (2, 4))
        self.assertEqual(c_out[1].shape, (2, 4))

        c1 = KVCache()
        c2 = KVCache()
        kv = mx.zeros((1, 4, 4, 4))
        c2.update_and_fetch(kv, kv)
        c_out = KVCache.merge((c1, c2))
        self.assertEqual(c_out.keys.shape, (2, 4, 4, 4))

        c1 = RotatingKVCache(max_size=4)
        c2 = RotatingKVCache(max_size=4)
        kv = mx.zeros((1, 4, 4, 4))
        c2.update_and_fetch(kv, kv)
        c_out = KVCache.merge((c1, c2))
        self.assertEqual(c_out.keys.shape, (2, 4, 4, 4))

    def test_arrays_cache_advance(self):
        """advance() runs once per layer per decode token and must not
        accumulate per-call graph state; these checks pin the arithmetic it
        performs on left_padding/lengths and its interplay with the other
        ArraysCache operations."""
        cache = ArraysCache(size=1, left_padding=[2, 0])
        cache[0] = mx.zeros((2, 4))
        cache.advance(1)
        self.assertEqual(cache.left_padding.tolist(), [1, -1])
        mask = cache.make_mask(4)
        expected = mx.arange(4)[None] >= mx.array([1, -1])[:, None]
        self.assertTrue(mx.array_equal(mask, expected))
        cache.advance(1)
        cache.advance(1)
        self.assertEqual(cache.left_padding.tolist(), [-1, -3])

        # lengths-based mask after advancing
        cache = ArraysCache(size=1)
        cache.prepare(lengths=[5, 3])
        cache.advance(2)
        self.assertEqual(cache.lengths.tolist(), [3, 1])
        mask = cache.make_mask(4)
        expected = mx.arange(4)[None] < mx.array([3, 1])[:, None]
        self.assertTrue(mx.array_equal(mask, expected))

        # in-place item assignment through the properties must reach the
        # real backing arrays even with a pending advance
        cache = ArraysCache(size=1, left_padding=[2, 0])
        cache.advance(1)
        cache.left_padding[0] = 7
        self.assertEqual(cache.left_padding.tolist(), [7, -1])
        cache = ArraysCache(size=1)
        cache.prepare(lengths=[5, 3])
        cache.advance(2)
        cache.lengths[1] = 9
        self.assertEqual(cache.lengths.tolist(), [3, 9])

        # the batch path arms left_padding even for all-empty merges
        merged = ArraysCache.merge((ArraysCache(1), ArraysCache(1)))
        self.assertEqual(merged.left_padding.tolist(), [0, 0])
        merged[0] = mx.zeros((2, 4))
        merged.advance(1)
        merged.filter(mx.array([1]))
        self.assertEqual(merged.left_padding.tolist(), [-1])

        # extend after both sides have advanced by different amounts
        a = ArraysCache(size=1, left_padding=[1])
        b = ArraysCache(size=1, left_padding=[3])
        a[0] = mx.zeros((1, 4))
        b[0] = mx.zeros((1, 4))
        a.advance(1)
        b.advance(2)
        a.extend(b)
        self.assertEqual(a.left_padding.tolist(), [0, 1])

        # repeated filter/extend churn with no intermediate reads (the
        # continuous-batching pattern) must stay correct; the mutation
        # methods materialize the metadata so batch churn also cannot
        # accumulate an unevaluated graph
        cache = ArraysCache(size=1, left_padding=[1, 2])
        cache[0] = mx.zeros((2, 4))
        for _ in range(50):
            cache.advance(1)
            other = ArraysCache(size=1, left_padding=[3])
            other[0] = mx.zeros((1, 4))
            cache.extend(other)
            cache.filter(mx.array([0, 1]))
        self.assertEqual(cache.left_padding.tolist(), [-49, -48])

        # assigning one field must not lose the other's pending decrement
        cache = ArraysCache(size=1, left_padding=[4])
        cache.lengths = mx.array([6])
        cache.advance(2)
        cache.left_padding = mx.array([9])
        self.assertEqual(cache.lengths.tolist(), [4])
        self.assertEqual(cache.left_padding.tolist(), [9])

        # finalize clears the state and advance becomes a no-op
        cache.finalize()
        self.assertIsNone(cache.left_padding)
        self.assertIsNone(cache.lengths)
        cache.advance(3)
        self.assertIsNone(cache.left_padding)
        self.assertIsNone(cache.make_mask(4))

        # deferred visibility (documented deviation from the eager
        # decrement): an alias shows the pre-advance value until the next
        # read of that field folds it, then observes it -- same object
        cache = ArraysCache(size=1, left_padding=[3])
        alias = cache.left_padding
        cache.advance(1)
        self.assertEqual(alias.tolist(), [3])
        self.assertEqual(cache.left_padding.tolist(), [2])
        self.assertIs(cache.left_padding, alias)
        self.assertEqual(alias.tolist(), [2])

        # make_mask() also folds the field it uses, so aliases observe
        # decrements after mask building too
        cache = ArraysCache(size=1, left_padding=[3])
        alias = cache.left_padding
        cache.advance(1)
        cache.make_mask(4)
        self.assertEqual(alias.tolist(), [2])

        # replacing a field with a pending decrement folds the outgoing
        # array first, so earlier aliases still observe the decrement
        cache = ArraysCache(size=1, left_padding=[3])
        alias = cache.left_padding
        cache.advance(1)
        cache.left_padding = mx.array([9])
        self.assertEqual(alias.tolist(), [2])
        self.assertEqual(cache.left_padding.tolist(), [9])

        # finalize() folds before clearing, with the same guarantee
        cache = ArraysCache(size=1, left_padding=[3])
        alias = cache.left_padding
        cache.advance(1)
        cache.finalize()
        self.assertIsNone(cache.left_padding)
        self.assertEqual(alias.tolist(), [2])

        # storing one array object in both fields: each field applies its
        # decrement at its own fold (documented deviation from stock's
        # eager double decrement at advance() time)
        shared = mx.array([5])
        cache = ArraysCache(size=1)
        cache.left_padding = shared
        cache.lengths = shared
        cache.advance(1)
        self.assertEqual(cache.lengths.tolist(), [4])
        self.assertEqual(cache.left_padding.tolist(), [3])

        # White-box guards for the leak fix itself: advance() rejects
        # mx.array and float offsets loudly (coercion would eval a lazy
        # scalar, truncate floats, and accept non-scalars), and it must
        # never rebind the stored arrays -- the per-call lazy rebind was
        # the leak.
        cache = ArraysCache(size=1, left_padding=[2, 0])
        stored = cache._left_padding
        for bad in (mx.array(1), mx.array([1]), mx.array([[1]]), 1.5):
            with self.assertRaises(TypeError):
                cache.advance(bad)
        cache.advance(1)
        cache.advance(1)
        self.assertIsInstance(cache._lp_advance, int)
        self.assertEqual(cache._lp_advance, 2)
        self.assertIs(cache._left_padding, stored)
        self.assertEqual(cache.left_padding.tolist(), [0, -2])

        # validation is state-independent: unarmed and finalized caches
        # reject the same arguments
        for bad in (mx.array(1), mx.array([1]), 1.5):
            with self.assertRaises(TypeError):
                ArraysCache(size=1).advance(bad)
        finalized = ArraysCache(size=1, left_padding=[1])
        finalized.finalize()
        with self.assertRaises(TypeError):
            finalized.advance(mx.array(1))

        # numpy integer scalars normalize through operator.index --
        # value preserved (documented: stock's array-promotion dtype
        # side effects for numpy offsets are not reproduced)
        cache = ArraysCache(size=1, left_padding=[5])
        cache.advance(np.int64(2))
        cache.advance(np.uint8(1))
        self.assertEqual(cache.left_padding.tolist(), [2])
        self.assertIsInstance(cache._lp_advance, int)

        # a discarded metadata read must not accumulate unevaluated folds:
        # each fold is applied in place and scheduled for evaluation
        # (regression: fold-on-read chained one 4-byte scalar per read)
        cache = ArraysCache(size=1)
        cache.prepare(lengths=[1])
        cache.lengths
        mx.synchronize()
        base_mem = mx.get_active_memory()
        for _ in range(2048):
            cache.advance(1)
            cache.lengths
        mx.synchronize()
        self.assertLess(mx.get_active_memory() - base_mem, 4096)

        # discarded make_mask() results must not accumulate unevaluated
        # folds either
        cache = ArraysCache(size=1, left_padding=[0])
        cache.make_mask(1)
        mx.synchronize()
        base_mem = mx.get_active_memory()
        for _ in range(2048):
            cache.advance(1)
            cache.make_mask(1)
        mx.synchronize()
        self.assertLess(mx.get_active_memory() - base_mem, 4096)

        # repeated public mutations must not accumulate unevaluated graph
        # either: item assignment through the property and reassignment of
        # lazy expressions are both scheduled for evaluation
        cache = ArraysCache(size=1, left_padding=[0])
        mx.synchronize()
        base_mem = mx.get_active_memory()
        for i in range(2048):
            cache.left_padding[0] = i
        mx.synchronize()
        self.assertLess(mx.get_active_memory() - base_mem, 4096)
        self.assertEqual(cache.left_padding.tolist(), [2047])

        cache = ArraysCache(size=1, left_padding=[0])
        mx.synchronize()
        base_mem = mx.get_active_memory()
        for _ in range(2048):
            cache.left_padding = cache.left_padding - 1
        mx.synchronize()
        self.assertLess(mx.get_active_memory() - base_mem, 4096)
        self.assertEqual(cache.left_padding.tolist(), [-2048])

        # with both fields populated, consuming only one must not rebind
        # the other's backing array (a rebind per read is an unbounded
        # lazy chain when that field is never read)
        cache = ArraysCache(size=1, left_padding=[2, 1])
        cache.prepare(lengths=[5, 3])
        stored = cache._left_padding
        for _ in range(5):
            cache.advance(1)
            cache.lengths
        self.assertIs(cache._left_padding, stored)
        self.assertEqual(cache._lp_advance, 5)
        self.assertEqual(cache.lengths.tolist(), [0, -2])
        self.assertEqual(cache.left_padding.tolist(), [-3, -4])

        # make_mask() straight after advance() (no property read in
        # between) must apply the pending offset; it folds the field it
        # uses in place (metadata-side subtraction, identical overflow
        # behavior to stock) and leaves the other field's counter alone
        cache = ArraysCache(size=1, left_padding=[2, 0])
        stored = cache._left_padding
        cache.advance(1)
        mask = cache.make_mask(4)
        self.assertEqual(cache._lp_advance, 0)
        self.assertIs(cache._left_padding, stored)
        expected = mx.arange(4)[None] >= mx.array([1, -1])[:, None]
        self.assertTrue(mx.array_equal(mask, expected))

        cache = ArraysCache(size=1)
        cache.prepare(lengths=[5, 3])
        cache.advance(2)
        mask = cache.make_mask(4)
        self.assertEqual(cache._len_advance, 0)
        expected = mx.arange(4)[None] < mx.array([3, 1])[:, None]
        self.assertTrue(mx.array_equal(mask, expected))

        # integer totals that overflow the dtype fold to stock's modular
        # result: sequential in-range subtractions wrap, and the deferred
        # fold must land on the same value (regression: the raw coalesced
        # total was rejected by MLX's scalar conversion)
        cache = ArraysCache(size=1)
        cache.left_padding = mx.array([0], dtype=mx.int32)
        cache.advance(2**31 - 1)
        cache.advance(1)
        self.assertEqual(cache.left_padding.tolist(), [-(2**31)])

        cache = ArraysCache(size=1)
        cache.lengths = mx.array([0], dtype=mx.int64)
        cache.advance(2**62)
        cache.advance(2**62)
        self.assertEqual(cache.lengths.tolist(), [-(2**63)])

        # sub-32-bit metadata wraps by the same modular rule
        cache = ArraysCache(size=1)
        cache.lengths = mx.array([0], dtype=mx.int8)
        for _ in range(3):
            cache.advance(100)
        self.assertEqual(cache.lengths.tolist(), [-44])

        cache = ArraysCache(size=1)
        cache.lengths = mx.array([5], dtype=mx.uint8)
        cache.advance(200)
        cache.advance(200)
        self.assertEqual(cache.lengths.tolist(), [117])

        # uint64 metadata: MLX converts python int scalars through
        # int64, so stock accepts negatives (wrapping modulo 2**64) and
        # rejects >= 2**63; the fold uses the congruent *signed*
        # representative so accumulated totals stay convertible
        cache = ArraysCache(size=1)
        cache.lengths = mx.array([0], dtype=mx.uint64)
        cache.advance(2**62)
        cache.advance(2**62)
        self.assertEqual(cache.lengths.tolist(), [2**63])
        cache = ArraysCache(size=1)
        cache.lengths = mx.array([1], dtype=mx.uint64)
        cache.advance(-1)
        self.assertEqual(cache.lengths.tolist(), [2])

        # the int64 scalar gate applies to floating metadata too (stock
        # rejects the python int before looking at the array dtype);
        # accumulated floating totals beyond the gate fold in gate-sized
        # chunks instead of failing the scalar conversion
        cache = ArraysCache(size=1)
        cache.lengths = mx.array([0.0], dtype=mx.float32)
        cache.advance(2**62)
        cache.advance(2**62)
        self.assertEqual(cache.lengths.tolist(), [float(-(2**63))])

        # a floating total spanning several gate-sized chunks folds to
        # the deterministic chunked value
        cache = ArraysCache(size=1)
        cache.lengths = mx.array([0.0], dtype=mx.float32)
        cache.advance(2**63 - 1)
        cache.advance(2**63 - 1)
        self.assertEqual(cache.lengths.tolist(), [float(-(2**64))])

        # a many-chunk fold schedules each chunk as it applies -- the
        # fold itself must not rebuild an unbounded unevaluated chain
        cache = ArraysCache(size=1)
        cache.lengths = mx.array([0.0], dtype=mx.float32)
        cache.lengths
        mx.synchronize()
        base_mem = mx.get_active_memory()
        for _ in range(2048):
            cache.advance(2**63 - 1)
        cache.lengths
        mx.synchronize()
        self.assertLess(mx.get_active_memory() - base_mem, 4096)

        # offsets outside the dtype's scalar range raise at the
        # advance() call, where stock's eager subtraction rejected them
        # (uniformly ValueError here; stock's int64-gate rejection
        # surfaced as an opaque std::bad_cast)
        cache = ArraysCache(size=1)
        cache.left_padding = mx.array([0], dtype=mx.int32)
        with self.assertRaises(ValueError):
            cache.advance(2**31)
        with self.assertRaises(ValueError):
            cache.advance(-(2**31) - 1)
        cache = ArraysCache(size=1)
        cache.lengths = mx.array([0], dtype=mx.uint16)
        with self.assertRaises(ValueError):
            cache.advance(-1)
        cache = ArraysCache(size=1)
        cache.lengths = mx.array([0], dtype=mx.uint64)
        with self.assertRaises(ValueError):
            cache.advance(2**63)
        cache = ArraysCache(size=1)
        cache.lengths = mx.array([0.0], dtype=mx.float32)
        with self.assertRaises(ValueError):
            cache.advance(2**63)

        # the range check is per armed field in stock's field order: an
        # offset valid for lengths but not for left_padding leaves the
        # lengths decrement applied, exactly like stock's sequential
        # eager subtractions did
        cache = ArraysCache(size=1)
        cache.lengths = mx.array([0], dtype=mx.int64)
        cache.left_padding = mx.array([0], dtype=mx.int32)
        with self.assertRaises(ValueError):
            cache.advance(2**31)
        self.assertEqual(cache.lengths.tolist(), [-(2**31)])
        self.assertEqual(cache.left_padding.tolist(), [0])

        # a shallow copy shares the backing arrays (as stock's does);
        # folding before copying prevents the pending decrement from
        # being applied once per object to the shared array
        cache = ArraysCache(size=1)
        cache.lengths = mx.array([5])
        cache.advance(1)
        cloned = copy.copy(cache)
        self.assertEqual(cache.lengths.tolist(), [4])
        self.assertEqual(cloned.lengths.tolist(), [4])
        self.assertIs(cloned._lengths, cache._lengths)
        self.assertIs(cloned._len_state, cache._len_state)
        # post-copy advances accumulate in shared field state, so either
        # alias folds the total exactly once into their shared array
        cloned.advance(2)
        cache.advance(1)
        self.assertEqual(cloned._len_advance, 3)
        self.assertEqual(cache.lengths.tolist(), [1])
        self.assertEqual(cloned.lengths.tolist(), [1])

        # deepcopy also folds first; the copy is an independent object
        # with its own counters and lock
        cache = ArraysCache(size=1)
        cache.lengths = mx.array([5])
        cache.advance(1)
        deep = copy.deepcopy(cache)
        self.assertEqual(deep.lengths.tolist(), [4])
        self.assertEqual(deep._len_advance, 0)
        self.assertIsNot(deep._lengths, cache._lengths)
        self.assertIsNot(deep._fold_lock, cache._fold_lock)
        deep.advance(2)
        self.assertEqual(deep.lengths.tolist(), [2])
        self.assertEqual(cache.lengths.tolist(), [4])

        # bool metadata -- which stock's arithmetic silently promotes to
        # int32 -- keeps its dtype through zero-net advance histories
        # (documented deviation); a nonzero net decrement promotes on
        # fold at int32 precision with stock's values
        cache = ArraysCache(size=1)
        cache.left_padding = mx.array([True], dtype=mx.bool_)
        cache.advance(0)
        self.assertEqual(cache.left_padding.dtype, mx.bool_)
        self.assertEqual(cache.left_padding.tolist(), [True])
        cache.advance(2)
        cache.advance(-1)
        self.assertEqual(cache.left_padding.dtype, mx.int32)
        self.assertEqual(cache.left_padding.tolist(), [0])

        # bool offsets are normalized by operator.index (documented:
        # stock's bool-scalar subtraction kept bool metadata bool)
        cache = ArraysCache(size=1)
        cache.left_padding = mx.array([False], dtype=mx.bool_)
        cache.advance(True)
        self.assertEqual(cache.left_padding.dtype, mx.int32)
        self.assertEqual(cache.left_padding.tolist(), [-1])

        # after the first advance() on bool metadata, later offsets
        # validate against the promoted int32 dtype, exactly as stock's
        # already-promoted stored array would
        cache = ArraysCache(size=1)
        cache.lengths = mx.array([True], dtype=mx.bool_)
        cache.advance(1)
        with self.assertRaises(ValueError):
            cache.advance(2**31)
        self.assertEqual(cache.lengths.tolist(), [0])

        # ... including zero and cancelling histories: stock promoted
        # the stored array on the first subtraction regardless of the
        # offset's value
        cache = ArraysCache(size=1)
        cache.lengths = mx.array([True], dtype=mx.bool_)
        cache.advance(1)
        cache.advance(-1)
        with self.assertRaises(ValueError):
            cache.advance(2**31)
        cache = ArraysCache(size=1)
        cache.lengths = mx.array([True], dtype=mx.bool_)
        cache.advance(0)
        with self.assertRaises(ValueError):
            cache.advance(2**31)

        # assigning a fresh bool array resets the promotion (stock's new
        # array had not been subtracted from yet); the first offset then
        # passes only the int64 gate, as stock's bool conversion did
        cache.lengths = mx.array([True], dtype=mx.bool_)
        cache.advance(2**31)
        self.assertEqual(cache.lengths.tolist(), [-2147483647])

        # re-assigning the SAME backing array is not a reset: the flag is
        # the only record of stock's physical promotion
        cache = ArraysCache(size=1)
        cache.lengths = mx.array([True], dtype=mx.bool_)
        cache.advance(0)
        cache.lengths = cache.lengths
        with self.assertRaises(ValueError):
            cache.advance(2**31)

        # a bool array shared by both fields promotes both: stock's
        # in-place promotion of the shared object made the second
        # field's subtraction range-check as int32
        shared = mx.array([True], dtype=mx.bool_)
        cache = ArraysCache(size=1)
        cache.left_padding = shared
        cache.lengths = shared
        with self.assertRaises(ValueError):
            cache.advance(2**31)
        self.assertEqual(cache._len_advance, 2**31)
        self.assertEqual(cache._lp_advance, 0)

        # shallow aliases share the bool-promotion record as well as the
        # backing array: once either alias has logically promoted bool to
        # int32, all aliases validate future offsets against int32
        cache = ArraysCache(size=1)
        cache.lengths = mx.array([True], dtype=mx.bool_)
        cloned = copy.copy(cache)
        cache.advance(0)
        with self.assertRaises(ValueError):
            cloned.advance(2**31)

        # extend must materialize a logical bool->int32 promotion before
        # mixed-dtype concatenation. Stock concatenates int32 with int8,
        # accepts 128, and produces this value.
        cache = ArraysCache(size=1)
        cache.lengths = mx.array([True], dtype=mx.bool_)
        cache.advance(0)
        other = ArraysCache(size=1)
        other.lengths = mx.array([1], dtype=mx.int8)
        cache.extend(other)
        self.assertEqual(cache.lengths.dtype, mx.int32)
        cache.advance(128)
        self.assertEqual(cache.lengths.tolist(), [-127, -127])

        # batch churn with the metadata never read must not accumulate
        # unevaluated gathers: filter/extend materialize the metadata.
        # The state slots are consumed every forward pass (mx.eval below)
        # but most layers' metadata never is
        cache = ArraysCache(size=1, left_padding=[1, 2])
        cache.prepare(lengths=[5, 6])
        cache[0] = mx.zeros((2, 4))
        mx.eval(cache[0], cache._left_padding, cache._lengths)
        mx.synchronize()
        base_mem = mx.get_active_memory()
        for _ in range(1024):
            cache.filter(mx.array([0, 1]))
            mx.eval(cache[0])
        mx.synchronize()
        self.assertLess(mx.get_active_memory() - base_mem, 4096)

    def test_arrays_cache_compile_captured_reads(self):
        """mx.compile captures state by temporarily swapping tracers into
        the captured containers; a metadata read during tracing must stay
        pure -- no fold, no async_eval (disallowed on tracers), counters
        intact -- exactly like stock's plain attribute access. The
        captured array observes a pending decrement at the next eager
        fold of the field (the documented deferred-visibility window)."""
        cache = ArraysCache(size=1, left_padding=[4])
        cache.advance(1)

        def read(x):
            return x + cache.left_padding

        compiled = mx.compile(read, inputs=vars(cache))
        out = compiled(mx.array([0]))
        # the traced read did not fold: pending decrement intact, the
        # pre-advance value used
        self.assertEqual(out.tolist(), [4])
        self.assertEqual(cache._lp_advance, 1)
        # an eager fold lands the decrement in place; the compiled
        # function reads the captured array's current value from then on
        self.assertEqual(cache.left_padding.tolist(), [3])
        self.assertEqual(compiled(mx.array([0])).tolist(), [3])

        # make_mask inside a trace behaves the same way
        cache = ArraysCache(size=1, left_padding=[1])
        cache.advance(1)

        def mask(x):
            return cache.make_mask(x.shape[-1])

        compiled = mx.compile(mask, inputs=vars(cache))
        m = compiled(mx.zeros((1, 2)))
        self.assertEqual(m.tolist(), [[False, True]])
        self.assertEqual(cache._lp_advance, 1)
        self.assertEqual(cache.left_padding.tolist(), [0])

    def test_arrays_cache_closure_captured_reads(self):
        """Purity during tracing applies only when the read encounters a
        tracer (metadata swapped in by mx.compile(..., inputs=...)). A
        closure-captured cache holds concrete arrays, so a read inside a
        trace folds eagerly -- producing exactly the values stock's
        eager decrement produced."""
        cache = ArraysCache(size=1, left_padding=[4])
        cache.advance(1)
        out = mx.compile(lambda x: x + cache.left_padding)(mx.array([0]))
        self.assertEqual(out.tolist(), [3])
        self.assertEqual(cache._lp_advance, 0)
        self.assertEqual(cache.left_padding.tolist(), [3])

        cache = ArraysCache(size=1, left_padding=[4])
        cache.advance(1)
        out = mx.vmap(lambda x: x + cache.left_padding)(mx.zeros((2, 1)))
        self.assertEqual(out.tolist(), [[3.0], [3.0]])
        self.assertEqual(cache._lp_advance, 0)
        self.assertEqual(cache.left_padding.tolist(), [3])

    def test_arrays_cache_compiled_advance_not_recorded(self):
        """Documented limitation: advance() is host-side bookkeeping and
        runs at trace time only, so a compiled function that calls it
        does not record the decrement in the traced graph -- call
        advance() outside compiled functions. (Stock incidentally
        recorded it as lazy graph arithmetic; that arithmetic is the
        leak this class exists to avoid.) The trace-time call itself is
        not lost: it stays pending and folds at the next eager read."""
        cache = ArraysCache(size=1, left_padding=[5])

        def step(x):
            cache.advance(1)
            return x + cache.left_padding

        compiled = mx.compile(step, inputs=vars(cache), outputs=vars(cache))
        outs = [compiled(mx.array([0])).item() for _ in range(3)]
        self.assertEqual(outs, [5, 5, 5])
        self.assertEqual(cache._lp_advance, 1)
        self.assertEqual(cache.left_padding.tolist(), [4])

    def test_arrays_cache_captured_mutation_guard(self):
        """Replacing metadata inside a graph transformation while a
        decrement is pending raises instead of silently discarding the
        decrement (the fold cannot land in a tracer). Only the raise and
        the preserved counter are asserted: an exception inside a traced
        function leaves the captured containers mid-swap -- an MLX-level
        property of compile, not specific to this class -- so the cache
        is not usable afterwards."""
        cache = ArraysCache(size=1, left_padding=[3])
        cache.advance(1)

        def replace(x):
            cache.left_padding = x
            return x

        compiled = mx.compile(replace, inputs=vars(cache))
        with self.assertRaises(RuntimeError):
            compiled(mx.array([9]))
        self.assertEqual(cache._lp_advance, 1)

        # copy.copy and copy.deepcopy guard the same way: a copy taken
        # mid-trace would escape holding the transient tracer
        for copier in (copy.copy, copy.deepcopy):
            cache = ArraysCache(size=1, left_padding=[3])
            cache.advance(1)

            def clone(x):
                copier(cache)
                return x

            compiled = mx.compile(clone, inputs=vars(cache))
            with self.assertRaises(RuntimeError):
                compiled(mx.array([0]))
            self.assertEqual(cache._lp_advance, 1)

    def test_arrays_cache_captured_item_assignment(self):
        """Documented window: in-place item assignment through a captured
        tracer read cannot fold first (__setitem__ on the returned array
        bypasses the property machinery and the mutation guard), so the
        write lands before the deferred decrement -- the alias-write
        window. Stock applied the decrement eagerly, so its assignment
        landed after it."""
        cache = ArraysCache(size=1, left_padding=[3])
        cache.advance(1)

        def mutate(x):
            cache.left_padding[0] = x[0]
            return cache.left_padding

        compiled = mx.compile(mutate, inputs=vars(cache), outputs=vars(cache))
        out = compiled(mx.array([9]))
        self.assertEqual(out.tolist(), [9])
        self.assertEqual(cache._lp_advance, 1)
        self.assertEqual(cache.left_padding.tolist(), [8])  # stock: [9]

    def test_arrays_cache_fold_failure_keeps_remainder(self):
        """The fold counter commits chunk-by-chunk: a failure mid-fold
        (injected into the intermediate scheduling call) leaves exactly
        the unapplied remainder pending -- never double-applied, never
        lost -- and a later fold completes it."""
        cache = ArraysCache(size=1)
        cache.lengths = mx.array([0.0], dtype=mx.float32)
        for _ in range(3):
            cache.advance(2**63 - 1)

        real_async_eval = mx.async_eval
        calls = [0]

        def fail_second(*args):
            calls[0] += 1
            if calls[0] == 2:
                raise RuntimeError("injected scheduling failure")
            return real_async_eval(*args)

        mx.async_eval = fail_second
        try:
            with self.assertRaises(RuntimeError):
                cache.lengths
        finally:
            mx.async_eval = real_async_eval
        # one chunk committed, two remain pending
        self.assertEqual(cache._len_advance, 2 * (2**63 - 1))
        # the next fold applies exactly the remainder
        self.assertEqual(cache.lengths.tolist(), [float(-(3 * 2**63))])
        self.assertEqual(cache._len_advance, 0)

    def test_arrays_cache_concurrent_reads(self):
        """Stock reads were pure attribute access and therefore trivially
        safe from multiple threads; deferred folding makes reads mutate,
        so the fold machinery is serialized per cache (unserialized
        concurrent folds of one array deadlock on their thread-local
        streams). Two concurrent readers racing one pending decrement
        must both complete and observe the folded value exactly once."""
        cache = ArraysCache(size=1)
        cache.lengths = mx.array([5])
        mx.eval(cache.lengths)
        cache.advance(1)

        results = []
        gate = threading.Barrier(3)

        def read():
            gate.wait()
            results.append(cache.lengths.tolist())

        threads = [threading.Thread(target=read, daemon=True) for _ in range(2)]
        for t in threads:
            t.start()
        gate.wait()
        for t in threads:
            t.join(timeout=60)
        self.assertTrue(all(not t.is_alive() for t in threads))
        self.assertEqual(results, [[4], [4]])
        self.assertEqual(cache._len_advance, 0)

    def test_arrays_cache_alias_copy_state(self):
        """Shallow aliases share pending state and their fold lock. A
        deepcopy of an alias graph recreates that topology with an
        independent array/lock group."""

        def concurrent_property_reads(caches):
            gate = threading.Barrier(3)
            arrays = []
            errors = []

            def read(cache):
                try:
                    gate.wait()
                    arrays.append(cache.lengths)
                except Exception as e:
                    errors.append(e)

            threads = [
                threading.Thread(target=read, args=(cache,), daemon=True)
                for cache in caches
            ]
            for thread in threads:
                thread.start()
            gate.wait()
            for thread in threads:
                thread.join(timeout=60)
            self.assertTrue(all(not thread.is_alive() for thread in threads))
            self.assertEqual(errors, [])
            mx.eval(*arrays)
            self.assertEqual([array.tolist() for array in arrays], [[3], [3]])

        cache = ArraysCache(size=1)
        cache.lengths = mx.array([5])
        aliases = [cache, copy.copy(cache)]
        self.assertIs(aliases[0]._lengths, aliases[1]._lengths)
        self.assertIs(aliases[0]._len_state, aliases[1]._len_state)
        self.assertIs(aliases[0]._fold_lock, aliases[1]._fold_lock)
        for alias in aliases:
            alias.advance(1)
        self.assertEqual(aliases[0]._len_advance, 2)
        concurrent_property_reads(aliases)
        self.assertEqual(aliases[0]._len_advance, 0)
        self.assertEqual(aliases[1]._len_advance, 0)

        cache = ArraysCache(size=1)
        cache.lengths = mx.array([5])
        original = [cache, copy.copy(cache)]
        aliases = copy.deepcopy(original)
        self.assertIs(aliases[0]._lengths, aliases[1]._lengths)
        self.assertIs(aliases[0]._len_state, aliases[1]._len_state)
        self.assertIs(aliases[0]._fold_lock, aliases[1]._fold_lock)
        self.assertIsNot(aliases[0]._lengths, original[0]._lengths)
        self.assertIsNot(aliases[0]._fold_lock, original[0]._fold_lock)
        for alias in aliases:
            alias.advance(1)
        concurrent_property_reads(aliases)

    def test_arrays_cache_pickle(self):
        """The deferred state round-trips without trying to pickle a
        native RLock, including the sharing topology of alias graphs."""
        cache = ArraysCache(size=1)
        cache.lengths = mx.array([5])
        cache.advance(2)
        restored = pickle.loads(pickle.dumps(cache))
        self.assertEqual(restored._len_advance, 2)
        self.assertEqual(restored.lengths.tolist(), [3])

        cache = ArraysCache(size=1)
        cache.lengths = mx.array([5])
        aliases = [cache, copy.copy(cache)]
        cache.advance(1)
        restored = pickle.loads(pickle.dumps(aliases))
        self.assertIs(restored[0]._lengths, restored[1]._lengths)
        self.assertIs(restored[0]._len_state, restored[1]._len_state)
        self.assertIs(restored[0]._fold_lock, restored[1]._fold_lock)
        self.assertEqual(restored[0].lengths.tolist(), [4])
        self.assertEqual(restored[1].lengths.tolist(), [4])

    def test_arrays_cache_metadata_serialization_is_atomic(self):
        """advance() cannot interleave between the two encoded fields."""
        cache = ArraysCache(size=1, left_padding=[10])
        cache.lengths = mx.array([20])
        cache.advance(1)
        left_folded = threading.Event()
        resume = threading.Event()
        advance_started = threading.Event()
        advance_done = threading.Event()
        serialized = []
        errors = []
        real_fold_left = cache._fold_lp

        def pause_after_left_fold():
            real_fold_left()
            left_folded.set()
            if not resume.wait(timeout=60):
                raise TimeoutError("serialization test did not resume")

        def serialize():
            try:
                serialized.append(cache.meta_state)
            except Exception as e:
                errors.append(e)

        def advance():
            advance_started.set()
            cache.advance(1)
            advance_done.set()

        cache._fold_lp = pause_after_left_fold
        save_thread = threading.Thread(target=serialize, daemon=True)
        save_thread.start()
        self.assertTrue(left_folded.wait(timeout=60))
        advance_thread = threading.Thread(target=advance, daemon=True)
        advance_thread.start()
        self.assertTrue(advance_started.wait(timeout=60))
        advance_done.wait(timeout=0.1)
        resume.set()
        save_thread.join(timeout=60)
        advance_thread.join(timeout=60)

        self.assertFalse(save_thread.is_alive())
        self.assertFalse(advance_thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(serialized, [("int32:9", "int32:19")])
        self.assertEqual(cache.left_padding.tolist(), [8])
        self.assertEqual(cache.lengths.tolist(), [18])

    def test_arrays_cache_metadata_serialization(self):
        cache_file = os.path.join(self.test_dir, "arrays_cache_meta.safetensors")

        # metadata round-trips, folding pending advances at save time
        cache = ArraysCache(size=1, left_padding=[3, 1])
        cache[0] = mx.zeros((2, 2))
        cache.prepare(lengths=[5, 4])
        cache.advance(2)
        save_prompt_cache(cache_file, [cache])
        loaded = load_prompt_cache(cache_file)[0]
        self.assertEqual(loaded.left_padding.tolist(), [1, -1])
        self.assertEqual(loaded.lengths.tolist(), [3, 2])
        mask = loaded.make_mask(4)
        expected = mx.arange(4)[None] >= mx.array([1, -1])[:, None]
        self.assertTrue(mx.array_equal(mask, expected))

        # absent fields stay absent through a round trip
        cache = ArraysCache(size=1)
        cache[0] = mx.zeros((1, 2))
        save_prompt_cache(cache_file, [cache])
        loaded = load_prompt_cache(cache_file)[0]
        self.assertIsNone(loaded.left_padding)
        self.assertIsNone(loaded.lengths)

        # dtype survives the round trip
        cache = ArraysCache(size=1)
        cache[0] = mx.zeros((1, 2))
        cache.lengths = mx.array([5], dtype=mx.int64)
        save_prompt_cache(cache_file, [cache])
        loaded = load_prompt_cache(cache_file)[0]
        self.assertEqual(loaded.lengths.dtype, mx.int64)
        self.assertEqual(loaded.lengths.tolist(), [5])

        # the codec keeps an empty metadata array distinct from an absent
        # field (zero-row caches cannot go through save_prompt_cache --
        # safetensors rejects empty state arrays -- so this is pinned at
        # the meta_state level)
        cache = ArraysCache(size=1, left_padding=[1])
        cache[0] = mx.zeros((1, 2))
        cache.filter(mx.array([], dtype=mx.int32))
        restored = ArraysCache(size=1)
        restored.meta_state = cache.meta_state
        self.assertIsNotNone(restored.left_padding)
        self.assertEqual(restored.left_padding.size, 0)
        self.assertIsNone(restored.lengths)

        # float metadata round-trips through the codec
        cache = ArraysCache(size=1)
        cache.lengths = mx.array([2.5], dtype=mx.float32)
        restored = ArraysCache(size=1)
        restored.meta_state = cache.meta_state
        self.assertEqual(restored.lengths.dtype, mx.float32)
        self.assertEqual(restored.lengths.tolist(), [2.5])

        # malformed or truncated entries raise instead of loading as
        # empty metadata; dtype names are whitelisted
        restored = ArraysCache(size=1)
        for bad in (("int32", ""), ("nosuchdtype:1", ""), ("eval:1", "")):
            with self.assertRaises(ValueError):
                restored.meta_state = bad

        # non-numeric metadata dtypes fail loudly at save time
        cache = ArraysCache(size=1)
        cache.lengths = mx.array([True], dtype=mx.bool_)
        with self.assertRaises(TypeError):
            cache.meta_state

        # non-1-D metadata fails loudly at save time too, rather than
        # emitting an entry the decoder cannot parse (a 0-d filter index
        # is the public route to scalar metadata)
        cache = ArraysCache(size=1, left_padding=[1])
        cache[0] = mx.zeros((1, 2))
        cache.filter(mx.array(0))
        with self.assertRaises(ValueError):
            cache.meta_state
        cache = ArraysCache(size=1)
        cache.lengths = mx.array([[1, 2]])
        with self.assertRaises(ValueError):
            cache.meta_state

    def test_extend_with_empty_and_nonempty_batch_caches(self):
        """Extending a batch cache when one side has keys=None should use the
        correct batch size for the placeholder, not the batch size from the
        non-None side. Regression test for broadcast error in dynamic_roll."""
        H, D = 8, 64
        max_size = 512

        # -- BatchRotatingKVCache --
        # Create 2 caches with content and 3 empty caches
        c1 = RotatingKVCache(max_size=max_size)
        c2 = RotatingKVCache(max_size=max_size)
        c1.update_and_fetch(mx.ones((1, H, 5, D)), mx.ones((1, H, 5, D)))
        c2.update_and_fetch(mx.ones((1, H, 3, D)), mx.ones((1, H, 3, D)))
        batch_full = BatchRotatingKVCache.merge([c1, c2])

        empty_caches = [RotatingKVCache(max_size=max_size) for _ in range(3)]
        batch_empty = BatchRotatingKVCache.merge(empty_caches)

        # Extend non-empty with empty (different batch sizes)
        batch_full.extend(batch_empty)
        self.assertEqual(batch_full.keys.shape[0], 5)
        self.assertEqual(batch_full.offset.shape[0], 5)

        # Prompt processing with right padding should not crash
        batch_full.prepare(lengths=[10, 8, 12, 7, 11], right_padding=[2, 4, 0, 5, 1])
        new_kv = mx.ones((5, H, 12, D))
        batch_full.update_and_fetch(new_kv, new_kv)

        # Also test empty extending non-empty
        batch_full2 = BatchRotatingKVCache.merge(
            [RotatingKVCache(max_size=max_size) for _ in range(3)]
        )
        c3 = RotatingKVCache(max_size=max_size)
        c4 = RotatingKVCache(max_size=max_size)
        c3.update_and_fetch(mx.ones((1, H, 4, D)), mx.ones((1, H, 4, D)))
        c4.update_and_fetch(mx.ones((1, H, 6, D)), mx.ones((1, H, 6, D)))
        batch_content = BatchRotatingKVCache.merge([c3, c4])
        batch_full2.extend(batch_content)
        self.assertEqual(batch_full2.keys.shape[0], 5)
        self.assertEqual(batch_full2.offset.shape[0], 5)

        # -- BatchKVCache --
        c1 = KVCache()
        c2 = KVCache()
        c1.update_and_fetch(mx.ones((1, H, 5, D)), mx.ones((1, H, 5, D)))
        c2.update_and_fetch(mx.ones((1, H, 3, D)), mx.ones((1, H, 3, D)))
        batch_full = BatchKVCache.merge([c1, c2])

        empty_caches = [KVCache() for _ in range(3)]
        batch_empty = BatchKVCache.merge(empty_caches)

        batch_full.extend(batch_empty)
        self.assertEqual(batch_full.keys.shape[0], 5)
        self.assertEqual(batch_full.offset.shape[0], 5)

    def test_arrays_cache_extend_with_empty(self):
        # test simple merge
        c1 = ArraysCache(2)
        c2 = ArraysCache(2)
        c1[0] = mx.zeros((1, 4, 8))
        c1[1] = mx.zeros((1, 4))
        c2[0] = mx.zeros((1, 4, 8))
        c2[1] = mx.zeros((1, 4))
        full = ArraysCache.merge((c1, c2))
        self.assertEqual(full[0].shape, (2, 4, 8))

        # extend with empty
        empty = ArraysCache.merge((ArraysCache(2),))
        full.extend(empty)
        self.assertEqual(full[0].shape, (3, 4, 8))
        self.assertEqual(full[1].shape, (3, 4))
        self.assertTrue(mx.all(full[0][2:] == 0))

        # making an empty cache with 2 sequences and merging it with
        # another one with 2 sequences
        empty2 = ArraysCache.merge((ArraysCache(2), ArraysCache(2)))
        content = ArraysCache.merge((c1, c2))
        empty2.extend(content)
        self.assertEqual(empty2[0].shape, (4, 4, 8))
        self.assertEqual(empty2[1].shape, (4, 4))

        # Extend content with empty
        content = ArraysCache.merge((c1, c2))
        empty2 = ArraysCache.merge((ArraysCache(2), ArraysCache(2)))
        content.extend(empty2)
        self.assertEqual(content[0].shape, (4, 4, 8))
        self.assertEqual(content[1].shape, (4, 4))
        self.assertEqual(content.make_mask(10).shape, (4, 10))

        # multiple empty extensions accumulate correctly
        stepwise = ArraysCache.merge((c1,))
        stepwise.extend(ArraysCache(2))
        stepwise.extend(ArraysCache.merge((ArraysCache(2), ArraysCache(2))))
        self.assertEqual(stepwise[0].shape, (4, 4, 8))
        self.assertEqual(stepwise[1].shape, (4, 4))

    def test_window_mask_with_full_kv_cache(self):
        c = KVCache()
        kv = mx.zeros((1, 1, 32, 128))
        c.update_and_fetch(kv, kv)

        h = mx.zeros((1, 1, 1, 128))
        mask = create_attention_mask(h, c, window_size=4)
        expected = create_causal_mask(1, offset=32, window_size=4)
        self.assertTrue(mx.array_equal(mask, expected))


if __name__ == "__main__":
    unittest.main()
