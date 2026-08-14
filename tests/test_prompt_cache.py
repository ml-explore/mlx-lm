# Copyright © 2024 Apple Inc.

import copy
import os
import tempfile
import unittest
from unittest import mock

import mlx.core as mx

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


def _pending_edges(*arrays):
    """Count pending lazy-graph edges. An evaluated array serializes with
    zero ``->`` edges in ``mx.export_to_dot`` output; any pending
    computation serializes with at least one."""
    arrays = [a for a in arrays if a is not None]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "graph.dot")
        mx.export_to_dot(path, *arrays)
        with open(path, encoding="utf-8") as f:
            return f.read().count("->")


class TestBatchCacheMetadataSync(unittest.TestCase):
    """Batch cache metadata (offset/left_padding/lengths) is rewritten on
    every decode step and membership change but never read by the decode
    forward graph, so left lazy its graphs accumulate and pin buffers for
    the lifetime of the batch (Metal buffer handles grow until the
    per-process resource limit aborts the process). These tests assert the
    metadata is tied into the state graph via mx.depends — mirroring the
    guard BatchRotatingKVCache already has — so the next forward pass
    evaluates it. The churn tests first self-check the edge-counting
    instrument on a deliberately lazy sentinel, so a change in
    ``export_to_dot``'s output format cannot silently make them vacuous."""

    def _assert_instrument_works(self):
        self.assertGreater(_pending_edges(mx.zeros((3,)) + 1), 0)

    def test_batch_kv_update_ties_metadata_into_keys(self):
        cache = BatchKVCache([1, 3, 0])
        k = mx.zeros((3, 2, 4, 8))
        with mock.patch.object(mx, "depends", wraps=mx.depends) as depends:
            keys, values = cache.update_and_fetch(k, k)
        deps = [a for call in depends.call_args_list for a in call.args[1]]
        self.assertTrue(any(a is cache.offset for a in deps))
        self.assertTrue(any(a is cache.left_padding for a in deps))
        mx.eval(keys, values)
        self.assertEqual(keys.shape, (3, 2, 4, 8))
        self.assertEqual(cache.offset.tolist(), [3, 1, 4])

    def test_arrays_cache_filter_extend_values(self):
        cache = ArraysCache(1, left_padding=[0, 1, 2])
        cache.lengths = mx.array([3, 4, 5])
        cache[0] = mx.zeros((3, 4))
        cache.filter(mx.array([1, 2]))
        self.assertEqual(cache.left_padding.tolist(), [1, 2])
        self.assertEqual(cache.lengths.tolist(), [4, 5])
        self.assertEqual(cache[0].shape, (2, 4))

        other = ArraysCache(1, left_padding=[7])
        other.lengths = mx.array([9])
        other[0] = mx.zeros((1, 4))
        cache.extend(other)
        self.assertEqual(cache.left_padding.tolist(), [1, 2, 7])
        self.assertEqual(cache.lengths.tolist(), [4, 5, 9])
        self.assertEqual(cache[0].shape, (3, 4))

    def _churn_arrays_cache(self, blind_entry, read_entry, cycles=16):
        """Join/leave churn where the layer overwrites one state entry
        WITHOUT reading it (a blind reset) and reads the other — the case
        the staple-every-entry design exists for. A guard on only the
        blind-written entry is silently dropped each cycle."""
        cache = ArraysCache(2, left_padding=[0, 1, 2])
        cache.lengths = mx.array([3, 4, 5])
        cache[0] = mx.zeros((3, 4))
        cache[1] = mx.zeros((3, 2))
        mx.eval(cache[0], cache[1])
        shapes = {0: (3, 4), 1: (3, 2)}

        for _ in range(cycles):
            other = ArraysCache(2, left_padding=[0])
            other.lengths = mx.array([6])
            other[0] = mx.zeros((1, 4))
            other[1] = mx.zeros((1, 2))
            cache.extend(other)
            cache.filter(mx.array([1, 2, 3], mx.int32))
            # forward pass: one entry blind-reset, the other read-then-written
            cache[blind_entry] = mx.zeros(shapes[blind_entry])
            cache[read_entry] = cache[read_entry] * 1
            mx.eval(cache[blind_entry], cache[read_entry])

        # A correct guard leaves the metadata fully materialized (0 pending
        # edges); without it this measures in the hundreds.
        self.assertEqual(_pending_edges(cache.left_padding, cache.lengths), 0)

    def test_arrays_cache_metadata_bounded_when_first_entry_blind_written(self):
        self._assert_instrument_works()
        self._churn_arrays_cache(blind_entry=0, read_entry=1)

    def test_arrays_cache_metadata_bounded_when_last_entry_blind_written(self):
        self._assert_instrument_works()
        self._churn_arrays_cache(blind_entry=1, read_entry=0)

    def test_arrays_cache_extend_alone_bounds_metadata(self):
        """extend() must carry the guard by itself — a guard wired only into
        filter() would be masked by any churn that interleaves the two."""
        self._assert_instrument_works()
        cache = ArraysCache(1, left_padding=[0])
        cache.lengths = mx.array([4])
        cache[0] = mx.zeros((1, 4))
        mx.eval(cache[0])
        for _ in range(16):
            other = ArraysCache(1, left_padding=[1])
            other.lengths = mx.array([5])
            other[0] = mx.zeros((1, 4))
            cache.extend(other)
            cache[0] = cache[0] * 1
            mx.eval(cache[0])
        self.assertEqual(_pending_edges(cache.left_padding, cache.lengths), 0)

    def test_batch_kv_metadata_graph_stays_bounded_under_churn(self):
        """Continuous-batching pattern: join (extend), leave (filter), decode
        step — evaluating only the forward outputs. The metadata graph must
        stay bounded instead of growing per membership change. 32 cycles so
        a guard that fires once and then latches off is also caught."""
        self._assert_instrument_works()
        layers = []
        for _ in range(2):
            c = BatchKVCache([0] * 3)
            k = mx.zeros((3, 2, 8, 4), mx.float16)
            c.update_and_fetch(k, k)
            layers.append(c)
        mx.eval(*[c.keys for c in layers])

        for _ in range(32):
            for c in layers:
                o = BatchKVCache([0])
                k = mx.zeros((1, 2, 8, 4), mx.float16)
                o.update_and_fetch(k, k)
                c.extend(o)
                c.filter(mx.array([1, 2, 3], mx.int32))
            step = mx.zeros((3, 2, 1, 4), mx.float16)
            outs = [c.update_and_fetch(step, step)[0] for c in layers]
            mx.eval(*outs)

        # A correct guard leaves the metadata fully materialized (0 pending
        # edges); without it this measures in the hundreds.
        for c in layers:
            self.assertEqual(_pending_edges(c.offset, c.left_padding), 0)

    def test_arrays_cache_without_metadata_filter(self):
        cache = ArraysCache(1)
        cache[0] = mx.zeros((2, 4))
        cache.filter(mx.array([0]))
        self.assertEqual(cache[0].shape, (1, 4))

    def test_arrays_cache_filter_with_unfilled_state(self):
        cache = ArraysCache(2, left_padding=[0, 1])
        cache.lengths = mx.array([3, 4])
        cache.filter(mx.array([1]))
        self.assertEqual(cache.left_padding.tolist(), [1])
        self.assertEqual(cache.lengths.tolist(), [4])


if __name__ == "__main__":
    unittest.main()
