# Copyright © 2024 Apple Inc.

import os
import tempfile
import unittest

import mlx.core as mx

from mlx_lm.generate import generate_step
from mlx_lm.models.base import hadamard_size_ok, rotate_last
from mlx_lm.models.cache import (
    QuantizedKVCache,
    load_prompt_cache,
    make_prompt_cache,
    save_prompt_cache,
)
from mlx_lm.utils import load

HF_MODEL_PATH = "mlx-community/Qwen1.5-0.5B-Chat-4bit"


class TestKVRotate(unittest.TestCase):
    """Coverage for the Hadamard-rotated quantized KV cache (``--kv-rotate``)."""

    @classmethod
    def setUpClass(cls):
        cls.model, cls.tokenizer = load(HF_MODEL_PATH)

    def setUp(self):
        self.test_dir_fid = tempfile.TemporaryDirectory()
        self.test_dir = self.test_dir_fid.name

    def tearDown(self):
        self.test_dir_fid.cleanup()

    def test_rotate_last_preserves_scores_across_head_dims(self):
        # Broadened head_dim coverage: the Hadamard rotation is orthonormal, so
        # rotating both Q and K leaves the full q.k score matrix unchanged for
        # every supported head_dim -- the invariant the quantized-SDPA fast path
        # relies on. Covers the sizes verified in the PR (64, 80, 96, 128).
        for d in (64, 80, 96, 128):
            self.assertTrue(hadamard_size_ok(d))
            q = mx.random.normal(shape=(2, 4, 3, d))
            k = mx.random.normal(shape=(2, 4, 5, d))
            s0 = q @ mx.swapaxes(k, -1, -2)
            s1 = rotate_last(q) @ mx.swapaxes(rotate_last(k), -1, -2)
            self.assertTrue(mx.allclose(s0, s1, atol=1e-3, rtol=1e-3))
            # orthonormal => per-vector L2 norm is preserved
            self.assertTrue(
                mx.allclose(
                    mx.sum(q * q, axis=-1),
                    mx.sum(rotate_last(q) * rotate_last(q), axis=-1),
                    atol=1e-3,
                    rtol=1e-3,
                )
            )
        # Unsupported sizes are rejected, so rotation is skipped rather than
        # silently applying a non-orthonormal transform (e.g. MLA's 576 latent).
        for d in (17, 100, 130, 576):
            self.assertFalse(hadamard_size_ok(d))

    def test_cache_to_quantized_rotated(self):
        # Dedicated coverage for the --quantized-kv-start rotation path:
        # KVCache.to_quantized(rotate=True) rotates the cached keys and the
        # quantized-SDPA fast path rotates Q, so at 8-bit (tiny quant error) the
        # rotated quantized cache still tracks the fp reference -- the rotation is
        # correct and neutral at high bits, exactly as intended (win is at 4-bit).
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
            i += 1

        prompt_cache = [
            c.to_quantized(bits=8, group_size=32, rotate=True) for c in prompt_cache
        ]
        for c in prompt_cache:
            self.assertIsInstance(c, QuantizedKVCache)
            self.assertTrue(c.rotate)
            # rotate flag is carried in meta_state (4 fields, not the legacy 3)
            self.assertEqual(len(c.meta_state), 4)

        for _, (tok, logits) in zip(
            range(1),
            generate_step(mx.array([toks[i]]), model, prompt_cache=prompt_cache),
        ):
            i += 1
            self.assertEqual(tok, toks[i])
            # 8-bit rotation is neutral: it preserves the argmax and stays near
            # fp, but redistributes quant error, so it sits a hair looser than
            # affine's 4e-2 (measured max-rel ~0.04 on this model) -- 6e-2 keeps
            # the assertion tight while non-flaky.
            self.assertTrue(mx.allclose(logits, all_logits[i], rtol=6e-2))

    def test_quantized_cache_rotate_roundtrip(self):
        # The rotate flag survives save/load, and a legacy 3-field meta_state
        # (written before rotate existed) loads as rotate=False.
        cache = [QuantizedKVCache(bits=4, group_size=32, rotate=True) for _ in range(2)]
        for c in cache:
            x = mx.random.uniform(shape=(1, 8, 10, 64))
            c.update_and_fetch(x, x)
        cache_file = os.path.join(self.test_dir, "rotate_cache.safetensors")
        save_prompt_cache(cache_file, cache)
        loaded = load_prompt_cache(cache_file)
        for c, lc in zip(cache, loaded):
            self.assertTrue(lc.rotate)
            self.assertEqual(lc.meta_state, c.meta_state)
        # Backward compatibility: pre-rotate 3-field meta_state -> rotate=False.
        legacy = QuantizedKVCache()
        legacy.meta_state = ("7", "32", "4")
        self.assertFalse(legacy.rotate)
        self.assertEqual(legacy.offset, 7)
        self.assertEqual(legacy.bits, 4)


if __name__ == "__main__":
    unittest.main()
