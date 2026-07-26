# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx

from mlx_lm.generate import (
    BatchGenerator,
    generate_step,
    maybe_quantize_kv_cache,
    speculative_generate_step,
)
from mlx_lm.models.cache import (
    ArraysCache,
    ChunkedKVCache,
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
    RotatingQuantizedKVCache,
    can_quantize_prompt_cache,
)


class CacheModel:
    def __init__(self, num_layers, cache_factory):
        self.layers = [object() for _ in range(num_layers)]
        self.cache_factory = cache_factory
        self.calls = 0

    def make_cache(self):
        return self.cache_factory()

    def __call__(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("cache validation must run before the model")


class TestCacheQuantizationCapability(unittest.TestCase):
    def test_mixed_cache_is_quantizable(self):
        prompt_cache = [
            KVCache(),
            RotatingKVCache(max_size=16, keep=0),
            QuantizedKVCache(group_size=32, bits=8),
            RotatingQuantizedKVCache(
                max_size=16,
                group_size=32,
                bits=8,
            ),
            ArraysCache(size=2),
        ]

        self.assertTrue(can_quantize_prompt_cache(prompt_cache))

    def test_rotating_cache_with_keep_is_not_quantizable(self):
        prompt_cache = [
            KVCache(),
            RotatingKVCache(max_size=16, keep=4),
        ]

        self.assertFalse(can_quantize_prompt_cache(prompt_cache))

    def test_non_kv_state_is_explicitly_ignored(self):
        state_cache = ArraysCache(size=2)
        prompt_cache = [state_cache, KVCache()]

        maybe_quantize_kv_cache(
            prompt_cache,
            quantized_kv_start=0,
            kv_group_size=32,
            kv_bits=8,
        )

        self.assertIs(prompt_cache[0], state_cache)
        self.assertIsInstance(prompt_cache[1], QuantizedKVCache)

    def test_maybe_quantize_does_not_silently_skip_unsupported_entry(self):
        prompt_cache = [KVCache(), ChunkedKVCache(chunk_size=16)]

        with self.assertRaisesRegex(
            ValueError,
            r"prompt cache layer 1 \(ChunkedKVCache: "
            r"KV-cache quantization is not implemented\)",
        ):
            maybe_quantize_kv_cache(
                prompt_cache,
                quantized_kv_start=0,
                kv_group_size=32,
                kv_bits=8,
            )

        self.assertIsInstance(prompt_cache[0], KVCache)

    def test_generate_step_fails_before_model_for_keep_cache(self):
        model = CacheModel(
            2,
            lambda: [
                KVCache(),
                RotatingKVCache(max_size=16, keep=4),
            ],
        )

        with self.assertRaisesRegex(
            ValueError,
            r"model prompt cache layer 1 \(RotatingKVCache: "
            r"keep=4 sink tokens are not supported\)",
        ):
            next(
                generate_step(
                    mx.array([1]),
                    model,
                    max_tokens=0,
                    kv_bits=8,
                    kv_group_size=32,
                )
            )

        self.assertEqual(model.calls, 0)

    def test_generate_step_rejects_quantized_cache_config_mismatch(self):
        model = CacheModel(1, lambda: [KVCache()])
        prompt_cache = [QuantizedKVCache(group_size=32, bits=4)]

        with self.assertRaisesRegex(
            ValueError,
            r"already quantized with bits=4, group_size=32; "
            r"requested bits=8, group_size=32",
        ):
            next(
                generate_step(
                    mx.array([1]),
                    model,
                    max_tokens=0,
                    prompt_cache=prompt_cache,
                    kv_bits=8,
                    kv_group_size=32,
                )
            )

        self.assertEqual(model.calls, 0)

    def test_speculative_generate_validates_draft_cache(self):
        model = CacheModel(1, lambda: [KVCache()])
        draft_model = CacheModel(1, lambda: [KVCache()])
        prompt_cache = [
            KVCache(),
            RotatingKVCache(max_size=16, keep=4),
        ]

        with self.assertRaisesRegex(
            ValueError,
            r"draft model prompt cache layer 0 \(RotatingKVCache: "
            r"keep=4 sink tokens are not supported\)",
        ):
            next(
                speculative_generate_step(
                    mx.array([1]),
                    model,
                    draft_model,
                    max_tokens=1,
                    prompt_cache=prompt_cache,
                    kv_bits=8,
                    kv_group_size=32,
                )
            )

        self.assertEqual(model.calls, 0)
        self.assertEqual(draft_model.calls, 0)

    def test_batch_generator_quantizes_every_mixed_cache_entry(self):
        model = CacheModel(
            2,
            lambda: [
                KVCache(),
                RotatingKVCache(max_size=16, keep=0),
            ],
        )
        generator = BatchGenerator(
            model,
            kv_bits=8,
            kv_group_size=32,
        )
        try:
            generator.insert([[1]])
            prompt_cache = generator._unprocessed_sequences[0][3]

            self.assertIsInstance(prompt_cache[0], QuantizedKVCache)
            self.assertIsInstance(prompt_cache[1], RotatingQuantizedKVCache)
            self.assertTrue(all(c.is_quantized() for c in prompt_cache))
        finally:
            generator.close()

    def test_batch_generator_does_not_skip_external_cache(self):
        model = CacheModel(2, lambda: [KVCache(), KVCache()])
        generator = BatchGenerator(
            model,
            kv_bits=8,
            kv_group_size=32,
        )
        prompt_cache = [KVCache(), ChunkedKVCache(chunk_size=16)]
        try:
            with self.assertRaisesRegex(
                ValueError,
                r"batch request 0 prompt cache layer 1 \(ChunkedKVCache: "
                r"KV-cache quantization is not implemented\)",
            ):
                generator.insert([[1]], caches=[prompt_cache])

            self.assertEqual(len(generator._unprocessed_sequences), 0)
            self.assertIsInstance(prompt_cache[0], KVCache)
        finally:
            generator.close()

    def test_batch_generator_quantizes_external_mixed_cache(self):
        model = CacheModel(2, lambda: [KVCache(), KVCache()])
        generator = BatchGenerator(
            model,
            kv_bits=8,
            kv_group_size=32,
        )
        prompt_cache = [
            KVCache(),
            RotatingKVCache(max_size=16, keep=0),
        ]
        try:
            generator.insert([[1]], caches=[prompt_cache])
            queued_cache = generator._unprocessed_sequences[0][3]

            self.assertIs(queued_cache, prompt_cache)
            self.assertIsInstance(queued_cache[0], QuantizedKVCache)
            self.assertIsInstance(queued_cache[1], RotatingQuantizedKVCache)
        finally:
            generator.close()


if __name__ == "__main__":
    unittest.main()
