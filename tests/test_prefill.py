# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.generate import generate_step
from mlx_lm.models import gemma3_text, gpt_oss
from mlx_lm.models.cache import BatchKVCache, KVCache, RotatingKVCache
from mlx_lm.models.prefill import prefill_query_lengths


def small_model(family, full_last=False):
    common = dict(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        sliding_window=8,
    )
    if family == "gemma3":
        return gemma3_text.Model(
            gemma3_text.ModelArgs(
                model_type="gemma3_text",
                num_hidden_layers=6 if full_last else 5,
                sliding_window_pattern=3,
                **common,
            )
        )
    layer_types = ["sliding_attention", "full_attention", "sliding_attention"]
    layer_types.append("full_attention" if full_last else "sliding_attention")
    model = gpt_oss.Model(
        gpt_oss.ModelArgs(
            num_hidden_layers=len(layer_types),
            layer_types=layer_types,
            num_local_experts=4,
            num_experts_per_tok=2,
            **common,
        )
    )
    for layer in model.layers:
        layer.self_attn.sinks = mx.linspace(-2, 2, 4)
    return model


class TestPrefill(unittest.TestCase):
    def test_plan(self):
        types = ["full_attention", "sliding_attention", "sliding_attention"]
        cache = [KVCache(), RotatingKVCache(8), RotatingKVCache(8)]
        self.assertEqual(prefill_query_lengths(64, types, 8, cache), [15, 8, 1])
        self.assertEqual(prefill_query_lengths(4, types, 8, cache), [4, 4, 1])
        cache[-1] = RotatingKVCache(8, keep=1)
        self.assertIsNone(prefill_query_lengths(64, types, 8, cache))
        cache[-1] = RotatingKVCache(8)
        cache[0] = BatchKVCache(left_padding=[0, 2])
        self.assertIsNone(prefill_query_lengths(64, types, 8, cache))
        cache[0] = KVCache()
        cache[0].offset = 3
        self.assertIsNone(prefill_query_lengths(64, types, 8, cache))

    def test_fallback(self):
        for family in ("gemma3", "gpt_oss"):
            model = small_model(family)
            baseline, compact = model.make_cache(), model.make_cache()
            for cache in (baseline, compact):
                for c in cache:
                    if isinstance(c, RotatingKVCache):
                        c.keep = 1
            tokens = mx.random.randint(0, 128, (1, 24))
            model(tokens, baseline)
            model.prefill(tokens, compact)
            token = mx.array([[1]])
            self.assertTrue(
                mx.array_equal(model(token, baseline), model(token, compact)).item()
            )

    def test_cached_logits(self):
        for family in ("gemma3", "gpt_oss"):
            for dtype in (mx.float32, mx.float16, mx.bfloat16):
                for full_last in (False, True):
                    with self.subTest(family=family, dtype=dtype, full_last=full_last):
                        mx.random.seed(7)
                        model = small_model(family, full_last)
                        model.set_dtype(dtype)
                        baseline, compact = model.make_cache(), model.make_cache()
                        # Include short chunks, window boundaries, and reused caches.
                        for length in (3, 33, 8, 9, 2, 25):
                            tokens = mx.random.randint(0, 128, (1, length))
                            model(tokens, cache=baseline)
                            model.prefill(tokens, cache=compact)
                            mx.eval([c.state for c in baseline + compact])
                            self.assertEqual(
                                [c.offset for c in baseline],
                                [c.offset for c in compact],
                            )
                            for _ in range(3):
                                token = mx.random.randint(0, 128, (1, 1))
                                expected = model(token, cache=baseline).astype(
                                    mx.float32
                                )
                                actual = model(token, cache=compact).astype(mx.float32)
                                tolerance = 2e-5 if dtype == mx.float32 else 0.04
                                self.assertTrue(
                                    mx.allclose(
                                        expected, actual, atol=tolerance, rtol=tolerance
                                    ).item(),
                                    f"max error: {mx.max(mx.abs(expected - actual)).item()}",
                                )

    def test_generation_and_embeddings(self):
        mx.random.seed(12)
        model = small_model("gemma3")
        prompt = mx.random.randint(0, 128, (40,))
        optimized = list(
            generate_step(prompt, model, max_tokens=4, prefill_step_size=16)
        )
        prefill = model.prefill
        model.prefill = model.__call__
        baseline = list(
            generate_step(prompt, model, max_tokens=4, prefill_step_size=16)
        )
        model.prefill = prefill
        self.assertEqual([t for t, _ in optimized], [t for t, _ in baseline])
        for (_, actual), (_, expected) in zip(optimized, baseline):
            self.assertTrue(mx.allclose(actual, expected, atol=2e-5).item())

        baseline, compact = model.make_cache(), model.make_cache()
        model(
            prompt[None],
            baseline,
            input_embeddings=model.model.embed_tokens(prompt[None]),
        )
        model.prefill(
            prompt[None],
            compact,
            input_embeddings=model.model.embed_tokens(prompt[None]),
        )
        token = mx.array([[1]])
        self.assertTrue(
            mx.allclose(model(token, baseline), model(token, compact), atol=2e-5).item()
        )

    def test_quantized_weights(self):
        for family in ("gemma3", "gpt_oss"):
            mx.random.seed(1)
            model = small_model(family)
            nn.quantize(model, group_size=32, bits=4)
            baseline, compact = model.make_cache(), model.make_cache()
            tokens = mx.random.randint(0, 128, (1, 32))
            model(tokens, baseline)
            model.prefill(tokens, compact)
            token = mx.array([[1]])
            self.assertTrue(
                mx.allclose(
                    model(token, baseline), model(token, compact), atol=2e-5
                ).item()
            )


if __name__ == "__main__":
    unittest.main()
