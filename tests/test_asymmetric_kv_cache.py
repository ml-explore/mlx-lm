# Copyright © 2026 Apple Inc.

import os
import tempfile
import unittest
from inspect import signature

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_lm.cache_prompt import setup_arg_parser as cache_prompt_arg_parser
from mlx_lm.generate import (
    DEFAULT_QUANTIZED_KV_START,
    generate_step,
    maybe_quantize_kv_cache,
)
from mlx_lm.generate import setup_arg_parser as generate_arg_parser
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.cache import (
    KVCache,
    QuantizedKVCache,
    load_prompt_cache,
    save_prompt_cache,
)
from mlx_lm.server import setup_arg_parser as server_arg_parser
from mlx_lm.server import validate_kv_args


class TestAsymmetricKVCache(unittest.TestCase):
    def test_scalar_bits_matches_explicit_symmetric_path(self):
        x = mx.arange(128, dtype=mx.float32).reshape(1, 1, 2, 64)
        scalar = QuantizedKVCache(bits=4, group_size=32)
        explicit = QuantizedKVCache(bits=4, key_bits=4, value_bits=4, group_size=32)

        scalar.update_and_fetch(x, x)
        explicit.update_and_fetch(x, x)
        expected = mx.quantize(x, group_size=32, bits=4)

        self.assertEqual(scalar.meta_state, explicit.meta_state)
        self.assertEqual(scalar.bits, 4)
        for scalar_side, explicit_side in zip(scalar.state, explicit.state):
            for actual, equivalent, baseline in zip(
                scalar_side, explicit_side, expected
            ):
                self.assertTrue(mx.array_equal(actual, equivalent))
                self.assertTrue(mx.array_equal(actual, baseline))

    def test_asymmetric_allocation_and_quantization(self):
        keys = mx.arange(128, dtype=mx.float32).reshape(1, 1, 2, 64)
        values = mx.arange(64, dtype=mx.float32).reshape(1, 1, 2, 32) * 0.5
        cache = QuantizedKVCache(group_size=32, key_bits=8, value_bits=4)

        cache.update_and_fetch(keys, values)

        self.assertEqual(cache.key_bits, 8)
        self.assertEqual(cache.value_bits, 4)
        self.assertIsNone(cache.bits)
        self.assertEqual(cache.keys[0].shape[-1], 16)
        self.assertEqual(cache.values[0].shape[-1], 4)
        for actual, expected in zip(
            cache.keys, mx.quantize(keys, group_size=32, bits=8)
        ):
            self.assertTrue(mx.array_equal(actual[..., :2, :], expected))
        for actual, expected in zip(
            cache.values, mx.quantize(values, group_size=32, bits=4)
        ):
            self.assertTrue(mx.array_equal(actual[..., :2, :], expected))

        queries = mx.ones((1, 1, 1, 64), dtype=mx.float32)
        output = scaled_dot_product_attention(
            queries,
            tuple(x[..., :2, :] for x in cache.keys),
            tuple(x[..., :2, :] for x in cache.values),
            cache,
            scale=64**-0.5,
            mask=None,
        )
        mx.eval(output)
        self.assertEqual(output.shape, (1, 1, 1, 32))

    def test_legacy_quantized_attention_duck(self):
        x = mx.arange(64, dtype=mx.float32).reshape(1, 1, 1, 64)
        state = mx.quantize(x, group_size=32, bits=4)

        class LegacyQuantizedDuck:
            bits = 4
            group_size = 32

        output = scaled_dot_product_attention(
            mx.ones_like(x),
            state,
            state,
            LegacyQuantizedDuck(),
            scale=64**-0.5,
            mask=None,
        )
        mx.eval(output)
        self.assertEqual(output.shape, x.shape)

    def test_serialization_round_trip_and_legacy_migration(self):
        x = mx.arange(128, dtype=mx.float32).reshape(1, 1, 2, 64)
        asymmetric = QuantizedKVCache(group_size=32, key_bits=8, value_bits=4)
        asymmetric.update_and_fetch(x, x)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "asymmetric.safetensors")
            save_prompt_cache(path, [asymmetric])
            loaded = load_prompt_cache(path)[0]
            self.assertEqual((loaded.key_bits, loaded.value_bits), (8, 4))
            self.assertEqual(loaded.meta_state, ("2", "2", "32", "8", "4"))
            for original_side, loaded_side in zip(asymmetric.state, loaded.state):
                for original, restored in zip(original_side, loaded_side):
                    self.assertTrue(mx.array_equal(original, restored))

            symmetric = QuantizedKVCache(bits=4, group_size=32)
            symmetric.update_and_fetch(x, x)
            legacy_path = os.path.join(directory, "legacy.safetensors")
            legacy_arrays = dict(tree_flatten([symmetric.state]))
            legacy_metadata = dict(
                tree_flatten([[("2", "32", "4")], {}, ["QuantizedKVCache"]])
            )
            mx.save_safetensors(legacy_path, legacy_arrays, legacy_metadata)

            legacy_loaded = load_prompt_cache(legacy_path)[0]
            self.assertEqual((legacy_loaded.key_bits, legacy_loaded.value_bits), (4, 4))
            migrated_path = os.path.join(directory, "migrated.safetensors")
            save_prompt_cache(migrated_path, [legacy_loaded])
            migrated = load_prompt_cache(migrated_path)[0]
            self.assertEqual(migrated.meta_state, ("2", "32", "4"))

            lying_path = os.path.join(directory, "lying-geometry.safetensors")
            lying_metadata = dict(
                tree_flatten([[("2", "2", "32", "8", "8")], {}, ["QuantizedKVCache"]])
            )
            mx.save_safetensors(lying_path, legacy_arrays, lying_metadata)
            with self.assertRaisesRegex(ValueError, "does not match 8-bit metadata"):
                load_prompt_cache(lying_path)

            geometry_path = os.path.join(directory, "cross-side-geometry.safetensors")
            key_state, value_state = asymmetric.state
            bad_values = tuple(
                mx.broadcast_to(array, (2, *array.shape[1:]))
                for array in value_state
            )
            geometry_arrays = dict(tree_flatten([(key_state, bad_values)]))
            geometry_metadata = dict(
                tree_flatten(
                    [[("2", "2", "32", "8", "4")], {}, ["QuantizedKVCache"]]
                )
            )
            mx.save_safetensors(geometry_path, geometry_arrays, geometry_metadata)
            with self.assertRaisesRegex(ValueError, "batch, head, or length mismatch"):
                load_prompt_cache(geometry_path)

            for name, invalid_meta, message in (
                ("bad-bits", ("2", "2", "32", "8", "7"), "Unsupported value bits"),
                ("bad-group", ("2", "2", "16", "8", "4"), "Unsupported group size"),
                ("bad-layout", ("2", "2", "32", "8"), "expected 3 legacy fields or 5"),
            ):
                corrupt_path = os.path.join(directory, f"{name}.safetensors")
                corrupt_metadata = dict(
                    tree_flatten([[invalid_meta], {}, ["QuantizedKVCache"]])
                )
                mx.save_safetensors(corrupt_path, legacy_arrays, corrupt_metadata)
                with self.assertRaisesRegex(ValueError, message):
                    load_prompt_cache(corrupt_path)

    def test_quantization_resolution_is_fail_closed(self):
        cache = KVCache()
        maybe_quantize_kv_cache([cache], 0, 64, None)
        self.assertIsInstance(cache, KVCache)

        with self.assertRaisesRegex(ValueError, "Both key and value bits"):
            maybe_quantize_kv_cache([cache], 0, 64, None, kv_key_bits=8)
        with self.assertRaisesRegex(ValueError, "Unsupported value bits"):
            QuantizedKVCache(key_bits=8, value_bits=7)
        with self.assertRaisesRegex(ValueError, "Unsupported group size"):
            QuantizedKVCache(group_size=16)

        x = mx.arange(128, dtype=mx.float32).reshape(1, 1, 1, 128)
        for bits in (2, 3, 4, 5, 6, 8):
            supported = QuantizedKVCache(bits=bits, group_size=32)
            supported.update_and_fetch(x, x)
            self.assertEqual(supported.keys[0].shape[-1], 128 * bits // 32)

        class LegacyDuckCache:
            offset = 0

            def to_quantized(self, group_size, bits):
                return (group_size, bits)

        legacy = [LegacyDuckCache()]
        maybe_quantize_kv_cache(legacy, 0, 64, 4)
        self.assertEqual(legacy, [(64, 4)])

    def test_cli_compatibility_and_defaults(self):
        parameters = list(signature(generate_step).parameters)
        self.assertLess(
            parameters.index("input_embeddings"), parameters.index("kv_key_bits")
        )

        generate_args = generate_arg_parser().parse_args(
            ["--kv-bits", "8", "--kv-value-bits", "4"]
        )
        self.assertEqual(generate_args.kv_bits, 8)
        self.assertEqual(generate_args.kv_key_bits, None)
        self.assertEqual(generate_args.kv_value_bits, 4)

        cache_args = cache_prompt_arg_parser().parse_args(
            [
                "--prompt-cache-file",
                "cache.safetensors",
                "--prompt",
                "hello",
                "--kv-key-bits",
                "8",
                "--kv-value-bits",
                "4",
            ]
        )
        self.assertEqual((cache_args.kv_key_bits, cache_args.kv_value_bits), (8, 4))

        server_args = server_arg_parser().parse_args([])
        self.assertEqual(server_args.top_k, 0)
        self.assertIsNone(server_args.kv_bits)
        self.assertIsNone(server_args.kv_key_bits)
        self.assertIsNone(server_args.kv_value_bits)
        self.assertEqual(server_args.quantized_kv_start, DEFAULT_QUANTIZED_KV_START)

        server_args.kv_key_bits = 8
        with self.assertRaisesRegex(ValueError, "Both key and value bits"):
            validate_kv_args(server_args)
        server_args.kv_key_bits = None
        server_args.kv_bits = 7
        with self.assertRaisesRegex(ValueError, "Unsupported key bits"):
            validate_kv_args(server_args)
        server_args.kv_bits = 4
        server_args.kv_group_size = 16
        with self.assertRaisesRegex(ValueError, "Unsupported group size"):
            validate_kv_args(server_args)


if __name__ == "__main__":
    unittest.main()
