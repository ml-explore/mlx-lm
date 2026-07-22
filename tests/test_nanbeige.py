# Copyright © 2026 Apple Inc.

import functools
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_lm.generate import BatchGenerator, generate_step
from mlx_lm.generate import main as generate_main
from mlx_lm.generate import speculative_generate_step
from mlx_lm.models import llama, nanbeige
from mlx_lm.models.cache import KVCache, load_prompt_cache, save_prompt_cache
from mlx_lm.utils import quantize_model


def make_nanbeige(num_hidden_layers=2):
    model = nanbeige.Model(
        nanbeige.ModelArgs(
            model_type="nanbeige",
            hidden_size=64,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=128,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            rms_norm_eps=1e-5,
            vocab_size=64,
            num_loops=2,
            skip_loop_final_norm=False,
            tie_word_embeddings=False,
        )
    )
    model.set_dtype(mx.float32)
    return model


class TestNanbeige(unittest.TestCase):
    def test_recurrent_forward_and_cache(self):
        model = make_nanbeige()
        inputs = mx.array([[0, 1, 2]])

        self.assertEqual(len(model.layers), 2)
        self.assertEqual(len(model.make_cache()), 4)
        self.assertEqual(len({id(layer) for layer in model.layers}), 2)
        parameter_names = dict(tree_flatten(model.parameters())).keys()
        self.assertEqual(
            sum(name.endswith("self_attn.q_proj.weight") for name in parameter_names),
            2,
        )

        full_logits = model(inputs)
        cache = model.make_cache()
        cached_logits = mx.concatenate(
            [model(inputs[:, i : i + 1], cache=cache) for i in range(inputs.shape[1])],
            axis=1,
        )
        self.assertTrue(mx.allclose(full_logits, cached_logits, rtol=1e-4, atol=1e-4))

        with self.assertRaisesRegex(ValueError, "requires 4 cache entries"):
            model(inputs, cache=model.make_cache()[:-1])
        with self.assertRaisesRegex(ValueError, "requires 4 cache entries"):
            model(inputs, cache=model.make_cache() + model.make_cache()[:1])

    def test_variable_length_batch_generation(self):
        model = make_nanbeige()
        prompts = [[0, 1, 2], [3, 4], [5]]
        generator = BatchGenerator(
            model,
            max_tokens=1,
            prefill_batch_size=3,
            completion_batch_size=3,
        )
        uids = generator.insert(prompts)
        responses = {}
        while batch := generator.next_generated():
            responses.update((response.uid, response) for response in batch)

        self.assertEqual(set(responses), set(uids))
        for uid, prompt in zip(uids, prompts):
            _, expected = next(generate_step(mx.array(prompt), model, max_tokens=1))
            self.assertTrue(mx.allclose(responses[uid].logprobs, expected))

    def test_quantized_kv_cache(self):
        model = make_nanbeige()
        inputs = mx.array([[0, 1, 2]])
        cache = model.make_cache()
        model(inputs[:, :2], cache=cache)
        quantized_cache = [entry.to_quantized(group_size=32, bits=4) for entry in cache]
        quantized_kv_logits = model(inputs[:, 2:], cache=quantized_cache)
        self.assertTrue(mx.all(mx.isfinite(quantized_kv_logits)).item())

    def _assert_quantized_weights(self, bits):
        model = make_nanbeige()
        quantized, _ = quantize_model(model, {"model_type": "nanbeige"}, 64, bits)
        logits = quantized(mx.array([[0, 1, 2]]))
        mx.eval(logits)
        self.assertIsInstance(
            quantized.model.layers[0].self_attn.q_proj, nn.QuantizedLinear
        )
        self.assertTrue(mx.all(mx.isfinite(logits)).item())

    def test_4_bit_weights(self):
        self._assert_quantized_weights(4)

    def test_8_bit_weights(self):
        self._assert_quantized_weights(8)

    def test_norm_runs_after_each_physical_pass(self):
        events = []

        class Layer(nn.Module):
            def __init__(self, index):
                super().__init__()
                self.index = index
                self.use_sliding = False

            def __call__(self, x, mask=None, cache=None):
                events.append(f"layer-{self.index}")
                return x

        class Norm(nn.Module):
            def __call__(self, x):
                events.append("norm")
                return x

        model = make_nanbeige().model
        model.layers = [Layer(0), Layer(1)]
        model.norm = Norm()
        model(None, input_embeddings=mx.zeros((1, 1, 64)))

        self.assertEqual(
            events,
            ["layer-0", "layer-1", "norm", "layer-0", "layer-1", "norm"],
        )

    def test_prompt_cache_round_trip_continues(self):
        model = make_nanbeige(num_hidden_layers=22)
        inputs = mx.array([[0, 1, 2]])
        cache = model.make_cache()
        model(inputs[:, :2], cache=cache)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "prompt.safetensors")
            save_prompt_cache(path, cache)
            loaded_cache = load_prompt_cache(path)

        expected = model(inputs[:, 2:], cache=cache)
        actual = model(inputs[:, 2:], cache=loaded_cache)
        self.assertEqual(len(loaded_cache), 44)
        self.assertTrue(mx.allclose(expected, actual, rtol=1e-5, atol=1e-5))

    def test_config_validation(self):
        config = {
            "model_type": "nanbeige",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "rms_norm_eps": 1e-5,
            "vocab_size": 64,
            "num_loops": 2,
            "loop_loss_weights": [],
            "skip_loop_final_norm": False,
        }
        self.assertEqual(nanbeige.ModelArgs.from_dict(config).num_loops, 2)

        for num_loops in (None, 0, 1, 3):
            invalid = dict(config)
            if num_loops is None:
                invalid.pop("num_loops")
            else:
                invalid["num_loops"] = num_loops
            with self.subTest(num_loops=num_loops), self.assertRaisesRegex(
                ValueError, "Only num_loops=2"
            ):
                nanbeige.ModelArgs.from_dict(invalid)

        with self.assertRaisesRegex(ValueError, "skip_loop_final_norm"):
            nanbeige.ModelArgs.from_dict({**config, "skip_loop_final_norm": True})
        with self.assertRaisesRegex(ValueError, "loop_loss_weights"):
            nanbeige.ModelArgs.from_dict({**config, "loop_loss_weights": [1.0]})

        for option, value in (
            ("qk_layernorm", True),
            ("emb_neighbor_num", 2),
            ("insert_ngram_layer_idx", [0]),
            ("ngram_insert_all_layers", True),
            ("enable_double_loop_split", True),
            ("loop_middle_layers", 1),
            ("loop_share_kv", True),
            ("enable_hyper_connection", True),
            ("enable_mhc", True),
            ("enable_h_res_identity", True),
            ("enable_depth_attention", True),
        ):
            with self.subTest(option=option), self.assertRaisesRegex(
                ValueError, rf"Unsupported Nanbeige architecture options:.*{option}"
            ):
                nanbeige.ModelArgs.from_dict({**config, option: value})

    def test_speculative_cache_uses_logical_cache_count(self):
        target = make_nanbeige()
        draft = llama.Model(
            llama.ModelArgs(
                model_type="llama",
                hidden_size=64,
                num_hidden_layers=1,
                intermediate_size=128,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=16,
                rms_norm_eps=1e-5,
                vocab_size=64,
            )
        )
        combined_cache = target.make_cache() + draft.make_cache()
        speculate = functools.partial(
            speculative_generate_step,
            mx.array([0, 1]),
            target,
            draft,
            max_tokens=1,
        )

        self.assertEqual(len(list(speculate(prompt_cache=combined_cache))), 1)

        wrong_sizes = (
            ("target", target.make_cache()[:-1] + draft.make_cache()),
            ("draft", target.make_cache() + draft.make_cache()[:-1]),
        )
        for cache_name, invalid_cache in wrong_sizes:
            with self.subTest(cache=cache_name), self.assertRaisesRegex(
                ValueError, "requires 5 prompt cache entries"
            ):
                next(speculate(prompt_cache=invalid_cache))

        class NonTrimmableKVCache(KVCache):
            def is_trimmable(self):
                return False

        for index, cache_name in ((0, "target"), (-1, "draft")):
            invalid_cache = list(combined_cache)
            invalid_cache[index] = NonTrimmableKVCache()
            with self.subTest(cache=cache_name), self.assertRaisesRegex(
                ValueError, rf"trimmable {cache_name} prompt cache"
            ):
                next(speculate(prompt_cache=invalid_cache))

    def test_cli_rejects_saved_cache_with_draft_model(self):
        argv = [
            "mlx_lm.generate",
            "--prompt-cache-file",
            "cache.safetensors",
            "--draft-model",
            "draft",
        ]
        with patch.object(sys, "argv", argv), self.assertRaisesRegex(
            ValueError, "cannot be combined"
        ):
            generate_main()


if __name__ == "__main__":
    unittest.main()
