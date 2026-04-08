# Copyright © 2024 Apple Inc.

import json
import os
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_lm import convert, utils

HF_MODEL_PATH = "mlx-community/Qwen1.5-0.5B-Chat-4bit"


class TinyArgs:
    def __init__(self, vocab_size, hidden_size):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    @classmethod
    def from_dict(cls, config):
        return cls(config["vocab_size"], config["hidden_size"])


class TinyModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed = nn.Embedding(args.vocab_size, args.hidden_size)
        self.proj = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def sanitize(self, weights):
        clean = {}
        for key, value in weights.items():
            if key.startswith("language_model."):
                key = key[len("language_model.") :]
            clean[key] = value
        return clean


def get_tiny_classes(config):
    return TinyModel, TinyArgs


class TestUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir_fid = tempfile.TemporaryDirectory()
        cls.test_dir = cls.test_dir_fid.name
        if not os.path.isdir(cls.test_dir):
            os.mkdir(cls.test_dir_fid.name)

    @classmethod
    def tearDownClass(cls):
        cls.test_dir_fid.cleanup()

    def test_load(self):
        from mlx_lm.models.qwen2 import Model as Qwen2Model

        model, _ = utils.load(HF_MODEL_PATH)
        self.assertIsInstance(model, Qwen2Model)

        model_lazy, _ = utils.load(HF_MODEL_PATH, lazy=True)

        mx.eval(model_lazy.parameters())

        p1 = model.layers[0].mlp.up_proj.weight
        p2 = model_lazy.layers[0].mlp.up_proj.weight
        self.assertTrue(mx.allclose(p1, p2))

    def test_make_shards(self):
        from mlx_lm.models import llama

        args = llama.ModelArgs(
            model_type="llama",
            hidden_size=2048,
            num_hidden_layers=32,
            intermediate_size=4096,
            num_attention_heads=32,
            rms_norm_eps=1e-5,
            vocab_size=30_000,
        )
        model = llama.Model(args)
        weights = tree_flatten(model.parameters())
        gb = sum(p.nbytes for _, p in weights) // 2**30
        shards = utils.make_shards(dict(weights), 1)
        self.assertTrue(gb <= len(shards) <= gb + 1)

    def test_quantize(self):
        from mlx_lm.models import llama

        args = llama.ModelArgs(
            model_type="llama",
            hidden_size=1024,
            num_hidden_layers=4,
            intermediate_size=2048,
            num_attention_heads=4,
            rms_norm_eps=1e-5,
            vocab_size=10_000,
        )
        model = llama.Model(args)
        model, config = utils.quantize_model(model, {}, 64, 4)
        weights = dict(tree_flatten(model.parameters()))
        self.assertTrue("model.layers.2.mlp.up_proj.scales" in weights)
        self.assertTrue("model.layers.2.mlp.up_proj.biases" in weights)
        self.assertEqual(config["quantization"]["group_size"], 64)
        self.assertEqual(config["quantization"]["bits"], 4)

    def test_convert(self):
        mlx_path = os.path.join(self.test_dir, "mlx_model")

        convert(HF_MODEL_PATH, mlx_path=mlx_path, quantize=False)
        model, _ = utils.load(mlx_path)
        self.assertTrue(isinstance(model.layers[0].mlp.up_proj, nn.QuantizedLinear))
        self.assertTrue(isinstance(model.layers[-1].mlp.up_proj, nn.QuantizedLinear))

        # Check model weights have right type
        mlx_path = os.path.join(self.test_dir, "mlx_model_bf16")
        convert(HF_MODEL_PATH, mlx_path=mlx_path, dtype="bfloat16")
        model, _ = utils.load(mlx_path)

        self.assertEqual(model.layers[0].mlp.up_proj.scales.dtype, mx.bfloat16)
        self.assertEqual(model.layers[-1].mlp.up_proj.scales.dtype, mx.bfloat16)

    def test_load_model_with_custom_get_classes(self):
        class CustomQwenModel(nn.Module):
            def __init__(self, args):
                super().__init__()
                self.config = args
                self.custom_attribute = "This is a custom model"

            def load_weights(self, weights, **kwargs):
                self.qwenWeights = weights

        class CustomQwenConfig:
            @classmethod
            def from_dict(cls, config):
                instance = cls()
                for k, v in config.items():
                    setattr(instance, k, v)
                return instance

        def custom_get_classes(config):
            return CustomQwenModel, CustomQwenConfig

        model_path = utils.hf_repo_to_path(HF_MODEL_PATH)
        model, _ = utils.load_model(model_path, get_model_classes=custom_get_classes)

        self.assertIsInstance(model, CustomQwenModel)
        self.assertTrue(hasattr(model, "custom_attribute"))
        self.assertEqual(model.custom_attribute, "This is a custom model")
        self.assertTrue(hasattr(model, "qwenWeights"))

    def _tiny_weights(self):
        model = TinyModel(TinyArgs(16, 32))
        mx.eval(model.parameters())
        return dict(tree_flatten(model.parameters()))

    def _quantized_tiny_weights(self):
        weights = self._tiny_weights()
        q_weight, scales, biases = mx.quantize(
            weights["proj.weight"],
            bits=4,
            group_size=32,
        )
        return {
            "embed.weight": weights["embed.weight"],
            "proj.weight": q_weight,
            "proj.scales": scales,
            "proj.biases": biases,
        }

    def _write_model_dir(self, weights, extra_config=None):
        model_dir = Path(tempfile.mkdtemp(dir=self.test_dir))
        config = {
            "model_type": "tiny",
            "vocab_size": 16,
            "hidden_size": 32,
        }
        if extra_config is not None:
            config.update(extra_config)
        with open(model_dir / "config.json", "w") as fid:
            json.dump(config, fid)
        mx.save_safetensors(str(model_dir / "model.safetensors"), weights)
        return model_dir

    def _load_tiny_model(self, model_dir):
        return utils.load_model(
            model_dir,
            get_model_classes=get_tiny_classes,
        )

    def test_load_model_drops_unknown_weights(self):
        base = self._tiny_weights()
        weights = dict(base)
        weights["vision_tower.encoder.weight"] = mx.zeros((1,), dtype=mx.float32)
        weights["audio_tower.encoder.weight"] = mx.zeros((1,), dtype=mx.float32)
        model_dir = self._write_model_dir(weights)

        model, _ = self._load_tiny_model(model_dir)

        loaded = dict(tree_flatten(model.parameters()))
        self.assertTrue(mx.allclose(loaded["embed.weight"], base["embed.weight"]))
        self.assertTrue(mx.allclose(loaded["proj.weight"], base["proj.weight"]))

    def test_load_model_drops_unknown_weights_after_sanitize(self):
        base = self._tiny_weights()
        weights = {f"language_model.{key}": value for key, value in base.items()}
        weights["vision_tower.encoder.weight"] = mx.zeros((1,), dtype=mx.float32)
        model_dir = self._write_model_dir(weights)

        model, _ = self._load_tiny_model(model_dir)

        loaded = dict(tree_flatten(model.parameters()))
        self.assertTrue(mx.allclose(loaded["embed.weight"], base["embed.weight"]))
        self.assertTrue(mx.allclose(loaded["proj.weight"], base["proj.weight"]))

    def test_load_model_still_fails_for_missing_supported_weights(self):
        weights = self._tiny_weights()
        weights.pop("proj.weight")
        model_dir = self._write_model_dir(weights)

        with self.assertRaises(ValueError):
            self._load_tiny_model(model_dir)

    def test_load_model_keeps_supported_quantized_weights(self):
        weights = self._quantized_tiny_weights()
        weights["vision_tower.encoder.weight"] = mx.zeros((1,), dtype=mx.float32)
        model_dir = self._write_model_dir(
            weights,
            extra_config={"quantization": {"bits": 4, "group_size": 32}},
        )

        model, _ = self._load_tiny_model(model_dir)

        loaded = dict(tree_flatten(model.parameters()))
        self.assertIn("proj.weight", loaded)
        self.assertIn("proj.scales", loaded)
        self.assertIn("proj.biases", loaded)
        self.assertTrue(mx.allclose(loaded["proj.scales"], weights["proj.scales"]))
        self.assertTrue(mx.allclose(loaded["proj.biases"], weights["proj.biases"]))

    def test_load_model_gemma4_with_per_layer_projection_quantization(self):
        from mlx_lm.models import gemma4

        args = gemma4.ModelArgs.from_dict(
            {
                "model_type": "gemma4",
                "vocab_size": 32,
                "text_config": {
                    "model_type": "gemma4_text",
                    "hidden_size": 32,
                    "num_hidden_layers": 2,
                    "intermediate_size": 64,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "num_global_key_value_heads": 1,
                    "head_dim": 16,
                    "global_head_dim": 16,
                    "sliding_window": 8,
                    "sliding_window_pattern": 1,
                    "layer_types": ["full_attention", "full_attention"],
                    "hidden_size_per_layer_input": 32,
                    "vocab_size_per_layer_input": 32,
                    "num_kv_shared_layers": 0,
                    "tie_word_embeddings": True,
                },
            }
        )
        model = gemma4.Model(args)
        model, config = utils.quantize_model(
            model,
            {
                "model_type": "gemma4",
                "vocab_size": args.vocab_size,
                "text_config": args.text_config,
            },
            group_size=32,
            bits=4,
        )

        config["quantization"]["language_model.model.per_layer_model_projection"] = {
            "group_size": 32,
            "bits": 4,
        }

        with tempfile.TemporaryDirectory(dir=self.test_dir) as mlx_path:
            utils.save_model(mlx_path, model)
            utils.save_config(config, os.path.join(mlx_path, "config.json"))

            loaded, loaded_config = utils.load_model(Path(mlx_path))

            self.assertIn(
                "language_model.model.per_layer_model_projection",
                loaded_config["quantization"],
            )

            logits = loaded(mx.array([[1, 2, 3]], dtype=mx.int32))
            mx.eval(logits)
            self.assertEqual(logits.shape, (1, 3, args.vocab_size))


if __name__ == "__main__":
    unittest.main()
