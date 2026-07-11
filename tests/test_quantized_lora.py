# Copyright © 2026 Apple Inc.

import json
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_lm.models import llama
from mlx_lm.tuner.lora import LoRALinear, QuantizedLoRALinear
from mlx_lm.tuner.utils import (
    linear_to_lora_layers,
    load_adapters,
    quantize_lora_layers,
    save_quantized_adapter,
)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(70, 45)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [Block(), Block(), Block()]

    def __call__(self, x):
        return sum(layer.proj(x) for layer in self.layers)


LORA_CONFIG = {
    "rank": 7,
    "dropout": 0.0,
    "scale": 2.0,
    "keys": ["proj"],
}


def make_lora_model(seed=19):
    mx.random.seed(seed)
    model = Model()
    model.freeze()
    linear_to_lora_layers(model, 3, LORA_CONFIG)
    for _, module in model.named_modules():
        if hasattr(module, "lora_b"):
            module.lora_b = mx.random.normal(module.lora_b.shape)
    model.eval()
    return model


def save_float_adapter(model, path: Path):
    path.mkdir()
    config = {
        "fine_tune_type": "lora",
        "num_layers": 3,
        "lora_parameters": LORA_CONFIG,
    }
    (path / "adapter_config.json").write_text(json.dumps(config))
    mx.save_safetensors(
        str(path / "adapters.safetensors"),
        dict(tree_flatten(model.trainable_parameters())),
    )


class TestQuantizedLoraAdapter(unittest.TestCase):
    def test_q8_adapter_preserves_tiny_transformer_logits(self):
        args = llama.ModelArgs(
            model_type="llama",
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            rms_norm_eps=1e-5,
            vocab_size=128,
            tie_word_embeddings=False,
        )
        mx.random.seed(47)
        model = llama.Model(args)
        model.freeze()
        linear_to_lora_layers(
            model,
            2,
            {
                "rank": 8,
                "dropout": 0.0,
                "scale": 2.0,
                "keys": ["self_attn.q_proj", "self_attn.v_proj"],
            },
        )
        for _, module in model.named_modules():
            if hasattr(module, "lora_b"):
                module.lora_b = 0.02 * mx.random.normal(module.lora_b.shape)
        model.eval()

        tokens = mx.array([[1, 7, 13, 21, 34]], dtype=mx.int32)
        reference = model(tokens).astype(mx.float32)
        mx.eval(reference)
        quantize_lora_layers(
            model,
            group_size=32,
            bits=8,
            rank_group_size=32,
        )
        candidate = model(tokens).astype(mx.float32)
        difference = candidate - reference
        mx.eval(candidate, difference)

        relative_l2 = float(mx.linalg.norm(difference)) / float(
            mx.linalg.norm(reference)
        )
        cosine = float(
            mx.sum(reference * candidate)
            / (mx.linalg.norm(reference) * mx.linalg.norm(candidate))
        )
        self.assertLess(relative_l2, 0.002)
        self.assertGreater(cosine, 0.99999)
        self.assertLess(float(mx.max(mx.abs(difference))), 0.005)

    def test_persist_and_directly_load_mixed_bits(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source_path = directory / "source"
            output_path = directory / "quantized"
            model = make_lora_model()
            save_float_adapter(model, source_path)

            layer_bits = {
                "layers.0.proj": 8,
                "layers.1.proj": 4,
            }
            quantize_lora_layers(
                model,
                group_size=32,
                rank_group_size=32,
                layer_bits=layer_bits,
            )
            x = mx.random.normal((2, 3, 70))
            reference = model(x)
            mx.eval(reference)
            save_quantized_adapter(model, source_path, output_path)

            mx.random.seed(19)
            loaded = Model()
            loaded.freeze()
            load_adapters(loaded, output_path)
            loaded.eval()
            candidate = loaded(x)
            mx.eval(candidate)

            self.assertTrue(mx.array_equal(reference, candidate))
            self.assertIsInstance(loaded.layers[0].proj, QuantizedLoRALinear)
            self.assertIsInstance(loaded.layers[1].proj, QuantizedLoRALinear)
            self.assertIsInstance(loaded.layers[2].proj, LoRALinear)
            self.assertEqual(loaded.layers[0].proj.bits, 8)
            self.assertEqual(loaded.layers[1].proj.bits, 4)

            saved_config = json.loads((output_path / "adapter_config.json").read_text())
            self.assertEqual(
                {
                    name: metadata["bits"]
                    for name, metadata in saved_config["adapter_quantization"][
                        "layers"
                    ].items()
                },
                layer_bits,
            )
            saved_weights = mx.load(str(output_path / "adapters.safetensors"))
            self.assertEqual(saved_weights["layers.0.proj.lora_a"].dtype, mx.uint32)
            self.assertEqual(saved_weights["layers.1.proj.lora_b"].dtype, mx.uint32)

            source_size = (source_path / "adapters.safetensors").stat().st_size
            output_size = (output_path / "adapters.safetensors").stat().st_size
            self.assertLess(output_size, source_size)

    def test_missing_quantized_weight_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source_path = directory / "source"
            output_path = directory / "quantized"
            model = make_lora_model()
            save_float_adapter(model, source_path)
            quantize_lora_layers(model, group_size=32, bits=8)
            save_quantized_adapter(model, source_path, output_path)

            adapter_file = output_path / "adapters.safetensors"
            weights = mx.load(str(adapter_file))
            del weights["layers.0.proj.lora_a_scales"]
            mx.save_safetensors(str(adapter_file), weights)

            mx.random.seed(19)
            loaded = Model()
            loaded.freeze()
            with self.assertRaisesRegex(ValueError, "missing weights"):
                load_adapters(loaded, output_path)


if __name__ == "__main__":
    unittest.main()
