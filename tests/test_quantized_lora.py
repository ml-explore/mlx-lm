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
from mlx_lm.tuner.lora_pack import (
    QuantizedLoRAAdapterPack,
    load_quantized_lora_adapter_bank,
)
from mlx_lm.tuner.utils import (
    linear_to_lora_layers,
    load_adapters,
    quantize_lora_layers,
    save_quantized_adapter,
    select_lora_layer_bits,
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


def make_quantized_layers(count, bits=8):
    layers = []
    for index in range(count):
        lora = LoRALinear.from_base(
            nn.Linear(64, 32, bias=False),
            r=8,
            dropout=0.0,
            scale=1.0 + index / 10,
        )
        lora.lora_a = mx.random.normal(lora.lora_a.shape)
        lora.lora_b = mx.random.normal(lora.lora_b.shape)
        layers.append(
            lora.to_quantized(
                group_size=32,
                bits=bits,
                rank_group_size=32,
            )
        )
    return layers


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

    def test_load_quantized_adapter_bank_without_base_models(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            adapter_paths = {}
            models = []
            for index, name in enumerate(("first", "second", "third")):
                source_path = directory / f"source-{name}"
                output_path = directory / f"quantized-{name}"
                model = make_lora_model(seed=31 + index)
                save_float_adapter(model, source_path)
                quantize_lora_layers(model, group_size=32, bits=8)
                save_quantized_adapter(model, source_path, output_path)
                adapter_paths[name] = output_path
                models.append(model)

            bank = load_quantized_lora_adapter_bank(adapter_paths)
            self.assertEqual(bank.adapter_names, ("first", "second", "third"))
            self.assertEqual(bank.adapter_index("second"), 1)
            self.assertGreater(bank.adapter_bytes, 0)
            with self.assertRaises(KeyError):
                bank.adapter_index("missing")

            x = mx.random.normal((2, 70))
            pack = bank.packs["layers.0.proj"]
            self.assertEqual(pack.input_dims, 70)
            self.assertEqual(pack.output_dims, 45)
            self.assertEqual(pack.rank, 7)
            for index, model in enumerate(models):
                expected = model.layers[0].proj.scale * model.layers[0].proj.lora_delta(
                    x
                )
                candidate = pack.delta_for_adapter(x, index)
                mx.eval(expected, candidate)
                self.assertTrue(mx.array_equal(expected, candidate))

    def test_quantized_adapter_pack_routes_mixed_rows(self):
        mx.random.seed(23)
        layers = make_quantized_layers(3)
        pack = QuantizedLoRAAdapterPack.from_layers(layers)
        x = mx.random.normal((2, 3, 64))
        indices = mx.array([[0, 1, 2], [2, 0, 1]], dtype=mx.int32)
        candidate = pack.delta(x, indices)

        flat_x = x.reshape((-1, 64))
        flat_indices = indices.reshape(-1)
        reference = mx.concatenate(
            [
                layers[int(flat_indices[row])].scale
                * layers[int(flat_indices[row])].lora_delta(flat_x[row : row + 1])
                for row in range(flat_x.shape[0])
            ],
            axis=0,
        ).reshape((2, 3, 32))
        mx.eval(candidate, reference)

        self.assertTrue(mx.allclose(candidate, reference, rtol=1e-5, atol=5e-5))
        self.assertEqual(candidate.shape, (2, 3, 32))
        self.assertEqual(pack.num_adapters, 3)

        homogeneous = pack.delta_for_adapter(x, 1)
        expected = layers[1].scale * layers[1].lora_delta(x)
        mx.eval(homogeneous, expected)
        self.assertTrue(mx.array_equal(homogeneous, expected))

    def test_quantized_adapter_pack_validates_its_contract(self):
        mx.random.seed(29)
        layers = make_quantized_layers(2)
        mismatched = make_quantized_layers(1, bits=4)[0]
        with self.assertRaisesRegex(ValueError, "share dimensions"):
            QuantizedLoRAAdapterPack.from_layers([layers[0], mismatched])

        pack = QuantizedLoRAAdapterPack.from_layers(layers)
        with self.assertRaisesRegex(ValueError, "adapter indices"):
            pack.delta(mx.zeros((3, 64)), [0, 1])
        with self.assertRaises(IndexError):
            pack.delta_for_adapter(mx.zeros((1, 64)), 2)

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

    def test_select_layer_bits_uses_error_thresholds(self):
        model = make_lora_model()
        layer_inputs = {
            "layers.0.proj": mx.random.normal((2, 3, 70)),
            "layers.1.proj": mx.random.normal((2, 3, 70)),
        }
        selected, report = select_lora_layer_bits(
            model,
            layer_inputs,
            candidate_bits=(4, 8),
            max_relative_l2=1.0,
            min_cosine=0.0,
            min_memory_reduction=-1.0,
        )
        self.assertEqual(
            selected,
            {
                "layers.0.proj": 4,
                "layers.1.proj": 4,
            },
        )
        self.assertEqual(report["layers.0.proj"]["selected_bits"], 4)

        selected, report = select_lora_layer_bits(
            model,
            layer_inputs,
            candidate_bits=(4, 8),
            max_relative_l2=0.0,
            min_cosine=1.0,
            min_memory_reduction=-1.0,
        )
        self.assertEqual(selected, {})
        self.assertIsNone(report["layers.0.proj"]["selected_bits"])

        model.layers[0].proj.lora_a = model.layers[0].proj.lora_a.astype(mx.float16)
        model.layers[0].proj.lora_b = model.layers[0].proj.lora_b.astype(mx.float16)
        selected, report = select_lora_layer_bits(
            model,
            {"layers.0.proj": layer_inputs["layers.0.proj"]},
            candidate_bits=(8,),
            max_relative_l2=1.0,
            min_cosine=0.0,
        )
        self.assertEqual(selected, {})
        self.assertLess(
            report["layers.0.proj"]["candidates"]["8"]["memory_reduction"],
            0.0,
        )

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

    def test_adapter_bank_rejects_shape_metadata_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source_path = directory / "source"
            output_path = directory / "quantized"
            model = make_lora_model()
            save_float_adapter(model, source_path)
            quantize_lora_layers(model, group_size=32, bits=8)
            save_quantized_adapter(model, source_path, output_path)

            config_path = output_path / "adapter_config.json"
            config = json.loads(config_path.read_text())
            config["adapter_quantization"]["layers"]["layers.0.proj"][
                "output_dims"
            ] += 1
            config_path.write_text(json.dumps(config))

            with self.assertRaisesRegex(ValueError, "shapes do not match metadata"):
                load_quantized_lora_adapter_bank({"invalid": output_path})


if __name__ == "__main__":
    unittest.main()
