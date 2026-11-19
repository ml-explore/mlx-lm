# Copyright © 2024 Apple Inc.

import json
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_lm.models import mistral3
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.tuner.utils import (
    linear_to_lora_layers,
    load_adapter_weights,
    load_adapters,
    print_trainable_parameters,
)


class TestTunerUtils(unittest.TestCase):
    def setUp(self):
        self.capturedOutput = StringIO()
        sys.stdout = self.capturedOutput

    def tearDown(self):
        sys.stdout = sys.__stdout__

    def test_quantized_print_trainable_parameters(self):
        model = MagicMock()
        quantized_linear = MagicMock(spec=nn.QuantizedLinear)
        quantized_linear.weight = MagicMock(size=1e6)
        quantized_linear.bits = 8
        lora_linear = MagicMock(spec=LoRALinear)
        lora_linear.weight = MagicMock(size=2e6)
        lora_linear.parameters.return_value = [lora_linear.weight]

        linear = MagicMock(spec=nn.Linear)
        linear.weight = MagicMock(size=3e6)
        linear.parameters.return_value = [linear.weight]

        model.leaf_modules.return_value = {
            "quantized_linear": quantized_linear,
            "lora_linear": lora_linear,
            "linear": linear,
        }

        model.trainable_parameters.return_value = {
            "layer1.weight": MagicMock(size=1e6),
            "layer3.weight": MagicMock(size=2e6),
        }
        expected_output_8bits = "Trainable parameters: 33.333% (3.000M/9.000M)\n"
        print_trainable_parameters(model)
        self.assertEqual(self.capturedOutput.getvalue(), expected_output_8bits)
        self.capturedOutput.truncate(0)
        self.capturedOutput.seek(0)

        quantized_linear.weight = MagicMock(size=1e6)
        quantized_linear.bits = 4
        expected_output_4bits = "Trainable parameters: 23.077% (3.000M/13.000M)\n"
        print_trainable_parameters(model)
        self.assertEqual(self.capturedOutput.getvalue(), expected_output_4bits)
        self.capturedOutput.truncate(0)
        self.capturedOutput.seek(0)

    def test_print_trainable_parameters(self):
        model = MagicMock()
        linear1 = MagicMock(spec=nn.Linear)
        linear1.weight = MagicMock(size=1e6)
        linear1.parameters.return_value = [linear1.weight]
        linear2 = MagicMock(spec=nn.Linear)
        linear2.weight = MagicMock(size=2e6)
        linear2.parameters.return_value = [linear2.weight]
        lora_linear = MagicMock(spec=LoRALinear)
        lora_linear.weight = MagicMock(size=3e6)
        lora_linear.parameters.return_value = [lora_linear.weight]
        model.leaf_modules.return_value = {
            "linear1": linear1,
            "linear2": linear2,
            "lora_linear": lora_linear,
        }

        model.trainable_parameters.return_value = {
            "layer1.weight": MagicMock(size=1e6),
            "layer3.weight": MagicMock(size=2e6),
        }
        expected_output = "Trainable parameters: 50.000% (3.000M/6.000M)\n"
        print_trainable_parameters(model)
        self.assertEqual(self.capturedOutput.getvalue(), expected_output)


class TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 3, bias=False)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [TinyLayer()]


def _make_adapter(weights):
    path = Path(tempfile.mkdtemp())
    config = {
        "fine_tune_type": "lora",
        "num_layers": 1,
        "lora_parameters": {
            "keys": ["proj"],
            "rank": 2,
            "scale": 1.0,
            "dropout": 0.0,
        },
    }
    (path / "adapter_config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(path / "adapters.safetensors"), weights)
    return path


LORA_PARAMETERS = {"rank": 2, "scale": 1.0, "dropout": 0.0}


def _mistral3() -> nn.Module:
    text_config = {
        "model_type": "mistral",
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 64,
        "rms_norm_eps": 1e-5,
    }
    args = mistral3.ModelArgs.from_dict(
        {"model_type": "mistral3", "text_config": text_config}
    )
    return mistral3.Model(args)


def _mistral3_lora_weights() -> dict[str, mx.array]:
    model = _mistral3()
    linear_to_lora_layers(model, 2, LORA_PARAMETERS)
    weights = {
        k: mx.random.normal(v.shape)
        for k, v in tree_flatten(model.parameters())
        if k.endswith(("lora_a", "lora_b"))
    }
    assert all(k.startswith("model.language_model.") for k in weights)
    return weights


def _older_names(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    older = {}
    for k, v in weights.items():
        if k.startswith("lm_head."):
            k = "language_model." + k
        older[k.replace("model.language_model.", "language_model.model.", 1)] = v
    return older


def _mistral3_adapter(weights: dict[str, mx.array], lora_parameters: dict) -> Path:
    path = Path(tempfile.mkdtemp())
    config = {
        "fine_tune_type": "lora",
        "num_layers": 2,
        "lora_parameters": lora_parameters,
    }
    (path / "adapter_config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(path / "adapters.safetensors"), weights)
    return path


class TestLoadAdapters(unittest.TestCase):
    def test_unknown_tensor_name_raises(self):
        path = _make_adapter(
            {
                "layers.0.proj.lora_a": mx.zeros((4, 2)),
                "layers.0.proj.lora_b": mx.zeros((2, 3)),
                "layers.0.proj.lora_typo": mx.zeros((1,)),
            }
        )
        with self.assertRaises(ValueError) as cm:
            load_adapters(TinyModel(), path)
        self.assertIn("lora_typo", str(cm.exception))

    def test_wrong_tensor_shape_raises(self):
        path = _make_adapter(
            {
                "layers.0.proj.lora_a": mx.zeros((5, 2)),
                "layers.0.proj.lora_b": mx.zeros((2, 3)),
            }
        )
        with self.assertRaises(ValueError) as cm:
            load_adapters(TinyModel(), path)
        self.assertIn("shape", str(cm.exception))

    def test_valid_adapter_loads(self):
        path = _make_adapter(
            {
                "layers.0.proj.lora_a": mx.zeros((4, 2)),
                "layers.0.proj.lora_b": mx.zeros((2, 3)),
            }
        )
        model = load_adapters(TinyModel(), path)
        self.assertIsInstance(model, nn.Module)

    def test_base_weight_tensor_rejected(self):
        path = _make_adapter(
            {
                "layers.0.proj.lora_a": mx.zeros((4, 2)),
                "layers.0.proj.lora_b": mx.zeros((2, 3)),
                "layers.0.proj.linear.weight": mx.full((3, 4), 42.0),
            }
        )
        with self.assertRaises(ValueError) as cm:
            load_adapters(TinyModel(), path)
        self.assertIn("linear.weight", str(cm.exception))

    def test_valid_dora_adapter_loads(self):
        path = Path(tempfile.mkdtemp())
        config = {
            "fine_tune_type": "dora",
            "num_layers": 1,
            "lora_parameters": {
                "keys": ["proj"],
                "rank": 2,
                "scale": 1.0,
                "dropout": 0.0,
            },
        }
        (path / "adapter_config.json").write_text(json.dumps(config))
        mx.save_safetensors(
            str(path / "adapters.safetensors"),
            {
                "layers.0.proj.lora_a": mx.zeros((4, 2)),
                "layers.0.proj.lora_b": mx.zeros((2, 3)),
                "layers.0.proj.m": mx.zeros((3,)),
            },
        )
        model = load_adapters(TinyModel(), path)
        self.assertIsInstance(model, nn.Module)

    def test_adapter_saved_under_older_layer_names_loads(self):
        adapters = _mistral3_lora_weights()
        path = _mistral3_adapter(_older_names(adapters), LORA_PARAMETERS)
        loaded = dict(tree_flatten(load_adapters(_mistral3(), path).parameters()))
        for k, v in adapters.items():
            self.assertTrue(mx.array_equal(loaded[k], v), k)

    def test_lm_head_adapter_saved_under_older_names_loads(self):
        model = _mistral3()
        linear_to_lora_layers(model, 2, {**LORA_PARAMETERS, "keys": ["lm_head"]})
        adapters = {
            k: mx.random.normal(v.shape)
            for k, v in tree_flatten(model.parameters())
            if k.endswith(("lora_a", "lora_b"))
        }
        older_keys = {**LORA_PARAMETERS, "keys": ["language_model.lm_head"]}
        path = _mistral3_adapter(_older_names(adapters), older_keys)
        loaded = dict(tree_flatten(load_adapters(_mistral3(), path).parameters()))
        for k, v in adapters.items():
            self.assertTrue(mx.array_equal(loaded[k], v), k)


class TestLoadAdapterWeights(unittest.TestCase):
    def test_older_layer_names_load(self):
        adapters = _mistral3_lora_weights()
        path = Path(tempfile.mkdtemp()) / "adapters.safetensors"
        mx.save_safetensors(str(path), _older_names(adapters))

        model = _mistral3()
        linear_to_lora_layers(model, 2, LORA_PARAMETERS)
        load_adapter_weights(model, path=path, fine_tune_type="lora")
        loaded = dict(tree_flatten(model.parameters()))
        for k, v in adapters.items():
            self.assertTrue(mx.array_equal(loaded[k], v), k)

    def test_unknown_tensor_name_raises(self):
        path = Path(tempfile.mkdtemp()) / "adapters.safetensors"
        mx.save_safetensors(
            str(path),
            {
                "layers.0.proj.lora_a": mx.zeros((4, 2)),
                "layers.0.proj.lora_b": mx.zeros((2, 3)),
                "layers.0.proj.lora_typo": mx.zeros((1,)),
            },
        )
        model = TinyModel()
        linear_to_lora_layers(model, 1, {"keys": ["proj"], **LORA_PARAMETERS})
        with self.assertRaises(ValueError) as cm:
            load_adapter_weights(model, path=path, fine_tune_type="lora")
        self.assertIn("lora_typo", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
