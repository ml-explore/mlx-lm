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

from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.tuner.utils import load_adapters, print_trainable_parameters


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


if __name__ == "__main__":
    unittest.main()
