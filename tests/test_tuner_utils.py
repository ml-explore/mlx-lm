# Copyright © 2024 Apple Inc.

import json
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.tuner.utils import (
    _convert_peft_config,
    _is_peft_config,
    _remap_peft_weights,
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


def _make_peft_config(r=8, lora_alpha=16, dropout=0.05):
    return {
        "peft_type": "LORA",
        "r": r,
        "lora_alpha": lora_alpha,
        "lora_dropout": dropout,
        "target_modules": ["q_proj", "v_proj"],
        "bias": "none",
    }


def _make_mlx_config():
    return {
        "fine_tune_type": "lora",
        "num_layers": 4,
        "lora_parameters": {"rank": 8, "scale": 20.0, "dropout": 0.0},
    }


def _make_peft_weights(num_layers=2, rank=8, in_dim=64, out_dim=64):
    weights = {}
    for i in range(num_layers):
        prefix = f"base_model.model.model.layers.{i}.self_attn.q_proj"
        weights[f"{prefix}.lora_A.weight"] = mx.zeros((rank, in_dim))
        weights[f"{prefix}.lora_B.weight"] = mx.zeros((out_dim, rank))
    return weights


class TestIsPeftConfig(unittest.TestCase):
    def test_returns_true_for_peft_config(self):
        self.assertTrue(_is_peft_config(_make_peft_config()))

    def test_returns_false_for_mlx_config(self):
        self.assertFalse(_is_peft_config(_make_mlx_config()))

    def test_returns_false_for_empty_dict(self):
        self.assertFalse(_is_peft_config({}))

    def test_case_sensitive(self):
        self.assertFalse(_is_peft_config({"PEFT_TYPE": "LORA"}))


class TestConvertPeftConfig(unittest.TestCase):
    @patch("mlx_lm.tuner.utils.mx.load")
    def test_basic_conversion(self, mock_load):
        mock_load.return_value = _make_peft_weights(num_layers=3, rank=8)
        with patch.object(Path, "exists", return_value=True):
            config = _convert_peft_config(_make_peft_config(r=8, lora_alpha=16), Path("/fake"))
        self.assertEqual(config.fine_tune_type, "lora")
        self.assertEqual(config.num_layers, 3)
        self.assertEqual(config.lora_parameters["rank"], 8)
        self.assertAlmostEqual(config.lora_parameters["scale"], 2.0)
        self.assertAlmostEqual(config.lora_parameters["dropout"], 0.05)

    @patch("mlx_lm.tuner.utils.mx.load")
    def test_scale_equals_one_when_alpha_equals_rank(self, mock_load):
        mock_load.return_value = _make_peft_weights(num_layers=2)
        with patch.object(Path, "exists", return_value=True):
            config = _convert_peft_config(_make_peft_config(r=16, lora_alpha=16), Path("/fake"))
        self.assertAlmostEqual(config.lora_parameters["scale"], 1.0)

    @patch("mlx_lm.tuner.utils.mx.load")
    def test_missing_lora_alpha_defaults_to_rank(self, mock_load):
        cfg = _make_peft_config(r=8)
        del cfg["lora_alpha"]
        mock_load.return_value = _make_peft_weights(num_layers=2)
        with patch.object(Path, "exists", return_value=True):
            config = _convert_peft_config(cfg, Path("/fake"))
        self.assertAlmostEqual(config.lora_parameters["scale"], 1.0)

    @patch("mlx_lm.tuner.utils.mx.load")
    def test_num_layers_from_max_index(self, mock_load):
        weights = {}
        for idx in [0, 2, 5]:
            weights[f"base_model.model.model.layers.{idx}.self_attn.q_proj.lora_A.weight"] = mx.zeros((8, 64))
            weights[f"base_model.model.model.layers.{idx}.self_attn.q_proj.lora_B.weight"] = mx.zeros((64, 8))
        mock_load.return_value = weights
        with patch.object(Path, "exists", return_value=True):
            config = _convert_peft_config(_make_peft_config(r=8), Path("/fake"))
        self.assertEqual(config.num_layers, 6)  # max(0,2,5) + 1

    def test_raises_for_unsupported_peft_type(self):
        cfg = _make_peft_config()
        cfg["peft_type"] = "ADALORA"
        with self.assertRaises(ValueError):
            _convert_peft_config(cfg, Path("/fake"))

    @patch("mlx_lm.tuner.utils.mx.load")
    def test_raises_for_missing_rank(self, mock_load):
        mock_load.return_value = _make_peft_weights(num_layers=2)
        cfg = _make_peft_config()
        del cfg["r"]
        with patch.object(Path, "exists", return_value=True):
            with self.assertRaises(ValueError):
                _convert_peft_config(cfg, Path("/fake"))

    def test_raises_when_weight_file_missing(self):
        with patch.object(Path, "exists", return_value=False):
            with self.assertRaises(FileNotFoundError):
                _convert_peft_config(_make_peft_config(), Path("/fake"))

    @patch("mlx_lm.tuner.utils.mx.load")
    def test_raises_when_no_layer_indices(self, mock_load):
        mock_load.return_value = {"base_model.model.lm_head.lora_A.weight": mx.zeros((8, 64))}
        with patch.object(Path, "exists", return_value=True):
            with self.assertRaises(ValueError):
                _convert_peft_config(_make_peft_config(), Path("/fake"))


class TestRemapPeftWeights(unittest.TestCase):
    def test_lora_a_key_and_shape(self):
        weights = {"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": mx.zeros((8, 64))}
        result = _remap_peft_weights(weights)
        self.assertIn("model.layers.0.self_attn.q_proj.lora_a", result)
        self.assertEqual(result["model.layers.0.self_attn.q_proj.lora_a"].shape, (64, 8))

    def test_lora_b_key_and_shape(self):
        weights = {"base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": mx.zeros((64, 8))}
        result = _remap_peft_weights(weights)
        self.assertIn("model.layers.0.self_attn.q_proj.lora_b", result)
        self.assertEqual(result["model.layers.0.self_attn.q_proj.lora_b"].shape, (8, 64))

    def test_non_peft_prefix_dropped(self):
        weights = {"model.embed_tokens.weight": mx.zeros((100, 64))}
        self.assertEqual(_remap_peft_weights(weights), {})

    def test_base_model_weight_without_lora_suffix_dropped(self):
        weights = {"base_model.model.model.embed_tokens.weight": mx.zeros((100, 64))}
        self.assertEqual(_remap_peft_weights(weights), {})

    def test_multiple_layers(self):
        result = _remap_peft_weights(_make_peft_weights(num_layers=4))
        self.assertEqual(len(result), 8)  # 4 layers × 2 keys

    def test_values_are_transposed(self):
        a = mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (3, 2)
        weights = {"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": a}
        result = _remap_peft_weights(weights)
        expected = mx.transpose(a)  # (2, 3)
        self.assertTrue(
            mx.all(result["model.layers.0.self_attn.q_proj.lora_a"] == expected).item()
        )

    def test_empty_input(self):
        self.assertEqual(_remap_peft_weights({}), {})


class TestLoadAdaptersPeft(unittest.TestCase):
    def _write_adapter_config(self, tmpdir, config_dict, weight_filename):
        with open(Path(tmpdir) / "adapter_config.json", "w") as f:
            json.dump(config_dict, f)
        (Path(tmpdir) / weight_filename).touch()

    @patch("mlx_lm.tuner.utils.linear_to_lora_layers")
    @patch("mlx_lm.tuner.utils.mx.load")
    def test_peft_path_invoked(self, mock_load, mock_lora_layers):
        mock_load.return_value = _make_peft_weights(num_layers=2, rank=8)
        mock_model = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_adapter_config(tmpdir, _make_peft_config(r=8, lora_alpha=16), "adapter_model.safetensors")
            load_adapters(mock_model, tmpdir)

        mock_lora_layers.assert_called_once()
        args = mock_lora_layers.call_args[0]
        self.assertEqual(args[1], 2)  # num_layers

    @patch("mlx_lm.tuner.utils.linear_to_lora_layers")
    @patch("mlx_lm.tuner.utils.mx.load")
    def test_peft_load_weights_called_with_list(self, mock_load, mock_lora_layers):
        mock_load.return_value = _make_peft_weights(num_layers=2, rank=8)
        mock_model = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_adapter_config(tmpdir, _make_peft_config(), "adapter_model.safetensors")
            load_adapters(mock_model, tmpdir)

        call_arg = mock_model.load_weights.call_args[0][0]
        self.assertIsInstance(call_arg, list)

    @patch("mlx_lm.tuner.utils.linear_to_lora_layers")
    def test_mlx_native_path_unchanged(self, mock_lora_layers):
        mock_model = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_adapter_config(tmpdir, _make_mlx_config(), "adapters.safetensors")
            load_adapters(mock_model, tmpdir)

        mock_lora_layers.assert_called_once()
        self.assertEqual(mock_lora_layers.call_args[0][1], 4)  # num_layers from config
        call_arg = mock_model.load_weights.call_args[0][0]
        self.assertIsInstance(call_arg, str)  # path string, not list

    def test_raises_for_missing_path(self):
        with self.assertRaises(FileNotFoundError):
            load_adapters(MagicMock(), "/nonexistent/path")


if __name__ == "__main__":
    unittest.main()
