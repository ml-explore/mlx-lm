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


CUSTOM_MODEL_FILE = """\
from pathlib import Path

import mlx.nn as nn

(Path(__file__).parent / "side_effect.txt").write_text("executed")


class ModelArgs:
    @classmethod
    def from_dict(cls, params):
        return cls()


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.linear = nn.Linear(8, 8)

    def __call__(self, x):
        return self.linear(x)
"""


class TestTrustRemoteCode(unittest.TestCase):
    def setUp(self):
        self.model_dir_fid = tempfile.TemporaryDirectory()
        self.model_path = Path(self.model_dir_fid.name)

    def tearDown(self):
        self.model_dir_fid.cleanup()

    @property
    def _side_effect_file(self) -> Path:
        return self.model_path / "side_effect.txt"

    def _write_custom_model_dir(self):
        config = {"model_type": "custom", "model_file": "arch.py"}
        with open(self.model_path / "config.json", "w") as f:
            json.dump(config, f)
        (self.model_path / "arch.py").write_text(CUSTOM_MODEL_FILE)
        mx.save_safetensors(
            str(self.model_path / "model.safetensors"),
            {"linear.weight": mx.zeros((8, 8)), "linear.bias": mx.zeros((8,))},
        )

    def test_model_file_blocked_by_default(self):
        """load_model must refuse to execute a custom model_file by default."""
        self._write_custom_model_dir()
        with self.assertRaises(ValueError) as cm:
            utils.load_model(self.model_path)
        self.assertIn("trust_remote_code", str(cm.exception))
        self.assertFalse(self._side_effect_file.exists())

    def test_model_file_loads_with_trust_remote_code(self):
        """Passing trust_remote_code=True opts in to executing model_file."""
        self._write_custom_model_dir()
        model, config = utils.load_model(self.model_path, trust_remote_code=True)
        self.assertIsInstance(model, nn.Module)
        self.assertEqual(config["model_file"], "arch.py")
        self.assertTrue(self._side_effect_file.exists())

    def test_normal_model_unaffected_by_default(self):
        """Models without model_file load fine with default arguments."""
        config = {
            "model_type": "llama",
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-5,
            "vocab_size": 64,
        }
        with open(self.model_path / "config.json", "w") as f:
            json.dump(config, f)
        model, loaded_config = utils.load_model(self.model_path, strict=False)
        self.assertIsInstance(model, nn.Module)
        self.assertEqual(loaded_config["model_type"], "llama")


class TestQ4_0Calibration(unittest.TestCase):
    """The q4_0 grid: scale from the *signed* element of largest magnitude."""

    def _negative_extremum_group(self):
        # extremum = -4 -> d = -4 / -8 = 0.5, code = floor(w / 0.5 + 8.5)
        weights = [0.0] * 32
        weights[0] = -4.0
        weights[2] = 0.5
        weights[3] = -0.5
        weights[4] = 3.5
        expected = [8] * 32  # w == 0 -> floor(8.5)
        expected[0] = 0  # floor(-8 + 8.5)
        expected[2] = 9  # floor(1 + 8.5)
        expected[3] = 7  # floor(-1 + 8.5)
        expected[4] = 15  # floor(7 + 8.5)
        return weights, expected

    def _unpack(self, packed):
        words = packed.flatten().tolist()
        return [(w >> (4 * i)) & 0xF for w in words for i in range(8)]

    def test_hand_computable_codes_scales_and_biases(self):
        weights, expected = self._negative_extremum_group()
        w = mx.array(weights).reshape(1, 32)

        packed, scales, biases = utils.q4_0_quantize(w)

        self.assertEqual(self._unpack(packed), expected)
        self.assertEqual(scales.tolist(), [[0.5]])
        self.assertEqual(biases.tolist(), [[-4.0]])

    def test_uses_signed_extremum_not_abs_max(self):
        """The regression that matters: -abs(w).max()/8 flips the sign on nearly half
        of all groups, silently producing a different grid."""
        weights, _ = self._negative_extremum_group()
        w = mx.array(weights).reshape(1, 32)

        packed, scales, _ = utils.q4_0_quantize(w)

        self.assertEqual(scales.tolist(), [[0.5]])
        self.assertEqual(self._unpack(packed)[0], 0)

        # -abs max would give d = -0.5, sending the same element to code 15.
        wrong_scale = -max(abs(v) for v in weights) / 8
        self.assertEqual(wrong_scale, -0.5)
        self.assertEqual(min(15, int(weights[0] / wrong_scale + 8.5)), 15)

    def test_positive_extremum_and_clipping(self):
        weights = [0.0] * 32
        weights[0] = 4.0
        weights[1] = -3.5
        w = mx.array(weights).reshape(1, 32)

        packed, scales, biases = utils.q4_0_quantize(w)
        codes = self._unpack(packed)

        self.assertEqual(scales.tolist(), [[-0.5]])
        self.assertEqual(biases.tolist(), [[4.0]])
        self.assertEqual(codes[0], 0)
        self.assertEqual(codes[1], 15)
        self.assertTrue(all(0 <= c <= 15 for c in codes))

    def test_all_zero_group_does_not_divide_by_zero(self):
        w = mx.zeros((1, 32))

        packed, scales, biases = utils.q4_0_quantize(w)

        self.assertEqual(scales.tolist(), [[0.0]])
        self.assertEqual(biases.tolist(), [[0.0]])
        self.assertEqual(self._unpack(packed), [8] * 32)

    def test_scales_are_per_group_across_rows(self):
        values = [0.0] * 128
        values[0] = -4.0  # row 0, group 0 -> 0.5
        values[32] = 2.0  # row 0, group 1 -> -0.25
        values[64] = 8.0  # row 1, group 0 -> -1.0
        values[96] = -1.0  # row 1, group 1 -> 0.125
        w = mx.array(values).reshape(2, 64)

        _, scales, biases = utils.q4_0_quantize(w)

        self.assertEqual(scales.tolist(), [[0.5, -0.25], [-1.0, 0.125]])
        self.assertEqual(biases.tolist(), [[-4.0, 2.0], [8.0, -1.0]])

    def test_dequantizes_onto_the_intended_lattice(self):
        weights, expected = self._negative_extremum_group()
        w = mx.array(weights).reshape(1, 32)

        packed, scales, biases = utils.q4_0_quantize(w)
        restored = mx.dequantize(
            packed, scales=scales, biases=biases, group_size=32, bits=4, mode="affine"
        )

        scale = scales.tolist()[0][0]
        want = [(c - 8) * scale for c in expected]
        self.assertTrue(mx.allclose(restored.flatten(), mx.array(want)).item())

    def test_rejects_incompatible_settings(self):
        model = nn.Sequential(nn.Linear(32, 8))
        for kwargs in (
            {"bits": 8, "group_size": 32},
            {"bits": 4, "group_size": 64},
        ):
            with self.assertRaises(ValueError):
                utils.quantize_model(
                    model, {}, calibration="q4_0", mode="affine", **kwargs
                )

    def test_accepts_omitted_settings(self):
        model = nn.Sequential(nn.Linear(32, 8))
        _, config = utils.quantize_model(
            model, {}, group_size=None, bits=None, calibration="q4_0"
        )
        self.assertEqual(config["quantization"]["bits"], 4)
        self.assertEqual(config["quantization"]["group_size"], 32)
        self.assertEqual(config["quantization"]["mode"], "affine")

    def test_calibrated_linear_matches_direct_derivation(self):
        mx.random.seed(0)
        linear = nn.Linear(64, 16, bias=True)
        original = mx.array(linear.weight)
        model = nn.Sequential(linear)

        model, _ = utils.quantize_model(model, {}, None, None, calibration="q4_0")

        want_w, want_s, want_b = utils.q4_0_quantize(original)
        self.assertTrue(mx.array_equal(model.layers[0].weight, want_w).item())
        self.assertTrue(mx.array_equal(model.layers[0].scales, want_s).item())
        self.assertTrue(mx.array_equal(model.layers[0].biases, want_b).item())

    def test_predicate_is_evaluated_once_per_module(self):
        """A predicate is not required to be pure; evaluating it twice can disagree
        between the capture and conversion passes and corrupt a module."""
        calls = {}

        def counting_predicate(path, module):
            calls[path] = calls.get(path, 0) + 1
            return True

        model = nn.Sequential(nn.Linear(32, 8), nn.Linear(32, 8))
        utils.quantize_model(
            model,
            {},
            None,
            None,
            calibration="q4_0",
            quant_predicate=counting_predicate,
        )

        self.assertTrue(calls)
        self.assertTrue(all(n == 1 for n in calls.values()), calls)

    def test_incompatible_per_layer_parameters_are_rejected(self):
        def mixed_predicate(path, module):
            return {"bits": 6, "group_size": 64}

        model = nn.Sequential(nn.Linear(64, 8))
        with self.assertRaises(ValueError):
            utils.quantize_model(
                model,
                {},
                None,
                None,
                calibration="q4_0",
                quant_predicate=mixed_predicate,
            )

    def test_compatible_per_layer_parameters_are_allowed(self):
        def matching_predicate(path, module):
            return {"bits": 4, "group_size": 32}

        model = nn.Sequential(nn.Linear(32, 8))
        model, _ = utils.quantize_model(
            model,
            {},
            None,
            None,
            calibration="q4_0",
            quant_predicate=matching_predicate,
        )
        self.assertEqual(model.layers[0].bits, 4)
        self.assertEqual(model.layers[0].group_size, 32)

    def test_helper_rejects_unsupported_shapes(self):
        with self.assertRaises(ValueError):
            utils.q4_0_quantize(mx.zeros((1, 20)))
        with self.assertRaises(ValueError):
            utils.q4_0_quantize(mx.zeros((2, 2, 32)))

    def test_unknown_calibration_is_rejected(self):
        with self.assertRaises(ValueError):
            utils.quantize_model(
                nn.Sequential(nn.Linear(32, 8)), {}, None, None, calibration="nope"
            )


if __name__ == "__main__":
    unittest.main()
