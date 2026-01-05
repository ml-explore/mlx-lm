# Copyright © 2024 Apple Inc.

import os
import tempfile
import unittest

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


    def test_awq_weight_transformation(self):
        """Test transformation of AutoAWQ packed weights to MLX format."""
        from mlx_lm.utils import _unpack_awq_weights, _transform_awq_weights

        # Test unpacking 4-bit weights
        bits = 4
        pack_factor = 32 // bits  # 8 values per uint32

        # Create a simple packed weight: [2, 4] containing 2*32=64 values
        # Each uint32 holds 8 4-bit values (0-15)
        out_features = 2
        packed_in = 4
        in_features = packed_in * pack_factor  # 32

        # Create test data: pack values 0,1,2,3,4,5,6,7 into each uint32
        packed_values = []
        for _ in range(out_features * packed_in):
            val = 0
            for i in range(pack_factor):
                val |= (i % 16) << (i * bits)
            packed_values.append(val)

        qweight = mx.array(packed_values, dtype=mx.uint32).reshape(out_features, packed_in)

        # Test unpacking function directly
        unpacked = _unpack_awq_weights(qweight, bits)
        self.assertEqual(unpacked.shape, (out_features, in_features))

        # Verify unpacked values
        for row in range(out_features):
            for col in range(in_features):
                expected = col % pack_factor
                self.assertEqual(unpacked[row, col].item(), expected)

        # Test full transformation
        group_size = 8
        n_groups = in_features // group_size

        # Create mock AWQ weights dict
        # qzeros shape: [n_groups, out_features // pack_factor]
        qzeros_packed_out = out_features // pack_factor if out_features >= pack_factor else 1
        weights = {
            "layer.qweight": qweight,
            "layer.scales": mx.ones((n_groups, out_features), dtype=mx.float16),
            "layer.qzeros": mx.zeros((n_groups, qzeros_packed_out), dtype=mx.uint32),
            "layer.other_param": mx.array([1.0, 2.0]),
        }

        quantization_config = {"bits": bits, "group_size": group_size}
        new_weights, mlx_quant = _transform_awq_weights(weights, quantization_config)

        # Verify transformation results
        self.assertIn("layer.weight", new_weights)
        self.assertIn("layer.scales", new_weights)
        self.assertIn("layer.biases", new_weights)
        self.assertIn("layer.other_param", new_weights)
        self.assertNotIn("layer.qweight", new_weights)
        self.assertNotIn("layer.qzeros", new_weights)

        # Verify weight is preserved (same packed format)
        self.assertTrue(mx.array_equal(new_weights["layer.weight"], qweight))

        # Verify scales shape is transposed to [out_features, n_groups]
        self.assertEqual(new_weights["layer.scales"].shape, (out_features, n_groups))
        self.assertEqual(new_weights["layer.biases"].shape, (out_features, n_groups))

        self.assertEqual(mlx_quant["bits"], bits)
        self.assertEqual(mlx_quant["group_size"], group_size)

        # Test that g_idx raises an error
        weights_with_gidx = {
            "layer.qweight": qweight,
            "layer.scales": mx.ones((n_groups, out_features), dtype=mx.float16),
            "layer.g_idx": mx.arange(in_features),
        }
        with self.assertRaises(ValueError) as context:
            _transform_awq_weights(weights_with_gidx, quantization_config)
        self.assertIn("g_idx", str(context.exception))


if __name__ == "__main__":
    unittest.main()
