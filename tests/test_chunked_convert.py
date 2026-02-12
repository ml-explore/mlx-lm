# Copyright © 2023-2024 Apple Inc.

import copy
import gc
import glob
import json
import shutil
import struct
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.chunked_convert import (
    WeightChunk,
    build_quant_plan,
    chunked_sanitize,
    convert_chunked,
    estimate_quantized_size,
    plan_layer_chunks,
    read_safetensors_header,
    scan_source_tensors,
    validate_chunked_compatible,
)
from mlx_lm.utils import MAX_FILE_SIZE_GB, save_config, save_model


def _make_llama_config(
    num_hidden_layers=4,
    hidden_size=256,
    intermediate_size=512,
    num_attention_heads=4,
    vocab_size=1000,
):
    return {
        "model_type": "llama",
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "intermediate_size": intermediate_size,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_attention_heads,
        "rms_norm_eps": 1e-5,
        "vocab_size": vocab_size,
        "tie_word_embeddings": False,
    }


def _save_synthetic_model(model_dir, config, model):
    """Save a synthetic model as sharded safetensors with index."""
    from mlx.utils import tree_flatten

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    weights = dict(tree_flatten(model.parameters()))
    mx.eval(*weights.values())

    # Split into 2 shards for testing
    keys = sorted(weights.keys())
    mid = len(keys) // 2
    shard1_keys = keys[:mid]
    shard2_keys = keys[mid:]

    shard1 = {k: weights[k] for k in shard1_keys}
    shard2 = {k: weights[k] for k in shard2_keys}

    mx.save_safetensors(str(model_dir / "model-00001-of-00002.safetensors"), shard1)
    mx.save_safetensors(str(model_dir / "model-00002-of-00002.safetensors"), shard2)

    weight_map = {}
    for k in shard1_keys:
        weight_map[k] = "model-00001-of-00002.safetensors"
    for k in shard2_keys:
        weight_map[k] = "model-00002-of-00002.safetensors"

    index = {
        "metadata": {"total_size": sum(v.nbytes for v in weights.values())},
        "weight_map": dict(sorted(weight_map.items())),
    }
    with open(model_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=4)

    save_config(config, model_dir / "config.json")

    return weight_map


class TestSafetensorsHeader(unittest.TestCase):
    def test_read_safetensors_header(self):
        """Create synthetic safetensors, verify header parsing returns correct shapes/dtypes."""
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            data = {"weight": mx.random.normal((128, 64))}
            mx.eval(data["weight"])
            mx.save_safetensors(f.name, data)
            header = read_safetensors_header(f.name)
            self.assertIn("weight", header)
            self.assertEqual(header["weight"]["shape"], [128, 64])
            self.assertIn("dtype", header["weight"])
            self.assertGreater(header["weight"]["size_bytes"], 0)

    def test_read_header_multiple_tensors(self):
        """Verify header parsing for multiple tensors."""
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            data = {
                "a.weight": mx.random.normal((32, 16)),
                "b.weight": mx.random.normal((64, 32)),
            }
            mx.eval(*data.values())
            mx.save_safetensors(f.name, data)
            header = read_safetensors_header(f.name)
            self.assertEqual(len(header), 2)
            self.assertEqual(header["a.weight"]["shape"], [32, 16])
            self.assertEqual(header["b.weight"]["shape"], [64, 32])


class TestPlanLayerChunks(unittest.TestCase):
    def test_basic_grouping(self):
        """Verify weight keys are grouped correctly by layer."""
        weight_map = {
            "model.embed_tokens.weight": "f1.safetensors",
            "model.layers.0.mlp.up_proj.weight": "f1.safetensors",
            "model.layers.0.mlp.down_proj.weight": "f1.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "f1.safetensors",
            "model.layers.1.mlp.up_proj.weight": "f2.safetensors",
            "model.layers.1.mlp.down_proj.weight": "f2.safetensors",
            "lm_head.weight": "f2.safetensors",
            "model.norm.weight": "f2.safetensors",
        }
        chunks = plan_layer_chunks(weight_map)
        # Should be: non-layer chunk, layer 0, layer 1
        self.assertEqual(len(chunks), 3)

        # Non-layer chunk first
        self.assertIsNone(chunks[0].layer_idx)
        self.assertIn("model.embed_tokens.weight", chunks[0].weight_keys)
        self.assertIn("lm_head.weight", chunks[0].weight_keys)
        self.assertIn("model.norm.weight", chunks[0].weight_keys)

        # Layer 0
        self.assertEqual(chunks[1].layer_idx, 0)
        self.assertEqual(len(chunks[1].weight_keys), 3)

        # Layer 1
        self.assertEqual(chunks[2].layer_idx, 1)
        self.assertEqual(len(chunks[2].weight_keys), 2)

    def test_no_non_layer_weights(self):
        """Model with only layer weights."""
        weight_map = {
            "model.layers.0.weight": "f1.safetensors",
            "model.layers.1.weight": "f1.safetensors",
        }
        chunks = plan_layer_chunks(weight_map)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].layer_idx, 0)
        self.assertEqual(chunks[1].layer_idx, 1)

    def test_source_files_tracked(self):
        """Source files are correctly tracked per chunk."""
        weight_map = {
            "model.layers.0.a": "f1.safetensors",
            "model.layers.0.b": "f2.safetensors",
        }
        chunks = plan_layer_chunks(weight_map)
        self.assertEqual(chunks[0].source_files, {"f1.safetensors", "f2.safetensors"})


class TestEstimateQuantizedSize(unittest.TestCase):
    def test_affine_4bit(self):
        """Verify size estimate matches actual mx.quantize output for affine 4-bit."""
        shape = [512, 256]
        bits = 4
        group_size = 64
        estimated = estimate_quantized_size(shape, bits=bits, group_size=group_size)

        w = mx.random.normal((512, 256))
        qw, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
        mx.eval(qw, scales, biases)
        actual = qw.nbytes + scales.nbytes + biases.nbytes
        self.assertEqual(estimated, actual)

    def test_affine_8bit(self):
        """Verify size estimate for 8-bit quantization."""
        shape = [256, 128]
        bits = 8
        group_size = 32
        estimated = estimate_quantized_size(shape, bits=bits, group_size=group_size)

        w = mx.random.normal((256, 128))
        qw, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
        mx.eval(qw, scales, biases)
        actual = qw.nbytes + scales.nbytes + biases.nbytes
        self.assertEqual(estimated, actual)


class TestBuildQuantPlan(unittest.TestCase):
    def test_basic_quant_plan(self):
        """Build quant plan for a small llama model."""
        config = _make_llama_config()
        plan, quant_config = build_quant_plan(
            config, q_bits=4, q_group_size=64, q_mode="affine"
        )
        # Should have entries for quantizable weight keys
        self.assertGreater(len(plan), 0)
        # All entries should be dicts or False
        for key, params in plan.items():
            self.assertTrue(
                params is False or isinstance(params, dict),
                f"Bad plan entry for {key}: {params}",
            )


class TestValidateChunkedCompatible(unittest.TestCase):
    def test_vl_model_rejected(self):
        """Vision-language models should be rejected."""
        config = {"model_type": "llama", "vision_config": {"hidden_size": 128}}
        with self.assertRaises(ValueError) as ctx:
            validate_chunked_compatible(config)
        self.assertIn("vision-language", str(ctx.exception))

    def test_model_file_rejected(self):
        """Custom model_file models should be rejected."""
        config = {"model_type": "llama", "model_file": "custom_model.py"}
        with self.assertRaises(ValueError) as ctx:
            validate_chunked_compatible(config)
        self.assertIn("custom model files", str(ctx.exception))

    def test_normal_model_accepted(self):
        """Normal text models should pass validation."""
        config = {"model_type": "llama"}
        # Should not raise
        validate_chunked_compatible(config)


class TestChunkedSanitize(unittest.TestCase):
    def test_no_sanitize(self):
        """Model without sanitize returns weights unchanged."""

        class FakeModel:
            pass

        model = FakeModel()
        weights = {"a": mx.zeros((2,)), "b": mx.ones((3,))}
        result = chunked_sanitize(model, weights, layer_idx=0)
        self.assertEqual(set(result.keys()), {"a", "b"})

    def test_passthrough_sanitize(self):
        """Model with identity sanitize returns weights unchanged."""

        class FakeModel:
            def sanitize(self, weights):
                return weights

        model = FakeModel()
        weights = {"a": mx.zeros((2,))}
        result = chunked_sanitize(model, weights, layer_idx=0)
        self.assertEqual(set(result.keys()), {"a"})


class TestChunkedConvertIntegration(unittest.TestCase):
    def test_chunked_vs_standard_quantized(self):
        """Compare chunked and standard conversion on synthetic Llama model."""
        from mlx_lm.convert import convert
        from mlx_lm.models.llama import Model, ModelArgs

        config = _make_llama_config()
        args = ModelArgs.from_dict(config)
        model = Model(args)
        mx.eval(model.parameters())

        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = Path(tmpdir) / "source"
            _save_synthetic_model(src_dir, config, model)

            # Also save a tokenizer (minimal)
            tok_dir = src_dir
            tok_config = {"model_type": "llama"}
            with open(tok_dir / "tokenizer_config.json", "w") as f:
                json.dump(tok_config, f)
            with open(tok_dir / "special_tokens_map.json", "w") as f:
                json.dump({}, f)

            # Standard conversion
            std_path = Path(tmpdir) / "standard"
            convert(
                str(src_dir),
                str(std_path),
                quantize=True,
                q_group_size=64,
                q_bits=4,
            )

            # Chunked conversion
            chunked_path = Path(tmpdir) / "chunked"
            convert_chunked(
                str(src_dir),
                str(chunked_path),
                quantize=True,
                q_group_size=64,
                q_bits=4,
            )

            # Load and compare all output weights
            std_weights = {}
            for f_path in sorted(glob.glob(str(std_path / "model*.safetensors"))):
                if "index" not in f_path:
                    std_weights.update(mx.load(f_path))
            chunked_weights = {}
            for f_path in sorted(glob.glob(str(chunked_path / "model*.safetensors"))):
                if "index" not in f_path:
                    chunked_weights.update(mx.load(f_path))

            self.assertEqual(
                set(std_weights.keys()),
                set(chunked_weights.keys()),
                f"Key mismatch: std has {set(std_weights.keys()) - set(chunked_weights.keys())}, "
                f"chunked has {set(chunked_weights.keys()) - set(std_weights.keys())}",
            )
            for key in std_weights:
                self.assertTrue(
                    mx.array_equal(std_weights[key], chunked_weights[key]),
                    f"Mismatch in {key}: shapes std={std_weights[key].shape} "
                    f"chunked={chunked_weights[key].shape}",
                )

    def test_chunked_dtype_only(self):
        """Dtype-only conversion (no quantize) produces correct output."""
        from mlx_lm.models.llama import Model, ModelArgs

        config = _make_llama_config(num_hidden_layers=2)
        args = ModelArgs.from_dict(config)
        model = Model(args)
        mx.eval(model.parameters())

        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = Path(tmpdir) / "source"
            _save_synthetic_model(src_dir, config, model)

            with open(src_dir / "tokenizer_config.json", "w") as f:
                json.dump({"model_type": "llama"}, f)
            with open(src_dir / "special_tokens_map.json", "w") as f:
                json.dump({}, f)

            chunked_path = Path(tmpdir) / "chunked"
            convert_chunked(
                str(src_dir),
                str(chunked_path),
                quantize=False,
                dtype="float16",
            )

            # Verify output exists and has weights
            out_files = glob.glob(str(chunked_path / "model*.safetensors"))
            out_files = [f for f in out_files if "index" not in f]
            self.assertGreater(len(out_files), 0)

            # Verify all weights are float16
            for f_path in out_files:
                weights = mx.load(f_path)
                for key, val in weights.items():
                    if mx.issubdtype(val.dtype, mx.floating):
                        self.assertEqual(
                            val.dtype,
                            mx.float16,
                            f"{key} has dtype {val.dtype}, expected float16",
                        )

    def test_chunked_moe_sanitize(self):
        """Chunked conversion correctly stacks MoE expert weights via sanitize."""
        from mlx_lm.models.mixtral import Model, ModelArgs

        config = {
            "model_type": "mixtral",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_local_experts": 2,
            "num_experts_per_tok": 1,
            "rms_norm_eps": 1e-5,
            "vocab_size": 256,
            "tie_word_embeddings": False,
        }
        args = ModelArgs.from_dict(config)
        model = Model(args)
        mx.eval(model.parameters())

        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = Path(tmpdir) / "source"
            src_dir.mkdir(parents=True)

            # Save the stacked-expert model, then unstack to per-expert format
            # to simulate the raw HF weight layout that sanitize processes
            from mlx.utils import tree_flatten

            weights = dict(tree_flatten(model.parameters()))
            mx.eval(*weights.values())

            # Unstack expert weights from switch_mlp back to per-expert keys
            raw_weights = {}
            for key, val in weights.items():
                if "switch_mlp" in key:
                    # e.g. model.layers.0.block_sparse_moe.switch_mlp.gate_proj.weight
                    # -> model.layers.0.block_sparse_moe.experts.{e}.w1.weight
                    name_map = {
                        "gate_proj": "w1",
                        "down_proj": "w2",
                        "up_proj": "w3",
                    }
                    for proj_name, expert_name in name_map.items():
                        if proj_name in key:
                            prefix = key.split(".switch_mlp.")[0]
                            suffix = key.split(f".{proj_name}.")[-1]
                            for e in range(config["num_local_experts"]):
                                expert_key = (
                                    f"{prefix}.experts.{e}.{expert_name}.{suffix}"
                                )
                                raw_weights[expert_key] = val[e]
                            break
                else:
                    raw_weights[key] = val

            mx.eval(*raw_weights.values())

            # Split into shards and save with index
            keys = sorted(raw_weights.keys())
            mid = len(keys) // 2
            shard1 = {k: raw_weights[k] for k in keys[:mid]}
            shard2 = {k: raw_weights[k] for k in keys[mid:]}

            mx.save_safetensors(
                str(src_dir / "model-00001-of-00002.safetensors"), shard1
            )
            mx.save_safetensors(
                str(src_dir / "model-00002-of-00002.safetensors"), shard2
            )

            weight_map = {}
            for k in keys[:mid]:
                weight_map[k] = "model-00001-of-00002.safetensors"
            for k in keys[mid:]:
                weight_map[k] = "model-00002-of-00002.safetensors"

            index = {
                "metadata": {"total_size": sum(v.nbytes for v in raw_weights.values())},
                "weight_map": dict(sorted(weight_map.items())),
            }
            with open(src_dir / "model.safetensors.index.json", "w") as f:
                json.dump(index, f, indent=4)
            save_config(config, src_dir / "config.json")

            # Create a minimal tokenizer using the tokenizers library
            from tokenizers import Tokenizer, models

            tok = Tokenizer(models.BPE())
            tok.save(str(src_dir / "tokenizer.json"))
            with open(src_dir / "tokenizer_config.json", "w") as f:
                json.dump({"tokenizer_class": "PreTrainedTokenizerFast"}, f)
            with open(src_dir / "special_tokens_map.json", "w") as f:
                json.dump({}, f)

            # Run chunked conversion (dtype only, no quant for simplicity)
            out_path = Path(tmpdir) / "chunked"
            convert_chunked(
                str(src_dir),
                str(out_path),
                quantize=False,
                dtype="float16",
            )

            # Load output and verify expert weights were stacked
            out_weights = {}
            for f_path in sorted(glob.glob(str(out_path / "model*.safetensors"))):
                if "index" not in f_path:
                    out_weights.update(mx.load(f_path))

            # Should have switch_mlp keys (stacked), not per-expert keys
            for key in out_weights:
                self.assertNotIn(
                    "experts.", key, f"Per-expert key found in output: {key}"
                )

            # Verify stacked expert weights exist for both layers
            for layer_idx in range(config["num_hidden_layers"]):
                for proj in ["gate_proj", "down_proj", "up_proj"]:
                    stacked_key = (
                        f"model.layers.{layer_idx}."
                        f"block_sparse_moe.switch_mlp.{proj}.weight"
                    )
                    self.assertIn(
                        stacked_key,
                        out_weights,
                        f"Missing stacked expert key: {stacked_key}",
                    )
                    # Should have shape [num_experts, ...]
                    self.assertEqual(
                        out_weights[stacked_key].shape[0],
                        config["num_local_experts"],
                        f"Wrong expert count in {stacked_key}",
                    )

    def test_chunked_dequantize_error(self):
        """--chunked with dequantize raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            convert_chunked(
                "/fake/path",
                "/fake/output",
                dequantize=True,
            )
        self.assertIn("--chunked", str(ctx.exception))


class TestChunkedConvertCLI(unittest.TestCase):
    def test_chunked_flag_in_parser(self):
        """Verify --chunked flag is registered in the argument parser."""
        from mlx_lm.convert import configure_parser

        parser = configure_parser()
        args = parser.parse_args(["--hf-path", "test/model", "--chunked"])
        self.assertTrue(args.chunked)

    def test_chunked_flag_default_false(self):
        """Verify --chunked defaults to False."""
        from mlx_lm.convert import configure_parser

        parser = configure_parser()
        args = parser.parse_args(["--hf-path", "test/model"])
        self.assertFalse(args.chunked)


if __name__ == "__main__":
    unittest.main()
