# Copyright © 2024 Apple Inc.

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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


class TestPreserveMTP(unittest.TestCase):
    def setUp(self):
        self.test_dir_fid = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.test_dir_fid.name)

    def tearDown(self):
        self.test_dir_fid.cleanup()

    def _write_source(self):
        """Write a tiny model whose source weights include dropped MTP weights."""
        from mlx_lm.models import mimo

        config = {
            "model_type": "mimo",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-5,
            "vocab_size": 320,
            "tie_word_embeddings": False,
            "num_nextn_predict_layers": 1,
        }
        model = mimo.Model(mimo.ModelArgs(**config))
        weights = dict(tree_flatten(model.parameters()))
        # MTP weights that `mimo.sanitize` drops, in mixed dtypes so the test
        # also covers that preservation keeps the source dtype.
        mtp_weights = {
            "model.mtp_layers.0.input_layernorm.weight": mx.random.normal((64,)).astype(
                mx.bfloat16
            ),
            "model.mtp_layers.0.token_layernorm.weight": mx.random.normal((64,)).astype(
                mx.float32
            ),
        }
        weights.update(mtp_weights)

        source = self.test_dir / "source"
        source.mkdir()
        mx.save_safetensors(str(source / "model.safetensors"), weights)
        with open(source / "config.json", "w") as f:
            json.dump(config, f)
        tokenizer = {
            "version": "1.0",
            "added_tokens": [],
            "pre_tokenizer": {"type": "Whitespace"},
            "model": {
                "type": "WordLevel",
                "vocab": {"<unk>": 0, "a": 1, "b": 2},
                "unk_token": "<unk>",
            },
        }
        with open(source / "tokenizer.json", "w") as f:
            json.dump(tokenizer, f)
        with open(source / "tokenizer_config.json", "w") as f:
            json.dump(
                {"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "<unk>"}, f
            )
        return source, mtp_weights

    def test_mtp_weight_matcher(self):
        """The MTP matcher covers each architecture's dropped weights."""
        from mlx_lm.convert import _is_mtp_weight

        mtp_names = [
            "model.mtp.embed_tokens.weight",  # qwen3_5, qwen3_next
            "mtp.layers.0.self_attn.q_proj.weight",  # nemotron_h, exaone_moe
            "model.mtp_layers.0.input_layernorm.weight",  # mimo
            "model.mtp.0.eh_proj.weight",  # mimo_v2_flash, kimi_linear, longcat_flash
            "model.layers.30.mtp.enorm.weight",  # step3p5
            "mtp_block.0.self_attn.qkv_proj.weight",  # ernie4_5_moe
            "model.mtp_hidden_norm.weight",  # ernie4_5_moe
        ]
        for name in mtp_names:
            self.assertTrue(_is_mtp_weight(name, None, None), name)

        # Trailing decoder layers past the dense stack.
        for model_type in [
            "deepseek_v3",
            "deepseek_v32",
            "glm4_moe",
            "glm4_moe_lite",
            "step3p5",
        ]:
            self.assertTrue(
                _is_mtp_weight(
                    "model.layers.30.self_attn.kv_b_proj.weight", model_type, 30
                ),
                model_type,
            )
        self.assertTrue(
            _is_mtp_weight("model.layers.61.eh_proj.weight", "deepseek_v3", 61)
        )

        # The trailing-layer rule only applies to the listed architectures.
        self.assertFalse(
            _is_mtp_weight("model.layers.30.self_attn.q_proj.weight", "llama", 30)
        )

        # Dense weights are never treated as MTP.
        for name in [
            "model.layers.29.self_attn.q_proj.weight",
            "model.embed_tokens.weight",
            "lm_head.weight",
            "model.layers.0.self_attn.rotary_emb.inv_freq",
        ]:
            self.assertFalse(_is_mtp_weight(name, "glm4_moe_lite", 30), name)

    def test_hf_repo_to_path_propagates_revision(self):
        """The preservation path resolves the source at the converted revision."""
        with mock.patch.object(
            utils, "snapshot_download", return_value=str(self.test_dir)
        ) as snapshot_download:
            path = utils.hf_repo_to_path("some/repo", revision="abc123")
        self.assertEqual(path, self.test_dir)
        self.assertEqual(snapshot_download.call_args.kwargs["revision"], "abc123")

    def test_convert_preserves_mtp_weights(self):
        """`preserve_mtp` writes the dropped MTP weights byte-identically."""
        source, mtp_weights = self._write_source()
        mlx_path = self.test_dir / "mlx_model"
        convert(str(source), mlx_path=str(mlx_path), dtype="float32", preserve_mtp=True)

        preserved = mx.load(str(mlx_path / "mtp.safetensors"))
        self.assertEqual(set(preserved), set(mtp_weights))
        for name, value in mtp_weights.items():
            self.assertEqual(preserved[name].dtype, value.dtype)  # source dtype kept
            self.assertTrue(mx.array_equal(preserved[name], value))

        # The sidecar has its own index covering exactly the preserved weights.
        with open(mlx_path / "mtp.safetensors.index.json") as f:
            index = json.load(f)
        self.assertEqual(set(index["weight_map"]), set(mtp_weights))
        for name in mtp_weights:
            self.assertEqual(index["weight_map"][name], "mtp.safetensors")

        # The converted model still loads and carries no MTP weights.
        model, _ = utils.load(str(mlx_path))
        loaded = dict(tree_flatten(model.parameters()))
        self.assertFalse(any("mtp" in name for name in loaded))

    def test_convert_without_flag_is_unchanged(self):
        """The output directory is identical with and without `preserve_mtp`."""
        source, _ = self._write_source()
        with_mtp = self.test_dir / "with_mtp"
        without_mtp = self.test_dir / "without_mtp"
        convert(str(source), mlx_path=str(with_mtp), dtype="float32", preserve_mtp=True)
        convert(
            str(source), mlx_path=str(without_mtp), dtype="float32", preserve_mtp=False
        )

        def digests(path):
            return {
                f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                for f in path.iterdir()
                if f.is_file()
            }

        with_digests = digests(with_mtp)
        without_digests = digests(without_mtp)

        # The flag only adds the mtp sidecar and its index; every other file,
        # model.safetensors.index.json included, is byte-identical.
        extra = set(with_digests) - set(without_digests)
        self.assertEqual(extra, {"mtp.safetensors", "mtp.safetensors.index.json"})
        for name, digest in without_digests.items():
            self.assertEqual(with_digests[name], digest, name)


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


if __name__ == "__main__":
    unittest.main()
