# Copyright © 2025 Apple Inc.

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_lm.models import llama
from mlx_lm.quant.awq import (
    AWQ_MODEL_CONFIGS,
    awq_quantize,
    update,
    update_config,
)
from mlx_lm.utils import save_config, save_model


def tiny_llama(tie_word_embeddings=False):
    args = llama.ModelArgs(
        model_type="llama",
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=256,
        tie_word_embeddings=tie_word_embeddings,
    )
    model = llama.Model(args)
    mx.eval(model.parameters())
    return model, args


def make_calibration_data(vocab_size, num_samples=4, seq_len=32):
    return mx.random.randint(0, vocab_size, shape=(num_samples, seq_len))


def model_config(model):
    config = {"model_type": "llama", "vocab_size": 256}
    if hasattr(model, "args"):
        config.update(vars(model.args))
    return config


def run_baseline_awq(
    model, calibration_data, output_path,
    bits=4, group_size=64, embed_bits=4, embed_group_size=32,
):
    """Current non-streaming AWQ path: quantize then save."""
    awq_config = AWQ_MODEL_CONFIGS["llama"]
    awq_quantize(
        model,
        calibration_data,
        awq_config,
        bits=bits,
        group_size=group_size,
        embed_bits=embed_bits,
        embed_group_size=embed_group_size,
    )
    config = update_config(model, model_config(model))
    save_model(output_path, model, donate_model=False)
    save_config(config, config_path=output_path / "config.json")
    return config


def run_streaming_awq(
    model, calibration_data, output_path,
    bits=4, group_size=64, embed_bits=4, embed_group_size=32,
):
    """Streaming AWQ path: incremental save per block, bounded memory."""
    from mlx_lm.quant.awq import awq_quantize_streaming

    awq_config = AWQ_MODEL_CONFIGS["llama"]
    config = model_config(model)
    awq_quantize_streaming(
        model,
        calibration_data,
        awq_config,
        output_path=output_path,
        config=config,
        bits=bits,
        group_size=group_size,
        embed_bits=embed_bits,
        embed_group_size=embed_group_size,
    )
    return config


class TestAWQStreaming(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir_fid = tempfile.TemporaryDirectory()
        cls.test_dir = Path(cls.test_dir_fid.name)

    @classmethod
    def tearDownClass(cls):
        cls.test_dir_fid.cleanup()

    def test_streaming_matches_baseline(self):
        """Streaming AWQ must produce identical weights, index, and config."""
        seed = 99
        bits = 4
        group_size = 32

        # Run baseline
        mx.random.seed(seed)
        model_a, _ = tiny_llama()
        data_a = make_calibration_data(256)
        baseline_path = self.test_dir / "baseline"
        baseline_path.mkdir(exist_ok=True)
        baseline_config = run_baseline_awq(
            model_a, data_a, baseline_path, bits=bits, group_size=group_size,
        )

        # Run streaming with identical model and data
        mx.random.seed(seed)
        model_b, _ = tiny_llama()
        data_b = make_calibration_data(256)
        streaming_path = self.test_dir / "streaming"
        streaming_path.mkdir(exist_ok=True)
        streaming_config = run_streaming_awq(
            model_b, data_b, streaming_path, bits=bits, group_size=group_size,
        )

        # -- Weight parity --
        baseline_weights = {}
        for f in sorted(baseline_path.glob("*.safetensors")):
            baseline_weights.update(mx.load(str(f)))

        streaming_weights = {}
        for f in sorted(streaming_path.glob("*.safetensors")):
            streaming_weights.update(mx.load(str(f)))

        self.assertEqual(
            set(baseline_weights.keys()),
            set(streaming_weights.keys()),
            "Weight keys must match between baseline and streaming outputs",
        )

        for key in sorted(baseline_weights.keys()):
            b = baseline_weights[key]
            s = streaming_weights[key]
            self.assertEqual(
                b.shape, s.shape,
                f"Shape mismatch for {key}: {b.shape} vs {s.shape}",
            )
            self.assertTrue(
                mx.array_equal(b, s),
                f"Value mismatch for {key}",
            )

        # -- Index parity --
        baseline_index_path = baseline_path / "model.safetensors.index.json"
        streaming_index_path = streaming_path / "model.safetensors.index.json"
        self.assertTrue(streaming_index_path.exists(), "Streaming must write index")

        with open(baseline_index_path) as f:
            baseline_index = json.load(f)
        with open(streaming_index_path) as f:
            streaming_index = json.load(f)

        self.assertEqual(
            set(baseline_index["weight_map"].keys()),
            set(streaming_index["weight_map"].keys()),
            "Index weight_map keys must match",
        )
        for shard_name in streaming_index["weight_map"].values():
            self.assertTrue(
                (streaming_path / shard_name).exists(),
                f"Missing shard from streaming weight_map: {shard_name}",
            )
        self.assertEqual(
            baseline_index["metadata"]["total_size"],
            streaming_index["metadata"]["total_size"],
            "total_size must match",
        )
        self.assertEqual(
            baseline_index["metadata"]["total_parameters"],
            streaming_index["metadata"]["total_parameters"],
            "total_parameters must match",
        )

        # -- Config parity --
        self.assertEqual(
            baseline_config["quantization"],
            streaming_config["quantization"],
            "Quantization config must match",
        )
        self.assertEqual(
            baseline_config,
            streaming_config,
            "Config dict must match between baseline and streaming",
        )

    def test_streaming_roundtrip_load(self):
        """Streaming output must be loadable and produce matching forward pass."""
        seed = 99
        bits = 4
        group_size = 32

        # Quantize via baseline, keep the in-memory model for forward pass
        mx.random.seed(seed)
        baseline_model, _ = tiny_llama()
        data = make_calibration_data(256)
        baseline_path = self.test_dir / "roundtrip_baseline"
        baseline_path.mkdir(exist_ok=True)
        run_baseline_awq(
            baseline_model, data, baseline_path, bits=bits, group_size=group_size,
        )

        # Quantize via streaming
        mx.random.seed(seed)
        streaming_model, _ = tiny_llama()
        data = make_calibration_data(256)
        streaming_path = self.test_dir / "roundtrip_streaming"
        streaming_path.mkdir(exist_ok=True)
        run_streaming_awq(
            streaming_model, data, streaming_path, bits=bits, group_size=group_size,
        )

        # Load streaming output via load_model (the real load path)
        from mlx_lm.utils import load_model

        loaded_model, _ = load_model(streaming_path)

        # Load baseline output the same way
        baseline_loaded, _ = load_model(baseline_path)

        # Forward pass on both with same input
        test_input = mx.array([[1, 2, 3, 4]])
        out_baseline = baseline_loaded(test_input)
        out_streaming = loaded_model(test_input)
        mx.eval(out_baseline, out_streaming)

        self.assertEqual(out_baseline.shape, out_streaming.shape)
        self.assertTrue(
            mx.allclose(out_baseline, out_streaming, atol=1e-6),
            "Forward pass output must match between baseline and streaming",
        )

    def test_streaming_fallback_path(self):
        """Streaming fallback path should be exercised and match baseline."""
        seed = 99
        bits = 4
        group_size = 32

        # Baseline run under forced fallback conditions
        mx.random.seed(seed)
        baseline_model, _ = tiny_llama()
        baseline_data = make_calibration_data(256)
        baseline_path = self.test_dir / "fallback_baseline"
        baseline_path.mkdir(exist_ok=True)

        # Streaming run under the same forced fallback conditions
        mx.random.seed(seed)
        streaming_model, _ = tiny_llama()
        streaming_data = make_calibration_data(256)
        fallback_path = self.test_dir / "fallback_streaming"
        fallback_path.mkdir(exist_ok=True)

        import mlx_lm.quant.awq as awq_mod
        from mlx_lm.quant.awq import awq_quantize_streaming
        from mlx_lm.utils import load_model

        streaming_config = model_config(streaming_model)
        baseline_config = model_config(baseline_model)

        def run_with_forced_fallback(run_fn):
            mse_calls = {"count": 0, "before": 0, "after": 0}
            quantize_calls = {"count": 0}
            loss_phase = {"after_clip": False}
            orig_quantize = awq_mod.nn.quantize

            def forced_mse(x, y):
                mse_calls["count"] += 1
                # Mark after-loss phase from clip_block rather than relying on
                # raw mse() call ordering.
                if loss_phase["after_clip"]:
                    mse_calls["after"] += 1
                    loss_phase["after_clip"] = False
                    loss_level = 2.0
                else:
                    mse_calls["before"] += 1
                    loss_level = 1.0
                return mx.ones_like((x - y).astype(mx.float32)) * loss_level

            def counting_quantize(*args, **kwargs):
                quantize_calls["count"] += 1
                return orig_quantize(*args, **kwargs)

            def passthrough_scale(*args, **kwargs):
                return None

            def mark_after_clip(*args, **kwargs):
                loss_phase["after_clip"] = True
                return None

            with (
                mock.patch(
                    "mlx_lm.quant.awq.scale_block",
                    side_effect=passthrough_scale,
                ),
                mock.patch(
                    "mlx_lm.quant.awq.clip_block",
                    side_effect=mark_after_clip,
                ),
                mock.patch("mlx_lm.quant.awq.mse", side_effect=forced_mse),
                mock.patch(
                    "mlx_lm.quant.awq.nn.quantize",
                    side_effect=counting_quantize,
                ),
            ):
                run_fn()

            return mse_calls, quantize_calls["count"]

        baseline_mse, baseline_quantize_calls = run_with_forced_fallback(
            lambda: awq_quantize(
                baseline_model,
                baseline_data,
                AWQ_MODEL_CONFIGS["llama"],
                bits=bits,
                group_size=group_size,
            )
        )
        baseline_config = update_config(baseline_model, baseline_config)
        save_model(baseline_path, baseline_model, donate_model=False)
        save_config(baseline_config, config_path=baseline_path / "config.json")

        streaming_mse, streaming_quantize_calls = run_with_forced_fallback(
            lambda: awq_quantize_streaming(
                streaming_model,
                streaming_data,
                AWQ_MODEL_CONFIGS["llama"],
                output_path=fallback_path,
                config=streaming_config,
                bits=bits,
                group_size=group_size,
            )
        )

        self.assertGreater(
            baseline_mse["count"], 0,
            "Expected baseline mse() calls under forced fallback",
        )
        self.assertEqual(
            baseline_mse["before"], baseline_mse["after"],
            "Expected paired baseline before/after loss computations",
        )
        expected_blocks = len(baseline_model.model.layers)
        self.assertEqual(
            baseline_mse["before"], expected_blocks,
            "Expected one baseline before-loss call per block",
        )
        self.assertGreater(
            baseline_quantize_calls, 2 * expected_blocks,
            "Expected extra baseline quantize calls from fallback branch",
        )
        self.assertGreater(
            streaming_mse["count"], 0,
            "Expected streaming mse() calls under forced fallback",
        )
        self.assertEqual(
            streaming_mse["before"], streaming_mse["after"],
            "Expected paired streaming before/after loss computations",
        )
        self.assertEqual(
            streaming_mse["before"], expected_blocks,
            "Expected one streaming before-loss call per block",
        )
        self.assertGreater(
            streaming_quantize_calls, 2 * expected_blocks,
            "Expected extra streaming quantize calls from fallback branch",
        )
        self.assertEqual(
            baseline_quantize_calls, streaming_quantize_calls,
            "Fallback quantize-call counts should match baseline and streaming",
        )
        self.assertTrue(
            (fallback_path / "model.safetensors.index.json").exists(),
            "Streaming fallback must write an index file",
        )
        self.assertTrue(
            (fallback_path / "config.json").exists(),
            "Streaming fallback must write config.json",
        )
        self.assertGreater(
            len(list(fallback_path.glob("*.safetensors"))), 0,
            "Streaming fallback must write at least one safetensors shard",
        )

        baseline_weights = {}
        for f in sorted(baseline_path.glob("*.safetensors")):
            baseline_weights.update(mx.load(str(f)))
        streaming_weights = {}
        for f in sorted(fallback_path.glob("*.safetensors")):
            streaming_weights.update(mx.load(str(f)))
        self.assertEqual(
            set(baseline_weights.keys()),
            set(streaming_weights.keys()),
            "Fallback weight keys must match baseline",
        )
        for key in sorted(baseline_weights.keys()):
            self.assertTrue(
                mx.array_equal(baseline_weights[key], streaming_weights[key]),
                f"Fallback value mismatch for {key}",
            )

        with open(baseline_path / "model.safetensors.index.json") as f:
            baseline_index = json.load(f)
        with open(fallback_path / "model.safetensors.index.json") as f:
            streaming_index = json.load(f)
        self.assertEqual(
            baseline_index["metadata"]["total_size"],
            streaming_index["metadata"]["total_size"],
            "Fallback total_size must match baseline",
        )
        self.assertEqual(
            baseline_index["metadata"]["total_parameters"],
            streaming_index["metadata"]["total_parameters"],
            "Fallback total_parameters must match baseline",
        )

        self.assertEqual(
            baseline_config["quantization"],
            streaming_config["quantization"],
            "Fallback quantization config must match baseline",
        )

        loaded_model, _ = load_model(fallback_path)
        test_input = mx.array([[1, 2, 3, 4]])
        out = loaded_model(test_input)
        mx.eval(out)
        self.assertEqual(out.shape[0], 1)


    def test_streaming_lm_key_prefix(self):
        """Keys must be correctly prefixed when lm_key is set."""
        seed = 99
        bits = 4
        group_size = 32

        # Build an AWQ config identical to llama but with lm_key set
        llama_awq = AWQ_MODEL_CONFIGS["llama"]
        lm_key_awq = update(llama_awq, lm_key="language_model")

        # Wrap a tiny llama in a container so model["language_model"] works
        class Wrapper(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.language_model = inner

        # -- Baseline (non-streaming) --
        mx.random.seed(seed)
        model_a, _ = tiny_llama()
        wrapped_a = Wrapper(model_a)
        data_a = make_calibration_data(256)
        baseline_path = self.test_dir / "lmkey_baseline"
        baseline_path.mkdir(exist_ok=True)

        awq_quantize(
            wrapped_a, data_a, lm_key_awq,
            bits=bits, group_size=group_size,
        )
        baseline_config = update_config(
            wrapped_a, {"model_type": "llama", "vocab_size": 256},
        )
        # save_model on the wrapper produces language_model.* keys
        save_model(baseline_path, wrapped_a, donate_model=False)

        # -- Streaming --
        mx.random.seed(seed)
        model_b, _ = tiny_llama()
        wrapped_b = Wrapper(model_b)
        data_b = make_calibration_data(256)
        streaming_path = self.test_dir / "lmkey_streaming"
        streaming_path.mkdir(exist_ok=True)

        from mlx_lm.quant.awq import awq_quantize_streaming

        config = {"model_type": "llama", "vocab_size": 256}
        awq_quantize_streaming(
            wrapped_b, data_b, lm_key_awq,
            output_path=streaming_path, config=config,
            bits=bits, group_size=group_size,
        )

        # -- Verify all streaming keys have the language_model prefix --
        baseline_weights = {}
        for f in sorted(baseline_path.glob("*.safetensors")):
            baseline_weights.update(mx.load(str(f)))

        streaming_weights = {}
        for f in sorted(streaming_path.glob("*.safetensors")):
            streaming_weights.update(mx.load(str(f)))

        for key in streaming_weights:
            self.assertTrue(
                key.startswith("language_model."),
                f"Key {key!r} missing language_model. prefix",
            )

        self.assertEqual(
            set(baseline_weights.keys()),
            set(streaming_weights.keys()),
            "Weight keys must match between baseline and streaming with lm_key",
        )

        for key in sorted(baseline_weights.keys()):
            self.assertTrue(
                mx.array_equal(baseline_weights[key], streaming_weights[key]),
                f"Value mismatch for {key}",
            )

        self.assertEqual(
            baseline_config["quantization"],
            config["quantization"],
            "lm_key quantization config must match baseline",
        )

    def test_streaming_tied_embeddings_matches_baseline(self):
        """Streaming must match baseline when embeddings are tied."""
        seed = 99
        bits = 4
        group_size = 32

        # -- Baseline --
        mx.random.seed(seed)
        model_a, _ = tiny_llama(tie_word_embeddings=True)
        data_a = make_calibration_data(256)
        baseline_path = self.test_dir / "tied_baseline"
        baseline_path.mkdir(exist_ok=True)
        baseline_config = run_baseline_awq(
            model_a, data_a, baseline_path, bits=bits, group_size=group_size,
        )

        # -- Streaming --
        mx.random.seed(seed)
        model_b, _ = tiny_llama(tie_word_embeddings=True)
        data_b = make_calibration_data(256)
        streaming_path = self.test_dir / "tied_streaming"
        streaming_path.mkdir(exist_ok=True)
        streaming_config = run_streaming_awq(
            model_b, data_b, streaming_path, bits=bits, group_size=group_size,
        )

        baseline_weights = {}
        for f in sorted(baseline_path.glob("*.safetensors")):
            baseline_weights.update(mx.load(str(f)))

        streaming_weights = {}
        for f in sorted(streaming_path.glob("*.safetensors")):
            streaming_weights.update(mx.load(str(f)))

        self.assertNotIn(
            "lm_head.weight",
            baseline_weights,
            "Tied embeddings baseline should not store lm_head.weight",
        )
        self.assertEqual(
            set(baseline_weights.keys()),
            set(streaming_weights.keys()),
            "Tied embeddings weight keys must match baseline",
        )
        for key in sorted(baseline_weights.keys()):
            self.assertTrue(
                mx.array_equal(baseline_weights[key], streaming_weights[key]),
                f"Tied embeddings value mismatch for {key}",
            )

        self.assertEqual(
            baseline_config["quantization"],
            streaming_config["quantization"],
            "Tied embeddings quantization config must match baseline",
        )

    def test_streaming_writes_multiple_shards_and_clears_layers(self):
        """Streaming should write multiple shards and clear processed blocks."""
        seed = 99
        bits = 4
        group_size = 32

        mx.random.seed(seed)
        model, _ = tiny_llama()
        data = make_calibration_data(256)
        output_path = self.test_dir / "streaming_multi_shard"
        output_path.mkdir(exist_ok=True)

        from mlx_lm.quant.awq import awq_quantize_streaming

        config = {"model_type": "llama", "vocab_size": 256}
        awq_quantize_streaming(
            model,
            data,
            AWQ_MODEL_CONFIGS["llama"],
            output_path=output_path,
            config=config,
            max_file_size_gb=1e-5,
            bits=bits,
            group_size=group_size,
        )

        shard_files = sorted(output_path.glob("*.safetensors"))
        self.assertGreater(
            len(shard_files), 1,
            "Streaming should write multiple safetensors shards",
        )

        for i, layer in enumerate(model.model.layers):
            self.assertTrue(
                all(v.size == 0 for _, v in tree_flatten(layer.parameters())),
                f"Layer {i} parameters should be cleared after streaming save",
            )
            for _, leaf in tree_flatten(
                layer.leaf_modules(), is_leaf=lambda m: isinstance(m, nn.Module)
            ):
                if hasattr(leaf, "input_feat"):
                    self.assertEqual(
                        leaf.input_feat.size, 0,
                        f"Layer {i} leaf input_feat should be cleared",
                    )
                if hasattr(leaf, "indices"):
                    self.assertEqual(
                        leaf.indices.size, 0,
                        f"Layer {i} leaf indices should be cleared",
                    )

    def test_streaming_nonzero_rank_skips_writes(self):
        """Non-zero rank should skip writing shard/index/config files."""
        seed = 99
        bits = 4
        group_size = 32

        mx.random.seed(seed)
        model, _ = tiny_llama()
        data = make_calibration_data(256)
        output_path = self.test_dir / "streaming_rank1_no_write"

        from mlx_lm.quant.awq import awq_quantize_streaming

        class FakeGroup:
            def rank(self):
                return 1

            def size(self):
                return 2

        config = model_config(model)
        with (
            mock.patch(
                "mlx_lm.quant.awq.mx.distributed.init", return_value=FakeGroup()
            ),
            mock.patch(
                "mlx_lm.quant.awq.mx.distributed.all_sum",
                side_effect=lambda x, *a, **k: x,
            ),
        ):
            awq_quantize_streaming(
                model,
                data,
                AWQ_MODEL_CONFIGS["llama"],
                output_path=output_path,
                config=config,
                bits=bits,
                group_size=group_size,
            )

        self.assertFalse(
            output_path.exists(),
            "Non-zero rank should not create the output directory",
        )
        self.assertIn(
            "quantization", config,
            "Non-zero rank should still update in-memory quantization config",
        )

    def test_streaming_invalid_shard_size_raises(self):
        """Invalid max_file_size_gb should raise ValueError."""
        model, _ = tiny_llama()
        data = make_calibration_data(256)
        output_path = self.test_dir / "streaming_invalid_size"
        output_path.mkdir(exist_ok=True)

        from mlx_lm.quant.awq import awq_quantize_streaming

        with self.assertRaises(ValueError):
            awq_quantize_streaming(
                model,
                data,
                AWQ_MODEL_CONFIGS["llama"],
                output_path=output_path,
                config=model_config(model),
                max_file_size_gb=0.0,
            )


if __name__ == "__main__":
    unittest.main()
