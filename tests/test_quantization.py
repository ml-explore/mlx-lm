# Copyright © 2026 Apple Inc.

import tempfile
import types
import unittest
from pathlib import Path

import mlx.core as mx

from mlx_lm.models import qwen3_5
from mlx_lm.quant.awq import AWQ_MODEL_CONFIGS, awq_quantize
from mlx_lm.quant.dwq import (
    build_target_metadata,
    compute_dwq_targets,
    save_target_metadata,
    validate_target_metadata,
)
from mlx_lm.quant.dynamic_quant import estimate_sensitivities


class DummyTokenizer:
    def __init__(self):
        self._tokenizer = types.SimpleNamespace(init_kwargs={"name": "dummy"})
        self.eos_token_id = 0

    def encode(self, text):
        return [min(ord(c), 127) for c in text]


class SmallVocabTeacher:
    def __call__(self, x):
        return mx.zeros((*x.shape, 16), dtype=mx.float32)


def tiny_qwen3_5_model():
    args = qwen3_5.ModelArgs.from_dict(
        {
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 4,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 32,
                "vocab_size": 128,
                "linear_num_value_heads": 2,
                "linear_num_key_heads": 1,
                "linear_key_head_dim": 32,
                "linear_value_head_dim": 32,
                "linear_conv_kernel_dim": 4,
                "full_attention_interval": 4,
                "tie_word_embeddings": False,
                "max_position_embeddings": 64,
            },
        }
    )
    return qwen3_5.Model(args)


class TestQuantization(unittest.TestCase):
    def test_awq_supports_qwen3_5_hybrid_layers(self):
        model = tiny_qwen3_5_model()
        inputs = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=mx.int32)

        awq_quantize(
            model,
            inputs,
            AWQ_MODEL_CONFIGS["qwen3_5"],
            group_size=32,
            bits=4,
            embed_group_size=32,
            embed_bits=4,
            n_grid=1,
        )
        logits = model(inputs)
        mx.eval(logits)

        self.assertEqual(logits.shape, (1, 8, 128))
        self.assertTrue(mx.all(mx.isfinite(logits)).item())

    def test_dynamic_quant_qwen3_5_sensitivity_avoids_custom_kernel_vjp(self):
        model = tiny_qwen3_5_model()
        model.eval()
        data = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=mx.int32)

        sensitivities = estimate_sensitivities(
            model,
            data,
            low_bits=4,
            low_group_size=32,
            high_bits=8,
            high_group_size=32,
            batch_size=1,
        )

        self.assertGreater(len(sensitivities), 0)
        self.assertTrue(any("linear_attn" in name for name, _ in sensitivities))

    def test_dwq_target_metadata_validates_matching_options(self):
        args = types.SimpleNamespace(
            model="model-a",
            data_path="dataset-a",
            max_seq_length=128,
            batch_size=1,
            seed=123,
        )
        metadata = build_target_metadata(args, DummyTokenizer(), num_samples=4)

        with tempfile.TemporaryDirectory() as tmp:
            target_dir = Path(tmp)
            save_target_metadata(target_dir, metadata)

            actual = validate_target_metadata(target_dir, metadata)
            self.assertEqual(actual["max_seq_length"], 128)

            expected = dict(metadata)
            expected["max_seq_length"] = 256
            with self.assertRaisesRegex(ValueError, "max_seq_length=128.*256"):
                validate_target_metadata(target_dir, expected)

    def test_dwq_target_generation_caps_top_k_to_vocab_size(self):
        data = [([1, 2, 3, 4], 0)]

        with tempfile.TemporaryDirectory() as tmp:
            target_dir = Path(tmp)
            compute_dwq_targets(
                SmallVocabTeacher(),
                target_dir,
                train_data=data,
                valid_data=data,
                batch_size=1,
                max_seq_length=8,
                seed=123,
            )
            targets = mx.load(target_dir / "train" / "0000000000.safetensors")

        self.assertEqual(targets["logits"].shape[-1], 16)
        self.assertEqual(targets["indices"].shape[-1], 16)


if __name__ == "__main__":
    unittest.main()
