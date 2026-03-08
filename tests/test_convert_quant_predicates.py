# Copyright © 2025 Apple Inc.

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mlx.nn as nn

from mlx_lm.convert import convert, jamba_int8_safe_predicate_builder
from mlx_lm.models.jamba import Model, ModelArgs


class TestConvertQuantPredicates(unittest.TestCase):
    def _build_tiny_jamba_model(self, *, tie_word_embeddings=True):
        args = ModelArgs(
            model_type="jamba",
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            attn_layer_offset=0,
            attn_layer_period=2,
            expert_layer_offset=0,
            expert_layer_period=2,
            mamba_d_conv=2,
            mamba_d_state=4,
            mamba_expand=2,
            num_experts=2,
            num_experts_per_tok=1,
            rms_norm_eps=1e-5,
            max_position_embeddings=16,
            vocab_size=32,
            tie_word_embeddings=tie_word_embeddings,
        )
        return Model(args)

    def test_jamba_int8_safe_predicate_paths(self):
        model = self._build_tiny_jamba_model()
        predicate = jamba_int8_safe_predicate_builder(group_size=32)

        dense_result = predicate(
            "model.layers.1.feed_forward.up_proj",
            model.model.layers[1].feed_forward.up_proj,
        )
        self.assertEqual(dense_result, {"group_size": 32, "bits": 8, "mode": "affine"})

        switch_result = predicate(
            "model.layers.0.feed_forward.switch_mlp.down_proj",
            model.model.layers[0].feed_forward.switch_mlp.down_proj,
        )
        self.assertEqual(
            switch_result,
            {"group_size": 32, "bits": 8, "mode": "affine"},
        )

        self.assertFalse(
            predicate(
                "model.layers.1.mamba.in_proj",
                model.model.layers[1].mamba.in_proj,
            )
        )
        self.assertFalse(
            predicate(
                "model.layers.0.feed_forward.router",
                model.model.layers[0].feed_forward.router,
            )
        )
        self.assertFalse(predicate("model.embed_tokens", model.model.embed_tokens))
        self.assertFalse(
            predicate(
                "model.layers.0.input_layernorm",
                model.model.layers[0].input_layernorm,
            )
        )
        self.assertFalse(predicate("model.final_layernorm", model.model.final_layernorm))

        untied_model = self._build_tiny_jamba_model(tie_word_embeddings=False)
        self.assertFalse(predicate("lm_head", untied_model.lm_head))

    def test_convert_rejects_jamba_recipe_for_non_jamba_architecture(self):
        fake_model = nn.Module()
        fake_model.model_type = "llama"

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "mlx_out"
            with patch("mlx_lm.convert.load", return_value=(fake_model, object(), {})):
                with self.assertRaisesRegex(ValueError, "only supports Jamba architectures"):
                    convert(
                        hf_path="dummy",
                        mlx_path=str(out_path),
                        quantize=True,
                        quant_predicate="jamba_int8_safe",
                    )

    def test_convert_rejects_non_affine_q_mode_for_quant_predicates(self):
        model = self._build_tiny_jamba_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "mlx_out"
            with patch("mlx_lm.convert.load", return_value=(model, object(), {})):
                with self.assertRaisesRegex(ValueError, "only support 'affine' quantization"):
                    convert(
                        hf_path="dummy",
                        mlx_path=str(out_path),
                        quantize=True,
                        q_mode="mxfp4",
                        quant_predicate="jamba_int8_safe",
                    )

    def test_convert_rejects_non_int8_q_bits_for_jamba_recipe(self):
        model = self._build_tiny_jamba_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "mlx_out"
            with patch("mlx_lm.convert.load", return_value=(model, object(), {})):
                with self.assertRaisesRegex(ValueError, "requires --q-bits 8"):
                    convert(
                        hf_path="dummy",
                        mlx_path=str(out_path),
                        quantize=True,
                        q_bits=4,
                        quant_predicate="jamba_int8_safe",
                    )


if __name__ == "__main__":
    unittest.main()
