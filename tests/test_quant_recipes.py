# Copyright © 2024 Apple Inc.

import unittest

import mlx.core as mx
import mlx.nn as nn

from mlx_lm import utils
from mlx_lm.convert import quant2_predicate_builder
from mlx_lm.models import llama


def _build_model(num_layers=4):
    args = llama.ModelArgs(
        model_type="llama",
        hidden_size=256,
        num_hidden_layers=num_layers,
        intermediate_size=512,
        num_attention_heads=4,
        rms_norm_eps=1e-5,
        vocab_size=1000,
        tie_word_embeddings=False,
    )
    return llama.Model(args)


class TestQuant2Recipes(unittest.TestCase):
    def test_quant2_bit_allocation(self):
        model = _build_model(num_layers=4)
        pred = quant2_predicate_builder("quant2", model, 64)
        model, _ = utils.quantize_model(model, {}, 64, 4, quant_predicate=pred)

        # First and last decoder layers are kept at 3-bit.
        self.assertEqual(model.layers[0].mlp.down_proj.bits, 3)
        self.assertEqual(model.layers[-1].mlp.down_proj.bits, 3)
        # Middle layers are quantized to 2-bit.
        self.assertEqual(model.layers[1].mlp.down_proj.bits, 2)
        self.assertEqual(model.layers[2].mlp.down_proj.bits, 2)
        self.assertEqual(model.layers[1].mlp.down_proj.group_size, 64)
        # Non-sensitive middle projection also at 2-bit.
        self.assertEqual(model.layers[1].self_attn.q_proj.bits, 2)

    def test_quant2_preserves_head(self):
        model = _build_model(num_layers=4)
        pred = quant2_predicate_builder("quant2", model, 64)
        model, _ = utils.quantize_model(model, {}, 64, 4, quant_predicate=pred)
        # lm_head is kept at 4-bit to preserve token resolution.
        self.assertEqual(model.lm_head.bits, 4)
        self.assertEqual(model.lm_head.group_size, 64)

    def test_quant2_128_group_size(self):
        model = _build_model(num_layers=4)
        pred = quant2_predicate_builder("quant2_128", model, 64)
        model, _ = utils.quantize_model(model, {}, 64, 4, quant_predicate=pred)
        # Same bit allocation as quant2 ...
        self.assertEqual(model.layers[0].mlp.down_proj.bits, 3)
        self.assertEqual(model.layers[1].mlp.down_proj.bits, 2)
        self.assertEqual(model.lm_head.bits, 4)
        # ... but the 2/3-bit decoder body uses group_size=128.
        self.assertEqual(model.layers[1].mlp.down_proj.group_size, 128)
        self.assertEqual(model.layers[0].mlp.down_proj.group_size, 128)
        # lm_head still uses the caller-supplied group_size.
        self.assertEqual(model.lm_head.group_size, 64)

    def test_quant2_first_and_last_layer_index(self):
        # With 6 layers, indices 0 and 5 are the sensitive 3-bit positions.
        model = _build_model(num_layers=6)
        pred = quant2_predicate_builder("quant2", model, 64)
        model, _ = utils.quantize_model(model, {}, 64, 4, quant_predicate=pred)
        self.assertEqual(model.layers[0].mlp.down_proj.bits, 3)
        self.assertEqual(model.layers[5].mlp.down_proj.bits, 3)
        for i in range(1, 5):
            self.assertEqual(model.layers[i].mlp.down_proj.bits, 2)

    def test_quant2_runs_inference(self):
        # The quantized model must still produce coherent output shapes.
        model = _build_model(num_layers=4)
        pred = quant2_predicate_builder("quant2", model, 64)
        model, _ = utils.quantize_model(model, {}, 64, 4, quant_predicate=pred)
        tokens = mx.array([[1, 2, 3]])
        logits = model(tokens)
        self.assertEqual(logits.shape, (1, 3, 1000))


if __name__ == "__main__":
    unittest.main()
