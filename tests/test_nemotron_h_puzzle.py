# Copyright © 2025-2026 Apple Inc.

import unittest

from mlx_lm.models.nemotron_h import Model


class _Stub:
    """Minimal stand-in carrying only the attribute quant_predicate reads."""

    def __init__(self, model_type):
        self.model_type = model_type


class TestNemotronHPuzzleQuantPredicate(unittest.TestCase):
    def test_puzzle_excludes_lm_head_from_quantization(self):
        pred = Model.quant_predicate.fget(_Stub("nemotron_h_puzzle"))
        # The 131k-token output projection must stay out of low-bit quant.
        self.assertFalse(pred("lm_head", object()))
        # Every other module is still eligible for quantization.
        self.assertTrue(pred("backbone.layers.0.mixer.fc1", object()))

    def test_non_puzzle_quantizes_everything(self):
        pred = Model.quant_predicate.fget(_Stub("nemotron_h"))
        self.assertTrue(pred("lm_head", object()))
        self.assertTrue(pred("backbone.layers.0.mixer.fc1", object()))


if __name__ == "__main__":
    unittest.main()
