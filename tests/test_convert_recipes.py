# Copyright © 2024 Apple Inc.

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlx_lm.convert import convert


class _DummyModel:
    def __init__(self, model_type=None):
        self.model_type = model_type


class TestConvertQuantRecipes(unittest.TestCase):
    def _make_mlx_path(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        return Path(td.name) / "mlx_model"

    @patch("mlx_lm.convert.save")
    @patch("mlx_lm.convert.load")
    def test_jamba_int8_safe_requires_jamba_architecture(self, mock_load, mock_save):
        mock_load.return_value = (_DummyModel(model_type="llama"), object(), {"model_type": "llama"})

        with self.assertRaisesRegex(ValueError, "only compatible with Jamba models"):
            convert(
                "dummy",
                mlx_path=self._make_mlx_path(),
                quant_predicate="jamba_int8_safe",
            )

        mock_save.assert_not_called()

    @patch("mlx_lm.convert.save")
    @patch("mlx_lm.convert.load")
    def test_jamba_int8_safe_requires_affine_mode(self, mock_load, mock_save):
        mock_load.return_value = (_DummyModel(model_type="jamba"), object(), {"model_type": "jamba"})

        with self.assertRaisesRegex(ValueError, "requires --q-mode affine"):
            convert(
                "dummy",
                mlx_path=self._make_mlx_path(),
                quant_predicate="jamba_int8_safe",
                q_mode="mxfp4",
            )

        mock_save.assert_not_called()

    @patch("mlx_lm.convert.save")
    @patch("mlx_lm.convert.load")
    def test_jamba_int8_safe_rejects_non_8_bits(self, mock_load, mock_save):
        mock_load.return_value = (_DummyModel(model_type="jamba"), object(), {"model_type": "jamba"})

        with self.assertRaisesRegex(ValueError, "Please use --q-bits 8"):
            convert(
                "dummy",
                mlx_path=self._make_mlx_path(),
                quant_predicate="jamba_int8_safe",
                q_bits=4,
            )

        mock_save.assert_not_called()

    @patch("mlx_lm.convert.save")
    @patch("mlx_lm.convert.load")
    def test_other_recipe_behavior_is_unchanged(self, mock_load, mock_save):
        mock_load.return_value = (_DummyModel(model_type="llama"), object(), {"model_type": "llama"})

        with self.assertRaisesRegex(ValueError, "Quant predicates only support 'affine' quantization"):
            convert(
                "dummy",
                mlx_path=self._make_mlx_path(),
                quant_predicate="mixed_3_4",
                q_mode="mxfp4",
            )

        mock_save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
