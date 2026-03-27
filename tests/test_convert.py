import re
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.convert import (
    apply_float_overrides,
    build_override_predicate,
    convert,
    parse_overrides,
    warn_mixed_mode_overrides,
    warn_mode_override_conflicts,
)


class TestParseOverrides(unittest.TestCase):

    def test_int_value(self):
        result = parse_overrides(["lm_head=8"])
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0][0], re.Pattern)
        self.assertEqual(result[0][1], 8)

    def test_float_dtype_value(self):
        for dtype in ("float16", "bfloat16", "float32"):
            result = parse_overrides([f"embed_tokens={dtype}"])
            self.assertEqual(result[0][1], dtype)

    def test_multiple_overrides(self):
        result = parse_overrides(["lm_head=8", "embed_tokens=float16"])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][1], 8)
        self.assertEqual(result[1][1], "float16")

    def test_regex_pattern(self):
        result = parse_overrides([r"layers\.0\..*=6"])
        self.assertTrue(result[0][0].search("model.layers.0.mlp.down_proj"))
        self.assertIsNone(result[0][0].search("model.layers.1.mlp.down_proj"))

    def test_missing_equals(self):
        with self.assertRaises(ValueError):
            parse_overrides(["lm_head"])

    def test_invalid_regex(self):
        with self.assertRaises(ValueError):
            parse_overrides(["[invalid=8"])

    def test_quant_mode_value(self):
        for mode in ("mxfp4", "nvfp4", "mxfp8"):
            result = parse_overrides([f"down_proj={mode}"])
            self.assertEqual(result[0][1], mode)

    def test_invalid_value(self):
        with self.assertRaises(ValueError):
            parse_overrides(["lm_head=garbage"])


class TestBuildOverridePredicate(unittest.TestCase):

    def test_int_override_matches(self):
        overrides = [(re.compile("lm_head"), 8)]
        pred = build_override_predicate(overrides, None, 64)
        result = pred("model.lm_head", None)
        self.assertEqual(result, {"group_size": 64, "bits": 8, "mode": "affine"})

    def test_float_override_returns_false(self):
        overrides = [(re.compile("embed_tokens"), "float16")]
        pred = build_override_predicate(overrides, None, 64)
        result = pred("model.embed_tokens", None)
        self.assertFalse(result)

    def test_no_match_delegates_to_base(self):
        overrides = [(re.compile("lm_head"), 8)]
        base = lambda path, module: {"group_size": 64, "bits": 4, "mode": "affine"}
        pred = build_override_predicate(overrides, base, 64)
        result = pred("model.layers.0.mlp.down_proj", None)
        self.assertEqual(result, {"group_size": 64, "bits": 4, "mode": "affine"})

    def test_no_match_no_base_returns_true(self):
        overrides = [(re.compile("lm_head"), 8)]
        pred = build_override_predicate(overrides, None, 64)
        result = pred("model.layers.0.mlp.down_proj", None)
        self.assertTrue(result)

    def test_first_match_wins(self):
        overrides = [
            (re.compile("lm_head"), 8),
            (re.compile("lm_head"), 6),
        ]
        pred = build_override_predicate(overrides, None, 64)
        result = pred("model.lm_head", None)
        self.assertEqual(result["bits"], 8)

    def test_quant_mode_override(self):
        overrides = [(re.compile("down_proj"), "mxfp4")]
        pred = build_override_predicate(overrides, None, 64)
        result = pred("model.layers.0.mlp.down_proj", None)
        self.assertEqual(result, {"group_size": 32, "bits": 4, "mode": "mxfp4"})

    def test_group_size_passthrough(self):
        overrides = [(re.compile("lm_head"), 4)]
        pred = build_override_predicate(overrides, None, 32)
        result = pred("model.lm_head", None)
        self.assertEqual(result["group_size"], 32)

    def test_int_override_uses_explicit_int_group_size(self):
        overrides = [(re.compile("lm_head"), 8)]
        pred = build_override_predicate(overrides, None, 32, int_group_size=64)
        result = pred("model.lm_head", None)
        self.assertEqual(result["group_size"], 64)


class _Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 64)


class TestApplyFloatOverrides(unittest.TestCase):

    def test_cast_matching_weight(self):
        model = _Wrapper()
        overrides = [(re.compile("linear"), "float16")]
        apply_float_overrides(model, overrides)
        self.assertEqual(model.linear.weight.dtype, mx.float16)

    def test_no_cast_non_matching(self):
        model = _Wrapper()
        original_dtype = model.linear.weight.dtype
        overrides = [(re.compile("lm_head"), "float16")]
        apply_float_overrides(model, overrides)
        self.assertEqual(model.linear.weight.dtype, original_dtype)

    def test_skips_int_overrides(self):
        model = _Wrapper()
        original_dtype = model.linear.weight.dtype
        overrides = [(re.compile("linear"), 8)]
        apply_float_overrides(model, overrides)
        self.assertEqual(model.linear.weight.dtype, original_dtype)


class TestModeConflictWarnings(unittest.TestCase):

    def test_non_affine_bits_conflict_warns(self):
        out = StringIO()
        with redirect_stdout(out):
            warn_mode_override_conflicts("mxfp4", None, 8, 32, 4)
        self.assertIn("--q-mode mxfp4 default is q-group-size=32, q-bits=4", out.getvalue())
        self.assertIn("q-bits=8", out.getvalue())

    def test_non_affine_group_size_conflict_warns(self):
        out = StringIO()
        with redirect_stdout(out):
            warn_mode_override_conflicts("mxfp4", 64, None, 32, 4)
        self.assertIn(
            "--q-mode mxfp4 default is q-group-size=32, q-bits=4", out.getvalue()
        )
        self.assertIn("q-group-size=64", out.getvalue())

    def test_non_affine_matching_values_no_warning(self):
        out = StringIO()
        with redirect_stdout(out):
            warn_mode_override_conflicts("mxfp4", 32, 4, 32, 4)
        self.assertEqual(out.getvalue(), "")

    def test_affine_no_warning(self):
        out = StringIO()
        with redirect_stdout(out):
            warn_mode_override_conflicts("affine", 32, 4, 64, 4)
        self.assertEqual(out.getvalue(), "")

    def test_non_affine_int_override_warns_mixed_mode(self):
        out = StringIO()
        with redirect_stdout(out):
            warn_mixed_mode_overrides("mxfp4", [(re.compile("lm_head"), 8)])
        self.assertIn(
            "Integer --q-override values force affine quantization",
            out.getvalue(),
        )
        self.assertIn("mixed quantization modes", out.getvalue())

    def test_non_affine_no_int_override_no_warning(self):
        out = StringIO()
        with redirect_stdout(out):
            warn_mixed_mode_overrides("mxfp4", [(re.compile("lm_head"), "mxfp8")])
        self.assertEqual(out.getvalue(), "")

    def test_affine_int_override_no_warning(self):
        out = StringIO()
        with redirect_stdout(out):
            warn_mixed_mode_overrides("affine", [(re.compile("lm_head"), 8)])
        self.assertEqual(out.getvalue(), "")


class TestConvertOverridePrecedence(unittest.TestCase):

    def test_float_override_applies_after_dtype_cast(self):
        model = _Wrapper()
        model.linear.weight = model.linear.weight.astype(mx.float16)
        captured = {}

        def fake_load(*_args, **_kwargs):
            return model, object(), {"torch_dtype": "float16"}

        def fake_save(_mlx_path, _hf_path, saved_model, *_args):
            captured["weight_dtype"] = saved_model.linear.weight.dtype

        with tempfile.TemporaryDirectory() as test_dir:
            mlx_path = f"{test_dir}/mlx_model"
            with patch("mlx_lm.convert.load", side_effect=fake_load):
                with patch("mlx_lm.convert.save", side_effect=fake_save):
                    convert(
                        "stub/repo",
                        mlx_path=mlx_path,
                        q_overrides=["linear=float32"],
                    )

        self.assertEqual(captured["weight_dtype"], mx.float32)

    def test_default_q_args_do_not_warn_on_non_affine_mode(self):
        model = _Wrapper()

        def fake_load(*_args, **_kwargs):
            return model, object(), {"torch_dtype": "float16"}

        with tempfile.TemporaryDirectory() as test_dir:
            out = StringIO()
            mlx_path = f"{test_dir}/mlx_model"
            with patch("mlx_lm.convert.load", side_effect=fake_load):
                with patch("mlx_lm.convert.save", side_effect=lambda *_args: None):
                    with redirect_stdout(out):
                        convert("stub/repo", mlx_path=mlx_path, q_mode="mxfp4")
            self.assertNotIn("--q-mode mxfp4 default is", out.getvalue())

    def test_default_q_args_use_mode_defaults(self):
        model = _Wrapper()
        captured = {}

        def fake_load(*_args, **_kwargs):
            return model, object(), {"torch_dtype": "float16"}

        def fake_quantize_model(
            model,
            config,
            group_size,
            bits,
            mode="affine",
            quant_predicate=None,
        ):
            captured["group_size"] = group_size
            captured["bits"] = bits
            captured["mode"] = mode
            return model, config

        with tempfile.TemporaryDirectory() as test_dir:
            mlx_path = f"{test_dir}/mlx_model"
            with patch("mlx_lm.convert.load", side_effect=fake_load):
                with patch(
                    "mlx_lm.convert.quantize_model", side_effect=fake_quantize_model
                ):
                    with patch("mlx_lm.convert.save", side_effect=lambda *_args: None):
                        convert(
                            "stub/repo",
                            mlx_path=mlx_path,
                            quantize=True,
                            q_mode="mxfp4",
                        )

        self.assertEqual(captured["group_size"], 32)
        self.assertEqual(captured["bits"], 4)
        self.assertEqual(captured["mode"], "mxfp4")

    def test_default_q_args_use_mxfp8_mode_defaults(self):
        model = _Wrapper()
        captured = {}

        def fake_load(*_args, **_kwargs):
            return model, object(), {"torch_dtype": "float16"}

        def fake_quantize_model(
            model,
            config,
            group_size,
            bits,
            mode="affine",
            quant_predicate=None,
        ):
            captured["group_size"] = group_size
            captured["bits"] = bits
            captured["mode"] = mode
            return model, config

        with tempfile.TemporaryDirectory() as test_dir:
            mlx_path = f"{test_dir}/mlx_model"
            with patch("mlx_lm.convert.load", side_effect=fake_load):
                with patch(
                    "mlx_lm.convert.quantize_model", side_effect=fake_quantize_model
                ):
                    with patch("mlx_lm.convert.save", side_effect=lambda *_args: None):
                        convert(
                            "stub/repo",
                            mlx_path=mlx_path,
                            quantize=True,
                            q_mode="mxfp8",
                        )

        self.assertEqual(captured["group_size"], 32)
        self.assertEqual(captured["bits"], 8)
        self.assertEqual(captured["mode"], "mxfp8")

    def test_explicit_q_bits_warn_on_non_affine_mode(self):
        model = _Wrapper()

        def fake_load(*_args, **_kwargs):
            return model, object(), {"torch_dtype": "float16"}

        with tempfile.TemporaryDirectory() as test_dir:
            out = StringIO()
            mlx_path = f"{test_dir}/mlx_model"
            with patch("mlx_lm.convert.load", side_effect=fake_load):
                with patch("mlx_lm.convert.save", side_effect=lambda *_args: None):
                    with redirect_stdout(out):
                        convert(
                            "stub/repo",
                            mlx_path=mlx_path,
                            q_mode="mxfp8",
                            q_bits=4,
                        )
            self.assertIn(
                "--q-mode mxfp8 default is q-group-size=32, q-bits=8",
                out.getvalue(),
            )
            self.assertIn("received q-bits=4", out.getvalue())
            self.assertNotIn("received q-group-size=", out.getvalue())

    def test_explicit_q_group_size_warn_on_non_affine_mode(self):
        model = _Wrapper()

        def fake_load(*_args, **_kwargs):
            return model, object(), {"torch_dtype": "float16"}

        with tempfile.TemporaryDirectory() as test_dir:
            out = StringIO()
            mlx_path = f"{test_dir}/mlx_model"
            with patch("mlx_lm.convert.load", side_effect=fake_load):
                with patch("mlx_lm.convert.save", side_effect=lambda *_args: None):
                    with redirect_stdout(out):
                        convert(
                            "stub/repo",
                            mlx_path=mlx_path,
                            q_mode="mxfp4",
                            q_group_size=64,
                        )
            self.assertIn(
                "--q-mode mxfp4 default is q-group-size=32, q-bits=4",
                out.getvalue(),
            )
            self.assertIn("received q-group-size=64", out.getvalue())
            self.assertNotIn("received q-bits=", out.getvalue())

    def test_non_affine_mode_int_override_uses_affine_group_size(self):
        model = _Wrapper()
        captured = {}

        def fake_load(*_args, **_kwargs):
            return model, object(), {"torch_dtype": "float16"}

        def fake_quantize_model(
            model,
            config,
            group_size,
            bits,
            mode="affine",
            quant_predicate=None,
        ):
            captured["override"] = quant_predicate("model.lm_head", None)
            return model, config

        with tempfile.TemporaryDirectory() as test_dir:
            mlx_path = f"{test_dir}/mlx_model"
            with patch("mlx_lm.convert.load", side_effect=fake_load):
                with patch(
                    "mlx_lm.convert.quantize_model", side_effect=fake_quantize_model
                ):
                    with patch("mlx_lm.convert.save", side_effect=lambda *_args: None):
                        convert(
                            "stub/repo",
                            mlx_path=mlx_path,
                            quantize=True,
                            q_mode="mxfp4",
                            q_overrides=["lm_head=8"],
                        )

        self.assertEqual(captured["override"]["mode"], "affine")
        self.assertEqual(captured["override"]["bits"], 8)
        self.assertEqual(captured["override"]["group_size"], 64)

    def test_affine_mode_quant_mode_override_uses_mode_defaults(self):
        model = _Wrapper()
        captured = {}

        def fake_load(*_args, **_kwargs):
            return model, object(), {"torch_dtype": "float16"}

        def fake_quantize_model(
            model,
            config,
            group_size,
            bits,
            mode="affine",
            quant_predicate=None,
        ):
            captured["override"] = quant_predicate("model.layers.0.mlp.down_proj", None)
            return model, config

        with tempfile.TemporaryDirectory() as test_dir:
            mlx_path = f"{test_dir}/mlx_model"
            with patch("mlx_lm.convert.load", side_effect=fake_load):
                with patch(
                    "mlx_lm.convert.quantize_model", side_effect=fake_quantize_model
                ):
                    with patch("mlx_lm.convert.save", side_effect=lambda *_args: None):
                        convert(
                            "stub/repo",
                            mlx_path=mlx_path,
                            quantize=True,
                            q_mode="affine",
                            q_overrides=["down_proj=mxfp4"],
                        )

        self.assertEqual(captured["override"]["mode"], "mxfp4")
        self.assertEqual(captured["override"]["bits"], 4)
        self.assertEqual(captured["override"]["group_size"], 32)


if __name__ == "__main__":
    unittest.main()
