import re
import unittest

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.convert import (
    apply_float_overrides,
    build_override_predicate,
    parse_overrides,
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


if __name__ == "__main__":
    unittest.main()
