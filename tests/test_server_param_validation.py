# Copyright © 2024 Apple Inc.

import math
import unittest

from mlx_lm.server import APIHandler


def _make_handler(**overrides):
    """Build an APIHandler without running BaseHTTPRequestHandler.__init__.

    validate_model_parameters() only reads plain attributes off ``self`` and
    does not touch the socket or the loaded model, so we can exercise it in
    isolation by populating a bare instance with valid defaults and overriding
    the field(s) under test.
    """
    handler = APIHandler.__new__(APIHandler)
    defaults = {
        "stream": False,
        "max_tokens": 16,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "num_draft_tokens": 3,
        "repetition_penalty": 1.0,
        "repetition_context_size": 20,
        "presence_penalty": 0.0,
        "presence_context_size": 20,
        "frequency_penalty": 0.0,
        "frequency_context_size": 20,
        "logprobs": False,
        "top_logprobs": -1,
        "xtc_probability": 0.0,
        "xtc_threshold": 0.0,
        "requested_model": "default_model",
        "adapter": None,
        "seed": None,
        "logit_bias": None,
    }
    defaults.update(overrides)
    for key, value in defaults.items():
        setattr(handler, key, value)
    return handler


class TestLogitBiasValidation(unittest.TestCase):
    def test_valid_logit_bias_is_normalized(self):
        handler = _make_handler(logit_bias={"1": 2, "3": -1.5})
        handler.validate_model_parameters()
        self.assertEqual(handler.logit_bias, {1: 2.0, 3: -1.5})
        for v in handler.logit_bias.values():
            self.assertIsInstance(v, float)

    def test_null_value_raises_valueerror_not_typeerror(self):
        # Upstream did `float(None)` inside a `except ValueError` guard, so a
        # null bias value raised an uncaught TypeError (500 / dropped socket).
        handler = _make_handler(logit_bias={"1": None})
        with self.assertRaises(ValueError):
            handler.validate_model_parameters()

    def test_non_finite_value_rejected(self):
        for bad in (float("inf"), float("-inf"), float("nan")):
            handler = _make_handler(logit_bias={"1": bad})
            with self.assertRaises(ValueError):
                handler.validate_model_parameters()

    def test_bool_value_rejected(self):
        # bool is a subclass of int; upstream let it pass silently.
        handler = _make_handler(logit_bias={"1": True})
        with self.assertRaises(ValueError):
            handler.validate_model_parameters()

    def test_non_integer_key_rejected(self):
        handler = _make_handler(logit_bias={"x": 2})
        with self.assertRaises(ValueError):
            handler.validate_model_parameters()

    def test_non_numeric_string_value_rejected(self):
        handler = _make_handler(logit_bias={"1": "nope"})
        with self.assertRaises(ValueError):
            handler.validate_model_parameters()

    def test_none_logit_bias_passes(self):
        handler = _make_handler(logit_bias=None)
        handler.validate_model_parameters()
        self.assertIsNone(handler.logit_bias)


class TestModelParameterValidation(unittest.TestCase):
    def test_top_p_out_of_range_raises(self):
        handler = _make_handler(top_p=5)
        with self.assertRaises(ValueError):
            handler.validate_model_parameters()

    def test_temperature_wrong_type_raises(self):
        handler = _make_handler(temperature="x")
        with self.assertRaises(ValueError):
            handler.validate_model_parameters()

    def test_valid_params_pass(self):
        handler = _make_handler(top_p=0.9, temperature=0.7)
        handler.validate_model_parameters()


if __name__ == "__main__":
    unittest.main()
