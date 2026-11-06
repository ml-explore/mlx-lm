# Copyright © 2026 Apple Inc.

import dataclasses
import unittest

import pydantic

from mlx_lm.server_v2 import settings
from mlx_lm.server_v2.engine import protocol


class TestSettings(unittest.TestCase):

    def test_params(self):
        params = settings.GenerationParams()
        self.assertEqual((params.max_tokens, params.temperature), (None, 0.0))
        request = protocol.Request([1])
        self.assertEqual((request.params, request.text_state), (params, None))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            params.max_tokens = 5
        bad = (
            {"max_tokens": "5"},
            {"max_tokens": 5.0},
            {"max_tokens": 0},
            {"other": 1},
        )
        for kwargs in bad:
            with self.assertRaises(pydantic.ValidationError):
                settings.GenerationParams(**kwargs)
        # A copy validates its values too.
        self.assertEqual(dataclasses.replace(params, top_k=3).top_k, 3)
        with self.assertRaises(pydantic.ValidationError):
            dataclasses.replace(params, top_k=-1)

    def test_defaults(self):
        expected = {
            "max_tokens": None,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "min_p": 0.0,
            "xtc_probability": 0.0,
            "xtc_threshold": 0.1,
            "repetition_penalty": None,
            "repetition_context_size": 20,
            "presence_penalty": None,
            "presence_context_size": 20,
            "frequency_penalty": None,
            "frequency_context_size": 20,
            "logit_bias": None,
            "stop": (),
            "logprobs": None,
        }
        self.assertEqual(dataclasses.asdict(settings.GenerationParams()), expected)

    def test_bounds(self):
        params = settings.GenerationParams(temperature=5.0, stop=["a"])
        self.assertEqual(params.stop, ("a",))
        cases = [
            {"temperature": -0.1},
            {"temperature": float("inf")},
            {"top_p": 0.0},
            {"top_p": 1.5},
            {"min_p": 1.5},
            {"xtc_probability": 1.5},
            {"repetition_penalty": -1.0},
            {"logprobs": 21},
            {"xtc_threshold": 0.6},
            {"presence_penalty": float("nan")},
            {"repetition_context_size": 3 * 10**9},
            {"logit_bias": {-1: 1.0}},
            {"stop": ("x",) * 17},
            {"stop": ("x" * 257,)},
            {"stop": (1,)},
        ]
        for kwargs in cases:
            with self.subTest(**kwargs), self.assertRaises(pydantic.ValidationError):
                settings.GenerationParams(**kwargs)

    def test_engine_config(self):
        config = settings.EngineConfig()
        self.assertIsNone(config.context_length)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            config.max_tokens = 5
        bad = (
            {"max_tokens": "5"},
            {"max_tokens": 5.0},
            {"max_tokens": 0},
            {"other": 1},
        )
        for kwargs in bad:
            with self.assertRaises(pydantic.ValidationError):
                settings.EngineConfig(**kwargs)
        settings.EngineConfig(
            context_length=2, prompt_cache_size=0, prompt_cache_bytes=0
        )
        cases = [{"context_length": 1}, {"decode_concurrency": 0}]
        for kwargs in cases + [{"prefill_step_size": 0}, {"keepalive": 0.0}]:
            with self.subTest(**kwargs), self.assertRaises(pydantic.ValidationError):
                settings.EngineConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
