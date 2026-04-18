# Copyright © 2024 Apple Inc.

import unittest
from pathlib import Path

from huggingface_hub import snapshot_download

from mlx_lm.tokenizer_utils import (
    BPEStreamingDetokenizer,
    NaiveStreamingDetokenizer,
    SPMStreamingDetokenizer,
)
from mlx_lm.utils import load_tokenizer


class TestTokenizers(unittest.TestCase):

    def check_tokenizer(self, tokenizer):
        def check(tokens):
            expected_text = tokenizer.decode(tokens)
            detokenizer = tokenizer.detokenizer
            detokenizer.reset()
            text = ""
            for e, t in enumerate(tokens):
                detokenizer.add_token(t)
                seg = detokenizer.last_segment
                text += seg
                self.assertEqual(detokenizer.tokens, tokens[: e + 1])
            detokenizer.finalize()
            text += detokenizer.last_segment
            self.assertEqual(text, expected_text)

        tokens = tokenizer.encode("こんにちは！私の名前はAI")
        check(tokens)

        tokens = tokenizer.encode("⊕ ⊻ ∧ ¬")
        check(tokens)

        tokens = tokenizer.encode("a ,b")
        check(tokens)

        tokens = tokenizer.encode('{"why_its_funny" :"a_joke_explainer" ,"rating":3.5}')
        check(tokens)

        tokens = tokenizer.encode("3 3")
        check(tokens)

        tokens = tokenizer.encode("import 'package:flutter/material.dart';")
        check(tokens)

        tokens = tokenizer.encode("hello\nworld")
        check(tokens)

    def test_tokenizers(self):
        tokenizer_repos = [
            ("mlx-community/Qwen1.5-0.5B-Chat-4bit", BPEStreamingDetokenizer),
            ("mlx-community/Mistral-7B-v0.2-4bit", SPMStreamingDetokenizer),
            ("mlx-community/Phi-3.5-mini-instruct-4bit", SPMStreamingDetokenizer),
            ("mlx-community/Mistral-7B-Instruct-v0.3", SPMStreamingDetokenizer),
            ("mlx-community/Llama-3.2-1B-Instruct-4bit", BPEStreamingDetokenizer),
            ("mlx-community/Falcon3-7B-Instruct-4bit", BPEStreamingDetokenizer),
        ]
        for tokenizer_repo, expected_detokenizer in tokenizer_repos:
            with self.subTest(tokenizer=tokenizer_repo):
                tokenizer = load_tokenizer(tokenizer_repo)
                tokenizer.decode([0, 1, 2])
                self.assertTrue(isinstance(tokenizer.detokenizer, expected_detokenizer))
                self.check_tokenizer(tokenizer)

        # Try one with a naive detokenizer
        tokenizer = load_tokenizer("mlx-community/Llama-3.2-1B-Instruct-4bit")
        tokenizer._detokenizer = NaiveStreamingDetokenizer(tokenizer)
        self.check_tokenizer(tokenizer)

    def test_special_tokens(self):
        tokenizer_repo = "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx"
        tokenizer = load_tokenizer(tokenizer_repo)

        detokenizer = tokenizer.detokenizer
        detokenizer.reset()
        detokenizer.add_token(tokenizer.eos_token_id)
        detokenizer.finalize()

        self.assertEqual(detokenizer.last_segment, tokenizer.eos_token)

    def test_tool_calling(self):
        tokenizer_repo = "mlx-community/Qwen3-4B-4bit"
        tokenizer = load_tokenizer(tokenizer_repo)
        self.assertTrue(tokenizer.has_tool_calling)
        self.assertEqual(tokenizer.tool_call_start, "<tool_call>")
        self.assertEqual(tokenizer.tool_call_end, "</tool_call>")

        tokenizer_repo = "mlx-community/Llama-3.2-1B-Instruct-4bit"
        tokenizer = load_tokenizer(tokenizer_repo)
        self.assertFalse(tokenizer.has_tool_calling)

    def test_thinking(self):
        tokenizer_repo = "mlx-community/Qwen3-4B-4bit"
        tokenizer = load_tokenizer(tokenizer_repo)
        self.assertTrue(tokenizer.has_thinking)
        self.assertEqual(tokenizer.think_start, "<think>")
        self.assertEqual(tokenizer.think_end, "</think>")


class _StubTokenizer:
    """Minimal tokenizer stub for testing marker-discovery in isolation.

    Mirrors the HuggingFace behaviour where unset extra special tokens
    (``boi_token``, ``stc_token`` etc.) return ``None`` via ``__getattr__``.
    """

    def __init__(self, **named_tokens):
        self._named_tokens = named_tokens
        self.eos_token_id = 0
        self.chat_template = None

    def __getattr__(self, name):
        # Instance dict is checked first; this only fires for unset names.
        if name in self._named_tokens:
            return self._named_tokens[name]
        if name.endswith("_token"):
            return None
        raise AttributeError(name)

    def get_vocab(self):
        return {}

    def encode(self, text, add_special_tokens=False):
        # Deterministic fake tokenisation; two IDs per input string.
        return [100, 101]


class TestMarkerDiscovery(unittest.TestCase):
    """Tests for _infer_markers_from_config and related wrapper plumbing."""

    def test_config_discovers_tool_markers(self):
        """stc_token / etc_token → tool_call_start / tool_call_end."""
        from mlx_lm.tokenizer_utils import _infer_markers_from_config

        tok = _StubTokenizer(
            stc_token="<|tool_call>",
            etc_token="<tool_call|>",
        )
        result = _infer_markers_from_config(tok)
        self.assertEqual(result["tool_call_start"], "<|tool_call>")
        self.assertEqual(result["tool_call_end"], "<tool_call|>")

    def test_config_no_markers_returns_none(self):
        """Tokenizer without config fields returns None markers."""
        from mlx_lm.tokenizer_utils import _infer_markers_from_config

        tok = _StubTokenizer()
        result = _infer_markers_from_config(tok)
        self.assertIsNone(result["tool_call_start"])
        self.assertIsNone(result["tool_call_end"])

    def test_config_partial_markers_ignored(self):
        """Only stc_token without etc_token → no markers set."""
        from mlx_lm.tokenizer_utils import _infer_markers_from_config

        tok = _StubTokenizer(stc_token="<|tool_call>")
        result = _infer_markers_from_config(tok)
        self.assertIsNone(result["tool_call_start"])
        self.assertIsNone(result["tool_call_end"])

    def test_config_markers_enable_tool_calling(self):
        """Markers passed to TokenizerWrapper should flip has_tool_calling."""
        from mlx_lm.tokenizer_utils import TokenizerWrapper

        wrapper = TokenizerWrapper(
            _StubTokenizer(),
            tool_call_start="<|tool_call>",
            tool_call_end="<tool_call|>",
        )
        self.assertTrue(wrapper.has_tool_calling)
        self.assertEqual(wrapper.tool_call_start, "<|tool_call>")
        self.assertEqual(wrapper.tool_call_end, "<tool_call|>")

    def test_think_start_end_params_override_inference(self):
        """Explicit think_start/think_end bypass _infer_thinking."""
        from mlx_lm.tokenizer_utils import TokenizerWrapper

        wrapper = TokenizerWrapper(
            _StubTokenizer(),
            think_start="<think>",
            think_end="</think>",
        )
        self.assertTrue(wrapper.has_thinking)
        self.assertEqual(wrapper.think_start, "<think>")
        self.assertEqual(wrapper.think_end, "</think>")

    def test_parser_markers_take_precedence(self):
        """Integration: when a parser module exists, its markers win.

        Verifies that adding config-based discovery does not regress any
        currently-supported model.  Qwen3 is matched by _infer_tool_parser
        and must keep using the parser module's markers.
        """
        tokenizer = load_tokenizer("mlx-community/Qwen3-4B-4bit")
        self.assertTrue(tokenizer.has_tool_calling)
        self.assertEqual(tokenizer.tool_call_start, "<tool_call>")
        self.assertEqual(tokenizer.tool_call_end, "</tool_call>")


if __name__ == "__main__":
    unittest.main()
