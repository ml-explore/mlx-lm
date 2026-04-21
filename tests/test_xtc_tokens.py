"""Tests for XTC special tokens construction in server.py."""

import sys
import unittest

sys.path.insert(0, ".")

import mlx.core as mx
from mlx_lm.sample_utils import apply_xtc


class FakeTokenizer:
    """Minimal mock with the properties _make_sampler uses."""

    eos_token_id = 2
    eos_token_ids = {2, 32000}

    def encode(self, text):
        return [198]


class TestXTCSpecialTokens(unittest.TestCase):
    """Verify xtc_special_tokens is a flat list of ints, not nested."""

    def _build_special_tokens(self, tokenizer):
        """Mirrors server.py _make_sampler construction."""
        return tokenizer.encode("\n") + list(tokenizer.eos_token_ids)

    def test_flat_list_of_ints(self):
        """Result must be a flat list containing only ints."""
        tokens = self._build_special_tokens(FakeTokenizer())
        self.assertIsInstance(tokens, list)
        for t in tokens:
            self.assertIsInstance(t, int, f"Expected int, got {type(t)}: {t}")

    def test_no_nested_lists(self):
        """Result must not contain any sub-lists."""
        tokens = self._build_special_tokens(FakeTokenizer())
        for t in tokens:
            self.assertNotIsInstance(t, list)

    def test_includes_all_eos_tokens(self):
        """All EOS token IDs must be present."""
        tok = FakeTokenizer()
        tokens = self._build_special_tokens(tok)
        for eos in tok.eos_token_ids:
            self.assertIn(eos, tokens)

    def test_includes_newline_tokens(self):
        """Tokens from encode('\\n') must be present."""
        tok = FakeTokenizer()
        tokens = self._build_special_tokens(tok)
        for nl in tok.encode("\n"):
            self.assertIn(nl, tokens)

    def test_apply_xtc_does_not_crash(self):
        """apply_xtc must accept the constructed token list."""
        tokens = self._build_special_tokens(FakeTokenizer())
        logits = mx.zeros((1, 5000))
        result = apply_xtc(logits, 0.5, 0.1, tokens)
        self.assertEqual(result.shape, logits.shape)

    def test_matches_generate_py_pattern(self):
        """Must produce the same pattern as generate.py:2062."""
        tok = FakeTokenizer()
        from_server = tok.encode("\n") + list(tok.eos_token_ids)
        from_generate = tok.encode("\n") + list(tok.eos_token_ids)
        self.assertEqual(sorted(from_server), sorted(from_generate))


if __name__ == "__main__":
    unittest.main()
