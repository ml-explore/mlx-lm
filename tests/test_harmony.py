# Copyright © 2026 Apple Inc.

import unittest

from mlx_lm.harmony import has_harmony_format, parse_harmony_response


class _FakeTokenizer:
    def __init__(self, vocab):
        self._vocab = vocab

    def get_vocab(self):
        return self._vocab


HARMONY_VOCAB = {
    "<|channel|>": 200005,
    "<|message|>": 200008,
    "<|start|>": 200006,
    "<|end|>": 200007,
    "<|return|>": 200002,
    "<|constrain|>": 200003,
}


class TestHasHarmonyFormat(unittest.TestCase):
    def test_full_control_set_detected(self):
        self.assertTrue(has_harmony_format(_FakeTokenizer(HARMONY_VOCAB)))

    def test_plain_vocab_not_detected(self):
        self.assertFalse(has_harmony_format(_FakeTokenizer({"hello": 1, "world": 2})))

    def test_partial_control_set_not_detected(self):
        # Missing <|end|> -- a tokenizer with only some of these strings
        # (coincidentally, for something unrelated) must not false-positive.
        partial = {k: v for k, v in HARMONY_VOCAB.items() if k != "<|end|>"}
        self.assertFalse(has_harmony_format(_FakeTokenizer(partial)))


class TestParseHarmonyResponse(unittest.TestCase):
    def test_plain_text_passthrough(self):
        text = "def f():\n    return 1"
        content, reasoning = parse_harmony_response(text)
        self.assertEqual(content, text)
        self.assertEqual(reasoning, "")

    def test_analysis_then_final_no_fence(self):
        text = (
            "<|channel|>analysis<|message|>Just provide function.<|end|>"
            "<|start|>assistant<|channel|>final<|message|>"
            "def square(n):\n    return n * n"
        )
        content, reasoning = parse_harmony_response(text)
        self.assertEqual(content, "def square(n):\n    return n * n")
        self.assertEqual(reasoning, "Just provide function.")
        self.assertNotIn("<|", content)

    def test_analysis_then_final_fenced(self):
        text = (
            "<|channel|>analysis<|message|>Reasoning about the discount "
            "algorithm.<|end|><|start|>assistant<|channel|>final<|message|>"
            "book_store.py\n```python\ndef total(basket):\n    return 0\n```"
        )
        content, reasoning = parse_harmony_response(text)
        self.assertTrue(content.startswith("book_store.py"))
        self.assertIn("discount algorithm", reasoning)
        self.assertNotIn("<|", content)

    def test_commentary_tool_call_kept_in_content(self):
        # ml-explore/mlx-lm#875's own expected behavior: a commentary
        # channel's tool-call payload belongs in content, not dropped.
        text = (
            "<|channel|>analysis<|message|>User wants Telegram message.<|end|>"
            "<|start|>assistant<|channel|>commentary<|constrain|>json<|message|>"
            "<send_telegram_message><chat_id>12345</chat_id></send_telegram_message>"
        )
        content, reasoning = parse_harmony_response(text)
        self.assertEqual(
            content,
            "<send_telegram_message><chat_id>12345</chat_id></send_telegram_message>",
        )
        self.assertIn("Telegram", reasoning)

    def test_truncated_mid_analysis_yields_empty_content(self):
        # Generation hit the completion length cap before any "final"
        # channel ever started -- must not fabricate content or raise.
        text = "<|channel|>analysis<|message|>Still reasoning about the appro"
        content, reasoning = parse_harmony_response(text)
        self.assertEqual(content, "")
        self.assertIn("Still reasoning", reasoning)

    def test_multiple_analysis_segments_concatenate(self):
        text = (
            "<|channel|>analysis<|message|>First thought.<|end|>"
            "<|start|>assistant<|channel|>analysis<|message|>Second thought.<|end|>"
            "<|start|>assistant<|channel|>final<|message|>answer"
        )
        content, reasoning = parse_harmony_response(text)
        self.assertEqual(content, "answer")
        self.assertEqual(reasoning, "First thought.Second thought.")


if __name__ == "__main__":
    unittest.main()
