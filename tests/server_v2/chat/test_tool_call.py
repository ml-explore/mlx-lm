# Copyright © 2026 Apple Inc.

import json
import unittest

from mlx_lm.generate import TextStateMachine
from mlx_lm.server_v2.chat import tool_call
from tests.server_v2 import helpers

Delta = TextStateMachine.Delta
TOOLS = [{"type": "function", "function": {"name": "f"}}]


class TestToolCallParser(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.parser = tool_call.ToolCallParser(
            helpers.load_chat_tokenizer(), tools=TOOLS
        )

    def test_arguments(self):
        cases = [
            ('{"name": "f"}', {}),
            ('{"name": "f", "arguments": "{\\"a\\": 1}"}', {"a": 1}),
            ('{"name": "f", "arguments": {"s": "\\u00e9"}}', {"s": "\u00e9"}),
        ]
        for text, arguments in cases:
            _, (call,) = self.parser.parse(Delta(tool_calls=(text,)))
            self.assertEqual(call.arguments, arguments)
        calls = [
            {"name": "f", "id": "abc"},
            {"name": "g", "id": 7},
            {"name": "h", "id": ""},
        ]
        _, calls = self.parser.parse(Delta(tool_calls=(json.dumps(calls),)))
        self.assertEqual([c.name for c in calls], ["f", "g", "h"])
        self.assertEqual(calls[0].id, "abc")
        for call in calls[1:]:
            self.assertRegex(call.id, "^[A-Za-z0-9]{9}$")

    def test_calls_that_stay_text(self):
        cases = [
            "{bad}",
            "[]",
            '{"name": 5}',
            '{"name": ""}',
            '{"name": "f", "arguments": [1]}',
            '{"name": "f", "arguments": "x"}',
            '{"name": "f", "arguments": {"a": NaN}}',
            # All calls of a block parse, or none of them.
            '[{"name": "f"}, {"name": "g", "arguments": 1}]',
        ]
        for text in cases:
            with self.subTest(text=text), self.assertLogs(level="WARNING"):
                parsed = self.parser.parse(Delta(content="Hi", tool_calls=(text,)))
            self.assertEqual(parsed, (f"Hi<tool_call>{text}</tool_call>", []))
        parser = tool_call.ToolCallParser(helpers.load_chat_tokenizer())
        parsed = parser.parse(Delta(tool_calls=('{"name": "f"}',)))
        self.assertEqual(parsed, ('<tool_call>{"name": "f"}</tool_call>', []))


if __name__ == "__main__":
    unittest.main()
