"""test_server_routing.py — Spec tests for server.py token routing state machine.

The routing loop classifies each generated token into one of four buckets:
  reasoning_text, tool_call text, visible text, or a state transition.

Confirmed facts (Qwen3.5-9B, verified with tokenizer):
  </think>     → single special token (ID 248069)
  <tool_call>  → single special token (ID 248058)
  </tool_call> → single special token (ID 248059)

So gen.text == ctx.tool_call_start is a single-token comparison that reliably
fires; it does not require multi-token buffering.

FIX (server.py, generation loop):
  Inside the `if in_reasoning` branch, an early-exit on tool_call_start ensures
  tool calls emitted before </think> are parsed correctly:

      elif ctx.has_tool_calling and gen.text == ctx.tool_call_start:
          in_reasoning = False
          made_tool_call = True
          in_tool_call = True
"""

import unittest

THINK_END = "</think>"
TC_START = "<tool_call>"
TC_END = "</tool_call>"


def _route_current(tokens: list[str]) -> dict:
    """Mirrors server.py routing loop (generation loop body).

    Update this function in lockstep with server.py lines 1595-1612.
    """
    in_reasoning = True  # Qwen3 chat template always opens <think> in assistant prefix
    in_tool_call = False
    made_tool_call = False
    reasoning_text = ""
    tool_calls: list[str] = []
    tool_text = ""
    text = ""

    for tok in tokens:
        if in_reasoning:
            if tok == THINK_END:
                in_reasoning = False
            elif tok == TC_START:
                in_reasoning = False
                made_tool_call = True
                in_tool_call = True
            else:
                reasoning_text += tok
        elif tok == TC_START:
            made_tool_call = True
            in_tool_call = True
        elif in_tool_call:
            if tok == TC_END:
                tool_calls.append(tool_text)
                tool_text = ""
                in_tool_call = False
            else:
                tool_text += tok
        else:
            text += tok

    return dict(
        text=text,
        reasoning=reasoning_text,
        tool_calls=tool_calls,
        finish_reason="tool_calls" if made_tool_call else "stop",
    )


class TestServerTokenRouting(unittest.TestCase):
    """Routing loop spec — all cases must pass with the fix applied."""

    def test_normal_flow_tool_call_after_think_close(self):
        """Happy path: </think> before <tool_call> — tool call parsed correctly."""
        tokens = [
            "I need to call something.\n",
            THINK_END,
            TC_START,
            '{"name":"run_cmd","arguments":{"cmd":"ls"}}',
            TC_END,
        ]
        r = _route_current(tokens)

        self.assertEqual(
            r["tool_calls"], ['{"name":"run_cmd","arguments":{"cmd":"ls"}}']
        )
        self.assertEqual(r["finish_reason"], "tool_calls")
        self.assertEqual(r["reasoning"], "I need to call something.\n")
        self.assertEqual(r["text"], "")

    def test_tool_call_inside_think_exits_reasoning_and_is_parsed(self):
        """<tool_call> seen while in_reasoning exits thinking and parses call."""
        tokens = [
            "Let me call the function.\n",
            TC_START,
            '{"name":"read_file","arguments":{"path":"/x"}}',
            TC_END,
        ]
        r = _route_current(tokens)

        self.assertEqual(
            r["tool_calls"], ['{"name":"read_file","arguments":{"path":"/x"}}']
        )
        self.assertEqual(r["finish_reason"], "tool_calls")
        self.assertEqual(r["reasoning"], "Let me call the function.\n")
        self.assertEqual(r["text"], "")

    def test_reasoning_text_before_tool_call_preserved(self):
        """Reasoning tokens before the in-think tool call appear in reasoning field."""
        tokens = [
            "step A\n",
            "step B\n",
            TC_START,
            '{"name":"f","arguments":{}}',
            TC_END,
        ]
        r = _route_current(tokens)

        self.assertEqual(r["reasoning"], "step A\nstep B\n")
        self.assertEqual(r["tool_calls"], ['{"name":"f","arguments":{}}'])
        self.assertEqual(r["finish_reason"], "tool_calls")

    def test_multiple_tool_calls_inside_think_all_parsed(self):
        """Multiple tool calls emitted inside thinking are all captured."""
        tokens = [
            "need two calls\n",
            TC_START,
            '{"name":"a"}',
            TC_END,
            TC_START,
            '{"name":"b"}',
            TC_END,
        ]
        r = _route_current(tokens)

        self.assertEqual(len(r["tool_calls"]), 2)
        self.assertEqual(r["tool_calls"][0], '{"name":"a"}')
        self.assertEqual(r["tool_calls"][1], '{"name":"b"}')
        self.assertEqual(r["finish_reason"], "tool_calls")

    def test_text_after_think_close_captured(self):
        """Visible text generated after </think> (no tool call) is captured."""
        tokens = ["reasoning\n", THINK_END, "Here is the answer.\n"]
        r = _route_current(tokens)

        self.assertEqual(r["text"], "Here is the answer.\n")
        self.assertEqual(r["reasoning"], "reasoning\n")
        self.assertEqual(r["finish_reason"], "stop")

    def test_no_tool_call_tokens_in_reasoning_field(self):
        """After fix: <tool_call> token is not absorbed into reasoning_text."""
        tokens = ["thinking\n", TC_START, '{"name":"f"}', TC_END]
        r = _route_current(tokens)

        self.assertNotIn(TC_START, r["reasoning"])
        self.assertNotIn(TC_END, r["reasoning"])


if __name__ == "__main__":
    unittest.main()
