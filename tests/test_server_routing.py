"""test_server_routing.py — Spec tests for server.py token routing state machine.

The routing loop classifies each generated token into one of four buckets:
  reasoning_text, tool_call text, visible text, or a state transition.

Confirmed facts (Qwen3.5-9B, verified with tokenizer):
  </think>     → single special token (ID 248069)
  <tool_call>  → single special token (ID 248058)
  </tool_call> → single special token (ID 248059)

So gen.text == ctx.tool_call_start is a single-token comparison that reliably
fires; it does not require multi-token buffering.

State machine paths (update when routing changes):

  REASONING state (in_reasoning=True):
    Path 1: </think>      → transition to NORMAL
    Path 2: <tool_call>   → transition to TOOL_CALL
    Path 3: other token   → accumulate to reasoning_text

  NORMAL state (in_reasoning=False, in_tool_call=False):
    Path 4: <tool_call>   → transition to TOOL_CALL
    Path 5: other token   → accumulate to text

  TOOL_CALL state (in_tool_call=True):
    Path 6: </tool_call>  → append tool_text, transition to NORMAL
    Path 7: other token   → accumulate to tool_text

  Edge cases:
    Path 8:  incomplete tool call (no </tool_call>) → flush tool_text
    Path 9:  empty reasoning (immediate </think>)
    Path 10: multiple sequential tool calls
    Path 11: text interleaved between tool calls

Note: </think> mid-tool-call (spurious tokens from budget enforcement or
quantised models) is prevented at the logits processor level by
ThinkingBudgetProcessor's tool-call awareness.  The routing does not need
to handle this case.
"""

import unittest

THINK_END = "</think>"
TC_START = "<tool_call>"
TC_END = "</tool_call>"


def _route_current(tokens: list[str]) -> dict:
    """Mirrors server.py routing loop (generation loop body).

    Update this function in lockstep with server.py lines 1595-1618.
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

    # Flush remaining tool text (server.py line 1674)
    if in_tool_call and tool_text:
        tool_calls.append(tool_text)

    return dict(
        text=text,
        reasoning=reasoning_text,
        tool_calls=tool_calls,
        finish_reason="tool_calls" if made_tool_call else "stop",
    )


# ── REASONING state (paths 1-3) ─────────────────────────────────────────


class TestReasoningState(unittest.TestCase):
    """Paths 1-3: token routing while in_reasoning=True."""

    def test_path1_think_end_exits_reasoning(self):
        """Path 1: </think> transitions from REASONING to NORMAL."""
        tokens = ["step A\n", THINK_END, "visible text"]
        r = _route_current(tokens)

        self.assertEqual(r["reasoning"], "step A\n")
        self.assertEqual(r["text"], "visible text")
        self.assertEqual(r["finish_reason"], "stop")

    def test_path2_tool_call_start_exits_reasoning(self):
        """Path 2: <tool_call> transitions from REASONING to TOOL_CALL."""
        tokens = ["thinking\n", TC_START, '{"name":"f"}', TC_END]
        r = _route_current(tokens)

        self.assertEqual(r["reasoning"], "thinking\n")
        self.assertEqual(r["tool_calls"], ['{"name":"f"}'])
        self.assertEqual(r["finish_reason"], "tool_calls")

    def test_path3_other_accumulates_reasoning(self):
        """Path 3: non-special tokens accumulate into reasoning_text."""
        tokens = ["step A\n", "step B\n", THINK_END]
        r = _route_current(tokens)

        self.assertEqual(r["reasoning"], "step A\nstep B\n")
        self.assertEqual(r["text"], "")

    def test_path2_reasoning_preserved_before_tool_call(self):
        """Path 2+3: reasoning tokens before tool call appear in reasoning field."""
        tokens = ["step A\n", "step B\n", TC_START, '{"name":"f"}', TC_END]
        r = _route_current(tokens)

        self.assertEqual(r["reasoning"], "step A\nstep B\n")
        self.assertEqual(r["tool_calls"], ['{"name":"f"}'])

    def test_path2_no_tool_call_tokens_in_reasoning(self):
        """Path 2: <tool_call>/<tool_call> tokens not absorbed into reasoning."""
        tokens = ["thinking\n", TC_START, '{"name":"f"}', TC_END]
        r = _route_current(tokens)

        self.assertNotIn(TC_START, r["reasoning"])
        self.assertNotIn(TC_END, r["reasoning"])


# ── NORMAL state (paths 4-5) ────────────────────────────────────────────


class TestNormalState(unittest.TestCase):
    """Paths 4-5: token routing after reasoning ends (in_reasoning=False)."""

    def test_path4_tool_call_after_think_close(self):
        """Path 4: <tool_call> after </think> transitions to TOOL_CALL."""
        tokens = [
            "reasoning\n", THINK_END,
            TC_START, '{"name":"run_cmd","arguments":{"cmd":"ls"}}', TC_END,
        ]
        r = _route_current(tokens)

        self.assertEqual(
            r["tool_calls"], ['{"name":"run_cmd","arguments":{"cmd":"ls"}}']
        )
        self.assertEqual(r["finish_reason"], "tool_calls")
        self.assertEqual(r["reasoning"], "reasoning\n")
        self.assertEqual(r["text"], "")

    def test_path5_text_after_think_close(self):
        """Path 5: non-special tokens after </think> accumulate as text."""
        tokens = ["reasoning\n", THINK_END, "Here is the answer.\n"]
        r = _route_current(tokens)

        self.assertEqual(r["text"], "Here is the answer.\n")
        self.assertEqual(r["reasoning"], "reasoning\n")
        self.assertEqual(r["finish_reason"], "stop")

    def test_path5_multiple_text_tokens(self):
        """Path 5: multiple text tokens all accumulate."""
        tokens = ["r\n", THINK_END, "Hello ", "world", "!"]
        r = _route_current(tokens)

        self.assertEqual(r["text"], "Hello world!")


# ── TOOL_CALL state (paths 6-7) ─────────────────────────────────────────


class TestToolCallState(unittest.TestCase):
    """Paths 6-7: token routing while in_tool_call=True."""

    def test_path6_tool_call_end_closes_and_appends(self):
        """Path 6: </tool_call> appends tool_text and transitions to NORMAL."""
        tokens = [
            THINK_END, TC_START,
            '{"name":"f","arguments":{"x":1}}',
            TC_END,
        ]
        r = _route_current(tokens)

        self.assertEqual(r["tool_calls"], ['{"name":"f","arguments":{"x":1}}'])
        self.assertEqual(r["finish_reason"], "tool_calls")

    def test_path7_other_accumulates_tool_text(self):
        """Path 7: non-special tokens accumulate into tool_text."""
        tokens = [
            THINK_END, TC_START,
            '{"name":"f",',
            '"arguments":',
            '{"x":1}}',
            TC_END,
        ]
        r = _route_current(tokens)

        self.assertEqual(r["tool_calls"], ['{"name":"f","arguments":{"x":1}}'])


# ── Edge cases (paths 8-11) ─────────────────────────────────────────────


class TestEdgeCases(unittest.TestCase):
    """Edge cases: flush, empty, multiple, interleaved."""

    def test_path8_incomplete_tool_call_flushed(self):
        """Path 8: tool call without </tool_call> is flushed at end."""
        tokens = [
            THINK_END, TC_START,
            '{"name":"f","arguments":{"cmd":"find .',
        ]
        r = _route_current(tokens)

        self.assertEqual(len(r["tool_calls"]), 1)
        self.assertIn("find .", r["tool_calls"][0])
        self.assertEqual(r["finish_reason"], "tool_calls")

    def test_path9_empty_reasoning(self):
        """Path 9: immediate </think> produces empty reasoning."""
        tokens = [THINK_END, "Hello"]
        r = _route_current(tokens)

        self.assertEqual(r["reasoning"], "")
        self.assertEqual(r["text"], "Hello")

    def test_path10_multiple_tool_calls(self):
        """Path 10: multiple sequential tool calls all captured."""
        tokens = [
            "thinking\n", THINK_END,
            TC_START, '{"name":"a"}', TC_END,
            TC_START, '{"name":"b"}', TC_END,
        ]
        r = _route_current(tokens)

        self.assertEqual(len(r["tool_calls"]), 2)
        self.assertEqual(r["tool_calls"][0], '{"name":"a"}')
        self.assertEqual(r["tool_calls"][1], '{"name":"b"}')
        self.assertEqual(r["finish_reason"], "tool_calls")

    def test_path10_multiple_tool_calls_inside_reasoning(self):
        """Path 10: multiple tool calls emitted inside reasoning."""
        tokens = [
            "need two calls\n",
            TC_START, '{"name":"a"}', TC_END,
            TC_START, '{"name":"b"}', TC_END,
        ]
        r = _route_current(tokens)

        self.assertEqual(len(r["tool_calls"]), 2)
        self.assertEqual(r["tool_calls"][0], '{"name":"a"}')
        self.assertEqual(r["tool_calls"][1], '{"name":"b"}')

    def test_path11_text_between_tool_calls(self):
        """Path 11: text tokens between tool calls captured."""
        tokens = [
            THINK_END,
            TC_START, '{"name":"a"}', TC_END,
            "some text",
            TC_START, '{"name":"b"}', TC_END,
        ]
        r = _route_current(tokens)

        self.assertEqual(r["tool_calls"], ['{"name":"a"}', '{"name":"b"}'])
        self.assertEqual(r["text"], "some text")

    def test_path6_tool_text_reset_between_calls(self):
        """Path 6: tool_text is reset after each </tool_call>."""
        tokens = [
            THINK_END,
            TC_START, "first", TC_END,
            TC_START, "second", TC_END,
        ]
        r = _route_current(tokens)

        self.assertEqual(r["tool_calls"], ["first", "second"])

    def test_finish_reason_stop_when_no_tool_calls(self):
        """finish_reason is 'stop' when no tool calls made."""
        tokens = ["thinking\n", THINK_END, "answer"]
        r = _route_current(tokens)
        self.assertEqual(r["finish_reason"], "stop")

    def test_finish_reason_tool_calls_when_tool_call_made(self):
        """finish_reason is 'tool_calls' when at least one tool call made."""
        tokens = [THINK_END, TC_START, '{"name":"f"}', TC_END]
        r = _route_current(tokens)
        self.assertEqual(r["finish_reason"], "tool_calls")


if __name__ == "__main__":
    unittest.main()
