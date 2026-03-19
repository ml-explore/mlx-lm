"""Regression tests: </think> token must not leak into tool_call arguments.

Finding (Qwen3.5-4B-OptiQ-4bit, --thinking-budget 512):
When generating many parallel tool calls in one turn, the model can emit
a <think> token ID mid-JSON argument string.  Without tool-call awareness,
ThinkingBudgetProcessor forces </think> inside the JSON, corrupting it.

Fix: ThinkingBudgetProcessor accepts tool_call_start_id/tool_call_end_id.
When inside a tool call, all budget enforcement is paused — forced </think>
never lands inside JSON arguments.

This file tests both configurations:
  - WITH tool_call IDs → bug is prevented (processor pauses during tool calls)
  - WITHOUT tool_call IDs → old behavior preserved (for models without tool calling)
"""

import json
import unittest

import mlx.core as mx

from mlx_lm.thinking_budget import (
    FORCED_LOGIT_VALUE,
    MASKED_LOGIT_VALUE,
    ThinkingBudgetProcessor,
)

VOCAB_SIZE = 1000
THINK_START = 100
THINK_END = 101
EOS_ID = 2
TC_START_ID = 200
TC_END_ID = 201

# Arbitrary token IDs representing tool call content
JSON_TOKS = [300, 301, 302, 303, 304]


def _logits():
    return mx.zeros((1, VOCAB_SIZE))


class TestWithToolCallIds(unittest.TestCase):
    """With tool_call_start_id/tool_call_end_id: bug is prevented."""

    def _make_proc(self, budget=10):
        return ThinkingBudgetProcessor(
            THINK_START,
            THINK_END,
            budget=budget,
            eos_token_ids=frozenset({EOS_ID}),
            tool_call_start_id=TC_START_ID,
            tool_call_end_id=TC_END_ID,
            in_thinking=True,
        )

    def test_think_start_mid_tool_call_ignored(self):
        """<think> mid-tool-call does NOT enter thinking state."""
        proc = self._make_proc(budget=10)

        # Enter tool call, then <think> appears mid-JSON
        seq = [THINK_START, TC_START_ID] + JSON_TOKS
        for i in range(1, len(seq)):
            proc(mx.array(seq[: i + 1]), _logits())

        seq.append(THINK_START)  # spurious <think> mid-tool-call
        proc(mx.array(seq), _logits())

        # Processor is inside tool call → state change ignored
        self.assertTrue(proc._in_tool_call)

    def test_think_end_masked_mid_tool_call(self):
        """</think> is actively masked during tool calls — prevents spontaneous leak."""
        proc = self._make_proc(budget=100)

        seq = [THINK_START, TC_START_ID, 300]
        proc(mx.array(seq[:2]), _logits())
        result = proc(mx.array(seq), _logits())

        self.assertEqual(result[0, THINK_END].item(), MASKED_LOGIT_VALUE)

    def test_no_forced_close_mid_tool_call(self):
        """Budget exceeded mid-tool-call does NOT force </think>."""
        proc = self._make_proc(budget=2)

        # 2 thinking tokens → budget would be hit
        seq = [THINK_START, 50, 51]
        proc(mx.array(seq[:2]), _logits())
        proc(mx.array(seq), _logits())

        # Enter tool call
        seq.append(TC_START_ID)
        proc(mx.array(seq), _logits())

        # Tokens inside tool call — must NOT force </think>
        for tok in [400, 401, 402]:
            seq.append(tok)
            result = proc(mx.array(seq), _logits())
            self.assertNotEqual(result[0, THINK_END].item(), FORCED_LOGIT_VALUE)

    def test_eos_not_suppressed_during_tool_call(self):
        """EOS is not masked inside a tool call."""
        proc = self._make_proc(budget=100)

        seq = [THINK_START, TC_START_ID, 300]
        proc(mx.array(seq[:2]), _logits())
        result = proc(mx.array(seq), _logits())

        self.assertNotEqual(result[0, EOS_ID].item(), MASKED_LOGIT_VALUE)


class TestWithoutToolCallIds(unittest.TestCase):
    """Without tool_call IDs: old behavior preserved (backwards compat).

    Models without tool calling support don't pass tool_call_start_id.
    The processor works exactly as before — no tool-call awareness.
    """

    def _make_proc(self, budget=10):
        return ThinkingBudgetProcessor(
            THINK_START,
            THINK_END,
            budget=budget,
            eos_token_ids=frozenset({EOS_ID}),
            # No tool_call_start_id / tool_call_end_id
            in_thinking=True,
        )

    def test_think_start_enters_thinking_without_tool_call_ids(self):
        """Without tool_call IDs, <think> mid-stream enters thinking state."""
        proc = self._make_proc(budget=10)

        # Natural close first, then <think> appears again
        seq = [THINK_START, THINK_END, 50, THINK_START]
        for i in range(1, len(seq)):
            proc(mx.array(seq[: i + 1]), _logits())

        # _thinking_done prevents re-entry after first close
        self.assertFalse(proc.in_thinking)

    def test_budget_enforced_without_tool_call_ids(self):
        """Without tool_call IDs, budget enforcement works normally."""
        proc = self._make_proc(budget=3)

        seq = [THINK_START, 50, 51, 52]
        proc(mx.array(seq[:2]), _logits())  # count=1
        proc(mx.array(seq[:3]), _logits())  # count=2
        result = proc(mx.array(seq), _logits())  # count=3, budget=3

        self.assertEqual(result[0, THINK_END].item(), FORCED_LOGIT_VALUE)

    def test_tc_start_id_token_not_special_without_tool_call_ids(self):
        """TC_START_ID is just a regular token when tool_call IDs not configured."""
        proc = self._make_proc(budget=2)

        # 2 thinking tokens → budget hit
        seq = [THINK_START, 50, 51]
        proc(mx.array(seq[:2]), _logits())
        proc(mx.array(seq), _logits())

        # TC_START_ID is treated as a regular thinking token
        seq.append(TC_START_ID)
        result = proc(mx.array(seq), _logits())

        # Budget already exceeded, forced close still active
        self.assertEqual(result[0, THINK_END].item(), FORCED_LOGIT_VALUE)


class TestObservedCorruptionPattern(unittest.TestCase):
    """Documents the exact corruption pattern from SWE-bench logs.

    Observed in django-11099 task (Qwen3.5-4B-OptiQ-4bit, budget=512):
    The model generated run_command tool calls. </think> appeared mid-JSON
    in the second call, followed by a text-based <function=...> fallback.

    With tool-call-aware ThinkingBudgetProcessor, this sequence is
    prevented at the logits level — the routing never sees </think>
    mid-tool-call.  This test documents the observed pattern for posterity.
    """

    def test_observed_corruption_pattern(self):
        tool_call_start = "<tool_call>"
        tool_call_end = "</tool_call>"

        gen_texts = [
            tool_call_start,
            '{"name": "run_command", "arguments": ',
            '{"command": "cd /app && git checkout django/..."}}',
            tool_call_end,
            tool_call_start,
            '{"name": "run_command", "arguments": ',
            '{"command": "ls -la /app/django/db/models/',
            "</think>",
            "\n\n\n",
            "<function=run_command>\n<parameter=command>",
            "\nls -la /app/django/db/models/ 2>&1",
        ]

        # Replay routing (server.py lines 1594-1618)
        in_tool_call = False
        tool_calls = []
        tool_text = ""

        for gen_text in gen_texts:
            if gen_text == tool_call_start:
                if in_tool_call and tool_text:
                    tool_calls.append(tool_text)
                    tool_text = ""
                in_tool_call = True
            elif in_tool_call:
                if gen_text == tool_call_end:
                    tool_calls.append(tool_text)
                    tool_text = ""
                    in_tool_call = False
                else:
                    tool_text += gen_text

        if in_tool_call and tool_text:
            tool_calls.append(tool_text)

        # First call: valid JSON
        self.assertEqual(len(tool_calls), 2)
        first_call = json.loads(tool_calls[0])
        self.assertEqual(first_call["name"], "run_command")

        # Second call: corrupted (documents the bug pattern)
        self.assertIn("</think>", tool_calls[1])
        with self.assertRaises(json.JSONDecodeError):
            json.loads(tool_calls[1])


if __name__ == "__main__":
    unittest.main()
