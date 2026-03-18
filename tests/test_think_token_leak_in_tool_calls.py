"""Regression test: </think> token leaks into tool_call arguments.

Finding (Qwen3.5-4B-OptiQ-4bit, --thinking-budget 512):
When generating many parallel tool calls in one turn, the model can emit
a <think> token ID mid-JSON argument string.  ThinkingBudgetProcessor has
no awareness of tool call boundaries, so it enters ``in_thinking=True``
and eventually forces ``</think>`` — which lands inside the JSON string,
corrupting the tool_call arguments.

Observed output (truncated):
  arguments='{"command": "cd /app/... && pytest -x</think>\\n\\n\\n
             <function=run_command>\\n<parameter=command>\\ncd /app/..."}'

Root cause:  ThinkingBudgetProcessor tracks only <think>/</think> token
IDs.  It cannot distinguish "model is thinking" from "model is generating
a tool_call argument that happens to contain the <think> token ID".

Full observed bug chain (from SWE-bench logs, django-11099 task):
  1. Model generates parallel tool calls (6-8 per turn normally)
  2. <think> token ID appears mid-JSON in one tool_call's arguments
  3. Processor enters in_thinking=True, budget counts down
  4. Forced </think> lands inside JSON → arguments corrupted
  5. Model sees </think> as end-of-thinking, switches to text-based
     <function=...> format for remaining calls
  6. Same commands get re-emitted (observed: ls -la repeated 60+ times,
     web_search repeated 6x in a single turn)
  7. tool_call index exceeds _MAX_TOOL_CALL_INDEX (64) → crash

Log evidence (Turn 2, django-11099):
  - ls -la .../django/db/models/   (repeated ~60 times)
  - Last instance has </think> pollution in arguments
  - Turn 3: index reaches 65 → crash
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

# Arbitrary token IDs representing tool call content
TOOL_CALL_START_TOK = 200
JSON_TOKS = [300, 301, 302, 303, 304]


def _logits():
    return mx.zeros((1, VOCAB_SIZE))


class TestThinkTokenLeakInToolCallArguments(unittest.TestCase):
    """Demonstrates that ThinkingBudgetProcessor corrupts tool_call arguments.

    The processor state machine reacts to <think>/<think> token IDs
    regardless of whether the model is currently generating tool_call JSON.
    If <think> appears mid-JSON (observed under context pressure with
    quantised Qwen3.5), the forced </think> lands inside the JSON string.
    """

    def _make_proc(self, budget: int) -> ThinkingBudgetProcessor:
        return ThinkingBudgetProcessor(
            THINK_START,
            THINK_END,
            budget=budget,
            eos_token_ids=frozenset({EOS_ID}),
        )

    def test_think_start_mid_tool_call_enters_thinking_state(self):
        """<think> token appearing during tool_call generation activates
        the thinking state machine — the processor has no tool-call awareness.
        """
        proc = self._make_proc(budget=10)

        # Simulate: <tool_call> + JSON tokens, then <think> appears
        seq = [TOOL_CALL_START_TOK] + JSON_TOKS
        for i in range(1, len(seq)):
            proc(mx.array(seq[: i + 1]), _logits())
        self.assertFalse(proc.in_thinking)

        # <think> token emitted mid-JSON argument string
        seq.append(THINK_START)
        proc(mx.array(seq), _logits())

        self.assertTrue(proc.in_thinking)
        self.assertEqual(proc.count, 0)

    def test_forced_close_corrupts_tool_call_argument_tokens(self):
        """After budget exhaustion, forced </think> is injected mid-JSON.

        Token sequence (conceptual):
          <tool_call>{"name":"run","arguments":{"cmd":"pytest <think> A B C
                                                        ^          ^
                                                   enters thinking  budget hit
                                                                    -> forces </think>
        """
        budget = 3
        proc = self._make_proc(budget=budget)

        # Tool call tokens before the spurious <think>
        seq = [TOOL_CALL_START_TOK] + JSON_TOKS
        for i in range(1, len(seq)):
            proc(mx.array(seq[: i + 1]), _logits())
        self.assertFalse(proc.in_thinking)

        # <think> appears mid-JSON
        seq.append(THINK_START)
        proc(mx.array(seq), _logits())
        self.assertTrue(proc.in_thinking)

        # Model continues generating (still inside JSON from server's view,
        # but processor now counts thinking tokens)
        for tok in [400, 401, 402]:
            seq.append(tok)
            result = proc(mx.array(seq), _logits())

        # Budget exhausted -> forces </think> as the only allowed next token
        self.assertEqual(proc.count, budget)
        self.assertEqual(result[0, THINK_END].item(), FORCED_LOGIT_VALUE)
        for tok_id in range(VOCAB_SIZE):
            if tok_id == THINK_END:
                continue
            self.assertEqual(
                result[0, tok_id].item(),
                MASKED_LOGIT_VALUE,
                f"token {tok_id} should be masked when forcing </think>",
            )

    def test_eos_masked_during_spurious_thinking_in_tool_call(self):
        """EOS is suppressed while processor is in spurious thinking mode.

        The model cannot cleanly finish the tool_call via EOS -- it is
        trapped in "thinking" mode until either the budget forces </think>
        or the model emits </think> on its own.  Both corrupt the JSON.
        """
        proc = self._make_proc(budget=100)

        seq = [TOOL_CALL_START_TOK] + JSON_TOKS
        seq.append(THINK_START)  # spurious <think> mid-tool-call
        proc(mx.array(seq), _logits())
        self.assertTrue(proc.in_thinking)

        seq.append(500)
        result = proc(mx.array(seq), _logits())
        self.assertEqual(result[0, EOS_ID].item(), MASKED_LOGIT_VALUE)

    def test_observed_corruption_pattern(self):
        """Replays the exact corruption pattern from SWE-bench logs.

        Observed in django-11099 task (Qwen3.5-4B-OptiQ-4bit, budget=512):
        The model generated run_command tool calls.  The last call's JSON
        was truncated at </think>, followed by a text-based <function=...>
        fallback.  The server's response routing (in_tool_call=True) appended
        everything — including </think> — to tool_text, producing unparseable
        JSON.

        Actual corrupted arguments from logs:
          {"command": "cd /app/... && pytest ... -x</think>\\n\\n\\n
           <function=run_command>\\n<parameter=command>\\n
           cd /app/... && pytest ... -xvs 2>&1"}

        This caused the model to re-emit the same calls (ls -la repeated
        ~60 times in Turn 2), eventually exceeding the 64 tool_call index
        limit and crashing.
        """
        tool_call_start = "<tool_call>"
        tool_call_end = "</tool_call>"

        # Simulated gen.text stream matching the observed log pattern.
        # Each string represents one decoded token chunk from the model.
        gen_texts = [
            # First tool call completes normally
            tool_call_start,
            '{"name": "run_command", "arguments": ',
            '{"command": "cd /app/.swe_bench_repos/django && ',
            'git checkout django/contrib/contenttypes/management/__init__.py"}}',
            tool_call_end,
            # Second tool call: </think> corrupts the arguments
            tool_call_start,
            '{"name": "run_command", "arguments": ',
            '{"command": "ls -la /app/.swe_bench_repos/django/db/models/',
            # --- Corruption point: forced </think> lands here ---
            "</think>",
            "\n\n\n",
            # Model switches to text-based format (no longer valid JSON)
            "<function=run_command>\n<parameter=command>",
            "\nls -la /app/.swe_bench_repos/django/db/models/ 2>&1",
            # Model never emits </tool_call> — generation continues
            # with repeated calls until index overflow
        ]

        # Replay the server's response routing state machine
        # (server.py lines 1594-1613)
        in_tool_call = False
        tool_calls = []
        tool_text = ""

        for gen_text in gen_texts:
            if gen_text == tool_call_start:
                if in_tool_call and tool_text:
                    # Flush previous incomplete tool call
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

        # Flush remaining (server.py line 1669)
        if in_tool_call and tool_text:
            tool_calls.append(tool_text)

        # First tool call: valid JSON, completed normally
        self.assertEqual(len(tool_calls), 2)
        first_call = json.loads(tool_calls[0])
        self.assertEqual(first_call["name"], "run_command")

        # Second tool call: corrupted by </think> injection
        self.assertIn("</think>", tool_calls[1])
        self.assertIn("<function=run_command>", tool_calls[1])

        with self.assertRaises(json.JSONDecodeError):
            json.loads(tool_calls[1])


if __name__ == "__main__":
    unittest.main()
