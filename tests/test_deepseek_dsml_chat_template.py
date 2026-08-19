"""Tests for the DeepSeek-V4 DSML chat template.

The byte-exact tests compare ``mlx_lm.chat_templates.deepseek_dsml`` against the
reference encoder (``encoding_dsv4.py``) shipped in the DeepSeek-V4-Flash model
repo, when that repo is present in the local HF cache; otherwise they skip. The
invariant tests always run.
"""

import glob
import os
import sys
import unittest

from mlx_lm.chat_templates import deepseek_dsml as v4


def _load_reference():
    bases = [
        os.environ.get("HF_HOME"),
        os.path.expanduser("~/.cache/huggingface"),
        "/Volumes/scratch-00001/hf",
    ]
    for base in bases:
        if not base:
            continue
        hits = glob.glob(
            os.path.join(base, "hub/models--deepseek-ai--DeepSeek-V4-Flash*/snapshots/*/encoding")
        )
        if hits:
            sys.path.insert(0, hits[0])
            import encoding_dsv4  # type: ignore

            return encoding_dsv4
    return None


_REF = _load_reference()

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a specific location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"},
            },
            "required": ["location"],
        },
    },
}]


def _msgs(extra=()):
    return [
        {"role": "system", "content": "You are a helpful assistant.", "tools": TOOLS},
        {"role": "user", "content": "What is the weather in Paris?"},
        *extra,
    ]


@unittest.skipUnless(_REF, "DeepSeek-V4-Flash encoding/ reference not in HF cache")
class TestChatTemplateVsReference(unittest.TestCase):
    def test_single_turn_both_modes(self):
        for mode in ("thinking", "chat"):
            self.assertEqual(
                v4.encode_messages(_msgs(), thinking_mode=mode),
                _REF.encode_messages(_msgs(), thinking_mode=mode),
                mode,
            )

    def test_apply_chat_template_generation_prompt(self):
        self.assertEqual(
            v4.apply_chat_template(_msgs(), add_generation_prompt=True, thinking_mode="thinking"),
            _REF.encode_messages(_msgs(), thinking_mode="thinking"),
        )

    def test_assistant_tool_call_replay_and_result(self):
        extra = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'}}]},
            {"role": "tool", "content": "15C and sunny", "tool_call_id": "c1"},
        ]
        for mode in ("chat", "thinking"):
            self.assertEqual(
                v4.encode_messages(_msgs(extra), thinking_mode=mode),
                _REF.encode_messages(_msgs(extra), thinking_mode=mode),
                mode,
            )


def _msgs_tools_kwarg():
    # System message WITHOUT tools attached; tools go in as a top-level kwarg,
    # the way the mlx-lm server calls apply_chat_template.
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the weather in Paris?"},
    ]


class TestServerCallingConvention(unittest.TestCase):
    """The mlx-lm server calls ``apply_chat_template(messages, tools=...,
    enable_thinking=..., add_generation_prompt=True)``: tools arrive as a
    top-level kwarg, and ``TokenizerWrapper`` injects ``enable_thinking``. These
    guard the two integration bugs those conventions exposed (a jinja template
    would have silently accepted both)."""

    def test_tools_kwarg_matches_tools_on_system_message(self):
        via_kwarg = v4.apply_chat_template(
            _msgs_tools_kwarg(), tools=TOOLS, add_generation_prompt=True, enable_thinking=True
        )
        via_message = v4.apply_chat_template(
            _msgs(), add_generation_prompt=True, enable_thinking=True
        )
        self.assertEqual(via_kwarg, via_message)
        self.assertIn("<｜DSML｜tool_calls>", via_kwarg)

    def test_enable_thinking_selects_mode(self):
        thinking = v4.apply_chat_template(
            _msgs_tools_kwarg(), tools=TOOLS, add_generation_prompt=True, enable_thinking=True
        )
        chat = v4.apply_chat_template(
            _msgs_tools_kwarg(), tools=TOOLS, add_generation_prompt=True, enable_thinking=False
        )
        self.assertTrue(thinking.endswith("<｜Assistant｜><think>"))
        self.assertTrue(chat.endswith("<｜Assistant｜></think>"))

    def test_tools_kwarg_without_system_message_renders(self):
        # A request with tools but no system message must still render the tool
        # instructions (a synthesized system block), not silently drop them.
        out = v4.apply_chat_template(
            [{"role": "user", "content": "What is the weather in Paris?"}],
            tools=TOOLS,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        self.assertIn("<｜DSML｜tool_calls>", out)
        self.assertIn("get_weather", out)

    def test_tool_call_replay_with_dict_arguments(self):
        # Agent loops replay their prior tool_calls with `arguments` as a dict,
        # not the OpenAI JSON string; the template must accept both (it used to
        # crash in encode_arguments_to_dsml's json.loads).
        msgs = [
            {"role": "system", "content": "s", "tools": TOOLS},
            {"role": "user", "content": "weather in Paris?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {
                    "name": "get_weather", "arguments": {"location": "Paris"}}}]},
            {"role": "tool", "content": "sunny", "tool_call_id": "c1"},
        ]
        out = v4.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=True)
        self.assertIn("get_weather", out)
        self.assertIn("Paris", out)

    def test_tolerates_injected_unknown_kwargs(self):
        # A jinja template silently ignores unknown template kwargs; the Python
        # equivalent must too, since the server/tokenizer may inject extras.
        out = v4.apply_chat_template(
            _msgs_tools_kwarg(),
            tools=TOOLS,
            add_generation_prompt=True,
            enable_thinking=True,
            some_unrecognized_kwarg="ignored",
        )
        self.assertIn("<｜DSML｜tool_calls>", out)


class TestChatTemplateInvariants(unittest.TestCase):
    def test_bos_tool_block_and_thinking_cue(self):
        out = v4.encode_messages(_msgs(), thinking_mode="thinking")
        self.assertTrue(out.startswith("<｜begin▁of▁sentence｜>"))
        self.assertIn("<｜DSML｜tool_calls>", out)
        self.assertTrue(out.endswith("<｜Assistant｜><think>"))

    def test_chat_mode_closes_think(self):
        out = v4.encode_messages([{"role": "user", "content": "hi"}], thinking_mode="chat")
        self.assertTrue(out.endswith("<｜Assistant｜></think>"))


if __name__ == "__main__":
    unittest.main()
