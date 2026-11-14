# Copyright © 2026 Apple Inc.

import unittest
from unittest import mock

from mlx_lm.server_v2 import settings
from mlx_lm.server_v2.chat import rendering
from tests.server_v2 import helpers

PARAMS = settings.GenerationParams()


class TestRendering(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = helpers.load_tokenizer()
        cls.chat_tokenizer = helpers.load_chat_tokenizer()

    def test_from_messages(self):
        messages = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
        ]
        params = settings.GenerationParams(temperature=0.5)
        result = rendering.from_messages(
            self.tokenizer, messages, params=params, parse_tool_calls=False
        )
        self.assertEqual((result.params, result.parse_tool_calls), (params, False))
        expected = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
        self.assertEqual(result.tokens, tuple(expected))
        self.assertEqual(result.text_state, "normal")

        # The system prompt and the prompt without its last token.
        (system, kind), (user, _) = result.checkpoints
        self.assertEqual(kind, "system")
        self.assertIn("Be brief.", self.tokenizer.decode(result.tokens[:system]))
        self.assertNotIn("Hi", self.tokenizer.decode(result.tokens[:system]))
        self.assertEqual(user, len(result.tokens) - 1)
        # One token early: the next token is shared with any other user text.
        other = [messages[0], {"role": "user", "content": "Zebra"}]
        other = rendering.from_messages(self.tokenizer, other, params=PARAMS)
        self.assertEqual(other.tokens[: system + 1], result.tokens[: system + 1])

    def test_bad_chats(self):
        user = [{"role": "user", "content": "Hi"}]
        tools = [{"type": "function", "function": {"name": "f"}}]
        with self.assertRaisesRegex(ValueError, "tool call parser"):
            rendering.from_messages(self.tokenizer, user, tools=tools, params=PARAMS)
        # With tool_choice "none" the calls stay text, so no parser is needed.
        rendering.from_messages(
            self.tokenizer, user, tools=tools, params=PARAMS, parse_tool_calls=False
        )
        with self.assertRaisesRegex(ValueError, "no chat template"):
            rendering.from_messages(
                mock.Mock(has_chat_template=False), user, params=PARAMS
            )
        broken = mock.Mock(side_effect=TypeError("Injected"))
        with mock.patch.object(self.tokenizer, "apply_chat_template", broken):
            with self.assertRaisesRegex(ValueError, "template failed"):
                rendering.from_messages(self.tokenizer, user, params=PARAMS)

    def test_system_turns(self):
        line = "{{ m.role }}: {{ m.content }}\n"
        shows = "{% for m in messages %}" + line + "{% endfor %}"
        drops = "{% for m in messages if m.role != 'system' %}" + line + "{% endfor %}"
        # As in Gemma 3: a later system message breaks the user/assistant order.
        alternates = (
            "{% set first = 1 if messages[0].role == 'system' else 0 %}"
            "{% if first %}{{ messages[0].content }}{% endif %}"
            "{% for m in messages[first:] %}"
            "{% if (m.role == 'user') != (loop.index0 % 2 == 0) %}"
            "{{ raise_exception('Conversation roles must alternate.') }}"
            "{% endif %}" + line + "{% endfor %}"
        )
        # As in DeepSeek-R1-Distill.
        last_only = (
            "{% for m in messages if m.role == 'system' %}{% if loop.last %}"
            + line
            + "{% endif %}{% endfor %}"
            + drops
        )
        # As in Devstral: a user message cannot follow a tool result.
        no_user_after_tool = (
            "{% for m in messages %}"
            "{% if m.role == 'user' and loop.index0 and loop.previtem.role == 'tool' %}"
            "{{ raise_exception('Roles must alternate.') }}"
            "{% endif %}{% if m.role != 'system' %}" + line + "{% endif %}{% endfor %}"
        )
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "system", "content": "Be brief."},
            {"role": "assistant", "content": "Yo"},
        ]
        merged = [{"role": "user", "content": "Hi\n\nBe brief."}, messages[2]]
        top = {"role": "system", "content": "Sys"}
        user, system, _ = messages
        call = {"role": "assistant", "content": "Look."}
        tool = {"role": "tool", "content": "Sun"}
        after_tool = [user, call, tool, system]
        as_user = [user, call, tool, {**system, "role": "user"}]
        joined = [user, call, {**tool, "content": "Sun\n\nBe brief."}]
        for template, chat, expected in [
            (shows, messages, messages),
            (alternates, messages, merged),
            (last_only, [top, *messages], [top, *merged]),
            (drops, [user, user], [user, user]),
            (drops, after_tool, as_user),
            (no_user_after_tool, after_tool, joined),
        ]:
            with self.subTest(template=template):
                tokenizer = helpers.load_tokenizer(chat_template=template)
                result = rendering.from_messages(tokenizer, chat, params=PARAMS)
                tokens = tokenizer.apply_chat_template(
                    expected, add_generation_prompt=True
                )
                self.assertEqual(result.tokens, tuple(tokens))

    def test_from_messages_without_checkpoints(self):
        messages = [{"role": "user", "content": "Hi"}]
        apply_template = mock.Mock(side_effect=[[1, 2, 3], ValueError])
        with mock.patch.object(self.tokenizer, "apply_chat_template", apply_template):
            with self.assertLogs(level="WARNING"):
                result = rendering.from_messages(
                    self.tokenizer, messages, params=PARAMS
                )
        self.assertEqual((result.tokens, result.checkpoints), ((1, 2, 3), ()))

    def test_agent_step_checkpoint(self):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "", "reasoning_content": "Look it up."},
            {"role": "tool", "content": "20 C"},
        ]
        result = rendering.from_messages(self.chat_tokenizer, messages, params=PARAMS)
        # The next agent step shares the whole prompt, with its reasoning.
        self.assertIn(len(result.tokens) - 1, dict(result.checkpoints))
        result = rendering.from_messages(
            self.chat_tokenizer, [messages[0], messages[2]], params=PARAMS
        )
        self.assertTrue(result.checkpoints)

    def test_text_state(self):
        messages = [{"role": "user", "content": "Hi"}]
        cases = [({}, "normal"), ({"think": True}, "reasoning")]
        for kwargs, state in cases:
            result = rendering.from_messages(
                self.chat_tokenizer, messages, template_kwargs=kwargs, params=PARAMS
            )
            self.assertEqual(result.text_state, state)
        # A marker in the user text does not open a block.
        messages = [{"role": "user", "content": "<think> is a tag. What does it do?"}]
        result = rendering.from_messages(self.chat_tokenizer, messages, params=PARAMS)
        self.assertEqual(result.text_state, "normal")
        text = rendering.from_text(self.chat_tokenizer, "One, two,", params=PARAMS)
        self.assertEqual(text.tokens, tuple(self.chat_tokenizer.encode("One, two,")))
        with self.assertRaises(ValueError):
            rendering.from_text(self.chat_tokenizer, "\ud800", params=PARAMS)
        with self.assertRaisesRegex(ValueError, "no tokens"):
            rendering.from_text(self.chat_tokenizer, "", params=PARAMS)


if __name__ == "__main__":
    unittest.main()
