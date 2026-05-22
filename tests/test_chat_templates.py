import unittest

from mlx_lm.chat_templates.deepseek_v32 import apply_chat_template


class TestDeepSeekV32ChatTemplate(unittest.TestCase):

    def setUp(self):
        self.messages = [
            {"role": "user", "content": "Hello"},
        ]

    def test_enable_thinking_true(self):
        result = apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        self.assertIn("<think>", result)

    def test_enable_thinking_false(self):
        result = apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        self.assertNotIn("<think>", result)

    def test_thinking_mode_kwarg_still_works(self):
        result = apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            thinking_mode="chat",
        )
        self.assertNotIn("<think>", result)

    def test_thinking_mode_takes_precedence(self):
        result = apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            enable_thinking=True,
            thinking_mode="chat",
        )
        self.assertNotIn("<think>", result)

    def test_thinking_mode_overrides_contradictory_enable_thinking(self):
        result = apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            enable_thinking=False,
            thinking_mode="thinking",
        )
        self.assertIn("<think>", result)

    def test_default_is_thinking(self):
        result = apply_chat_template(
            self.messages,
            add_generation_prompt=True,
        )
        self.assertIn("<think>", result)


if __name__ == "__main__":
    unittest.main()
