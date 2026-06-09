import argparse
import unittest
from unittest.mock import MagicMock, patch

from mlx_lm.chat import broadcast_string, setup_arg_parser


class TestChat(unittest.TestCase):

    def test_setup_arg_parser_system_prompt(self):
        parser = setup_arg_parser()

        # Test default (no system prompt)
        args = parser.parse_args([])
        self.assertIsNone(args.system_prompt)

        # Test with system prompt
        args = parser.parse_args(["--system-prompt", "You are a helpful assistant."])
        self.assertEqual(args.system_prompt, "You are a helpful assistant.")

    def test_setup_arg_parser_all_args(self):
        parser = setup_arg_parser()
        args = parser.parse_args(
            [
                "--model",
                "test-model",
                "--adapter-path",
                "/path/to/adapter",
                "--temp",
                "0.7",
                "--top-p",
                "0.9",
                "--xtc-probability",
                "0.1",
                "--xtc-threshold",
                "0.1",
                "--seed",
                "42",
                "--max-kv-size",
                "1024",
                "--max-tokens",
                "512",
                "--system-prompt",
                "You are a helpful assistant.",
            ]
        )

        self.assertEqual(args.model, "test-model")
        self.assertEqual(args.adapter_path, "/path/to/adapter")
        self.assertEqual(args.temp, 0.7)
        self.assertEqual(args.top_p, 0.9)
        self.assertEqual(args.xtc_probability, 0.1)
        self.assertEqual(args.xtc_threshold, 0.1)
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.max_kv_size, 1024)
        self.assertEqual(args.max_tokens, 512)
        self.assertEqual(args.system_prompt, "You are a helpful assistant.")

    def test_broadcast_string_single_rank_returns_input(self):
        group = MagicMock()
        group.size.return_value = 1

        self.assertEqual(broadcast_string("hello", group), "hello")

    @patch("mlx_lm.chat.sharded_load")
    @patch("mlx_lm.chat.make_prompt_cache")
    @patch("mlx_lm.chat.stream_generate")
    @patch("mlx_lm.chat.broadcast_string")
    @patch("mlx_lm.chat.ChatUI")
    def test_system_prompt_in_messages(
        self,
        mock_chat_ui,
        mock_broadcast_string,
        mock_stream_generate,
        mock_make_prompt_cache,
        mock_sharded_load,
    ):
        from mlx_lm.chat import main

        group = MagicMock()
        group.rank.return_value = 0
        group.size.return_value = 2

        # Mock the model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "processed_prompt"
        mock_sharded_load.return_value = (mock_model, mock_tokenizer)

        # Mock prompt cache
        mock_prompt_cache = MagicMock()
        mock_make_prompt_cache.return_value = mock_prompt_cache

        ui = MagicMock()
        ui.prompt.side_effect = ["What is the weather?", "q"]
        mock_chat_ui.return_value.__enter__.return_value = ui

        # Mock stream_generate to return some responses
        mock_response = MagicMock()
        mock_response.text = "Hello there!"
        mock_response.generation_tokens = 1
        mock_response.generation_tps = 1.0
        mock_response.prompt_tps = 1.0
        mock_response.peak_memory = 1.0
        mock_stream_generate.return_value = [mock_response]
        mock_broadcast_string.side_effect = lambda query, *_args, **_kwargs: query

        # Test with system prompt
        with patch(
            "mlx_lm.chat.mx.distributed.init",
            return_value=group,
        ), patch(
            "sys.argv", ["chat.py", "--system-prompt", "You are a weather assistant."]
        ):
            try:
                main()
            except SystemExit:
                pass

        # Verify that apply_chat_template was called with system prompt
        mock_tokenizer.apply_chat_template.assert_called()
        call_args = mock_tokenizer.apply_chat_template.call_args[0][
            0
        ]  # First positional arg (messages)

        # Check that the messages contain both system and user messages
        self.assertEqual(len(call_args), 2)
        self.assertEqual(call_args[0]["role"], "system")
        self.assertEqual(call_args[0]["content"], "You are a weather assistant.")
        self.assertEqual(call_args[1]["role"], "user")
        self.assertEqual(call_args[1]["content"], "What is the weather?")
        mock_broadcast_string.assert_any_call("What is the weather?", group)

    @patch("mlx_lm.chat.sharded_load")
    @patch("mlx_lm.chat.make_prompt_cache")
    @patch("mlx_lm.chat.stream_generate")
    @patch("mlx_lm.chat.broadcast_string")
    @patch("mlx_lm.chat.ChatUI")
    def test_non_root_rank_uses_broadcast_without_prompt(
        self,
        mock_chat_ui,
        mock_broadcast_string,
        mock_stream_generate,
        mock_make_prompt_cache,
        mock_sharded_load,
    ):
        from mlx_lm.chat import main

        group = MagicMock()
        group.rank.return_value = 1
        group.size.return_value = 2

        mock_sharded_load.return_value = (MagicMock(), MagicMock())
        mock_make_prompt_cache.return_value = MagicMock()
        mock_broadcast_string.return_value = "q"

        ui = MagicMock()
        mock_chat_ui.return_value.__enter__.return_value = ui

        with patch("mlx_lm.chat.mx.distributed.init", return_value=group), patch(
            "sys.argv", ["chat.py"]
        ):
            try:
                main()
            except SystemExit:
                pass

        ui.prompt.assert_not_called()
        mock_broadcast_string.assert_called_once_with("", group)
        mock_stream_generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
