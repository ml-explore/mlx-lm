import argparse
import unittest
from unittest.mock import MagicMock, patch

from rich.markdown import Markdown
from rich.panel import Panel

from mlx_lm.cli_ui import ChatUI, make_console


class TestCliUI(unittest.TestCase):

    def setUp(self):
        make_console.cache_clear()

    def tearDown(self):
        make_console.cache_clear()

    def test_make_console_forces_terminal_interactive_mode(self):
        console = make_console()

        self.assertTrue(console.is_terminal)
        self.assertTrue(console.is_interactive)

    @patch("mlx_lm.cli_ui.Live")
    def test_rank_zero_streams_with_live_markdown_buffer(self, mock_live):
        live = MagicMock()
        mock_live.return_value = live
        args = argparse.Namespace(window_size=20, refresh_rate=7, max_tokens=128)

        ui = ChatUI(args, rank=0)
        ui.stream_token("hello")
        ui.stream_token(" **world**")

        mock_live.assert_called_once()
        live.start.assert_called_once()
        self.assertEqual(live.update.call_count, 2)

        initial_panel = mock_live.call_args.args[0]
        self.assertIsInstance(initial_panel, Panel)
        self.assertIsInstance(initial_panel.renderable, Markdown)
        self.assertEqual(mock_live.call_args.kwargs["refresh_per_second"], 7)

        updated_panel = live.update.call_args.args[0]
        self.assertIsInstance(updated_panel, Panel)
        self.assertIsInstance(updated_panel.renderable, Markdown)
        self.assertEqual(ui._response_text, "hello **world**")

    @patch("mlx_lm.cli_ui.Live")
    def test_non_root_stream_does_not_start_live(self, mock_live):
        args = argparse.Namespace(window_size=20, refresh_rate=7, max_tokens=128)

        ui = ChatUI(args, rank=1)
        ui.stream_token("hidden")

        mock_live.assert_not_called()
        self.assertEqual(ui._response_text, "")


if __name__ == "__main__":
    unittest.main()
