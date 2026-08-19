"""Regression test: the deepseek_dsml parser tolerates the truncated invoke
close tag (</｜DSML｜inv>) that DeepSeek-V4 quants systematically emit instead of
the canonical </｜DSML｜invoke>."""

import unittest

from mlx_lm.tool_parsers import deepseek_dsml


class TestDeepSeekDSMLCloseTag(unittest.TestCase):
    def test_truncated_invoke_close_tag(self):
        # Exact shape captured from 4/5/6-bit: close tag truncated to </｜DSML｜inv>.
        text = (
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="get_weather">\n'
            '<｜DSML｜parameter name="location" string="true">Paris, France</｜DSML｜parameter>\n'
            "</｜DSML｜inv>\n"
            "</｜DSML｜tool_calls>"
        )
        self.assertEqual(
            deepseek_dsml.parse_tool_call(text),
            {"name": "get_weather", "arguments": {"location": "Paris, France"}},
        )

    def test_wellformed_close_still_parses(self):
        text = (
            '<｜DSML｜invoke name="get_weather">\n'
            '<｜DSML｜parameter name="location" string="true">Paris</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>"
        )
        self.assertEqual(
            deepseek_dsml.parse_tool_call(text),
            {"name": "get_weather", "arguments": {"location": "Paris"}},
        )


if __name__ == "__main__":
    unittest.main()
