import unittest
from pathlib import Path

from mlx_lm.tool_parsers import function_gemma, json_tools, qwen3_coder


class TestToolParsing(unittest.TestCase):

    def test_json_parse(self):
        tool_text = (
            '{"name": "multiply", "arguments": {"a": 12234585, "b": 48838483920}}'
        )
        tool_call = json_tools.parse_tool_text(tool_text, None)
        expected = {
            "name": "multiply",
            "arguments": '{"a": 12234585, "b": 48838483920}',
        }
        self.assertEqual(tool_call, expected)

    def test_qwen3_coder_parse(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "multiply",
                    "description": "Multiply two numbers.",
                    "parameters": {
                        "type": "object",
                        "required": ["a", "b"],
                        "properties": {
                            "a": {"type": "number", "description": "a is a number"},
                            "b": {"type": "number", "description": "b is a number"},
                        },
                    },
                },
            }
        ]
        tool_text = r"""
        <function=multiply>
        <parameter=a>
        12234585
        </parameter>
        <parameter=b>
        48838483920
        </parameter>
        </function>"""
        tool_call = qwen3_coder.parse_tool_call(tool_text, tools)
        expected = {
            "name": "multiply",
            "arguments": '{"a": 12234585, "b": 48838483920}',
        }
        self.assertEqual(tool_call, expected)

    def test_function_gemma(self):
        text = "call:get_current_temperature{location:<escape>London<escape>}"
        tool_call = function_gemma.parse_tool_calls(text)
        expected = {
            "name": "get_current_temperature",
            "arguments": '{"location": "London"}',
        }
        self.assertEqual(tool_call, expected)


if __name__ == "__main__":
    unittest.main()
