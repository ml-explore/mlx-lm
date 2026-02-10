import unittest
from pathlib import Path

from mlx_lm.tool_parsers import (
    function_gemma,
    glm47,
    json_tools,
    kimi_k2,
    longcat,
    minimax_m2,
    pythonic,
    qwen3_coder,
)


class TestToolParsing(unittest.TestCase):

    def test_parsers(self):
        test_cases = [
            ("call:multiply{a:12234585,b:48838483920}", function_gemma),
            (
                '{"name": "multiply", "arguments": {"a": 12234585, "b": 48838483920}}',
                glm47,
            ),
            ("multiply a=12234585 b=48838483920", glm47),
            (
                "multiply<arg_key>a</arg_key><arg_value>12234585</arg_value><arg_key>b</arg_key><arg_value>48838483920</arg_value>",
                glm47,
            ),
            (
                '{"name": "multiply", "arguments": {"a": 12234585, "b": 48838483920}}',
                json_tools,
            ),
            (
                '<invoke name="multiply">\n<parameter name="a">12234585</parameter>\n<parameter name="b">48838483920</parameter>\n</invoke>',
                minimax_m2,
            ),
            (
                "<function=multiply>\n<parameter=a>\n12234585\n</parameter>\n<parameter=b>\n48838483920\n</parameter>\n</function>",
                qwen3_coder,
            ),
            (
                "multiply<longcat_arg_key>a</longcat_arg_key>\n<longcat_arg_value>12234585</longcat_arg_value>\n<longcat_arg_key>b</longcat_arg_key>\n<longcat_arg_value>48838483920</longcat_arg_value>",
                longcat,
            ),
            (
                '{"name": "multiply", "arguments": {"a": 12234585, "b": 48838483920}}',
                longcat,
            ),
            (
                "[multiply(a=12234585, b=48838483920)]",
                pythonic,
            ),
        ]

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

        for test_case, parser in test_cases:
            with self.subTest(parser=parser):
                tool_call = parser.parse_tool_call(test_case, tools)
                expected = {
                    "name": "multiply",
                    "arguments": {"a": 12234585, "b": 48838483920},
                }
                self.assertEqual(tool_call, expected)

        test_cases = [
            (
                "call:get_current_temperature{location:<escape>London<escape>}",
                function_gemma,
            ),
            (
                'get_current_temperature<arg_key>location</arg_key><arg_value>"London"</arg_value>',
                glm47,
            ),
            (
                '{"name": "get_current_temperature", "arguments": {"location": "London"}}',
                json_tools,
            ),
            (
                '<invoke name="get_current_temperature">\n<parameter name="location">London</parameter>\n</invoke>',
                minimax_m2,
            ),
            (
                "<function=get_current_temperature>\n<parameter=location>\nLondon\n</parameter>\n</function>",
                qwen3_coder,
            ),
            (
                "get_current_temperature<longcat_arg_key>location</longcat_arg_key>\n<longcat_arg_value>London</longcat_arg_value>",
                longcat,
            ),
            (
                '{"name": "get_current_temperature", "arguments": {"location": "London"}}',
                longcat,
            ),
            (
                '[get_current_temperature(location="London")]',
                pythonic,
            ),
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_temperature",
                    "description": "Get the current temperature.",
                    "parameters": {
                        "type": "object",
                        "required": ["location"],
                        "properties": {
                            "location": {"type": "str", "description": "The location."},
                        },
                    },
                },
            }
        ]

        for test_case, parser in test_cases:
            with self.subTest(parser=parser):
                tool_call = parser.parse_tool_call(test_case, tools)
                expected = {
                    "name": "get_current_temperature",
                    "arguments": {"location": "London"},
                }
                self.assertEqual(tool_call, expected)

    def test_kimi_k2(self):
        # Single tool call
        test_case = (
            "<|tool_call_begin|>functions.multiply:0<|tool_call_argument_begin|>"
            '{"a": 12234585, "b": 48838483920}<|tool_call_end|>'
        )
        tool_calls = kimi_k2.parse_tool_call(test_case, None)
        expected = [
            {
                "id": "functions.multiply:0",
                "name": "multiply",
                "arguments": {"a": 12234585, "b": 48838483920},
            }
        ]
        self.assertEqual(tool_calls, expected)

        # Multiple tool calls
        test_case = (
            "<|tool_call_begin|>functions.search:0<|tool_call_argument_begin|>"
            '{"query": "weather"}<|tool_call_end|>'
            "<|tool_call_begin|>functions.read_file:1<|tool_call_argument_begin|>"
            '{"path": "/tmp/test.txt"}<|tool_call_end|>'
        )
        tool_calls = kimi_k2.parse_tool_call(test_case, None)
        expected = [
            {
                "id": "functions.search:0",
                "name": "search",
                "arguments": {"query": "weather"},
            },
            {
                "id": "functions.read_file:1",
                "name": "read_file",
                "arguments": {"path": "/tmp/test.txt"},
            },
        ]
        self.assertEqual(tool_calls, expected)


if __name__ == "__main__":
    unittest.main()
