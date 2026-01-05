import unittest
from pathlib import Path

from mlx_lm.tool_parsers import (
    function_gemma,
    glm47,
    json_tools,
    minimax_m2,
    qwen3_coder,
)


class TestToolParsing(unittest.TestCase):

    def test_parsers(self):
        parsers = [function_gemma, glm47, json_tools, minimax_m2, qwen3_coder]

        test_cases = [
            "call:multiply{a:12234585,b:48838483920}",
            "multiply<arg_key>a</arg_key><arg_value>12234585</arg_value><arg_key>b</arg_key><arg_value>48838483920</arg_value>",
            '{"name": "multiply", "arguments": {"a": 12234585, "b": 48838483920}}',
            '<invoke name="multiply">\n<parameter name="a">12234585</parameter>\n<parameter name="b">48838483920</parameter>\n</invoke>',
            "<function=multiply>\n<parameter=a>\n12234585\n</parameter>\n<parameter=b>\n48838483920\n</parameter>\n</function>",
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

        for parser, test_case in zip(parsers, test_cases):
            with self.subTest(parser=parser):
                tool_call = parser.parse_tool_call(test_case, tools)
                expected = {
                    "name": "multiply",
                    "arguments": {"a": 12234585, "b": 48838483920},
                }
                self.assertEqual(tool_call, expected)

        test_cases = [
            "call:get_current_temperature{location:<escape>London<escape>}",
            'get_current_temperature<arg_key>location</arg_key><arg_value>"London"</arg_value>',
            '{"name": "get_current_temperature", "arguments": {"location": "London"}}',
            '<invoke name="get_current_temperature">\n<parameter name="location">London</parameter>\n</invoke>',
            "<function=get_current_temperature>\n<parameter=location>\nLondon\n</parameter>\n</function>",
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

        for parser, test_case in zip(parsers, test_cases):
            with self.subTest(parser=parser):
                tool_call = parser.parse_tool_call(test_case, tools)
                expected = {
                    "name": "get_current_temperature",
                    "arguments": {"location": "London"},
                }
                self.assertEqual(tool_call, expected)


if __name__ == "__main__":
    unittest.main()
