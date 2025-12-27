# Copyright © 2025 Apple Inc.

"""Tests for tool call parsers."""

import unittest

from mlx_lm.tool_parsers import (
    MinimaxM2ToolParser,
    ToolCall,
    get_tool_parser,
)


class TestMinimaxM2ToolParser(unittest.TestCase):
    """Tests for the MiniMax M2 XML tool call parser."""

    def setUp(self):
        self.parser = MinimaxM2ToolParser()

    def test_tool_call_tokens(self):
        """Test that the correct start/end tokens are defined."""
        self.assertEqual(self.parser.tool_call_start, "<minimax:tool_call>")
        self.assertEqual(self.parser.tool_call_end, "</minimax:tool_call>")

    def test_parse_simple_tool_call(self):
        """Test parsing a simple MiniMax XML tool call."""
        tool_text = '''
        <invoke name="get_weather">
        <parameter name="location">Boston</parameter>
        </invoke>
        '''
        result = self.parser.parse_tool_calls(tool_text)

        self.assertTrue(result.tools_called)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].name, "get_weather")
        self.assertEqual(result.tool_calls[0].arguments, {"location": "Boston"})

    def test_parse_multiple_parameters(self):
        """Test parsing a tool call with multiple parameters."""
        tool_text = '''
        <invoke name="search_web">
        <parameter name="query">machine learning</parameter>
        <parameter name="max_results">10</parameter>
        <parameter name="include_images">true</parameter>
        </invoke>
        '''
        result = self.parser.parse_tool_calls(tool_text)

        self.assertTrue(result.tools_called)
        args = result.tool_calls[0].arguments
        self.assertEqual(args["query"], "machine learning")
        self.assertEqual(args["max_results"], 10)
        self.assertEqual(args["include_images"], True)

    def test_parse_multiple_invokes(self):
        """Test parsing multiple invoke blocks."""
        tool_text = '''
        <invoke name="func1">
        <parameter name="x">1</parameter>
        </invoke>
        <invoke name="func2">
        <parameter name="y">2</parameter>
        </invoke>
        '''
        result = self.parser.parse_tool_calls(tool_text)

        self.assertTrue(result.tools_called)
        self.assertEqual(len(result.tool_calls), 2)
        self.assertEqual(result.tool_calls[0].name, "func1")
        self.assertEqual(result.tool_calls[0].arguments, {"x": 1})
        self.assertEqual(result.tool_calls[1].name, "func2")
        self.assertEqual(result.tool_calls[1].arguments, {"y": 2})

    def test_parse_json_array_parameter(self):
        """Test parsing a parameter containing a JSON array."""
        tool_text = '''
        <invoke name="search">
        <parameter name="tags">["tech", "ai", "ml"]</parameter>
        </invoke>
        '''
        result = self.parser.parse_tool_calls(tool_text)

        self.assertTrue(result.tools_called)
        self.assertEqual(result.tool_calls[0].arguments["tags"], ["tech", "ai", "ml"])

    def test_parse_json_object_parameter(self):
        """Test parsing a parameter containing a JSON object."""
        tool_text = '''
        <invoke name="configure">
        <parameter name="settings">{"theme": "dark", "fontSize": 14}</parameter>
        </invoke>
        '''
        result = self.parser.parse_tool_calls(tool_text)

        self.assertTrue(result.tools_called)
        self.assertEqual(
            result.tool_calls[0].arguments["settings"],
            {"theme": "dark", "fontSize": 14}
        )

    def test_parse_empty_text(self):
        """Test parsing empty text."""
        result = self.parser.parse_tool_calls("")

        self.assertFalse(result.tools_called)
        self.assertEqual(len(result.tool_calls), 0)

    def test_parse_with_single_quotes(self):
        """Test parsing with single quotes in attribute."""
        tool_text = '''
        <invoke name='get_data'>
        <parameter name='id'>123</parameter>
        </invoke>
        '''
        result = self.parser.parse_tool_calls(tool_text)

        self.assertTrue(result.tools_called)
        self.assertEqual(result.tool_calls[0].name, "get_data")
        self.assertEqual(result.tool_calls[0].arguments["id"], 123)

    def test_to_openai_format(self):
        """Test conversion to OpenAI format."""
        tool_text = '''
        <invoke name="test_func">
        <parameter name="x">1</parameter>
        </invoke>
        '''
        result = self.parser.parse_tool_calls(tool_text)

        openai_format = result.tool_calls[0].to_openai_format()
        self.assertEqual(openai_format["type"], "function")
        self.assertEqual(openai_format["function"]["name"], "test_func")
        self.assertEqual(openai_format["function"]["arguments"], '{"x": 1}')

    def test_extract_from_full_response(self):
        """Test extracting tool calls from a full model response."""
        full_response = '''Let me search for that information.
<minimax:tool_call>
<invoke name="search_web">
<parameter name="query">MLX framework</parameter>
</invoke>
</minimax:tool_call>
I'll get back to you with the results.'''

        content, tool_calls = self.parser.extract_tool_calls_from_response(
            full_response
        )

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].name, "search_web")
        self.assertEqual(tool_calls[0].arguments, {"query": "MLX framework"})
        self.assertIn("Let me search", content)
        self.assertIn("get back to you", content)


class TestGetToolParser(unittest.TestCase):
    """Tests for the get_tool_parser factory function."""

    def test_get_minimax_parser(self):
        """Test getting the MiniMax parser by name."""
        parser = get_tool_parser("minimax_m2")
        self.assertIsInstance(parser, MinimaxM2ToolParser)

    def test_invalid_parser_name(self):
        """Test that invalid parser name raises ValueError."""
        with self.assertRaises(ValueError):
            get_tool_parser("invalid_parser")


class TestToolCall(unittest.TestCase):
    """Tests for the ToolCall dataclass."""

    def test_to_openai_format_basic(self):
        """Test basic OpenAI format conversion."""
        tc = ToolCall(name="test", arguments={"a": 1})
        result = tc.to_openai_format()

        self.assertEqual(result["type"], "function")
        self.assertEqual(result["function"]["name"], "test")
        self.assertEqual(result["function"]["arguments"], '{"a": 1}')

    def test_to_openai_format_with_id(self):
        """Test OpenAI format conversion with ID."""
        tc = ToolCall(name="test", arguments={}, id="call_123")
        result = tc.to_openai_format()

        self.assertEqual(result["id"], "call_123")

    def test_to_openai_format_complex_arguments(self):
        """Test OpenAI format with complex nested arguments."""
        tc = ToolCall(
            name="complex",
            arguments={
                "nested": {"deep": {"value": 42}},
                "array": [1, 2, 3],
                "mixed": [{"a": 1}, {"b": 2}]
            }
        )
        result = tc.to_openai_format()

        self.assertEqual(result["function"]["name"], "complex")
        # Arguments should be JSON string
        self.assertIn('"nested"', result["function"]["arguments"])
        self.assertIn('"array"', result["function"]["arguments"])


if __name__ == "__main__":
    unittest.main()
