import unittest
from pathlib import Path

from mlx_lm.tokenizer_utils import _infer_tool_parser, _resolve_tool_parser_type
from mlx_lm.tool_parsers import (
    function_gemma,
    gemma4,
    glm47,
    json_tools,
    kimi_k2,
    longcat,
    minimax_m2,
    mistral,
    pythonic,
    qwen3_coder,
)

# Minimal chat-template fragments carrying the markers _infer_tool_parser keys on.
# Qwen3-Coder emits XML tool calls; Hermes-style emits JSON.
_QWEN3_CODER_TEMPLATE = "{% for tool in tools %}<tool_call>\n<function={{ tool.name }}>{% endfor %}"
_HERMES_JSON_TEMPLATE = '<tool_call>{"name": {{ tool_call.name }}}</tool_call>'


class TestToolParsing(unittest.TestCase):
    def test_parsers(self):
        test_cases = [
            ("call:multiply{a:12234585,b:48838483920}", function_gemma),
            ("call:multiply{a:12234585,b:48838483920}", gemma4),
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
            (
                'multiply[ARGS]{"a": 12234585, "b": 48838483920}',
                mistral,
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
                'call:get_current_temperature{location:<|"|>London<|"|>}',
                gemma4,
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
            (
                'get_current_temperature[ARGS]{"location": "London"}',
                mistral,
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

    def test_qwen3_coder_single_quoted_params(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filters": {"type": "object"},
                            "tags": {"type": "array"},
                        },
                    },
                },
            }
        ]

        # single-quoted dict (python-style, not valid JSON)
        test_case = (
            "<function=search>"
            "<parameter=filters>{'category': 'books', 'in_stock': True}</parameter>"
            "<parameter=tags>['fiction', 'new']</parameter>"
            "</function>"
        )
        tool_call = qwen3_coder.parse_tool_call(test_case, tools)
        self.assertEqual(tool_call["name"], "search")
        self.assertEqual(
            tool_call["arguments"]["filters"],
            {"category": "books", "in_stock": True},
        )
        self.assertEqual(tool_call["arguments"]["tags"], ["fiction", "new"])

        # valid JSON (double-quoted) should still work
        test_case = (
            "<function=search>"
            '<parameter=filters>{"category": "books"}</parameter>'
            '<parameter=tags>["fiction", "new"]</parameter>'
            "</function>"
        )
        tool_call = qwen3_coder.parse_tool_call(test_case, tools)
        self.assertEqual(tool_call["arguments"]["filters"], {"category": "books"})
        self.assertEqual(tool_call["arguments"]["tags"], ["fiction", "new"])

    def test_gemma4(self):
        # Nested object
        test_case = 'call:configure{settings:{enabled:true,name:<|"|>test<|"|>}}'
        tool_call = gemma4.parse_tool_call(test_case, None)
        self.assertEqual(tool_call["name"], "configure")
        self.assertEqual(
            tool_call["arguments"],
            {"settings": {"enabled": True, "name": "test"}},
        )

        # Array of strings
        test_case = 'call:tag{items:[<|"|>foo<|"|>,<|"|>bar<|"|>]}'
        tool_call = gemma4.parse_tool_call(test_case, None)
        self.assertEqual(tool_call["name"], "tag")
        self.assertEqual(tool_call["arguments"], {"items": ["foo", "bar"]})

        # Mixed types
        test_case = 'call:search{query:<|"|>hello world<|"|>,limit:10,verbose:false}'
        tool_call = gemma4.parse_tool_call(test_case, None)
        self.assertEqual(tool_call["name"], "search")
        self.assertEqual(
            tool_call["arguments"],
            {"query": "hello world", "limit": 10, "verbose": False},
        )

        # Multiple tool calls in a single block (no delimiter between them)
        test_case = (
            'call:glob{pattern:<|"|>README*.md<|"|>}'
            'call:glob{pattern:<|"|>CONTRIBUTING.md<|"|>}'
        )
        tool_calls = gemma4.parse_tool_call(test_case, None)
        self.assertIsInstance(tool_calls, list)
        self.assertEqual(len(tool_calls), 2)
        self.assertEqual(tool_calls[0]["name"], "glob")
        self.assertEqual(tool_calls[0]["arguments"], {"pattern": "README*.md"})
        self.assertEqual(tool_calls[1]["name"], "glob")
        self.assertEqual(tool_calls[1]["arguments"], {"pattern": "CONTRIBUTING.md"})

        # Multiple tool calls with nested args
        test_case = (
            'call:search{query:<|"|>weather<|"|>,limit:5}'
            'call:configure{settings:{enabled:true,name:<|"|>test<|"|>}}'
        )
        tool_calls = gemma4.parse_tool_call(test_case, None)
        self.assertIsInstance(tool_calls, list)
        self.assertEqual(len(tool_calls), 2)
        self.assertEqual(tool_calls[0]["name"], "search")
        self.assertEqual(
            tool_calls[0]["arguments"],
            {"query": "weather", "limit": 5},
        )
        self.assertEqual(tool_calls[1]["name"], "configure")
        self.assertEqual(
            tool_calls[1]["arguments"],
            {"settings": {"enabled": True, "name": "test"}},
        )

        # Hyphenated function name (e.g. manim-video)
        test_case = (
            'call:manim-video{mode:<|"|>plan<|"|>,prompt:<|"|>explain KV caching<|"|>}'
        )
        tool_call = gemma4.parse_tool_call(test_case, None)
        self.assertEqual(tool_call["name"], "manim-video")
        self.assertEqual(
            tool_call["arguments"],
            {"mode": "plan", "prompt": "explain KV caching"},
        )

        # Braces inside a string argument (e.g. code snippets or markdown in content)
        test_case = (
            'call:skill_manage{action:<|"|>create<|"|>,'
            'content:<|"|>use a dict like {key: value} in your code<|"|>}'
        )
        tool_call = gemma4.parse_tool_call(test_case, None)
        self.assertEqual(tool_call["name"], "skill_manage")
        self.assertEqual(tool_call["arguments"]["action"], "create")
        self.assertIn("{", tool_call["arguments"]["content"])

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

    def test_minimax_m2(self):
        test_case = (
            '<invoke name="search">\n'
            '<parameter name="query">weather</parameter>\n'
            "</invoke>\n"
            '<invoke name="read_file">\n'
            '<parameter name="path">/tmp/test.txt</parameter>\n'
            "</invoke>"
        )
        expected = [
            {"name": "search", "arguments": {"query": "weather"}},
            {"name": "read_file", "arguments": {"path": "/tmp/test.txt"}},
        ]
        tool_calls = minimax_m2.parse_tool_call(test_case, None)
        self.assertEqual(expected, tool_calls)


class TestToolParserSelection(unittest.TestCase):
    """Regression guard for parser selection precedence (tokenizer_utils).

    The Qwen3-Coder-Next repos ship a chat template that emits XML tool calls
    but mislabel `tool_parser_type` as the generic "json_tools", which sent
    the server into json.loads() on XML and silently dropped every tool call.
    """

    def test_template_inference(self):
        # The marker the XML grammar is detected by.
        self.assertEqual(_infer_tool_parser(_QWEN3_CODER_TEMPLATE), "qwen3_coder")
        self.assertEqual(_infer_tool_parser(_HERMES_JSON_TEMPLATE), "json_tools")
        self.assertIsNone(_infer_tool_parser("no tools here"))
        self.assertIsNone(_infer_tool_parser(None))

    def test_generic_label_yields_to_specific_template(self):
        # The actual bug: config says json_tools, template is XML -> use XML.
        self.assertEqual(
            _resolve_tool_parser_type("json_tools", _QWEN3_CODER_TEMPLATE),
            "qwen3_coder",
        )

    def test_missing_label_uses_inference(self):
        self.assertEqual(
            _resolve_tool_parser_type(None, _QWEN3_CODER_TEMPLATE), "qwen3_coder"
        )
        self.assertEqual(
            _resolve_tool_parser_type(None, _HERMES_JSON_TEMPLATE), "json_tools"
        )

    def test_hermes_json_unchanged(self):
        # A genuine Hermes/JSON model must be untouched: json_tools stays
        # json_tools (only a *generic* label yields, and only to a *specific*
        # template-inferred parser).
        self.assertEqual(
            _resolve_tool_parser_type("json_tools", _HERMES_JSON_TEMPLATE),
            "json_tools",
        )
        self.assertEqual(
            _resolve_tool_parser_type("json_tools", "no tool markers"),
            "json_tools",
        )

    def test_deliberate_specific_label_wins(self):
        # A specific, deliberately-set parser is never overridden by inference.
        self.assertEqual(
            _resolve_tool_parser_type("minimax_m2", _QWEN3_CODER_TEMPLATE),
            "minimax_m2",
        )
        self.assertEqual(
            _resolve_tool_parser_type("qwen3_coder", _HERMES_JSON_TEMPLATE),
            "qwen3_coder",
        )

    def test_no_parser_anywhere(self):
        self.assertIsNone(_resolve_tool_parser_type(None, "plain template"))


class TestQwen3CoderServerHandoff(unittest.TestCase):
    """Parse tool_text exactly as server.py's state machine hands it off:
    <tool_call>/</tool_call> stripped, with the leading/trailing newlines the
    model emits around the <function=...> block."""

    def test_server_framed_tool_text(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "limit": {"type": "integer"},
                            "verbose": {"type": "boolean"},
                        },
                    },
                },
            }
        ]
        tool_text = (
            "\n<function=read_file>\n"
            "<parameter=path>\n/tmp/x.txt\n</parameter>\n"
            "<parameter=limit>\n42\n</parameter>\n"
            "<parameter=verbose>\ntrue\n</parameter>\n"
            "</function>\n"
        )
        tool_call = qwen3_coder.parse_tool_call(tool_text, tools)
        self.assertEqual(tool_call["name"], "read_file")
        self.assertEqual(
            tool_call["arguments"],
            {"path": "/tmp/x.txt", "limit": 42, "verbose": True},
        )

    def test_no_schema_defaults_to_string(self):
        tool_text = "<function=ping>\n<parameter=host>\nlocalhost\n</parameter>\n</function>"
        tool_call = qwen3_coder.parse_tool_call(tool_text, None)
        self.assertEqual(
            tool_call, {"name": "ping", "arguments": {"host": "localhost"}}
        )


if __name__ == "__main__":
    unittest.main()
