import unittest

from mlx_lm.tool_parsers import (
    function_gemma,
    gemma4,
    glm47,
    json_tools,
    kimi_k2,
    kimi_k3,
    longcat,
    minicpm5,
    minimax_m2,
    mistral,
    pythonic,
    qwen3_coder,
)


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
            (
                '<function name="multiply"><param name="a">12234585</param><param name="b">48838483920</param></function>',
                minicpm5,
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
            (
                '<function name="get_current_temperature"><param name="location">London</param></function>',
                minicpm5,
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

    def test_pythonic_single_quoted_args_with_commas(self):
        # LFM2.5 emits single-quoted strings; embedded commas must not truncate
        test_case = "[write(filePath='/tmp/hello.py', " "content='# Hello, world!')]"
        tool_call = pythonic.parse_tool_call(test_case, None)
        self.assertEqual(tool_call["name"], "write")
        self.assertEqual(tool_call["arguments"]["filePath"], "/tmp/hello.py")
        self.assertEqual(tool_call["arguments"]["content"], "# Hello, world!")

        # Double-quoted still works
        test_case = '[search(query="hello, world")]'
        tool_call = pythonic.parse_tool_call(test_case, None)
        self.assertEqual(tool_call["arguments"]["query"], "hello, world")

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

    def test_kimi_k3(self):
        # Typed per-key arguments
        test_case = (
            '<|open|>call tool="multiply" index="1"<|sep|>'
            '<|open|>argument key="a" type="number"<|sep|>12234585<|close|>argument<|sep|>'
            '<|open|>argument key="b" type="number"<|sep|>48838483920<|close|>argument<|sep|>'
            "<|close|>call<|sep|>"
        )
        tool_calls = kimi_k3.parse_tool_call(test_case, None)
        expected = [
            {"name": "multiply", "arguments": {"a": 12234585, "b": 48838483920}}
        ]
        self.assertEqual(tool_calls, expected)

        # String argument stays raw, other types decode as JSON
        test_case = (
            '<|open|>call tool="search" index="1"<|sep|>'
            '<|open|>argument key="query" type="string"<|sep|>{"not": "json"}<|close|>argument<|sep|>'
            '<|open|>argument key="limit" type="number"<|sep|>5<|close|>argument<|sep|>'
            '<|open|>argument key="safe" type="boolean"<|sep|>true<|close|>argument<|sep|>'
            '<|open|>argument key="filters" type="array"<|sep|>["a", "b"]<|close|>argument<|sep|>'
            "<|close|>call<|sep|>"
        )
        tool_calls = kimi_k3.parse_tool_call(test_case, None)
        expected = [
            {
                "name": "search",
                "arguments": {
                    "query": '{"not": "json"}',
                    "limit": 5,
                    "safe": True,
                    "filters": ["a", "b"],
                },
            }
        ]
        self.assertEqual(tool_calls, expected)

        # Raw JSON object block
        test_case = (
            '<|open|>call tool="get_weather" index="1"<|sep|>'
            '<|open|>json type="object"<|sep|>{"city": "Tokyo"}<|close|>json<|sep|>'
            "<|close|>call<|sep|>"
        )
        tool_calls = kimi_k3.parse_tool_call(test_case, None)
        expected = [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]
        self.assertEqual(tool_calls, expected)

        # Multiple calls in one tools section, escaped attribute values
        test_case = (
            '<|open|>call tool="say" index="1"<|sep|>'
            '<|open|>argument key="text" type="string"<|sep|>hi<|close|>argument<|sep|>'
            "<|close|>call<|sep|>"
            '<|open|>call tool="echo&amp;log" index="2"<|sep|>'
            "<|close|>call<|sep|>"
        )
        tool_calls = kimi_k3.parse_tool_call(test_case, None)
        expected = [
            {"name": "say", "arguments": {"text": "hi"}},
            {"name": "echo&log", "arguments": {}},
        ]
        self.assertEqual(tool_calls, expected)

        # Malformed call (missing tool name) does not discard valid siblings
        test_case = (
            '<|open|>call index="1"<|sep|>'
            "<|close|>call<|sep|>"
            '<|open|>call tool="say" index="2"<|sep|>'
            '<|open|>argument key="text" type="string"<|sep|>hi<|close|>argument<|sep|>'
            "<|close|>call<|sep|>"
        )
        tool_calls = kimi_k3.parse_tool_call(test_case, None)
        self.assertEqual(tool_calls, [{"name": "say", "arguments": {"text": "hi"}}])

        # Valid call followed by a call with a bad JSON block
        test_case = (
            '<|open|>call tool="say" index="1"<|sep|>'
            '<|open|>argument key="text" type="string"<|sep|>hi<|close|>argument<|sep|>'
            "<|close|>call<|sep|>"
            '<|open|>call tool="broken" index="2"<|sep|>'
            '<|open|>json type="object"<|sep|>{not json<|close|>json<|sep|>'
            "<|close|>call<|sep|>"
        )
        tool_calls = kimi_k3.parse_tool_call(test_case, None)
        self.assertEqual(tool_calls, [{"name": "say", "arguments": {"text": "hi"}}])

        # Truncated trailing call after a complete one
        test_case = (
            '<|open|>call tool="say" index="1"<|sep|>'
            '<|open|>argument key="text" type="string"<|sep|>hi<|close|>argument<|sep|>'
            "<|close|>call<|sep|>"
            '<|open|>call tool="cut" index="2"<|sep|>'
            '<|open|>argument key="x" type="num'
        )
        tool_calls = kimi_k3.parse_tool_call(test_case, None)
        self.assertEqual(tool_calls, [{"name": "say", "arguments": {"text": "hi"}}])

        # Nothing parseable still raises
        with self.assertRaises(ValueError):
            kimi_k3.parse_tool_call(
                '<|open|>call index="1"<|sep|><|close|>call<|sep|>', None
            )

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

    def test_minicpm5(self):
        # Multiple tool calls
        test_case = (
            '<function name="search"><param name="query">weather</param></function>'
            '<function name="read_file"><param name="path">/tmp/test.txt</param></function>'
        )
        tool_calls = minicpm5.parse_tool_call(test_case, None)
        self.assertIsInstance(tool_calls, list)
        self.assertEqual(len(tool_calls), 2)
        self.assertEqual(tool_calls[0]["name"], "search")
        self.assertEqual(tool_calls[0]["arguments"], {"query": "weather"})
        self.assertEqual(tool_calls[1]["name"], "read_file")
        self.assertEqual(tool_calls[1]["arguments"], {"path": "/tmp/test.txt"})

        # Numeric argument (not string type from schema)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "multiply",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                        },
                    },
                },
            }
        ]
        test_case = '<function name="multiply"><param name="a">12234585</param><param name="b">48838483920</param></function>'
        tool_call = minicpm5.parse_tool_call(test_case, tools)
        self.assertEqual(tool_call["name"], "multiply")
        self.assertEqual(tool_call["arguments"]["a"], 12234585)
        self.assertEqual(tool_call["arguments"]["b"], 48838483920)

        # CDATA block for string with special chars
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "write",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                        },
                    },
                },
            }
        ]
        test_case = '<function name="write"><param name="content"><![CDATA[hello & world <foo>]]></param></function>'
        tool_call = minicpm5.parse_tool_call(test_case, tools)
        self.assertEqual(tool_call["name"], "write")
        self.assertEqual(tool_call["arguments"]["content"], "hello & world <foo>")

        # Empty arguments
        test_case = '<function name="ping"></function>'
        tool_call = minicpm5.parse_tool_call(test_case, None)
        self.assertEqual(tool_call["name"], "ping")
        self.assertEqual(tool_call["arguments"], {})

    @staticmethod
    def _weather_tools():
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "date": {"type": "string"},
                        },
                        "required": ["city"],
                    },
                },
            }
        ]

    def _run_state_machine(self, model_output, chunk_size=3):
        """Replicate the server's tool-call collection path (server.py).

        The text state machine strips the ``<function`` / ``</function>``
        markers and the payload between them is collected while in the "tool"
        state, exactly as done by the OpenAI-compatible server. This exercises
        the real ``mlx_lm.generate`` machinery rather than only calling
        ``parse_tool_call`` directly.
        """
        from mlx_lm.generate import TextStateMachine, make_text_state_machine

        class _Tokenizer:
            has_thinking = False
            has_tool_calling = True
            tool_call_start = "<function"
            tool_call_end = "</function>"
            structural_markers = ()

        sm = make_text_state_machine(_Tokenizer())
        state = sm.make_state("normal")
        tool_text = ""
        tool_calls = []
        prev_state = "normal"
        text = ""
        for i in range(0, len(model_output), chunk_size):
            state, clean_text, current_state = TextStateMachine.step(
                state, model_output[i : i + chunk_size]
            )
            if current_state == "tool":
                tool_text += clean_text
            elif current_state == "normal":
                if prev_state == "tool":
                    tool_calls.append(tool_text)
                    tool_text = ""
                text += clean_text
            prev_state = current_state

        # Trailing tool text (e.g. truncated at finish_reason="length").
        if prev_state == "tool" and tool_text:
            tool_calls.append(tool_text)
        return tool_calls, text

    def test_minicpm5_payload_format(self):
        """parse_tool_call consumes the state-machine payload directly.

        The tokenizer text state machine strips the <function/</function>
        markers, so callers pass only the payload between them. This is the
        format the parser must accept — not only complete XML blocks.
        """
        payload = 'name="get_weather"><param name="city">Tokyo</param>'
        tool_call = minicpm5.parse_tool_call(payload, self._weather_tools())
        self.assertEqual(
            tool_call, {"name": "get_weather", "arguments": {"city": "Tokyo"}}
        )

    def test_minicpm5_integration_state_machine(self):
        """Single call with text before/after through the real state machine."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ]
        model_output = (
            "Sure, I'll look that up.\n"
            '<function name="search"><param name="query">weather</param></function>\n'
            "It looks sunny."
        )
        tool_calls, text = self._run_state_machine(model_output)
        self.assertEqual(len(tool_calls), 1)
        self.assertIn("Sure, I'll look that up.", text)
        self.assertIn("It looks sunny.", text)
        parsed = minicpm5.parse_tool_call(tool_calls[0], tools)
        self.assertEqual(parsed, {"name": "search", "arguments": {"query": "weather"}})

    def test_minicpm5_integration_multiple_calls(self):
        """Multiple calls split by the state machine each parse correctly."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
        ]
        model_output = (
            '<function name="search"><param name="query">weather</param></function>'
            '<function name="read_file"><param name="path">/tmp/test.txt</param></function>'
        )
        tool_calls, _ = self._run_state_machine(model_output)
        self.assertEqual(len(tool_calls), 2)
        parsed = [minicpm5.parse_tool_call(tc, tools) for tc in tool_calls]
        self.assertEqual(
            parsed,
            [
                {"name": "search", "arguments": {"query": "weather"}},
                {"name": "read_file", "arguments": {"path": "/tmp/test.txt"}},
            ],
        )

    def test_minicpm5_integration_truncated(self):
        """A call cut off at finish_reason='length' (trailing tool text).

        With a schema the truncated call is rejected; without one the function
        name is salvaged and incomplete parameters are not fabricated.
        """
        model_output = '<function name="get_weather"><param name="city">To'
        tool_calls, _ = self._run_state_machine(model_output)
        self.assertEqual(len(tool_calls), 1)
        with self.assertRaises(ValueError):
            minicpm5.parse_tool_call(tool_calls[0], self._weather_tools())
        parsed = minicpm5.parse_tool_call(tool_calls[0], None)
        self.assertEqual(parsed["name"], "get_weather")
        self.assertEqual(parsed["arguments"], {})

    def test_minicpm5_typed_parameters(self):
        """Schema-driven deserialization for typed params (review #5)."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "book",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "count": {"type": "integer"},
                            "price": {"type": "number"},
                            "in_stock": {"type": "boolean"},
                            "tags": {"type": "array"},
                            "metadata": {"type": "object"},
                            "note": {},
                            "empty": {"type": "null"},
                        },
                        "required": ["title"],
                    },
                },
            }
        ]
        test_case = (
            '<function name="book">'
            '<param name="title">The Hobbit &amp; friends</param>'
            '<param name="count">2</param>'
            '<param name="price">19.99</param>'
            '<param name="in_stock">true</param>'
            '<param name="tags">["fiction", "classic"]</param>'
            '<param name="metadata">{"pages": 310}</param>'
            '<param name="note">{"looks": "like json"}</param>'
            '<param name="empty">null</param>'
            "</function>"
        )
        arguments = minicpm5.parse_tool_call(test_case, tools)["arguments"]
        self.assertEqual(arguments["title"], "The Hobbit & friends")
        self.assertEqual(arguments["count"], 2)
        self.assertEqual(arguments["price"], 19.99)
        self.assertIs(arguments["in_stock"], True)
        self.assertEqual(arguments["tags"], ["fiction", "classic"])
        self.assertEqual(arguments["metadata"], {"pages": 310})
        # No declared type: preserved as a string, JSON-looking or not.
        self.assertEqual(arguments["note"], '{"looks": "like json"}')
        self.assertIsNone(arguments["empty"])

    def test_minicpm5_integer_accepts_whole_number_floats(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "book",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "count": {"type": "integer"},
                        },
                        "required": ["title"],
                    },
                },
            }
        ]
        tool_call = minicpm5.parse_tool_call(
            '<function name="book"><param name="title">x</param>'
            '<param name="count">1.0</param></function>',
            tools,
        )
        self.assertEqual(tool_call["arguments"]["count"], 1)

    def test_minicpm5_malformed_typed_value_rejected(self):
        """A value that does not match its declared type is rejected (review #5)."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "book",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "count": {"type": "integer"},
                        },
                        "required": ["title"],
                    },
                },
            }
        ]
        with self.assertRaises(ValueError):
            minicpm5.parse_tool_call(
                '<function name="book"><param name="title">x</param>'
                '<param name="count">abc</param></function>',
                tools,
            )

    def test_minicpm5_strings_preserved_without_schema(self):
        """Without a schema, values are preserved as strings (review #4).

        Numeric- or JSON-looking text must not be silently coerced when the
        parser has no evidence about the parameter type.
        """
        test_case = (
            '<function name="echo">'
            '<param name="message">123</param>'
            '<param name="code">00123</param>'
            '<param name="payload">{"a": 1}</param>'
            "</function>"
        )
        tool_call = minicpm5.parse_tool_call(test_case, None)
        self.assertEqual(
            tool_call["arguments"],
            {"message": "123", "code": "00123", "payload": '{"a": 1}'},
        )

    def test_minicpm5_string_type_keeps_json_looking_value(self):
        """A declared string parameter keeps JSON-looking text as a string."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "send",
                    "parameters": {
                        "type": "object",
                        "properties": {"message": {"type": "string"}},
                        "required": ["message"],
                    },
                },
            }
        ]
        test_case = (
            '<function name="send">'
            '<param name="message">{"foo": "bar"}</param>'
            "</function>"
        )
        tool_call = minicpm5.parse_tool_call(test_case, tools)
        self.assertEqual(tool_call["arguments"]["message"], '{"foo": "bar"}')

    def test_minicpm5_unknown_function_rejected(self):
        """Calls to functions outside the requested tool schema are rejected."""
        with self.assertRaises(ValueError):
            minicpm5.parse_tool_call(
                '<function name="not_a_requested_tool"><param name="x">1</param></function>',
                self._weather_tools(),
            )

    def test_minicpm5_unknown_parameter_dropped(self):
        """Parameters absent from the schema are dropped (vLLM semantics)."""
        test_case = (
            '<function name="get_weather">'
            '<param name="city">Tokyo</param>'
            '<param name="made_up_argument">foo</param>'
            "</function>"
        )
        tool_call = minicpm5.parse_tool_call(test_case, self._weather_tools())
        self.assertEqual(tool_call["arguments"], {"city": "Tokyo"})

    def test_minicpm5_unknown_parameters_do_not_satisfy_required(self):
        """Dropped unknown params cannot satisfy required params."""
        with self.assertRaises(ValueError):
            minicpm5.parse_tool_call(
                '<function name="get_weather"><param name="made_up_argument">foo</param></function>',
                self._weather_tools(),
            )

    def test_minicpm5_missing_required_parameter_rejected(self):
        with self.assertRaises(ValueError):
            minicpm5.parse_tool_call(
                '<function name="get_weather"><param name="date">2024-06-27</param></function>',
                self._weather_tools(),
            )

    def test_minicpm5_duplicate_parameter_rejected(self):
        with self.assertRaises(ValueError):
            minicpm5.parse_tool_call(
                '<function name="get_weather">'
                '<param name="city">Tokyo</param>'
                '<param name="city">Osaka</param>'
                "</function>",
                self._weather_tools(),
            )

    def test_minicpm5_tokenizer_space_variants(self):
        """SentencePiece/GPT decoders may emit \u0120/\u010a (review #9)."""
        test_case = (
            '<function\u0120name="get_weather">'
            '<param\u0120name="city">Tokyo</param>'
            "</function>"
        )
        tool_call = minicpm5.parse_tool_call(test_case, self._weather_tools())
        self.assertEqual(
            tool_call, {"name": "get_weather", "arguments": {"city": "Tokyo"}}
        )
        test_case = (
            '<function name="get_weather">\u010a'
            '<param name="city">Tokyo</param>\u010a'
            "</function>"
        )
        tool_call = minicpm5.parse_tool_call(test_case, self._weather_tools())
        self.assertEqual(tool_call["arguments"], {"city": "Tokyo"})

    def test_minicpm5_collapsed_tags(self):
        """Model output that collapses tag names/attributes is normalized."""
        test_case = (
            '<functionname="get_weather">'
            '<paramname="city">Tokyo</param>'
            "</function>"
        )
        tool_call = minicpm5.parse_tool_call(test_case, self._weather_tools())
        self.assertEqual(
            tool_call, {"name": "get_weather", "arguments": {"city": "Tokyo"}}
        )

    def test_minicpm5_cdata_whitespace_preserved(self):
        """CDATA content is taken verbatim so whitespace survives (review #8)."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "write",
                    "parameters": {
                        "type": "object",
                        "properties": {"content": {"type": "string"}},
                        "required": ["content"],
                    },
                },
            }
        ]
        test_case = (
            '<function name="write">'
            '<param name="content"><![CDATA[    def foo():\n        pass\n]]></param>'
            "</function>"
        )
        tool_call = minicpm5.parse_tool_call(test_case, tools)
        self.assertEqual(
            tool_call["arguments"]["content"], "    def foo():\n        pass\n"
        )

    def test_minicpm5_not_a_tool_call_raises(self):
        with self.assertRaises(ValueError):
            minicpm5.parse_tool_call("random text that is not a tool call", None)

    def test_minicpm5_autodetection(self):
        """Auto-detection requires a MiniCPM5-specific template marker."""
        from mlx_lm.tokenizer_utils import _infer_tool_parser

        class _Tokenizer:
            def __init__(self, chat_template):
                self.chat_template = chat_template

            def get_vocab(self):
                return {}

        # The real MiniCPM5 chat template carries this system-prompt sentence
        # together with the <function ...> XML tool-call markers.
        tokenizer = _Tokenizer(
            "{%- if tools %}# Tools\n\n"
            "You are provided with function signatures within <tools></tools> "
            "XML tags:\n<tools>...<function name=\"' ~ tool_call.name ~ '\">"
            "...</function>..."
        )
        self.assertEqual(_infer_tool_parser(tokenizer), "minicpm5")

        # A template that merely contains <function name= must NOT match.
        tokenizer = _Tokenizer(
            "{% if not tools %}<function name='x'></function>{% endif %}"
        )
        self.assertNotEqual(_infer_tool_parser(tokenizer), "minicpm5")

    def test_qwen3_coder_iso_date(self):
        """Qwen3 coder parser should not crash on ISO 8601 dates."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "schedule",
                    "description": "Schedule a task",
                    "parameters": {
                        "type": "object",
                        "required": ["name", "deadline"],
                        "properties": {
                            "name": {"type": "string"},
                            "deadline": {"type": "string"},
                        },
                    },
                },
            }
        ]
        test_case = (
            "<function=schedule>\n"
            "<parameter=name>\n"
            "deploy\n"
            "</parameter>\n"
            "<parameter=deadline>\n"
            "2025-06-15T10:30:00Z\n"
            "</parameter>\n"
            "</function>"
        )
        tool_calls = qwen3_coder.parse_tool_call(test_case, tools)
        # parse_tool_call returns dict, not list
        self.assertEqual(tool_calls["name"], "schedule")
        self.assertEqual(tool_calls["arguments"]["name"], "deploy")
        self.assertEqual(tool_calls["arguments"]["deadline"], "2025-06-15T10:30:00Z")

    def test_qwen3_coder_partial_number(self):
        """Qwen3 coder parser should handle partial number-like strings."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "log",
                    "description": "Log a message",
                    "parameters": {
                        "type": "object",
                        "required": ["msg"],
                        "properties": {
                            "msg": {"type": "string"},
                        },
                    },
                },
            }
        ]
        test_case = (
            "<function=log>\n"
            "<parameter=msg>\n"
            "version 3.10.5-beta\n"
            "</parameter>\n"
            "</function>"
        )
        tool_calls = qwen3_coder.parse_tool_call(test_case, tools)
        # parse_tool_call returns dict, not list
        self.assertEqual(tool_calls["arguments"]["msg"], "version 3.10.5-beta")

    def test_qwen3_coder_missing_function_tag_close(self):
        """Recover the function name when the model drops the ">" after it."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "Get the current time",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        # Missing ">" after the name plus an orphan "</parameter>"
        test_case = "<function=get_current_time\n</parameter>\n</function>"
        tool_call = qwen3_coder.parse_tool_call(test_case, tools)
        self.assertEqual(tool_call["name"], "get_current_time")
        self.assertEqual(tool_call["arguments"], {})


if __name__ == "__main__":
    unittest.main()
