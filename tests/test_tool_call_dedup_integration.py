"""Integration tests for consecutive duplicate tool call detection.

Tests go through the real HTTP pipeline by monkeypatching
ResponseGenerator.generate() to inject controlled token sequences.

Observed problem: Qwen3.5-9B generates 43-64 identical tool calls in a
single response until max_tokens. The server faithfully routes all of
them — wasting compute and confusing the agent loop.

Fix: ToolCallDedup in the generation loop compares each completed tool
call with the previous one. On consecutive duplicate, stop generation
with finish_reason=tool_calls.

See: https://github.com/ml-explore/mlx-lm/issues/613
"""

import http.server
import json
import threading
import unittest

import requests
from unittest.mock import MagicMock

from mlx_lm.server import (
    APIHandler,
    GenerationContext,
    LRUPromptCache,
    Response,
    ResponseGenerator,
)

# Token IDs
EOS_ID = 2
THINK_START_ID = 100
THINK_END_ID = 101

# String tokens
THINK_END_STR = "</think>"
TC_START_STR = "<tool_call>"
TC_END_STR = "</tool_call>"

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
            },
        },
    }
]


def _r(text, token=0, finish_reason=None):
    return Response(
        text=text, token=token, logprob=0.0,
        finish_reason=finish_reason, top_tokens=(),
    )


def _eos():
    return _r("", token=EOS_ID)


def _make_ctx(has_tool_calling=True, has_thinking=True):
    from mlx_lm.tool_parsers.json_tools import parse_tool_call

    return GenerationContext(
        has_tool_calling=has_tool_calling,
        tool_call_start=TC_START_STR,
        tool_call_end=TC_END_STR,
        tool_parser=parse_tool_call,
        has_thinking=has_thinking,
        think_start_id=THINK_START_ID,
        think_end_id=THINK_END_ID,
        think_end=THINK_END_STR,
        eos_token_ids={EOS_ID},
        stop_token_sequences=[],
        prompt=[1, 2, 3, THINK_START_ID],
    )


def _make_mock_provider():
    provider = MagicMock()
    provider.is_batchable = True
    provider.cli_args = type("obj", (object,), {
        "thinking_budget": None,
        "allowed_origins": ["*"],
        "decode_concurrency": 32,
        "prompt_concurrency": 8,
        "prefill_step_size": 2048,
        "prompt_cache_size": 10,
        "prompt_cache_bytes": 1 << 63,
        "prompt_cache_total_bytes": None,
        "temp": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "max_tokens": 512,
        "num_draft_tokens": 0,
        "adapter_path": None,
        "chat_template": None,
        "use_default_chat_template": False,
        "trust_remote_code": False,
        "draft_model": None,
        "chat_template_args": {},
        "model": None,
    })
    return provider


class _ServerFixture:
    def __init__(self):
        self.provider = _make_mock_provider()
        self.response_generator = ResponseGenerator.__new__(ResponseGenerator)
        self.response_generator.model_provider = self.provider
        self.response_generator.prompt_cache = LRUPromptCache()
        self.response_generator._stop = False

        self.httpd = http.server.HTTPServer(
            ("localhost", 0),
            lambda *args, **kwargs: APIHandler(
                self.response_generator, *args, **kwargs
            ),
        )
        self.port = self.httpd.server_port
        self._thread = threading.Thread(target=self.httpd.serve_forever)
        self._thread.daemon = True
        self._thread.start()

    def set_generate(self, ctx, responses):
        def mock_generate(request, args, progress_callback=None):
            return ctx, iter(responses)
        self.response_generator.generate = mock_generate

    def chat(self, tools=None, max_tokens=100, stream=False):
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools
        return requests.post(
            f"http://localhost:{self.port}/v1/chat/completions",
            json=payload,
            timeout=5,
            stream=stream,
        )

    def shutdown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self._thread.join(timeout=3)


_server = None


def setUpModule():
    global _server
    _server = _ServerFixture()


def tearDownModule():
    _server.shutdown()


def _make_repeated_tool_calls(n, unique_first=1):
    """Generate token sequence with n tool calls.

    First `unique_first` have distinct commands, rest are identical
    copies of the last unique one.
    """
    tokens = [_r("thinking\n"), _r(THINK_END_STR)]
    for i in range(n):
        cmd = f"cmd_{i}" if i < unique_first else f"cmd_{unique_first - 1}"
        tokens.extend([
            _r(TC_START_STR),
            _r(f'{{"name": "run_command", "arguments": {{"command": "{cmd}"}}}}'),
            _r(TC_END_STR),
        ])
    tokens.append(_eos())
    return tokens


class TestDuplicateToolCallStop(unittest.TestCase):
    """Consecutive duplicate tool calls trigger early stop."""

    def test_consecutive_duplicates_stopped(self):
        """20 identical calls → only 1 returned (2nd triggers stop)."""
        ctx = _make_ctx()
        _server.set_generate(ctx, _make_repeated_tool_calls(20, unique_first=1))

        body = _server.chat(tools=_TOOLS, max_tokens=4096).json()
        tc = body["choices"][0]["message"]["tool_calls"]

        self.assertEqual(len(tc), 1)
        self.assertEqual(body["choices"][0]["finish_reason"], "tool_calls")

    def test_unique_then_duplicates_keeps_unique(self):
        """1 unique + 10 identical → unique preserved, dupes stopped."""
        ctx = _make_ctx()
        tokens = [_r("plan\n"), _r(THINK_END_STR)]
        tokens.extend([
            _r(TC_START_STR),
            _r('{"name": "run_command", "arguments": {"command": "find /app"}}'),
            _r(TC_END_STR),
        ])
        for _ in range(10):
            tokens.extend([
                _r(TC_START_STR),
                _r('{"name": "run_command", "arguments": {"command": "ls"}}'),
                _r(TC_END_STR),
            ])
        tokens.append(_eos())

        _server.set_generate(ctx, tokens)
        body = _server.chat(tools=_TOOLS).json()
        tc = body["choices"][0]["message"]["tool_calls"]

        self.assertEqual(len(tc), 2)
        cmds = [json.loads(t["function"]["arguments"])["command"] for t in tc]
        self.assertEqual(cmds, ["find /app", "ls"])
        self.assertEqual(body["choices"][0]["finish_reason"], "tool_calls")

    def test_distinct_calls_not_affected(self):
        """All-different tool calls pass through unchanged."""
        ctx = _make_ctx()
        cmds = [
            "find /app -name '*.py'",
            "cat /app/validators.py",
            "grep -r URLValidator /app",
            "python -m pytest tests/",
        ]
        tokens = [_r("plan\n"), _r(THINK_END_STR)]
        for cmd in cmds:
            tokens.extend([
                _r(TC_START_STR),
                _r(f'{{"name": "run_command", "arguments": {{"command": "{cmd}"}}}}'),
                _r(TC_END_STR),
            ])
        tokens.append(_eos())

        _server.set_generate(ctx, tokens)
        body = _server.chat(tools=_TOOLS).json()
        tc = body["choices"][0]["message"]["tool_calls"]

        self.assertEqual(len(tc), 4)
        returned = [json.loads(t["function"]["arguments"])["command"] for t in tc]
        self.assertEqual(returned, cmds)

    def test_swe_bench_scale_43_duplicates(self):
        """Reproduce SWE-bench: 43 identical calls → early stop."""
        ctx = _make_ctx()
        _server.set_generate(ctx, _make_repeated_tool_calls(43, unique_first=1))

        body = _server.chat(tools=_TOOLS, max_tokens=4096).json()
        tc = body["choices"][0]["message"]["tool_calls"]

        self.assertEqual(len(tc), 1)
        self.assertEqual(body["choices"][0]["finish_reason"], "tool_calls")

    def test_streaming_duplicate_stop(self):
        """Streaming mode also stops on consecutive duplicates."""
        ctx = _make_ctx()
        _server.set_generate(ctx, _make_repeated_tool_calls(20, unique_first=1))

        resp = _server.chat(tools=_TOOLS, max_tokens=4096, stream=True)
        self.assertEqual(resp.status_code, 200)

        tool_call_count = 0
        last_finish_reason = None
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            delta = chunk["choices"][0]["delta"]
            fr = chunk["choices"][0].get("finish_reason")
            if fr:
                last_finish_reason = fr
            for tc in delta.get("tool_calls") or []:
                tool_call_count += 1

        self.assertLess(tool_call_count, 5)
        self.assertEqual(last_finish_reason, "tool_calls")

    def test_no_tools_unaffected(self):
        """Without tool calling, dedup has no effect."""
        ctx = _make_ctx(has_tool_calling=False)
        tokens = [
            _r("thinking\n"), _r(THINK_END_STR),
            _r("Hello world!"),
            _eos(),
        ]
        _server.set_generate(ctx, tokens)
        body = _server.chat().json()

        self.assertEqual(body["choices"][0]["finish_reason"], "stop")
        self.assertIn("Hello", body["choices"][0]["message"]["content"])


if __name__ == "__main__":
    unittest.main()
