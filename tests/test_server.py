# Copyright © 2024 Apple Inc.

import http
import io
import json
import threading
import time
import unittest
from queue import Queue
from types import SimpleNamespace
from unittest import mock

import mlx.core as mx
import requests

from mlx_lm.generate import TextStateMachine
from mlx_lm.models.cache import KVCache, QuantizedKVCache
from mlx_lm.server import (
    APIHandler,
    GenerationContext,
    LRUPromptCache,
    Response,
    ResponseGenerator,
    SamplingArguments,
    ToolCallFormatter,
    _apply_stop_words,
    _make_sampler,
)
from mlx_lm.tool_parsers import json_tools, pythonic
from mlx_lm.utils import load


class DummyModelProvider:
    def __init__(self, with_draft=False, kv_bits=None, quantized_kv_start=0):
        HF_MODEL_PATH = "mlx-community/Qwen1.5-0.5B-Chat-4bit"
        self.model, self.tokenizer = load(HF_MODEL_PATH)
        self.processor = None
        self.model_key = (HF_MODEL_PATH, None)
        self.is_batchable = True

        # Add draft model support
        self.draft_model = None
        self.draft_model_key = None
        self.cli_args = type(
            "obj",
            (object,),
            {
                "adapter_path": None,
                "chat_template": None,
                "use_default_chat_template": False,
                "trust_remote_code": False,
                "draft_model": None,
                "num_draft_tokens": 3,
                "temp": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "min_p": 0.0,
                "max_tokens": 512,
                "chat_template_args": {},
                "model": None,
                "decode_concurrency": 32,
                "prompt_concurrency": 8,
                "prefill_step_size": 2048,
                "prompt_cache_size": 10,
                "prompt_cache_bytes": 1 << 63,
                "prompt_cache_total_bytes": None,
                "allowed_origins": ["*"],
                "kv_bits": kv_bits,
                "kv_group_size": 64,
                "quantized_kv_start": quantized_kv_start,
            },
        )

        if with_draft:
            # Use the same model as the draft model for testing
            self.draft_model, _ = load(HF_MODEL_PATH)
            self.draft_model_key = HF_MODEL_PATH
            self.cli_args.draft_model = HF_MODEL_PATH

    def load(self, model, adapter=None, draft_model=None):
        assert model in ["default_model", "chat_model"]
        return self.model, self.tokenizer

    def load_default(self):
        return self.load("default_model", None, "default_model")

    def reset(self) -> None:
        self.model_key = None
        self.model = None
        self.tokenizer = None
        self.draft_model = None
        self.is_batchable = False


class MockCache:
    def __init__(self, value, is_trimmable: bool = True):
        self.value = value
        self._is_trimmable = is_trimmable

    @property
    def nbytes(self):
        return len(self.value)

    def __eq__(self, other):
        return other.value == self.value

    def is_trimmable(self):
        return self._is_trimmable

    def trim(self, n):
        assert self._is_trimmable
        return n


TOOL_CALLS = {
    "normal": [("<tool_call>", "tool")],
    "tool": [("</tool_call>", "normal")],
}


class TestTextStateMachine(unittest.TestCase):
    """Test the TextStateMachine buffering and stripping behavior."""

    def test_strips_control_sequences(self):
        reader = TextStateMachine(TOOL_CALLS).reader()
        self.assertEqual(
            reader.feed("hi <tool_call>a</tool_call><tool_call>b</tool_call> bye"),
            TextStateMachine.Delta(content="hi  bye", tool_calls=("a", "b")),
        )

    def test_finish_releases_held_text(self):
        # "</" can start "</tool_call>".
        reader = TextStateMachine(TOOL_CALLS).reader()
        reader.feed("<tool_call>body</")
        self.assertEqual(
            reader.finish(), TextStateMachine.Delta(tool_calls=("body</",))
        )

    def test_empty_tool_calls_are_dropped(self):
        reader = TextStateMachine(TOOL_CALLS).reader()
        self.assertEqual(
            reader.feed("a<tool_call></tool_call>b<tool_call>\n"),
            TextStateMachine.Delta(content="ab"),
        )
        self.assertEqual(reader.finish(), TextStateMachine.Delta())

    def test_holds_only_what_can_start_a_marker(self):
        reader = TextStateMachine(TOOL_CALLS).reader()
        self.assertEqual(reader.feed("a <<"), TextStateMachine.Delta(content="a <"))

    def test_held_text_is_read_once(self):
        # A marker split over two chunks.
        reader = TextStateMachine({"normal": [("##", "normal")]}).reader()
        self.assertEqual(reader.feed("a #"), TextStateMachine.Delta(content="a "))
        self.assertEqual(reader.feed("# b"), TextStateMachine.Delta(content=" b"))

    def test_held_text_keeps_its_state(self):
        sm = TextStateMachine({**TOOL_CALLS, "reasoning": [("</think>", "normal")]})
        reader = sm.reader(state="reasoning")
        self.assertEqual(
            reader.feed("hmm</think>Hi <"),
            TextStateMachine.Delta(content="Hi ", reasoning="hmm"),
        )
        self.assertEqual(
            reader.feed("<tool_call>{}"), TextStateMachine.Delta(content="<")
        )

    def test_tool_call_inside_reasoning(self):
        tokenizer = SimpleNamespace(
            has_thinking=True,
            think_start="<think>",
            think_end="</think>",
            has_tool_calling=True,
            tool_call_start="<tool_call>",
            tool_call_end="</tool_call>",
            structural_markers=(),
            response_transitions={},
        )
        reader = TextStateMachine.from_tokenizer(tokenizer).reader(state="reasoning")
        self.assertEqual(
            reader.feed("a<tool_call>b</tool_call>c</think>d"),
            TextStateMachine.Delta(content="d", reasoning="ac", tool_calls=("b",)),
        )

    def test_tool_call_without_end_marker(self):
        tokenizer = SimpleNamespace(
            has_thinking=False,
            has_tool_calling=True,
            tool_call_start="[TOOL_CALLS]",
            tool_call_end="",
            structural_markers=(),
            response_transitions={},
        )
        reader = TextStateMachine.from_tokenizer(tokenizer).reader()
        self.assertEqual(
            reader.feed("hello[TOOL_CALLS]body"),
            TextStateMachine.Delta(content="hello"),
        )
        self.assertEqual(reader.finish(), TextStateMachine.Delta(tool_calls=("body",)))

    def test_deltas_add(self):
        a = TextStateMachine.Delta(content="a", tool_calls=("x",))
        b = TextStateMachine.Delta(reasoning="r", tool_calls=("y",))
        self.assertEqual(
            a + b,
            TextStateMachine.Delta(content="a", reasoning="r", tool_calls=("x", "y")),
        )

    def test_unknown_state_raises(self):
        with self.assertRaises(ValueError):
            TextStateMachine({"normal": [("<x>", "other")]})
        with self.assertRaises(ValueError):
            TextStateMachine({}).reader(state="other")


class TestApplyStopWords(unittest.TestCase):
    def test_stop_words(self):
        sm = TextStateMachine.from_stop_words(["\n\n"])
        # The last chunk and its finish reason after "a\n".
        cases = [
            (("\nb", None), ("", "stop")),
            (("", "stop"), ("\n", "stop")),  # EOS: "\n" is part of the text
            (("\nb", "length"), ("", "stop")),
        ]
        for (text, finish_reason), expected in cases:
            reader = sm.reader()
            _apply_stop_words(reader, "a\n", finish_reason=None)
            self.assertEqual(
                _apply_stop_words(reader, text, finish_reason=finish_reason),
                expected,
            )


class TestHandleCompletion(unittest.TestCase):
    def test_stream_drops_tool_calls_that_do_not_parse(self):
        ctx = GenerationContext(
            has_tool_calling=True,
            has_thinking=False,
            tool_parser=json_tools.parse_tool_call,
            text_sm=TextStateMachine(TOOL_CALLS),
            initial_state="normal",
            prompt=[0],
        )
        call = '<tool_call>{"name": "f", "arguments": {}}</tool_call>'
        chunks = ["Hi.", "<tool_call>{oops}</tool_call>", call]
        response = [Response(c, 0, 0.0, None, ()) for c in chunks]
        response.append(Response("", 0, 0.0, "stop", ()))
        handler = mock.MagicMock(
            stream=True, logprobs=False, top_logprobs=0, stream_options=None
        )
        handler.response_generator.generate.return_value = (ctx, iter(response))
        sent = []

        def generate_response(text, finish_reason, *args, tool_calls=(), **kwargs):
            names = [tc["function"]["name"] for tc in tool_calls]
            sent.append((text, finish_reason, names))
            return {}

        handler.generate_response.side_effect = generate_response
        with self.assertLogs(level="WARNING"):
            APIHandler.handle_completion(handler, SimpleNamespace(tools=None), [])
        self.assertEqual(
            sent, [("Hi.", None, []), ("", None, ["f"]), ("", "tool_calls", [])]
        )

    def test_held_text_read_once(self):
        reader = TextStateMachine({"normal": [("##", "normal")]}).reader()
        delta = reader.feed("a #") + reader.feed("# b") + reader.finish()
        self.assertEqual(delta, TextStateMachine.Delta(content="a  b"))


class TestToolCallFormatter(unittest.TestCase):
    def test_formats_parallel_tool_calls(self):
        formatter = ToolCallFormatter(pythonic.parse_tool_call, tools=None)
        raw_tool_call = (
            '[get_time(location="Paris"), '
            'grocery.order(items=[{"name": "noodles", "organic": true}])]'
        )

        tool_calls = formatter([raw_tool_call])

        self.assertEqual(
            [tc["function"]["name"] for tc in tool_calls],
            ["get_time", "grocery.order"],
        )
        self.assertTrue(all(tc["type"] == "function" for tc in tool_calls))
        self.assertEqual(
            json.loads(tool_calls[1]["function"]["arguments"]),
            {"items": [{"name": "noodles", "organic": True}]},
        )

    def test_drops_broken_tool_calls(self):
        formatter = ToolCallFormatter(json_tools.parse_tool_call, None, True)
        broken = ["{oops}", "null", '{"name": "f"}']
        with self.assertLogs(level="WARNING"):
            tool_calls = formatter([*broken, '{"name": "g", "arguments": {}}'])
        self.assertEqual(
            [(tc["function"]["name"], tc["index"]) for tc in tool_calls], [("g", 0)]
        )


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.response_generator = ResponseGenerator(
            DummyModelProvider(), LRUPromptCache()
        )
        cls.server_address = ("localhost", 0)
        cls.httpd = http.server.HTTPServer(
            cls.server_address,
            lambda *args, **kwargs: APIHandler(cls.response_generator, *args, **kwargs),
        )
        cls.port = cls.httpd.server_port
        cls.server_thread = threading.Thread(target=cls.httpd.serve_forever)
        cls.server_thread.daemon = True
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.server_thread.join()
        cls.response_generator.stop_and_join()

    def test_max_tokens_zero_is_rejected(self):
        url = f"http://localhost:{self.port}/v1/completions"
        response = requests.post(url, json={"prompt": "hi", "max_tokens": 0})
        self.assertEqual(response.status_code, 400)

    def test_malformed_stop_is_rejected(self):
        url = f"http://localhost:{self.port}/v1/completions"
        for stop in (5, [["a"]]):
            response = requests.post(url, json={"prompt": "hi", "stop": stop})
            self.assertEqual(response.status_code, 400)

    def test_handle_completions(self):
        url = f"http://localhost:{self.port}/v1/completions"

        post_data = {
            "model": "default_model",
            "prompt": "Once upon a time",
            "max_tokens": 10,
            "temperature": 0.5,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "repetition_context_size": 20,
            "seed": 999,
            "stop": "stop sequence",
        }

        response = requests.post(url, json=post_data)

        response_body = json.loads(response.text)

        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)
        first_text = response_body["choices"][0]["text"]
        self.assertEqual(
            first_text,
            json.loads(requests.post(url, json=post_data).text)["choices"][0]["text"],
        )

    def test_handle_chat_completions(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.7,
            "top_p": 0.85,
            "repetition_penalty": 1.2,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
        }
        response = requests.post(url, json=chat_post_data)
        response_body = response.text
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)

    def test_handle_chat_completions_with_content_fragments(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.7,
            "top_p": 0.85,
            "repetition_penalty": 1.2,
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are a helpful assistant."}
                    ],
                },
                {"role": "user", "content": [{"type": "text", "text": "Hello!"}]},
            ],
        }
        response = requests.post(url, json=chat_post_data)
        response_body = response.text
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)

    def test_handle_chat_completions_with_null_tool_content(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.7,
            "top_p": 0.85,
            "repetition_penalty": 1.2,
            "messages": [
                {"role": "user", "content": "what is 2+3?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": "123",
                            "function": {
                                "name": "add",
                                "arguments": '{"a": 2, "b": 3}',
                            },
                        }
                    ],
                },
                {"role": "tool", "content": "5", "tool_call_id": "123"},
            ],
        }
        response = requests.post(url, json=chat_post_data)
        response_body = response.text
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)

    def test_handle_chat_completions_with_dict_tool_arguments(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "messages": [
                {"role": "user", "content": "what is 2+3?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": "123",
                            "function": {
                                "name": "add",
                                "arguments": {"a": 2, "b": 3},
                            },
                        }
                    ],
                },
                {"role": "tool", "content": "5", "tool_call_id": "123"},
            ],
        }
        response = requests.post(url, json=chat_post_data)
        response_body = response.text
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)

    def test_generation_thread_exit_releases_model(self):
        response_generator = ResponseGenerator(DummyModelProvider(), LRUPromptCache())
        provider = response_generator.model_provider
        response_generator.stop_and_join()

        # The weights are dropped even though this frame still holds the
        # provider, and the args survive for request handler threads.
        self.assertIsNone(provider.model)
        self.assertIsNone(provider.tokenizer)
        self.assertFalse(provider.is_batchable)
        self.assertIsNotNone(response_generator.cli_args.allowed_origins)

    def test_handle_models(self):
        url = f"http://localhost:{self.port}/v1/models"
        response = requests.get(url)
        self.assertEqual(response.status_code, 200)
        response_body = json.loads(response.text)
        self.assertEqual(response_body["object"], "list")
        self.assertIsInstance(response_body["data"], list)
        self.assertGreater(len(response_body["data"]), 0)
        model = response_body["data"][0]
        self.assertIn("id", model)
        self.assertEqual(model["object"], "model")
        self.assertIn("created", model)

    def test_health_endpoint(self):
        url = f"http://localhost:{self.port}/health"

        response = requests.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

        self.response_generator.stop_and_join()
        response = requests.get(url)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"status": "unavailable"})


class TestServerWithDraftModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.response_generator = ResponseGenerator(
            DummyModelProvider(with_draft=True), LRUPromptCache()
        )
        cls.server_address = ("localhost", 0)
        cls.httpd = http.server.HTTPServer(
            cls.server_address,
            lambda *args, **kwargs: APIHandler(cls.response_generator, *args, **kwargs),
        )
        cls.port = cls.httpd.server_port
        cls.server_thread = threading.Thread(target=cls.httpd.serve_forever)
        cls.server_thread.daemon = True
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.server_thread.join()
        cls.response_generator.stop_and_join()

    def test_handle_completions_with_draft_model(self):
        url = f"http://localhost:{self.port}/v1/completions"

        post_data = {
            "model": "default_model",
            "prompt": "Once upon a time",
            "max_tokens": 10,
            "temperature": 0.0,
            "top_p": 1.0,
        }

        response = requests.post(url, json=post_data)
        self.assertEqual(response.status_code, 200)

        response_body = json.loads(response.text)
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)
        self.assertIn("usage", response_body)

        # Check that tokens were generated
        self.assertTrue(response_body["usage"]["completion_tokens"] > 0)

    def test_handle_chat_completions_with_draft_model(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"

        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
        }

        response = requests.post(url, json=chat_post_data)
        self.assertEqual(response.status_code, 200)

        response_body = json.loads(response.text)
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)
        self.assertIn("usage", response_body)

        # Check that tokens were generated
        self.assertTrue(response_body["usage"]["completion_tokens"] > 0)

    def test_streaming_with_draft_model(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"

        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.0,
            "stream": True,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
        }

        response = requests.post(url, json=chat_post_data, stream=True)
        self.assertEqual(response.status_code, 200)

        chunk_count = 0
        for chunk in response.iter_lines():
            if chunk:
                data = chunk.decode("utf-8")
                if data.startswith("data: ") and data != "data: [DONE]":
                    chunk_data = json.loads(data[6:])  # Skip the "data: " prefix
                    self.assertIn("choices", chunk_data)
                    self.assertEqual(len(chunk_data["choices"]), 1)
                    self.assertIn("delta", chunk_data["choices"][0])
                    chunk_count += 1

        # Make sure we got some streaming chunks
        self.assertGreater(chunk_count, 0)

    def test_prompt_cache_with_draft_model(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"

        # First request to initialize cache
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 5,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me a story about"},
            ],
        }

        first_response = requests.post(url, json=chat_post_data)
        self.assertEqual(first_response.status_code, 200)

        # Second request with same prefix should use cache
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 5,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me a story about dragons."},
            ],
        }

        second_response = requests.post(url, json=chat_post_data)
        self.assertEqual(second_response.status_code, 200)

        # Both responses should have content
        first_response_body = json.loads(first_response.text)
        second_response_body = json.loads(second_response.text)

        self.assertIn("choices", first_response_body)
        self.assertIn("choices", second_response_body)
        self.assertIn("message", first_response_body["choices"][0])
        self.assertIn("message", second_response_body["choices"][0])
        self.assertIn("content", first_response_body["choices"][0]["message"])
        self.assertIn("content", second_response_body["choices"][0]["message"])

        # Ensure both generated content
        self.assertIsNotNone(first_response_body["choices"][0]["message"]["content"])
        self.assertIsNotNone(second_response_body["choices"][0]["message"]["content"])


class TestServerKVCacheQuantization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_provider = DummyModelProvider(kv_bits=4, quantized_kv_start=0)
        cls.prompt_cache = LRUPromptCache()
        cls.response_generator = ResponseGenerator(cls.model_provider, cls.prompt_cache)
        cls.httpd = http.server.HTTPServer(
            ("localhost", 0),
            lambda *args, **kwargs: APIHandler(cls.response_generator, *args, **kwargs),
        )
        cls.port = cls.httpd.server_port
        cls.server_thread = threading.Thread(target=cls.httpd.serve_forever)
        cls.server_thread.daemon = True
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.server_thread.join()
        cls.response_generator.stop_and_join()

    def test_quantized_kv_disables_batching(self):
        args = type("args", (object,), {"seed": None})
        self.assertFalse(self.response_generator._is_batchable(args))

    def test_completion_quantizes_the_cache(self):
        url = f"http://localhost:{self.port}/v1/completions"
        prompt = "Once upon a time"
        response = requests.post(
            url,
            json={"model": "default_model", "prompt": prompt, "max_tokens": 8},
        )
        self.assertIn("choices", json.loads(response.text))

        tokens = self.model_provider.tokenizer.encode(prompt)
        cache, _ = self.prompt_cache.fetch_nearest_cache(
            self.model_provider.model_key, tokens
        )
        self.assertIsNotNone(cache)
        for c in cache:
            self.assertIsInstance(c, QuantizedKVCache)
            self.assertEqual(c.bits, 4)
            self.assertEqual(c.group_size, 64)


class TestServerWithoutKVCacheQuantization(unittest.TestCase):
    def test_batching_stays_enabled(self):
        prompt_cache = LRUPromptCache()
        response_generator = ResponseGenerator(DummyModelProvider(), prompt_cache)
        try:
            args = type("args", (object,), {"seed": None})
            self.assertTrue(response_generator._is_batchable(args))
        finally:
            response_generator.stop_and_join()


class TestKeepalive(unittest.TestCase):
    def test_keepalive_callback(self):
        """Test keepalive callback sends SSE comments and handles errors"""
        from unittest.mock import Mock

        # Mock handler
        mock_wfile = io.BytesIO()
        handler = Mock()
        handler.wfile = mock_wfile

        # Test callback logic (same as in server.py)
        def keepalive_callback(processed_tokens, total_tokens):
            if handler.stream:
                try:
                    handler.wfile.write(
                        f": keepalive {processed_tokens}/{total_tokens}\n\n".encode()
                    )
                    handler.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

        # Test streaming enabled
        handler.stream = True
        keepalive_callback(1024, 4096)

        output = mock_wfile.getvalue().decode("utf-8")
        self.assertEqual(output, ": keepalive 1024/4096\n\n")

        # Test streaming disabled
        handler.stream = False
        mock_wfile.seek(0)
        mock_wfile.truncate(0)
        keepalive_callback(2048, 4096)

        output = mock_wfile.getvalue().decode("utf-8")
        self.assertEqual(output, "")

        # Test error handling
        handler.stream = True
        handler.wfile = Mock()
        handler.wfile.write.side_effect = BrokenPipeError("Connection broken")

        # Should not raise exception
        try:
            keepalive_callback(3072, 4096)
        except Exception as e:
            self.fail(f"Callback should handle BrokenPipeError: {e}")


class TestLRUPromptCache(unittest.TestCase):
    def test_caching(self):
        cache = LRUPromptCache(max_size=10)

        def get_kv(n):
            keys = mx.arange(n).reshape(1, 1, n, 1)
            return keys, keys

        model = ("test", None, None)
        tokens = [10] * 24

        c, t = cache.fetch_nearest_cache(model, tokens)
        self.assertTrue(c is None)
        self.assertEqual(t, tokens)

        c = [KVCache()]
        c[0].update_and_fetch(*get_kv(24))
        cache.insert_cache(model, t, c)

        # Fetching a cache that is strictly a prefix doesn't remove it from the
        # lru cache
        tokens = tokens + [20] * 5
        c, t = cache.fetch_nearest_cache(model, tokens)
        k, v = c[0].keys_and_values()
        self.assertTrue((k == v).all().item())
        self.assertTrue((k.flatten() == mx.arange(24)).all().item())
        self.assertEqual(t, [20] * 5)
        self.assertEqual(len(cache), 1)

        # Inserting a trimmable cache with shared prefix removes the prefixes
        tokens = tokens + [30] * 3
        c[0].update_and_fetch(*get_kv(8))
        cache.insert_cache(model, tokens, c)
        self.assertEqual(len(cache), 1)

        # Fetching a cache with a shared prefix doesn't remove it either
        tokens = tokens[:26] + [40] * 8
        c, t = cache.fetch_nearest_cache(model, tokens)
        k, v = c[0].keys_and_values()
        self.assertTrue((k == v).all().item())
        self.assertTrue(
            (k.flatten() == mx.concatenate([mx.arange(24), mx.arange(2)])).all().item()
        )
        self.assertEqual(t, [40] * 8)
        self.assertEqual(len(cache), 1)

        # Inserting a diverged cache actually creates another entry
        c[0].update_and_fetch(*get_kv(8))
        cache.insert_cache(model, tokens, c)
        self.assertEqual(len(cache), 2)

    def test_lru(self):
        cache = LRUPromptCache(max_size=2)
        model = ("test", None, None)
        cache.insert_cache(model, [1, 2], [MockCache("test1")])
        cache.insert_cache(model, [2, 3], [MockCache("test2")])

        c, t = cache.fetch_nearest_cache(model, [1, 2])
        self.assertEqual(c, [MockCache("test1")])
        self.assertEqual(t, [])
        c, t = cache.fetch_nearest_cache(model, [1])
        self.assertEqual(c, [MockCache("test1")])
        self.assertEqual(t, [1])
        c, t = cache.fetch_nearest_cache(model, [1, 3, 4])
        self.assertEqual(c, [MockCache("test1")])
        self.assertEqual(t, [3, 4])
        c, t = cache.fetch_nearest_cache(model, [2, 3, 4])
        self.assertEqual(c, [MockCache("test2")])
        self.assertEqual(t, [4])
        c, t = cache.fetch_nearest_cache(model, [2, 4, 5])
        self.assertEqual(c, [MockCache("test2")])
        self.assertEqual(t, [4, 5])

        cache.insert_cache(model, [1, 2], [MockCache("test1")])
        cache.insert_cache(model, [2, 3], [MockCache("test2")])
        cache.insert_cache(model, [3, 4], [MockCache("test3")])

        c, t = cache.fetch_nearest_cache(model, [1, 2])
        self.assertEqual(c, None)
        self.assertEqual(t, [1, 2])
        c, t = cache.fetch_nearest_cache(model, [2, 3])
        self.assertEqual(c, [MockCache("test2")])
        self.assertEqual(t, [])
        c, t = cache.fetch_nearest_cache(model, [3, 4])
        self.assertEqual(c, [MockCache("test3")])
        self.assertEqual(t, [])

        cache.insert_cache(model, [4, 5], [MockCache("test4")], cache_type="user")
        c, t = cache.fetch_nearest_cache(model, [2, 3])
        self.assertEqual(c, None)
        self.assertEqual(t, [2, 3])
        c, t = cache.fetch_nearest_cache(model, [3, 4])
        self.assertEqual(c, [MockCache("test3")])
        self.assertEqual(t, [])
        c, t = cache.fetch_nearest_cache(model, [4, 5])
        self.assertEqual(c, [MockCache("test4")])
        self.assertEqual(t, [])

        cache.insert_cache(model, [5, 6], [MockCache("test5")])
        cache.insert_cache(model, [6, 7], [MockCache("test6")])
        c, t = cache.fetch_nearest_cache(model, [5, 6])
        self.assertEqual(c, None)
        self.assertEqual(t, [5, 6])
        c, t = cache.fetch_nearest_cache(model, [6, 7])
        self.assertEqual(c, [MockCache("test6")])
        self.assertEqual(t, [])
        c, t = cache.fetch_nearest_cache(model, [4, 5])
        self.assertEqual(c, [MockCache("test4")])
        self.assertEqual(t, [])

    def test_insert_trimmable_cache_removes_immediate_prefix(self):
        cache = LRUPromptCache(max_size=10)
        model = ("test", None, None)

        cache.insert_cache(model, [1, 2], [MockCache("ab")])
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.nbytes, 2)

        cache.insert_cache(model, [1, 2, 3], [MockCache("abc")])
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.nbytes, 3)

    def test_insert_empty_tokens_does_not_self_destruct(self):
        cache = LRUPromptCache(max_size=10)
        model = ("test", None, None)

        cache.insert_cache(model, [], [MockCache("root")])
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.nbytes, 4)

        c, t = cache.fetch_nearest_cache(model, [])
        self.assertIsNotNone(c)
        self.assertEqual(t, [])

    def test_fetch_empty_tokens_after_root_eviction(self):
        cache = LRUPromptCache(max_size=10)
        model = ("test", None, None)

        cache.insert_cache(model, [], [MockCache("root")])
        cache.insert_cache(model, [1], [MockCache("a")])

        c, t = cache.fetch_nearest_cache(model, [])
        self.assertIsNone(c)
        self.assertEqual(t, [])

    def test_lru_bytes(self):
        cache = LRUPromptCache(max_size=100, max_bytes=10)
        model = ("test", None, None)

        cache.insert_cache(model, [1, 2], [MockCache("aaa")])
        cache.insert_cache(model, [3, 4], [MockCache("bbb")])
        cache.insert_cache(model, [4, 5], [MockCache("ccc")])
        cache.insert_cache(model, [6, 7], [MockCache("ddd")])

        self.assertEqual(len(cache), 3)
        self.assertEqual(cache.nbytes, 9)

        cache.trim_to(n_bytes=7)
        self.assertEqual(len(cache), 2)
        self.assertEqual(cache.nbytes, 6)

        c, t = cache.fetch_nearest_cache(model, [1, 2])
        self.assertEqual(c, None)
        self.assertEqual(t, [1, 2])
        c, t = cache.fetch_nearest_cache(model, [3, 4])
        self.assertEqual(c, None)
        self.assertEqual(t, [3, 4])


class TestMakeSampler(unittest.TestCase):
    def test_xtc_special_tokens(self):
        class FakeTokenizer:
            eos_token_ids = [0, 1, 9]

            def encode(self, text, add_special_tokens=False):
                return [3]

        sampling = SamplingArguments(
            temperature=0.6,
            top_p=1.0,
            top_k=0,
            min_p=0.0,
            xtc_probability=1.0,
            xtc_threshold=0.1,
        )
        args = type("obj", (object,), {"sampling": sampling})
        sampler = _make_sampler(args, FakeTokenizer())
        logits = mx.log(
            mx.array([[0.4, 0.2, 0.1, 0.1, 0.05, 0.05, 0.03, 0.03, 0.02, 0.02]])
        )
        token = sampler(logits)
        mx.eval(token)
        self.assertEqual(token.shape, (1,))


class TestModelSwapClearsCache(unittest.TestCase):
    @mock.patch("mlx_lm.server.mx.clear_cache")
    @mock.patch("mlx_lm.server.make_prompt_cache", return_value=[])
    @mock.patch("mlx_lm.server.load")
    @mock.patch("mlx_lm.server.mx.distributed.init")
    def test_load_clears_mlx_buffer_pool(self, init, load, make_cache, clear_cache):
        import argparse

        from mlx_lm.server import ModelProvider

        args = argparse.Namespace(
            adapter_path=None,
            chat_template=None,
            draft_model=None,
            model="model-a",
            pipeline=False,
            trust_remote_code=False,
            use_default_chat_template=False,
        )
        tokenizer = mock.Mock()
        tokenizer.chat_template = None
        tokenizer.default_chat_template = None
        load.return_value = (mock.Mock(), tokenizer, {})
        init.return_value.size.return_value = 1

        provider = ModelProvider(args)
        provider._load("model-a")
        provider._load("model-b")
        self.assertGreaterEqual(clear_cache.call_count, 2)


class FailingModelProvider:
    def load_default(self):
        raise RuntimeError("simulated generate crash")


class TestGenerationThreadDeath(unittest.TestCase):
    def _crashed_generator(self):
        rg = ResponseGenerator(FailingModelProvider(), LRUPromptCache())
        rg.join()
        time.sleep(0.05)
        return rg

    def test_generation_unavailable_after_thread_crash(self):
        rg = self._crashed_generator()
        self.assertTrue(rg._generation_failed)
        self.assertFalse(rg.generation_available())
        with self.assertRaisesRegex(RuntimeError, "generation thread died"):
            rg.generate(None, None)

    def test_inflight_request_does_not_hang_after_thread_crash(self):
        rg = self._crashed_generator()
        # A request dequeued before the crash never gets a response queued, so
        # waiting on it must give up instead of blocking forever.
        with self.assertRaisesRegex(RuntimeError, "generation thread died"):
            rg._await_response(Queue())


if __name__ == "__main__":
    unittest.main()
