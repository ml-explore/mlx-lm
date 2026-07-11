# Copyright © 2024-2026 Apple Inc.

import http
import io
import json
import socket
import threading
import time
import unittest
from unittest import mock

import mlx.core as mx
import requests

from mlx_lm.generate import TextStateMachine
from mlx_lm.models.cache import KVCache
from mlx_lm.server import (
    APIHandler,
    CacheEvictor,
    LRUPromptCache,
    Response,
    ResponseGenerator,
)
from mlx_lm.utils import load


class DummyModelProvider:
    def __init__(self, with_draft=False):
        HF_MODEL_PATH = "mlx-community/Qwen1.5-0.5B-Chat-4bit"
        self.model, self.tokenizer = load(HF_MODEL_PATH)
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
                "cache_idle_evict_s": 0.0,
                "allowed_origins": ["*"],
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


class TestTextStateMachine(unittest.TestCase):
    """Test the TextStateMachine buffering and stripping behavior."""

    def test_strips_control_sequences(self):
        sm = TextStateMachine(
            {
                "normal": [("<tool_call>", "tool")],
                "tool": [("</tool_call>", "normal")],
            }
        )
        state = sm.make_state()
        state, text, s = sm.step(state, "hi <tool_call>body</tool_call> bye")
        state, rest, s = sm.flush(state)
        full = text + rest
        self.assertEqual(full, "hi body bye")

    def test_back_to_back_tool_calls(self):
        sm = TextStateMachine(
            {
                "normal": [("<tool_call>", "tool")],
                "tool": [("</tool_call>", "normal")],
            }
        )
        state = sm.make_state()
        state, t1, s = sm.step(state, "<tool_call>call1</tool_call>")
        state, t2, s = sm.step(state, "<tool_call>call2</tool_call>")
        state, rest, s = sm.flush(state)
        full = t1 + t2 + rest
        self.assertEqual(full, "call1call2")

    def test_partial_match_buffered_then_flushed(self):
        sm = TextStateMachine(
            {
                "normal": [("<tool_call>", "tool")],
                "tool": [("</tool_call>", "normal")],
            }
        )
        # First enter tool state
        state = sm.make_state()
        state, text, s = sm.step(state, "<tool_call>body</")
        self.assertEqual(s, "tool")
        # 'body' is emitted, '</' is buffered (partial match of '</tool_call>')
        self.assertEqual(text, "body")
        # flush releases the buffered text
        state, rest, s = sm.flush(state)
        self.assertEqual(rest, "</")

    def test_discard_drops_buffer(self):
        sm = TextStateMachine(
            {
                "normal": [("STOP", "normal")],
            }
        )
        state = sm.make_state()
        state, text, s = sm.step(state, "hello ST")
        self.assertEqual(text, "hello ")
        # discard drops the buffered 'ST'
        state, s = sm.discard(state)
        self.assertEqual(s, "normal")

    def test_stop_words_stripped(self):
        sm = TextStateMachine(
            {
                "normal": [("STOP", "normal")],
            }
        )
        state = sm.make_state()
        state, text, s = sm.step(state, "hello STOP world")
        state, rest, s = sm.flush(state)
        self.assertEqual(text + rest, "hello  world")

    def test_reasoning_to_tool_transition(self):
        # A tool call started inside a reasoning block must enter "tool".
        sm = TextStateMachine(
            {
                "normal": [("<think>", "reasoning"), ("<tool>", "tool")],
                "reasoning": [("</think>", "normal"), ("<tool>", "tool")],
                "tool": [("</tool>", "normal")],
            }
        )
        state = sm.make_state()
        state, _, s = sm.step(state, "<think>hmm")
        self.assertEqual(s, "reasoning")
        state, _, s = sm.step(state, "<tool>")
        self.assertEqual(s, "tool")
        state, _, s = sm.step(state, "</tool>")
        self.assertEqual(s, "normal")

    def test_empty_end_marker_stays_in_tool_on_discard(self):
        # Models with an empty tool_call_end (e.g. Mistral) never leave "tool";
        # discard on stop must preserve the state so the tool call is flushed.
        sm = TextStateMachine(
            {
                "normal": [("[TOOL_CALLS]", "tool")],
                "tool": [],
            }
        )
        state = sm.make_state()
        state, text, s = sm.step(state, "[TOOL_CALLS]f[ARGS]{}")
        self.assertEqual(s, "tool")
        self.assertEqual(text, "f[ARGS]{}")
        state, s = sm.discard(state)
        self.assertEqual(s, "tool")


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

    def test_make_state_machine_empty_tool_call_end(self):
        class FakeTokenizer:
            has_thinking = False
            has_tool_calling = True
            tool_call_start = "[TOOL_CALLS]"
            tool_call_end = ""
            tool_call_start_tokens = (100,)
            tool_call_end_tokens = ()
            eos_token_ids = [2]

            def convert_ids_to_tokens(self, t):
                return f"<eos{t}>"

            def encode(self, text, add_special_tokens=False):
                return []

        stop_matcher, text_sm = self.response_generator._make_state_machine(
            ("fake-empty-end", None, None),
            FakeTokenizer(),
            stop_words=[],
        )

        # Verify the text state machine strips tool call markers
        text_state = text_sm.make_state()
        text_state, clean_text, s = text_sm.step(text_state, "hello[TOOL_CALLS]body")
        self.assertEqual(s, "tool")
        # 'hello' is before the match, 'body' flows through (no tool_call_end)
        self.assertEqual(clean_text, "hellobody")

        # Verify EOS stops via the stop matcher
        stop_state = stop_matcher.make_state()
        stop_state, matched = stop_matcher.match(stop_state, stop_matcher._trie, 2)
        self.assertTrue(matched)

    def test_no_idle_eviction_when_disabled(self):
        # --cache-idle-evict-s defaults to 0 (off): the generation loop keeps
        # polling while idle but must never call mx.clear_cache().
        with mock.patch.object(mx, "clear_cache") as clear_cache:
            time.sleep(0.6)
            clear_cache.assert_not_called()

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
        k, v = c[0].state
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
        k, v = c[0].state
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


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class TestCacheEvictor(unittest.TestCase):
    """Unit tests for the idle/on-demand eviction state machine."""

    def _make_evictor(self, idle_s, prompt_cache=None):
        clock = FakeClock()
        evictor = CacheEvictor(
            prompt_cache or LRUPromptCache(), idle_s=idle_s, clock=clock
        )
        return evictor, clock

    def _request_evict_async(self, evictor, **kwargs):
        """Call request_evict on a helper thread (it blocks until the
        generation-loop side services it via step)."""
        results = []
        thread = threading.Thread(
            target=lambda: results.append(evictor.request_evict(**kwargs))
        )
        thread.start()
        # Wait until the request is enqueued before stepping
        while evictor._requests.empty():
            time.sleep(0.005)
        return thread, results

    def test_disabled_never_evicts(self):
        evictor, clock = self._make_evictor(idle_s=0.0)
        with mock.patch.object(mx, "clear_cache") as clear_cache:
            for _ in range(3):
                clock.advance(1e6)
                evictor.step(busy=False)
            clear_cache.assert_not_called()

    def test_idle_eviction_fires_once_after_threshold(self):
        evictor, clock = self._make_evictor(idle_s=60.0)
        with mock.patch.object(mx, "clear_cache") as clear_cache:
            evictor.step(busy=False)
            clear_cache.assert_not_called()

            clock.advance(59.0)
            evictor.step(busy=False)
            clear_cache.assert_not_called()

            clock.advance(2.0)
            evictor.step(busy=False)
            self.assertEqual(clear_cache.call_count, 1)

            # Only once per idle period, no matter how long the idle lasts
            clock.advance(1e6)
            evictor.step(busy=False)
            self.assertEqual(clear_cache.call_count, 1)

            # Activity re-arms the timer
            evictor.touch()
            clock.advance(61.0)
            evictor.step(busy=False)
            self.assertEqual(clear_cache.call_count, 2)

    def test_no_eviction_while_busy(self):
        evictor, clock = self._make_evictor(idle_s=60.0)
        with mock.patch.object(mx, "clear_cache") as clear_cache:
            # Way past the threshold, but there is active work
            clock.advance(120.0)
            evictor.step(busy=True)
            clear_cache.assert_not_called()

            # The busy step re-armed the timer, so going idle does not evict
            evictor.step(busy=False)
            clear_cache.assert_not_called()

            # Only after a full idle period does the eviction fire
            clock.advance(61.0)
            evictor.step(busy=False)
            self.assertEqual(clear_cache.call_count, 1)

    def test_on_demand_eviction_returns_stats(self):
        prompt_cache = LRUPromptCache()
        prompt_cache.insert_cache(("test", None, None), [1, 2], [MockCache("abcd")])
        evictor, _ = self._make_evictor(idle_s=0.0, prompt_cache=prompt_cache)

        # Plain eviction keeps the prompt cache
        with mock.patch.object(mx, "clear_cache") as clear_cache:
            thread, results = self._request_evict_async(evictor)
            evictor.step(busy=False)
            thread.join()
            self.assertEqual(clear_cache.call_count, 1)
        result = results[0]
        self.assertTrue(result["evicted"])
        for key in ("cache_bytes", "active_bytes", "prompt_cache_bytes"):
            self.assertIn(key, result["before"])
            self.assertIn(key, result["after"])
        self.assertEqual(result["before"]["prompt_cache_bytes"], 4)
        self.assertEqual(result["after"]["prompt_cache_bytes"], 4)

        # clear_prompt_cache=True also drops the stored prompt caches
        with mock.patch.object(mx, "clear_cache"):
            thread, results = self._request_evict_async(
                evictor, clear_prompt_cache=True
            )
            evictor.step(busy=False)
            thread.join()
        result = results[0]
        self.assertTrue(result["evicted"])
        self.assertEqual(result["before"]["prompt_cache_bytes"], 4)
        self.assertEqual(result["after"]["prompt_cache_bytes"], 0)
        self.assertEqual(len(prompt_cache), 0)

    def test_on_demand_eviction_refused_while_busy(self):
        prompt_cache = LRUPromptCache()
        prompt_cache.insert_cache(("test", None, None), [1, 2], [MockCache("abcd")])
        evictor, _ = self._make_evictor(idle_s=0.0, prompt_cache=prompt_cache)

        with mock.patch.object(mx, "clear_cache") as clear_cache:
            thread, results = self._request_evict_async(
                evictor, clear_prompt_cache=True
            )
            evictor.step(busy=True)
            thread.join()
            clear_cache.assert_not_called()
        result = results[0]
        self.assertFalse(result["evicted"])
        self.assertEqual(result["reason"], "busy")
        # Nothing was touched
        self.assertEqual(len(prompt_cache), 1)
        self.assertEqual(prompt_cache.nbytes, 4)

    def test_timed_out_evict_request_is_discarded(self):
        # Regression: if the generation thread only reaches a queued evict
        # request after the handler's timeout expired (e.g. it was blocked in
        # a long non-batched generation), the client was already answered
        # `evicted: false` — servicing the stale message anyway would be a
        # ghost eviction the client was told did not happen.
        prompt_cache = LRUPromptCache()
        prompt_cache.insert_cache(("test", None, None), [1, 2], [MockCache("abcd")])
        evictor, clock = self._make_evictor(idle_s=0.0, prompt_cache=prompt_cache)

        with mock.patch.object(mx, "clear_cache") as clear_cache:
            # No step() runs while the handler waits: the "loop" is blocked.
            thread, results = self._request_evict_async(
                evictor, clear_prompt_cache=True, timeout=0.05
            )
            thread.join()
            result = results[0]
            self.assertFalse(result["evicted"])
            self.assertEqual(result["reason"], "timeout")

            # The loop unblocks after the deadline: the stale message must be
            # discarded, not serviced.
            clock.advance(1.0)
            evictor.step(busy=False)
            clear_cache.assert_not_called()
            self.assertEqual(len(prompt_cache), 1)
            self.assertEqual(prompt_cache.nbytes, 4)
            self.assertTrue(evictor._requests.empty())

            # A fresh request afterwards works normally.
            thread, results = self._request_evict_async(
                evictor, clear_prompt_cache=True
            )
            evictor.step(busy=False)
            thread.join()
            self.assertTrue(results[0]["evicted"])
            self.assertEqual(clear_cache.call_count, 1)


class TestIdleEviction(unittest.TestCase):
    """Integration tests: eviction wired into the server's generation loop."""

    @classmethod
    def setUpClass(cls):
        provider = DummyModelProvider()
        provider.cli_args.cache_idle_evict_s = 0.25
        cls.response_generator = ResponseGenerator(provider, LRUPromptCache())
        cls.server_address = ("localhost", 0)
        # ThreadingHTTPServer (as in production) so that /v1/cache/evict can
        # be exercised while a completion request is streaming.
        cls.httpd = http.server.ThreadingHTTPServer(
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

    def _completion(self, max_tokens=5):
        url = f"http://localhost:{self.port}/v1/completions"
        response = requests.post(
            url,
            json={
                "model": "default_model",
                "prompt": "Once upon a time",
                "max_tokens": max_tokens,
            },
        )
        self.assertEqual(response.status_code, 200)

    def test_idle_timer_fires_in_generation_loop(self):
        with mock.patch.object(mx, "clear_cache") as clear_cache:
            self._completion()
            # Generation itself may call mx.clear_cache; only count calls
            # made after the request completed (i.e. while idle).
            clear_cache.reset_mock()

            deadline = time.time() + 5.0
            while clear_cache.call_count == 0 and time.time() < deadline:
                time.sleep(0.05)
            self.assertEqual(clear_cache.call_count, 1)

            # Once per idle period: staying idle must not evict again
            time.sleep(0.6)
            self.assertEqual(clear_cache.call_count, 1)

    def test_evict_endpoint_returns_before_after_stats(self):
        # Populate the prompt cache with a completed request
        self._completion()

        url = f"http://localhost:{self.port}/v1/cache/evict"
        response = requests.post(url, json={})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["evicted"])
        for key in (
            "cache_bytes",
            "active_bytes",
            "prompt_cache_sequences",
            "prompt_cache_bytes",
        ):
            self.assertIsInstance(body["before"][key], int)
            self.assertIsInstance(body["after"][key], int)
        # The prompt cache is kept by default
        self.assertGreater(body["after"]["prompt_cache_bytes"], 0)

        # An empty body works too, and clear_prompt_cache drops the KV caches
        response = requests.post(url, json={"clear_prompt_cache": True})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["evicted"])
        self.assertGreater(body["before"]["prompt_cache_bytes"], 0)
        self.assertEqual(body["after"]["prompt_cache_bytes"], 0)

        response = requests.post(url)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["evicted"])

    def test_evict_endpoint_rejects_invalid_body(self):
        url = f"http://localhost:{self.port}/v1/cache/evict"
        response = requests.post(url, data=b"not json")
        self.assertEqual(response.status_code, 400)
        response = requests.post(url, json=["not", "a", "dict"])
        self.assertEqual(response.status_code, 400)

    def _raw_status(self, raw_request: bytes) -> bytes:
        """Send a raw HTTP request and return the response status line."""
        with socket.create_connection(("localhost", self.port), timeout=5) as sock:
            sock.sendall(raw_request)
            sock.settimeout(5)
            data = b""
            while b"\r\n" not in data:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk
        return data.split(b"\r\n", 1)[0]

    def test_evict_endpoint_invalid_content_length(self):
        # A malformed Content-Length must get a 400, not a connection reset
        status = self._raw_status(
            b"POST /v1/cache/evict HTTP/1.1\r\n"
            b"Host: localhost\r\n"
            b"Content-Length: abc\r\n"
            b"Connection: close\r\n\r\n"
        )
        self.assertIn(b" 400 ", status + b" ")

    def test_evict_endpoint_requires_content_length(self):
        # A chunked (length-less) body is never read, so silently treating it
        # as empty would drop a `clear_prompt_cache` the client asked for:
        # require Content-Length like the completion endpoints do (411).
        status = self._raw_status(
            b"POST /v1/cache/evict HTTP/1.1\r\n"
            b"Host: localhost\r\n"
            b"Transfer-Encoding: chunked\r\n"
            b"Connection: close\r\n\r\n"
            b'1c\r\n{"clear_prompt_cache": true}\r\n0\r\n\r\n'
        )
        self.assertIn(b" 411 ", status + b" ")

    def test_cache_endpoints_wrong_method(self):
        response = requests.get(f"http://localhost:{self.port}/v1/cache/evict")
        self.assertEqual(response.status_code, 404)
        response = requests.post(f"http://localhost:{self.port}/v1/cache/stats")
        self.assertEqual(response.status_code, 404)

    def test_evict_endpoint_busy_refusal_over_http(self):
        # While a batched request is generating, the endpoint must refuse
        # immediately with `evicted: false` and touch nothing.
        completion_url = f"http://localhost:{self.port}/v1/completions"
        evict_url = f"http://localhost:{self.port}/v1/cache/evict"
        started = threading.Event()

        def long_request():
            with requests.post(
                completion_url,
                json={
                    "model": "default_model",
                    "prompt": "Once upon a time",
                    "max_tokens": 1000,
                    "stream": True,
                },
                stream=True,
            ) as response:
                # Streaming headers are sent only after the request has been
                # inserted into the batch, so the server is busy from here
                # until the (long) generation finishes.
                started.set()
                for _ in response.iter_content(chunk_size=None):
                    pass

        thread = threading.Thread(target=long_request)
        thread.start()
        try:
            self.assertTrue(started.wait(timeout=15))
            response = requests.post(evict_url, json={"clear_prompt_cache": True})
            body = response.json()
        finally:
            thread.join()

        self.assertEqual(response.status_code, 200)
        self.assertFalse(body["evicted"])
        self.assertEqual(body["reason"], "busy")

    def test_cache_stats_endpoint(self):
        url = f"http://localhost:{self.port}/v1/cache/stats"
        response = requests.get(url)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        for key in (
            "cache_bytes",
            "active_bytes",
            "prompt_cache_sequences",
            "prompt_cache_bytes",
        ):
            self.assertIsInstance(body[key], int)


if __name__ == "__main__":
    unittest.main()
