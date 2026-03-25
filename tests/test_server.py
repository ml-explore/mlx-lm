# Copyright © 2024 Apple Inc.

import http
import io
import json
import threading
import unittest
from queue import Empty, Queue
from unittest.mock import patch

import mlx.core as mx
import requests

from mlx_lm.generate import BatchGenerator
from mlx_lm.models.cache import KVCache
from mlx_lm.server import (
    APIHandler,
    CompletionRequest,
    GenerationArguments,
    LogitsProcessorArguments,
    LRUPromptCache,
    ModelDescription,
    ResponseGenerator,
    SamplingArguments,
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

    def test_sequence_overlap(self):
        from mlx_lm.server import sequence_overlap

        self.assertTrue(sequence_overlap([1], [1]))
        self.assertTrue(sequence_overlap([1, 2], [1, 2]))
        self.assertTrue(sequence_overlap([1, 3], [3, 4]))
        self.assertTrue(sequence_overlap([1, 2, 3], [2, 3]))

        self.assertFalse(sequence_overlap([1], [2]))
        self.assertFalse(sequence_overlap([1, 2], [3, 4]))
        self.assertFalse(sequence_overlap([1, 2, 3], [4, 1, 2, 3]))


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

        # Fetching a strict shorter-prefix hit consumes the only stored entry.
        tokens = tokens + [20] * 5
        c, t = cache.fetch_nearest_cache(model, tokens)
        k, v = c[0].state
        self.assertTrue((k == v).all().item())
        self.assertTrue((k.flatten() == mx.arange(24)).all().item())
        self.assertEqual(t, [20] * 5)
        self.assertEqual(len(cache), 0)

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
        cache.insert_cache(model, [1, 2], [MockCache("test1")])

        c, t = cache.fetch_nearest_cache(model, [1, 2])
        self.assertEqual(c, [MockCache("test1")])
        self.assertEqual(t, [])
        c, t = cache.fetch_nearest_cache(model, [1, 2])
        self.assertEqual(c, [MockCache("test1")])
        self.assertEqual(t, [])
        c, t = cache.fetch_nearest_cache(model, [1, 2])
        self.assertIsNone(c)
        self.assertEqual(t, [1, 2])

        cache.insert_cache(model, [1, 2], [MockCache("test1")])
        cache.insert_cache(model, [2, 3], [MockCache("test2")])

        c, t = cache.fetch_nearest_cache(model, [1, 2])
        self.assertEqual(c, [MockCache("test1")])
        self.assertEqual(t, [])
        c, t = cache.fetch_nearest_cache(model, [2, 3, 4])
        self.assertEqual(c, [MockCache("test2")])
        self.assertEqual(t, [4])

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

        cache.insert_cache(model, [4, 5], [MockCache("test4")], checkpoint=True)
        c, t = cache.fetch_nearest_cache(model, [2, 3])
        self.assertEqual(c, None)
        self.assertEqual(t, [2, 3])
        c, t = cache.fetch_nearest_cache(model, [3, 4])
        self.assertEqual(c, None)
        self.assertEqual(t, [3, 4])
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


class TestResponseGeneratorBatchPromptCheckpoints(unittest.TestCase):
    @staticmethod
    def _generation_args():
        return GenerationArguments(
            model=ModelDescription("default_model", None, None),
            sampling=SamplingArguments(0.0, 1.0, 0, 0.0, 0.0, 0.0),
            logits=LogitsProcessorArguments(None, 1.0, 20, 0.0, 20, 0.0, 20),
            stop_words=[],
            max_tokens=2,
            num_draft_tokens=3,
            logprobs=False,
            top_logprobs=0,
            seed=None,
            chat_template_kwargs=None,
        )

    @staticmethod
    def _make_text_request(prompt="hello"):
        return CompletionRequest(
            request_type="text",
            prompt=prompt,
            messages=[],
            tools=None,
            role_mapping=None,
        )

    def _build_response_generator(self):
        class FakeModel:
            def make_cache(self):
                return [KVCache()]

        class FakeTokenizer:
            has_tool_calling = False
            tool_call_start = ""
            tool_call_end = ""
            tool_parser = staticmethod(lambda text, _: {})
            detokenizer = None
            has_thinking = False
            think_start_id = 0
            think_end_id = 0
            think_end = ""
            eos_token_id = 0
            eos_token_ids = set()

            def encode(self, text, add_special_tokens=False):
                return [1, 2, 3]

        class FakeProvider:
            is_batchable = True

            def __init__(self):
                self.cli_args = type(
                    "obj",
                    (object,),
                    {
                        "decode_concurrency": 4,
                        "prompt_concurrency": 2,
                        "prefill_step_size": 77,
                        "prompt_cache_bytes": None,
                    },
                )
                self.model = FakeModel()
                self.tokenizer = FakeTokenizer()
                self.draft_model = None
                self.model_key = ("fake-model", None, None)

            def load(self, model, adapter=None, draft_model=None):
                return self.model, self.tokenizer

        generator = ResponseGenerator.__new__(ResponseGenerator)
        generator.model_provider = FakeProvider()
        generator.prompt_cache = LRUPromptCache(max_size=10)
        generator.requests = Queue()
        generator._is_distributed = False
        generator._rank = 0
        generator._stop = False
        generator._time_budget = []
        return generator

    def _run_batch_checkpoint_probe(
        self,
        *,
        request,
        tokenized_prompt,
        callback_prompt_end=None,
        has_thinking=False,
        think_start_id=0,
        seeded_entries=None,
    ):
        generator = self._build_response_generator()
        generator._time_budget = [None]
        generator.model_provider.tokenizer.detokenizer = type(
            "FakeDetokenizer",
            (),
            {
                "last_segment": "",
                "add_token": lambda self, token: None,
            },
        )()
        generator.model_provider.tokenizer.has_thinking = has_thinking
        generator.model_provider.tokenizer.think_start_id = think_start_id
        for entry in seeded_entries or []:
            if len(entry) == 3:
                tokens, prompt_cache, checkpoint = entry
            else:
                tokens, prompt_cache = entry
                checkpoint = False
            generator.prompt_cache.insert_cache(
                generator.model_provider.model_key,
                tokens,
                prompt_cache,
                checkpoint=checkpoint,
            )

        request_queue = Queue()
        request_args = self._generation_args()
        request_seen = False
        captured = {"request_queue": request_queue}

        def next_request(timeout=None):
            nonlocal request_seen
            if request_seen:
                return None
            request_seen = True
            return (request_queue, request, request_args)

        class FakeBatchGenerator:
            prompt_cache_nbytes = 0

            def __init__(self, *args, **kwargs):
                captured["constructor_kwargs"] = kwargs
                captured["prompt_checkpoint_callback"] = kwargs.get(
                    "prompt_checkpoint_callback"
                )

            def insert(
                self,
                prompts,
                max_tokens=None,
                caches=None,
                samplers=None,
                logits_processors=None,
                prompt_checkpoints=None,
            ):
                captured["insert_prompts"] = prompts
                captured["insert_caches"] = caches
                captured["insert_prompt_checkpoints"] = prompt_checkpoints
                prompt_checkpoint = None
                if prompt_checkpoints is not None:
                    prompt_checkpoint = prompt_checkpoints[0]
                if callback_prompt_end is not None:
                    captured["effective_prompt_end"] = callback_prompt_end
                elif prompt_checkpoint is None:
                    captured["effective_prompt_end"] = 1
                elif prompt_checkpoint > 0:
                    captured["effective_prompt_end"] = max(
                        1, len(prompts[0]) - prompt_checkpoint
                    )
                else:
                    captured["effective_prompt_end"] = -prompt_checkpoint
                return [123]

            def next(self):
                checkpoint_callback = captured.get("prompt_checkpoint_callback")
                if (
                    checkpoint_callback is not None
                    and captured.get("insert_prompt_checkpoints") is not None
                ):
                    checkpoint_callback(
                        [
                            (
                                123,
                                captured["effective_prompt_end"],
                                iter([MockCache("checkpoint")]),
                            )
                        ]
                    )
                generator._stop = True
                return [
                    type(
                        "BatchResponse",
                        (),
                        {
                            "uid": 123,
                            "token": 0,
                            "logprobs": mx.array([0.0], dtype=mx.float32),
                            "finish_reason": "stop",
                            "prompt_cache": [MockCache("final")],
                        },
                    )()
                ]

        generator._next_request = next_request
        with patch.object(generator, "_tokenize", return_value=tokenized_prompt):
            with patch("mlx_lm.server.BatchGenerator", FakeBatchGenerator):
                generator._generate()

        return generator, captured

    def _run_malformed_then_valid_batch_probe(self, malformed_request):
        generator = self._build_response_generator()
        generator._time_budget = [None]
        generator.model_provider.tokenizer.detokenizer = type(
            "FakeDetokenizer",
            (),
            {
                "last_segment": "",
                "add_token": lambda self, token: None,
            },
        )()
        malformed_queue = Queue()
        valid_queue = Queue()
        request_args = self._generation_args()
        valid_request = CompletionRequest(
            request_type="chat",
            prompt="",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            role_mapping=None,
        )
        requests = [
            (malformed_queue, malformed_request, request_args),
            (valid_queue, valid_request, request_args),
        ]
        captured = {"insert_prompts": []}

        def next_request(timeout=None):
            if requests:
                return requests.pop(0)
            generator._stop = True
            return None

        class FakeBatchGenerator:
            prompt_cache_nbytes = 0

            def __init__(self, *args, **kwargs):
                self._done = False

            def insert(
                self,
                prompts,
                max_tokens=None,
                caches=None,
                samplers=None,
                logits_processors=None,
                prompt_checkpoints=None,
            ):
                captured["insert_prompts"].append(prompts)
                return [123]

            def next(self):
                if self._done:
                    return []
                self._done = True
                return [
                    type(
                        "BatchResponse",
                        (),
                        {
                            "uid": 123,
                            "token": 0,
                            "logprobs": mx.array([0.0], dtype=mx.float32),
                            "finish_reason": "stop",
                            "prompt_cache": [MockCache("final")],
                        },
                    )()
                ]

        generator._next_request = next_request
        with patch.object(
            generator, "_tokenize", side_effect=[[11, 12, 13, 14], [11, 12, 13, 14]]
        ):
            with patch("mlx_lm.server.BatchGenerator", FakeBatchGenerator):
                generator._generate()

        return malformed_queue, valid_queue, captured

    def test_generate_batch_mode_forwards_checkpoint_callback_and_prompt_checkpoints(
        self,
    ):
        request = CompletionRequest(
            request_type="chat",
            prompt="",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            role_mapping=None,
        )
        generator, captured = self._run_batch_checkpoint_probe(
            request=request,
            tokenized_prompt=[11, 12, 13, 14],
            has_thinking=True,
            think_start_id=99,
        )

        self.assertIn("prompt_checkpoint_callback", captured["constructor_kwargs"])
        self.assertEqual(captured["insert_prompt_checkpoints"], [-1])
        self.assertEqual(len(generator.prompt_cache), 2)

        checkpoint_cache, rest = generator.prompt_cache.fetch_nearest_cache(
            generator.model_provider.model_key,
            [11, 12, 13],
        )
        self.assertEqual(rest, [])
        self.assertEqual([cache.value for cache in checkpoint_cache], ["checkpoint"])

    def test_generate_batch_mode_non_thinking_model_stores_checkpoint(self):
        """Non-thinking models still save a checkpoint at -1 (last token).

        This is important for models with non-trimmable caches (ArraysCache)
        where the completion entry can't be rewound, but a checkpoint entry
        at the prompt boundary enables reuse via the shorter-cache path.
        """
        request = CompletionRequest(
            request_type="chat",
            prompt="",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            role_mapping=None,
        )
        generator, captured = self._run_batch_checkpoint_probe(
            request=request,
            tokenized_prompt=[11, 12, 13, 14],
        )

        self.assertIn("prompt_checkpoint_callback", captured["constructor_kwargs"])
        self.assertEqual(captured["insert_prompt_checkpoints"], [-1])
        self.assertEqual(len(generator.prompt_cache), 2)

        checkpoint_cache, rest = generator.prompt_cache.fetch_nearest_cache(
            generator.model_provider.model_key,
            [11, 12, 13],
        )
        self.assertEqual(rest, [])
        self.assertEqual([cache.value for cache in checkpoint_cache], ["checkpoint"])

    def test_generate_batch_mode_does_not_store_checkpoint_for_non_user_terminal_chat(
        self,
    ):
        request = CompletionRequest(
            request_type="chat",
            prompt="",
            messages=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
            tools=None,
            role_mapping=None,
        )
        generator, captured = self._run_batch_checkpoint_probe(
            request=request,
            tokenized_prompt=[11, 12, 13, 14],
        )

        self.assertIn("prompt_checkpoint_callback", captured["constructor_kwargs"])
        self.assertEqual(captured["insert_prompt_checkpoints"], [None])
        self.assertEqual(len(generator.prompt_cache), 1)

        self.assertIsNone(
            generator.prompt_cache._search(
                generator.model_provider.model_key, [11, 12, 13]
            ).exact
        )

    def test_generate_batch_mode_uses_think_start_checkpoint_offset(self):
        request = CompletionRequest(
            request_type="chat",
            prompt="",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            role_mapping=None,
        )
        generator, captured = self._run_batch_checkpoint_probe(
            request=request,
            tokenized_prompt=[11, 12, 99, 13, 14],
            callback_prompt_end=4,
            has_thinking=True,
            think_start_id=99,
        )

        self.assertIn("prompt_checkpoint_callback", captured["constructor_kwargs"])
        self.assertEqual(captured["insert_prompt_checkpoints"], [-4])
        self.assertEqual(len(generator.prompt_cache), 2)

        checkpoint_cache, rest = generator.prompt_cache.fetch_nearest_cache(
            generator.model_provider.model_key,
            [11],
        )
        self.assertEqual(rest, [])
        self.assertEqual([cache.value for cache in checkpoint_cache], ["checkpoint"])

    def test_generate_batch_mode_does_not_store_empty_key_checkpoint_entry(self):
        request = CompletionRequest(
            request_type="chat",
            prompt="",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            role_mapping=None,
        )
        generator, captured = self._run_batch_checkpoint_probe(
            request=request,
            tokenized_prompt=[11, 12, 13, 14],
            callback_prompt_end=4,
            has_thinking=True,
            think_start_id=99,
        )

        self.assertIn("prompt_checkpoint_callback", captured["constructor_kwargs"])
        self.assertEqual(captured["insert_prompt_checkpoints"], [-1])
        self.assertEqual(len(generator.prompt_cache), 1)

        root_cache, root_rest = generator.prompt_cache.fetch_nearest_cache(
            generator.model_provider.model_key, []
        )
        self.assertIsNone(root_cache)
        self.assertEqual(root_rest, [])

        final_cache, final_rest = generator.prompt_cache.fetch_nearest_cache(
            generator.model_provider.model_key,
            [11, 12, 13, 14, 0],
        )
        self.assertEqual(final_rest, [])
        self.assertEqual(final_cache, [MockCache("final")])

    def test_generate_batch_mode_does_not_store_checkpoint_for_text_requests(self):
        request = self._make_text_request(prompt="hello world")
        generator, captured = self._run_batch_checkpoint_probe(
            request=request,
            tokenized_prompt=[11, 12, 13, 14],
        )

        self.assertIn("prompt_checkpoint_callback", captured["constructor_kwargs"])
        self.assertEqual(captured["insert_prompt_checkpoints"], [None])
        self.assertEqual(len(generator.prompt_cache), 1)

        self.assertIsNone(
            generator.prompt_cache._search(
                generator.model_provider.model_key, [11, 12, 13]
            ).exact
        )

    def test_generate_batch_mode_empty_chat_messages_reports_request_error(self):
        request = CompletionRequest(
            request_type="chat",
            prompt="",
            messages=[],
            tools=None,
            role_mapping=None,
        )
        error_queue, valid_queue, captured = self._run_malformed_then_valid_batch_probe(
            request
        )

        error = error_queue.get_nowait()
        self.assertIsInstance(error, ValueError)
        self.assertEqual(str(error), "Chat request messages must be a non-empty list")
        with self.assertRaises(Empty):
            error_queue.get_nowait()
        self.assertEqual(captured["insert_prompts"], [[[11, 12, 13, 14]]])
        valid_ctx = valid_queue.get_nowait()
        valid_response = valid_queue.get_nowait()
        self.assertFalse(isinstance(valid_ctx, Exception))
        self.assertTrue(hasattr(valid_ctx, "prompt"))
        self.assertEqual(valid_response.finish_reason, "stop")
        self.assertIsNone(valid_queue.get_nowait())

    def test_generate_batch_mode_missing_last_role_reports_request_error(self):
        request = CompletionRequest(
            request_type="chat",
            prompt="",
            messages=[{"content": "hello"}],
            tools=None,
            role_mapping=None,
        )
        error_queue, valid_queue, captured = self._run_malformed_then_valid_batch_probe(
            request
        )

        error = error_queue.get_nowait()
        self.assertIsInstance(error, ValueError)
        self.assertEqual(str(error), "Chat request last message must include a role")
        with self.assertRaises(Empty):
            error_queue.get_nowait()
        self.assertEqual(captured["insert_prompts"], [[[11, 12, 13, 14]]])
        valid_ctx = valid_queue.get_nowait()
        valid_response = valid_queue.get_nowait()
        self.assertFalse(isinstance(valid_ctx, Exception))
        self.assertTrue(hasattr(valid_ctx, "prompt"))
        self.assertEqual(valid_response.finish_reason, "stop")
        self.assertIsNone(valid_queue.get_nowait())

    def test_generate_batch_mode_does_not_forward_impossible_checkpoint_with_warm_cache(
        self,
    ):
        request = CompletionRequest(
            request_type="chat",
            prompt="",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            role_mapping=None,
        )
        generator, captured = self._run_batch_checkpoint_probe(
            request=request,
            tokenized_prompt=[11, 12, 99, 13, 14],
            callback_prompt_end=4,
            has_thinking=True,
            think_start_id=99,
            seeded_entries=[([11, 12], [MockCache("seed")])],
        )

        self.assertIn("prompt_checkpoint_callback", captured["constructor_kwargs"])
        self.assertEqual(captured["insert_prompts"], [[99, 13, 14]])
        self.assertEqual(captured["insert_prompt_checkpoints"], [None])

    def test_localize_prompt_checkpoint_at_prompt_start_suppressed_with_warm_cache(
        self,
    ):
        """When checkpoint_position = -len(prompt) (position 0 = start of
        prompt) and a warm cache covers a prefix, _localize_prompt_checkpoint
        returns None because the checkpoint falls before the reused region."""
        generator = self._build_response_generator()
        prompt = [11, 12, 13, 14]
        rest = [13, 14]  # cache hit covered [11, 12]

        # checkpoint_position = -4 means position 0 (start of prompt).
        # rest_offset = 4 - 2 = 2, checkpoint_prefix = 0 < 2 → suppressed.
        result = generator._localize_prompt_checkpoint(prompt, rest, -len(prompt))
        self.assertIsNone(result)

        # Also verify that -1 (last token) is NOT suppressed in same scenario.
        result2 = generator._localize_prompt_checkpoint(prompt, rest, -1)
        self.assertIsNotNone(result2)
        self.assertEqual(result2, -1)

    def test_generate_batch_mode_real_generator_stores_checkpoint_cache(self):
        class DeterministicBatchModel:
            layers = [object()]

            def make_cache(self):
                return [KVCache()]

            def __call__(self, input_tokens, cache=None, input_embeddings=None):
                if cache is not None:
                    for layer_cache in cache:
                        kv = mx.zeros(
                            (input_tokens.shape[0], 1, input_tokens.shape[1], 1),
                            dtype=mx.float32,
                        )
                        layer_cache.update_and_fetch(kv, kv)
                batch, seq_len = input_tokens.shape
                vocab_size = 4
                logits = -1000.0 * mx.ones((vocab_size,), dtype=mx.float32)
                logits = logits + (2000.0 * (mx.arange(vocab_size) == 0))
                return mx.broadcast_to(logits, (batch, seq_len, vocab_size))

        generator = self._build_response_generator()
        generator._time_budget = [None]
        generator.model_provider.model = DeterministicBatchModel()
        generator.model_provider.tokenizer.has_thinking = True
        generator.model_provider.tokenizer.think_start_id = 99
        generator.model_provider.tokenizer.detokenizer = type(
            "FakeDetokenizer",
            (),
            {
                "last_segment": "",
                "add_token": lambda self, token: None,
            },
        )()
        request_queue = Queue()
        request_args = self._generation_args()
        request_args.max_tokens = 1
        request = CompletionRequest(
            request_type="chat",
            prompt="",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            role_mapping=None,
        )
        request_seen = False
        original_next = BatchGenerator.next

        def next_request(timeout=None):
            nonlocal request_seen
            if request_seen:
                return None
            request_seen = True
            return (request_queue, request, request_args)

        def stopping_next(batch_generator):
            responses = original_next(batch_generator)
            if responses and all(r.finish_reason is not None for r in responses):
                generator._stop = True
            return responses

        generator._next_request = next_request
        with patch.object(generator, "_tokenize", return_value=[11, 12, 13, 14]):
            with patch.object(BatchGenerator, "next", new=stopping_next):
                generator._generate()

        self.assertEqual(len(generator.prompt_cache), 2)
        checkpoint_cache, rest = generator.prompt_cache.fetch_nearest_cache(
            generator.model_provider.model_key,
            [11, 12, 13],
        )
        self.assertEqual(rest, [])
        self.assertIsNotNone(checkpoint_cache)
        self.assertEqual([layer.offset for layer in checkpoint_cache], [3])
        self.assertEqual(len(generator.prompt_cache), 2)

    def test_generate_batch_mode_exact_checkpoint_hit_does_not_forward_empty_prompt(
        self,
    ):
        request = CompletionRequest(
            request_type="chat",
            prompt="",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            role_mapping=None,
        )
        generator, captured = self._run_batch_checkpoint_probe(
            request=request,
            tokenized_prompt=[11, 12, 13],
            seeded_entries=[([11, 12, 13], [MockCache("checkpoint")], True)],
        )

        self.assertTrue(captured["insert_prompts"][0])
        request_queue = captured["request_queue"]
        ctx = request_queue.get_nowait()
        response = request_queue.get_nowait()
        self.assertFalse(isinstance(ctx, Exception))
        self.assertEqual(response.finish_reason, "stop")
        self.assertIsNone(request_queue.get_nowait())

    def test_serve_single_exact_checkpoint_hit_does_not_forward_empty_prompt(self):
        generator = self._build_response_generator()
        generator.prompt_cache.insert_cache(
            generator.model_provider.model_key,
            [11, 12, 13],
            [MockCache("checkpoint")],
            checkpoint=True,
        )
        request_queue = Queue()
        gen_result = type(
            "GenResult",
            (),
            {
                "text": "x",
                "token": 0,
                "logprobs": mx.array([0.0], dtype=mx.float32),
                "finish_reason": "stop",
            },
        )()

        def stream_generate_probe(**stream_kwargs):
            self.assertTrue(stream_kwargs["prompt"])
            yield gen_result

        with patch("mlx_lm.server.stream_generate", side_effect=stream_generate_probe):
            with patch.object(generator, "_tokenize", return_value=[11, 12, 13]):
                generator._serve_single(
                    (request_queue, self._make_text_request(), self._generation_args())
                )

        ctx = request_queue.get_nowait()
        response = request_queue.get_nowait()
        self.assertFalse(isinstance(ctx, Exception))
        self.assertFalse(isinstance(response, Exception))
        self.assertEqual(response.finish_reason, "stop")
        self.assertIsNone(request_queue.get_nowait())

    def test_generate_batch_mode_real_generator_suppresses_impossible_warm_cache_checkpoint(
        self,
    ):
        class DeterministicBatchModel:
            layers = [object()]

            def make_cache(self):
                return [KVCache()]

            def __call__(self, input_tokens, cache=None, input_embeddings=None):
                if cache is not None:
                    for layer_cache in cache:
                        kv = mx.zeros(
                            (input_tokens.shape[0], 1, input_tokens.shape[1], 1),
                            dtype=mx.float32,
                        )
                        layer_cache.update_and_fetch(kv, kv)
                batch, seq_len = input_tokens.shape
                vocab_size = 4
                logits = -1000.0 * mx.ones((vocab_size,), dtype=mx.float32)
                logits = logits + (2000.0 * (mx.arange(vocab_size) == 0))
                return mx.broadcast_to(logits, (batch, seq_len, vocab_size))

        generator = self._build_response_generator()
        generator._time_budget = [None]
        generator.model_provider.model = DeterministicBatchModel()
        generator.model_provider.tokenizer.detokenizer = type(
            "FakeDetokenizer",
            (),
            {
                "last_segment": "",
                "add_token": lambda self, token: None,
            },
        )()
        generator.model_provider.tokenizer.has_thinking = True
        generator.model_provider.tokenizer.think_start_id = 99

        seeded_cache = generator.model_provider.model.make_cache()
        generator.model_provider.model(
            mx.array([[11, 12]], dtype=mx.uint32), cache=seeded_cache
        )
        generator.prompt_cache.insert_cache(
            generator.model_provider.model_key,
            [11, 12],
            seeded_cache,
        )

        request_queue = Queue()
        request_args = self._generation_args()
        request_args.max_tokens = 1
        request = CompletionRequest(
            request_type="chat",
            prompt="",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            role_mapping=None,
        )
        request_seen = False
        original_next = BatchGenerator.next

        def next_request(timeout=None):
            nonlocal request_seen
            if request_seen:
                return None
            request_seen = True
            return (request_queue, request, request_args)

        def stopping_next(batch_generator):
            responses = original_next(batch_generator)
            if responses and all(r.finish_reason is not None for r in responses):
                generator._stop = True
            return responses

        generator._next_request = next_request
        with patch.object(generator, "_tokenize", return_value=[11, 12, 99, 13, 14]):
            with patch.object(BatchGenerator, "next", new=stopping_next):
                generator._generate()

        self.assertEqual(len(generator.prompt_cache), 1)
        self.assertIsNone(
            generator.prompt_cache._search(
                generator.model_provider.model_key, [11]
            ).exact
        )


if __name__ == "__main__":
    unittest.main()
