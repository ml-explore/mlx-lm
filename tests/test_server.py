# Copyright © 2024 Apple Inc.

import http
import io
import json
import threading
import unittest
from unittest.mock import Mock, patch

import mlx.core as mx
import requests

from mlx_lm.models.cache import KVCache
from mlx_lm.server import (
    APIHandler,
    CompletionRequest,
    GenerationArguments,
    GenerationContext,
    LogitsProcessorArguments,
    LRUPromptCache,
    ModelDescription,
    Response as ServerResponse,
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
                "prompt_cache_warmup": False,
                "prompt_cache_total_bytes": None,
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


class MockPromptCacheManager:
    def __init__(self, cache, rest):
        self.cache = cache
        self.rest = rest
        self.fetch_calls = []
        self.insert_calls = []

    def log_cache_stats(self):
        return None

    def fetch_nearest_cache(self, model_key, tokens):
        self.fetch_calls.append((model_key, tokens))
        return self.cache, self.rest

    def insert_cache(self, model_key, tokens, prompt_cache, checkpoint=False):
        self.insert_calls.append((model_key, tokens, prompt_cache, checkpoint))


class SequencedTokenizer:
    def __init__(self, prompts):
        self.prompts = iter(prompts)
        self.has_chat_template = True
        self.has_tool_calling = True

    def apply_chat_template(
        self,
        messages,
        tools=None,
        add_generation_prompt=True,
        tokenize=True,
        **kwargs,
    ):
        return next(self.prompts)


class TestPromptCacheWarmup(unittest.TestCase):
    def make_args(self):
        return GenerationArguments(
            model=ModelDescription(model="default_model", adapter=None, draft=None),
            sampling=SamplingArguments(
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                min_p=0.0,
                xtc_probability=0.0,
                xtc_threshold=0.1,
            ),
            logits=LogitsProcessorArguments(
                logit_bias=None,
                repetition_penalty=1.0,
                repetition_context_size=20,
            ),
            stop_words=[],
            max_tokens=16,
            num_draft_tokens=0,
            logprobs=False,
            top_logprobs=0,
            seed=None,
            chat_template_kwargs={},
        )

    def make_generator(self, prompt_cache):
        cli_args = type(
            "obj",
            (object,),
            {
                "chat_template_args": {},
                "draft_model": None,
                "prefill_step_size": 16,
                "prompt_cache_warmup": True,
            },
        )
        model_provider = type(
            "obj",
            (object,),
            {
                "cli_args": cli_args,
                "draft_model": None,
                "model": object(),
                "model_key": ("default_model", None, None),
                "load": Mock(
                    return_value=(object(), SequencedTokenizer(([1], [1])))
                ),
            },
        )
        generator = ResponseGenerator.__new__(ResponseGenerator)
        generator.model_provider = model_provider
        generator.prompt_cache = prompt_cache
        generator._prompt_cache_warmup = None
        generator._prompt_cache_warmup_lock = threading.Lock()
        generator._is_distributed = False
        return generator

    def make_context(self, *, has_thinking=False):
        return GenerationContext(
            has_tool_calling=True,
            tool_call_start="<tool>",
            tool_call_end="</tool>",
            tool_parser=lambda *_: {"name": "demo", "arguments": {"value": 1}},
            has_thinking=has_thinking,
            think_start_id=1,
            think_end_id=2,
            think_end="</think>",
            eos_token_ids=set(),
            stop_token_sequences=[],
            prompt=[1] if has_thinking else [3],
        )

    def make_handler(self, response_generator):
        handler = APIHandler.__new__(APIHandler)
        handler.created = 0
        handler.response_generator = response_generator
        handler.system_fingerprint = "fp"
        handler.stream = False
        handler.stream_options = None
        handler.request_id = "chatcmpl-test"
        handler.object_type = "chat.completion"
        handler.requested_model = "chat_model"
        handler.requested_draft_model = "default_model"
        handler.adapter = None
        handler.max_tokens = 16
        handler.temperature = 0.0
        handler.top_p = 1.0
        handler.top_k = 0
        handler.min_p = 0.0
        handler.repetition_penalty = 1.0
        handler.repetition_context_size = 20
        handler.xtc_probability = 0.0
        handler.xtc_threshold = 0.1
        handler.logit_bias = None
        handler.logprobs = False
        handler.top_logprobs = 0
        handler.seed = None
        handler.chat_template_kwargs = {}
        handler.num_draft_tokens = 0
        handler.wfile = io.BytesIO()
        handler._set_completion_headers = Mock()
        handler._set_stream_headers = Mock()
        handler.send_header = Mock()
        handler.end_headers = Mock()
        return handler

    def test_handle_completion_enqueues_warmup_for_reasoning_reply(self):
        response_generator = Mock()
        response_generator.generate.return_value = (
            self.make_context(has_thinking=True),
            iter(
                [
                    ServerResponse("trace", 1, 0.0, None, ()),
                    ServerResponse("</think>", 2, 0.0, None, ()),
                    ServerResponse("answer", 3, 0.0, "stop", ()),
                ]
            ),
        )
        handler = self.make_handler(response_generator)
        request = CompletionRequest(
            "chat",
            "",
            [{"role": "user", "content": "Hello"}],
            None,
            None,
        )

        handler.handle_completion(request, [])

        response_generator.enqueue_prompt_cache_warmup.assert_called_once()
        assistant_message = response_generator.enqueue_prompt_cache_warmup.call_args.args[
            2
        ]
        self.assertEqual(assistant_message["role"], "assistant")
        self.assertEqual(assistant_message["content"], "answer")
        self.assertEqual(assistant_message["reasoning"], "trace")
        self.assertEqual(assistant_message["reasoning_content"], "trace")

    def test_handle_completion_skips_warmup_for_tool_call_finish_reason(self):
        response_generator = Mock()
        response_generator.generate.return_value = (
            self.make_context(),
            iter(
                [
                    ServerResponse("<tool>", 1, 0.0, None, ()),
                    ServerResponse('{"value": 1}', 2, 0.0, None, ()),
                    ServerResponse("</tool>", 3, 0.0, "tool_calls", ()),
                ]
            ),
        )
        handler = self.make_handler(response_generator)
        request = CompletionRequest(
            "chat",
            "",
            [{"role": "user", "content": "Hello"}],
            [{"type": "function", "function": {"name": "demo"}}],
            None,
        )

        handler.handle_completion(request, [])

        response_generator.enqueue_prompt_cache_warmup.assert_not_called()

    def test_run_prompt_cache_warmup_prefills_partial_cache_hit(self):
        prompt_cache = MockPromptCacheManager(["base-cache"], [7])
        generator = self.make_generator(prompt_cache)
        generator.model_provider.load = Mock(
            return_value=(
                generator.model_provider.model,
                SequencedTokenizer(([1, 2, 3, 7, 10], [1, 2, 3, 7, 20])),
            )
        )
        request = CompletionRequest(
            "chat",
            "",
            [
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
            None,
            None,
        )
        generator.enqueue_prompt_cache_warmup(
            request,
            self.make_args(),
            {"role": "assistant", "content": "Hi"},
        )

        with patch("mlx_lm.server.generate_step", return_value=iter(())) as generate_step:
            served = generator._run_prompt_cache_warmup()

        self.assertTrue(served)
        self.assertEqual(
            prompt_cache.fetch_calls,
            [(("default_model", None, None), [1, 2, 3, 7])],
        )
        generate_step.assert_called_once()
        self.assertEqual(len(prompt_cache.insert_calls), 1)
        model_key, tokens, _, checkpoint = prompt_cache.insert_calls[0]
        self.assertEqual(model_key, ("default_model", None, None))
        self.assertEqual(tokens, [1, 2, 3, 7])
        self.assertTrue(checkpoint)

    def test_run_prompt_cache_warmup_skips_exact_cache_hit(self):
        prompt_cache = MockPromptCacheManager(["exact-cache"], [])
        generator = self.make_generator(prompt_cache)
        generator.model_provider.load = Mock(
            return_value=(
                generator.model_provider.model,
                SequencedTokenizer(([1, 2, 3, 9], [1, 2, 3, 8])),
            )
        )
        request = CompletionRequest(
            "chat",
            "",
            [{"role": "user", "content": "Hello"}],
            None,
            None,
        )
        generator.enqueue_prompt_cache_warmup(
            request,
            self.make_args(),
            {"role": "assistant", "content": "Hi"},
        )

        with patch("mlx_lm.server.generate_step", return_value=iter(())) as generate_step:
            served = generator._run_prompt_cache_warmup()

        self.assertTrue(served)
        self.assertEqual(
            prompt_cache.fetch_calls,
            [(("default_model", None, None), [1, 2, 3])],
        )
        generate_step.assert_not_called()
        self.assertEqual(prompt_cache.insert_calls, [])


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

        cache.insert_cache(model, [4, 5], [MockCache("test4")], checkpoint=True)
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


if __name__ == "__main__":
    unittest.main()
