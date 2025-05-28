# Copyright © 2024 Apple Inc.

import http
import json
import threading
import unittest

import requests

from mlx_lm.server import APIHandler
from mlx_lm.utils import load


class DummyModelProvider:
    def __init__(self, with_draft=False):
        HF_MODEL_PATH = "mlx-community/Qwen1.5-0.5B-Chat-4bit"
        self.model, self.tokenizer = load(HF_MODEL_PATH)
        self.model_key = (HF_MODEL_PATH, None)

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
                "num_draft_tokens": 3,
                "temp": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "min_p": 0.0,
                "max_tokens": 512,
                "chat_template_args": {},
            },
        )

        if with_draft:
            # Use the same model as the draft model for testing
            self.draft_model, _ = load(HF_MODEL_PATH)
            self.draft_model_key = HF_MODEL_PATH

    def load(self, model, adapter=None, draft_model=None):
        assert model in ["default_model", "chat_model"]
        return self.model, self.tokenizer


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_provider = DummyModelProvider()
        cls.server_address = ("localhost", 0)
        cls.httpd = http.server.HTTPServer(
            cls.server_address,
            lambda *args, **kwargs: APIHandler(cls.model_provider, *args, **kwargs),
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
            "stop": "stop sequence",
        }

        response = requests.post(url, json=post_data)

        response_body = response.text

        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)

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
        cls.model_provider = DummyModelProvider(with_draft=True)
        cls.server_address = ("localhost", 0)
        cls.httpd = http.server.HTTPServer(
            cls.server_address,
            lambda *args, **kwargs: APIHandler(cls.model_provider, *args, **kwargs),
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


# --- Tests for get_prompt_cache ---

from unittest.mock import MagicMock, patch

from mlx_lm.server import PromptCache


class TestGetPromptCache(unittest.TestCase):

    def setUp(self):
        """Set up mocks and a handler instance for each test."""
        self.mock_model_provider = MagicMock()
        # Simulate tokenizer needed for decoding in original debug logs (though not strictly needed for cache logic)
        self.mock_model_provider.tokenizer = MagicMock()
        self.mock_model_provider.tokenizer.decode = lambda x: f"decoded({x})"
        self.mock_model_provider.model_key = ("model_v1", None, None)
        self.mock_model_provider.draft_model = None  # Start without draft model

        # --- Prevent BaseHTTPRequestHandler.__init__ from running ---
        # It tries to handle a request immediately, which fails with mocks.
        # We only need the APIHandler instance with its attributes set.
        with patch(
            "http.server.BaseHTTPRequestHandler.__init__", lambda *args, **kwargs: None
        ):
            # APIHandler init still requires args for BaseHTTPRequestHandler signature,
            # but they won't be used by the patched __init__.
            mock_request = MagicMock()
            mock_client_address = ("127.0.0.1", 8080)
            mock_server = MagicMock()

            self.prompt_cache_instance = PromptCache()
            self.handler = APIHandler(
                self.mock_model_provider,
                mock_request,
                mock_client_address,
                mock_server,
                prompt_cache=self.prompt_cache_instance,  # Inject our cache instance
            )
            # Manually set attributes usually set by APIHandler.__init__ if needed
            # self.handler.created = MagicMock()
            # self.handler.system_fingerprint = MagicMock()
            # (Not strictly necessary for get_prompt_cache testing)

    @patch("mlx_lm.server.make_prompt_cache")
    def test_initial_request_empty_cache(self, mock_make_cache):
        """Test first request when the cache is empty."""
        mock_make_cache.return_value = "new_cache_obj"
        prompt = [1, 2, 3]

        processed_prompt = self.handler.get_prompt_cache(prompt)

        self.assertEqual(processed_prompt, [1, 2, 3])
        self.assertEqual(self.handler.prompt_cache.tokens, [1, 2, 3])
        self.assertEqual(self.handler.prompt_cache.cache, "new_cache_obj")
        self.assertEqual(self.handler.prompt_cache.model_key, ("model_v1", None, None))
        mock_make_cache.assert_called_once()

    @patch("mlx_lm.server.trim_prompt_cache")
    @patch("mlx_lm.server.can_trim_prompt_cache", return_value=True)
    def test_identical_request_full_hit(self, mock_can_trim, mock_trim_cache):
        """Test when the new prompt is identical to the cached one."""
        self.handler.prompt_cache.tokens = [1, 2, 3]
        self.handler.prompt_cache.model_key = ("model_v1", None, None)
        self.handler.prompt_cache.cache = "existing_cache_obj"
        prompt = [1, 2, 3]

        # Mock common_prefix_len to return the full length
        with patch("mlx_lm.server.common_prefix_len", return_value=3):
            processed_prompt = self.handler.get_prompt_cache(prompt)

        mock_trim_cache.assert_called_once_with("existing_cache_obj", 1)
        self.assertEqual(processed_prompt, [3])
        self.assertEqual(self.handler.prompt_cache.tokens, [1, 2, 3])

    def test_cache_is_prefix(self):
        """Test when the cached prompt is a prefix of the new prompt."""
        self.handler.prompt_cache.tokens = [1, 2, 3]
        self.handler.prompt_cache.model_key = ("model_v1", None, None)
        self.handler.prompt_cache.cache = "existing_cache_obj"
        prompt = [1, 2, 3, 4, 5]

        with patch("mlx_lm.server.common_prefix_len", return_value=3):
            processed_prompt = self.handler.get_prompt_cache(prompt)

        # Should process the suffix, cache tokens updated
        self.assertEqual(processed_prompt, [4, 5])
        self.assertEqual(self.handler.prompt_cache.tokens, [1, 2, 3, 4, 5])
        self.assertEqual(self.handler.prompt_cache.cache, "existing_cache_obj")

    @patch("mlx_lm.server.trim_prompt_cache")
    @patch("mlx_lm.server.can_trim_prompt_cache", return_value=True)
    def test_partial_match_trim_success(self, mock_can_trim, mock_trim_cache):
        """Test partial match where cache is longer and trimming succeeds."""
        self.handler.prompt_cache.tokens = [1, 2, 3, 4, 5]
        self.handler.prompt_cache.model_key = ("model_v1", None, None)
        self.handler.prompt_cache.cache = "existing_cache_obj"
        prompt = [1, 2, 3, 6, 7]  # Diverges after token 3

        with patch("mlx_lm.server.common_prefix_len", return_value=3):
            processed_prompt = self.handler.get_prompt_cache(prompt)

        # Should process the new suffix, cache trimmed and updated
        self.assertEqual(processed_prompt, [6, 7])
        self.assertEqual(self.handler.prompt_cache.tokens, [1, 2, 3, 6, 7])
        mock_can_trim.assert_called_once_with("existing_cache_obj")
        # Called with cache object and num_to_trim (5 - 3 = 2)
        mock_trim_cache.assert_called_once_with("existing_cache_obj", 2)
        self.assertEqual(
            self.handler.prompt_cache.cache, "existing_cache_obj"
        )  # Cache obj itself isn't changed by mock

    @patch("mlx_lm.server.make_prompt_cache")
    @patch("mlx_lm.server.trim_prompt_cache")
    @patch("mlx_lm.server.can_trim_prompt_cache", return_value=False)
    def test_partial_match_trim_fail(
        self, mock_can_trim, mock_trim_cache, mock_make_cache
    ):
        """Test partial match where cache is longer but trimming fails."""
        mock_make_cache.return_value = "new_cache_obj_on_reset"
        self.handler.prompt_cache.tokens = [1, 2, 3, 4, 5]
        self.handler.prompt_cache.model_key = ("model_v1", None, None)
        self.handler.prompt_cache.cache = "existing_cache_obj"
        prompt = [1, 2, 3, 6, 7]  # Diverges after token 3

        with patch("mlx_lm.server.common_prefix_len", return_value=3):
            processed_prompt = self.handler.get_prompt_cache(prompt)

        # Should process the full prompt, cache reset
        self.assertEqual(processed_prompt, [1, 2, 3, 6, 7])
        self.assertEqual(self.handler.prompt_cache.tokens, [1, 2, 3, 6, 7])
        mock_can_trim.assert_called_once_with("existing_cache_obj")
        mock_trim_cache.assert_not_called()
        mock_make_cache.assert_called_once()  # Cache was reset
        self.assertEqual(self.handler.prompt_cache.cache, "new_cache_obj_on_reset")

    @patch("mlx_lm.server.make_prompt_cache")
    def test_no_common_prefix(self, mock_make_cache):
        """Test when there is no common prefix between cache and prompt."""
        mock_make_cache.return_value = "new_cache_obj"
        self.handler.prompt_cache.tokens = [1, 2, 3]
        self.handler.prompt_cache.model_key = ("model_v1", None, None)
        self.handler.prompt_cache.cache = "existing_cache_obj"
        prompt = [4, 5, 6]

        with patch("mlx_lm.server.common_prefix_len", return_value=0):
            processed_prompt = self.handler.get_prompt_cache(prompt)

        # Should process the full prompt, cache reset
        self.assertEqual(processed_prompt, [4, 5, 6])
        self.assertEqual(self.handler.prompt_cache.tokens, [4, 5, 6])
        mock_make_cache.assert_called_once()
        self.assertEqual(self.handler.prompt_cache.cache, "new_cache_obj")

    @patch("mlx_lm.server.make_prompt_cache")
    def test_model_changed(self, mock_make_cache):
        """Test cache reset when the model key changes."""
        mock_make_cache.return_value = "new_cache_obj_model_change"
        self.handler.prompt_cache.tokens = [1, 2, 3]
        self.handler.prompt_cache.model_key = ("model_v1", None, None)  # Original key
        self.handler.prompt_cache.cache = "existing_cache_obj"

        # Simulate model provider having a new key
        self.mock_model_provider.model_key = ("model_v2", None, None)
        prompt = [1, 2, 3, 4]

        # No need to mock common_prefix_len, model check happens first
        processed_prompt = self.handler.get_prompt_cache(prompt)

        # Should process the full prompt, cache reset
        self.assertEqual(processed_prompt, [1, 2, 3, 4])
        self.assertEqual(self.handler.prompt_cache.tokens, [1, 2, 3, 4])
        mock_make_cache.assert_called_once()
        self.assertEqual(self.handler.prompt_cache.cache, "new_cache_obj_model_change")
        self.assertEqual(self.handler.prompt_cache.model_key, ("model_v2", None, None))


if __name__ == "__main__":
    unittest.main()
