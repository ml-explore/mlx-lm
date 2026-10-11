# Copyright © 2024 Apple Inc.

import http
import io
import json
import threading
import sys
import time
import types
import unittest
from queue import Queue
from unittest import mock

import mlx.core as mx
import requests

from mlx_lm.generate import TextStateMachine, generate_step, maybe_quantize_kv_cache
from mlx_lm.models.cache import (
    CacheList,
    KVCache,
    QuantizedKVCache,
    QuantizedRotatingKVCache,
    RotatingKVCache,
    make_prompt_cache,
    stays_trimmable,
)
from mlx_lm.sample_utils import make_logits_processors
from mlx_lm.server import (
    APIHandler,
    GenerationContext,
    LRUPromptCache,
    ResponseGenerator,
    SamplingArguments,
    ToolCallFormatter,
    _make_sampler,
    checkpoint_room,
)
from mlx_lm.tool_parsers import pythonic
from mlx_lm.utils import load


class DummyModelProvider:
    def __init__(self, with_draft=False, kv_bits=None, quantized_kv_start=0):
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
        state, text, _ = sm.step(state, "hi <tool_call>body</tool_call> bye")
        state, rest, _ = sm.flush(state)
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
        state, t1, _ = sm.step(state, "<tool_call>call1</tool_call>")
        state, t2, _ = sm.step(state, "<tool_call>call2</tool_call>")
        state, rest, _ = sm.flush(state)
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
        state, text, _ = sm.step(state, "hello STOP world")
        state, rest, _ = sm.flush(state)
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


def _complete(port, prompt, max_tokens=8):
    url = f"http://localhost:{port}/v1/completions"
    body = {"model": "default_model", "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0}
    response = requests.post(url, json=body)
    return response.status_code, json.loads(response.text)



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

    def test_exact_prompt_cache_hit_keeps_serving(self):
        # A prompt equal to a stored prompt+completion is an exact cache
        # hit; the batched path used to strip every segment and kill the
        # generation thread ("empty prompt") for all later requests.
        status, first = _complete(self.port, "Once upon a time")
        self.assertEqual(status, 200)
        prompt = "Once upon a time" + first["choices"][0]["text"]
        status, second = _complete(self.port, prompt)
        self.assertEqual(status, 200)
        self.assertIn("choices", second)
        self.assertTrue(self.response_generator.generation_available())
        status, _ = _complete(self.port, "Hello")
        self.assertEqual(status, 200)

    def test_tool_call_keeps_stream_alive(self):
        # A tool call is sent only once it closes; while a long one is being
        # generated the stream carries keepalive comments, or a client's
        # stall timeout cuts it off mid-call.
        sm = TextStateMachine(
            {
                "normal": [("<tool_call>", "tool")],
                "tool": [("</tool_call>", "normal")],
            }
        )
        ctx = GenerationContext(
            has_tool_calling=True,
            has_thinking=False,
            tool_parser=pythonic.parse_tool_call,
            text_sm=sm,
            initial_state="normal",
            prompt=[1, 2, 3],
        )
        pieces = ["<tool_call>", "[get_weather(", 'city="Pa', 'ris"', ")]"]
        pieces.append("</tool_call>")
        gens = [
            types.SimpleNamespace(
                text=t, token=i, logprob=0.0, finish_reason=None, top_tokens=()
            )
            for i, t in enumerate(pieces)
        ]
        gens.append(
            types.SimpleNamespace(
                text="", token=99, logprob=0.0, finish_reason="stop", top_tokens=()
            )
        )
        clock = iter(range(0, 1000, 3))
        with mock.patch.object(
            self.response_generator, "generate", return_value=(ctx, iter(gens))
        ), mock.patch("mlx_lm.server.time.monotonic", lambda: next(clock)):
            response = requests.post(
                f"http://localhost:{self.port}/v1/chat/completions",
                json={
                    "model": "default_model",
                    "messages": [{"role": "user", "content": "weather?"}],
                    "stream": True,
                },
            )
        self.assertEqual(response.status_code, 200)
        events = response.text.split("\n\n")
        self.assertIn(": keepalive tool", events)
        data = [json.loads(e[6:]) for e in events if e.startswith("data: {")]
        calls = [
            call
            for d in data
            for choice in d["choices"]
            for call in choice["delta"].get("tool_calls") or []
        ]
        self.assertEqual([c["function"]["name"] for c in calls], ["get_weather"])
        # The keepalives come while the call is generated, before it's sent.
        first_call = next(i for i, e in enumerate(events) if "get_weather" in e)
        self.assertLess(events.index(": keepalive tool"), first_call)

    def test_thinking_budget_closes_the_reasoning(self):
        # At the budget the server closes the reasoning itself and the
        # answer goes on as a new request of the prompt, the tokens so far
        # and the close -- not a logits processor, which would turn off
        # MTP and speculative decoding.
        sm = TextStateMachine({"normal": [("<think>", "reasoning")], "reasoning": [("</think>", "normal")]})
        def ctx(prompt):
            return GenerationContext(has_tool_calling=False, has_thinking=True, tool_parser=None,
                                     text_sm=sm, initial_state="reasoning", prompt=prompt,
                                     model_key=self.response_generator.model_provider.model_key)
        def gen(text, token, finish=None):
            return types.SimpleNamespace(text=text, token=token, logprob=0.0, finish_reason=finish, top_tokens=())
        thinking = [gen(" step", 10 + i) for i in range(50)]
        answer = [gen("Answer", 40), gen("", 41, "stop")]
        first = ctx([1, 2, 3])
        calls = []
        def fake_generate(request, args, progress_callback=None):
            calls.append((request, args))
            return (first, iter(thinking)) if len(calls) == 1 else (ctx(request.prompt_tokens), iter(answer))
        tokenizer = types.SimpleNamespace(has_thinking=True, think_end="</think>",
                                          encode=lambda s, add_special_tokens=False: [7, 8])
        with mock.patch.object(self.response_generator, "generate", side_effect=fake_generate), \
                mock.patch.object(self.response_generator.model_provider, "tokenizer", tokenizer):
            response = requests.post(
                f"http://localhost:{self.port}/v1/chat/completions",
                json={"model": "default_model", "messages": [{"role": "user", "content": "hi"}],
                      "thinking_budget": 3, "max_tokens": 100},
            )
        self.assertEqual(response.status_code, 200, response.text)
        message = response.json()["choices"][0]["message"]
        self.assertEqual(message["content"].strip(), "Answer")
        self.assertEqual(message["reasoning"], " step step step\n\nI have to answer now.\n")
        self.assertEqual(len(calls), 2)
        carried, args = calls[1]
        self.assertEqual(carried.prompt_tokens, [1, 2, 3, 10, 11, 12, 7, 8])
        self.assertIsNone(args.thinking_budget)
        self.assertEqual(args.max_tokens, 100 - 5)
        # Usage counts the request's own prompt, not the carry-on's.
        self.assertEqual(response.json()["usage"]["prompt_tokens"], 3)

    def _budget_run(self, thinking, budget, max_tokens=100, answer=None):
        sm = TextStateMachine({"normal": [("<think>", "reasoning")], "reasoning": [("</think>", "normal")]})
        def ctx(prompt):
            return GenerationContext(has_tool_calling=False, has_thinking=True, tool_parser=None,
                                     text_sm=sm, initial_state="reasoning", prompt=prompt,
                                     model_key=self.response_generator.model_provider.model_key)
        def gen(text, token, finish=None):
            return types.SimpleNamespace(text=text, token=token, logprob=0.0, finish_reason=finish, top_tokens=())
        gens = [gen(t, 10 + i) for i, t in enumerate(thinking)]
        answer = answer or [gen("Answer", 40), gen("", 41, "stop")]
        calls = []
        def fake_generate(request, args, progress_callback=None):
            calls.append((request, args))
            return (ctx([1, 2, 3]), iter(gens)) if len(calls) == 1 else (ctx(request.prompt_tokens), iter(answer))
        tokenizer = types.SimpleNamespace(has_thinking=True, think_end="</think>",
                                          encode=lambda s, add_special_tokens=False: [7, 8])
        with mock.patch.object(self.response_generator, "generate", side_effect=fake_generate), \
                mock.patch.object(self.response_generator.model_provider, "tokenizer", tokenizer):
            response = requests.post(
                f"http://localhost:{self.port}/v1/chat/completions",
                json={"model": "default_model", "messages": [{"role": "user", "content": "hi"}],
                      "thinking_budget": budget, "max_tokens": max_tokens},
            )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json(), calls

    def test_thinking_budget_edges(self):
        # Zero: closed before any reasoning token.
        out, calls = self._budget_run([" a", " b"], 0)
        self.assertEqual(out["choices"][0]["message"]["reasoning"], "\n\nI have to answer now.\n")
        self.assertEqual(calls[1][0].prompt_tokens, [1, 2, 3, 7, 8])
        # The close uses what max_tokens leaves: the answer ends there.
        out, calls = self._budget_run([" a", " b", " c"], 2, max_tokens=4)
        self.assertEqual(len(calls), 1)
        self.assertEqual(out["choices"][0]["finish_reason"], "length")
        # Not in the middle of a close the model writes itself.
        # (at three, "</th" is held as a partial close: the next token waits)
        out, calls = self._budget_run([" a", " b", "</th", "ink>", " ok"], 3)
        self.assertEqual(len(calls), 1, "the model closed it: no carry-on")
        self.assertIn("ok", out["choices"][0]["message"]["content"])

    def test_thinking_budget_is_checked(self):
        for bad in (-1, "2", 1.5, True):
            response = requests.post(
                f"http://localhost:{self.port}/v1/chat/completions",
                json={"model": "default_model", "messages": [{"role": "user", "content": "hi"}], "thinking_budget": bad},
            )
            self.assertEqual(response.status_code, 400, bad)

    def test_request_stats_line(self):
        # One log line per request: what a client needs for speed statistics.
        ctx = GenerationContext(
            has_tool_calling=False,
            has_thinking=False,
            tool_parser=None,
            text_sm=TextStateMachine({}),
            initial_state="normal",
            prompt=[1, 2, 3, 4, 5],
            prompt_cache_count=2,
        )
        gens = [
            types.SimpleNamespace(
                text=t, token=i, logprob=0.0, finish_reason=None, top_tokens=(),
                from_draft=d,
            )
            for i, (t, d) in enumerate([("a", False), ("b", True), ("c", True)])
        ]
        gens.append(
            types.SimpleNamespace(
                text="", token=99, logprob=0.0, finish_reason="stop", top_tokens=(),
                from_draft=False,
            )
        )
        # started, then each of the four tokens.
        clock = iter([10.0, 12.0, 13.0, 14.0, 15.5])
        with mock.patch.object(
            self.response_generator, "generate", return_value=(ctx, iter(gens))
        ), mock.patch("mlx_lm.server._now", lambda: next(clock)), \
                self.assertLogs(level="INFO") as logs:
            response = requests.post(
                f"http://localhost:{self.port}/v1/chat/completions",
                json={
                    "model": "default_model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
        self.assertEqual(response.status_code, 200)
        lines = [l for l in logs.output if "Request stats:" in l]
        self.assertEqual(len(lines), 1)
        self.assertIn("prompt=5 cached=2 first_token_s=2.000 tokens=4", lines[0])
        self.assertIn("decode_s=3.500 drafted=2", lines[0])

    def test_request_stats_use_generation_times(self):
        # A slow client doesn't stretch the decode time: tokens carry when
        # the generation thread made them.
        ctx = GenerationContext(
            has_tool_calling=False, has_thinking=False, tool_parser=None,
            text_sm=TextStateMachine({}), initial_state="normal", prompt=[1, 2], prompt_cache_count=0,
        )
        gens = [
            types.SimpleNamespace(text=t, token=i, logprob=0.0, finish_reason=f, top_tokens=(),
                                  from_draft=False, made_at=at)
            for i, (t, f, at) in enumerate([("a", None, 11.0), ("b", None, 11.5), ("", "stop", 12.0)])
        ]
        clock = iter([10.0] + [99.0] * 20)   # the handler's own clock: much later
        with mock.patch.object(
            self.response_generator, "generate", return_value=(ctx, iter(gens))
        ), mock.patch("mlx_lm.server._now", lambda: next(clock)), \
                self.assertLogs(level="INFO") as logs:
            requests.post(
                f"http://localhost:{self.port}/v1/chat/completions",
                json={"model": "default_model", "messages": [{"role": "user", "content": "hi"}], "stream": True},
            )
        line = next(l for l in logs.output if "Request stats:" in l)
        self.assertIn("first_token_s=1.000 tokens=3 decode_s=1.000", line)

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

    def test_batched_chat_ending_with_a_tool_message(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "List the files."},
            {"role": "assistant", "content": "Calling list_files."},
            {"role": "tool", "content": "a.py b.py"},
        ]
        body = {"model": "chat_model", "max_tokens": 4, "messages": messages}
        for _ in range(2):
            response = requests.post(url, json=body)
            self.assertEqual(response.status_code, 200)
        usage = response.json()["usage"]
        self.assertGreater(usage["prompt_tokens_details"]["cached_tokens"], 0)

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

    def test_make_state_machine_empty_tool_call_end(self):
        class FakeTokenizer:
            has_thinking = False
            has_tool_calling = True
            tool_call_start = "[TOOL_CALLS]"
            tool_call_end = ""
            tool_call_start_tokens = (100,)
            tool_call_end_tokens = ()
            eos_token_ids = [2]
            structural_markers = ()

            def convert_ids_to_tokens(self, t):
                return f"<eos{t}>"

            def encode(self, text, add_special_tokens=False):
                return []

        stop_sequences, text_sm = self.response_generator._make_state_machine(
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
        self.assertTrue(stop_sequences.matcher().advance(2))

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

        base = f"http://localhost:{self.port}"
        ids = [m["id"] for m in response_body["data"]]
        self.assertEqual([m["id"] for m in requests.get(f"{base}/api/v0/models").json()["data"]], ids)
        self.assertEqual([m["id"] for m in requests.get(f"{base}/v1/models/?x=1").json()["data"]], ids)
        for model_id in ids:   # a local model's id is an absolute path
            listed = requests.get(f"{base}/v1/models/{model_id.lstrip('/')}").json()["data"]
            self.assertEqual([m["id"] for m in listed], [model_id])
        self.assertEqual(requests.get(f"{base}/v1/modelsX").status_code, 404)

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

    def test_exact_prompt_cache_hit_single_path(self):
        # An exact hit used to hand stream_generate an empty prompt (an
        # error for a valid request); with quantized KV, batched too.
        status, first = _complete(self.port, "In a small village")
        self.assertEqual(status, 200)
        status, second = _complete(self.port, "In a small village" + first["choices"][0]["text"])
        self.assertEqual(status, 200)
        self.assertIn("choices", second)

    def test_quantized_kv_keeps_batching(self):
        # The batch quantizes its KV cache too (BatchQuantizedKVCache).
        args = type("args", (object,), {"seed": None, "num_draft_tokens": 0})
        self.assertTrue(self.response_generator._is_batchable(args))

    def test_completion_quantizes_the_cache(self):
        url = f"http://localhost:{self.port}/v1/completions"
        prompt = "Once upon a time"
        response = requests.post(
            url,
            json={"model": "default_model", "prompt": prompt, "max_tokens": 8},
        )
        self.assertIn("choices", json.loads(response.text))

        tokens = self.model_provider.tokenizer.encode(prompt)
        # The response ends before the generation thread stores the cache.
        for _ in range(100):
            cache, _ = self.prompt_cache.fetch_nearest_cache(
                self.model_provider.model_key, tokens
            )
            if cache is not None:
                break
            time.sleep(0.02)
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


class TestServerSingleCheckpoints(unittest.TestCase):
    """The single-request path saves system / user checkpoints. A sliding
    window cache can't be trimmed back from the whole sequence, so without
    them a turn whose history differs from the cached answer reused nothing."""

    SYSTEM = "You are a helpful assistant. " * 8

    def _serve(self, kv_bits=None, prompt_cache_bytes=None, batchable=False):
        provider = DummyModelProvider(kv_bits=kv_bits, quantized_kv_start=0)
        provider.is_batchable = batchable
        provider.cli_args.prompt_cache_bytes = prompt_cache_bytes
        n_layers = len(provider.model.layers)
        provider.model.make_cache = lambda: [
            RotatingKVCache(max_size=16) for _ in range(n_layers)
        ]
        prompt_cache = LRUPromptCache()
        generator = ResponseGenerator(provider, prompt_cache)
        httpd = http.server.HTTPServer(
            ("localhost", 0),
            lambda *args, **kwargs: APIHandler(generator, *args, **kwargs),
        )
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()

        def stop():
            httpd.shutdown()
            httpd.server_close()
            thread.join()
            generator.stop_and_join()

        self.addCleanup(stop)
        return provider, prompt_cache, f"http://localhost:{httpd.server_port}"

    def _chat(self, url, messages, max_tokens=4, content=False, **extra):
        response = requests.post(
            f"{url}/v1/chat/completions",
            json={
                "model": "chat_model",
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "messages": messages,
                **extra,
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        cached = body["usage"]["prompt_tokens_details"]["cached_tokens"]
        if content:
            return cached, body["choices"][0]["message"]["content"]
        return cached

    def _check_reuse(self, kv_bits=None):
        provider, prompt_cache, url = self._serve(kv_bits)
        first = [
            {"role": "system", "content": self.SYSTEM},
            {"role": "user", "content": "Hello!"},
        ]
        tokenizer = provider.tokenizer
        prompt = tokenizer.apply_chat_template(first, add_generation_prompt=True)
        sys_tokens = tokenizer.apply_chat_template(
            first[:1] + [{"role": "user", "content": ""}]
        )
        n_system = next(i for i, (a, b) in enumerate(zip(sys_tokens, prompt)) if a != b)
        self.assertGreater(n_system, 16)

        self.assertEqual(self._chat(url, first), 0)
        second = first + [
            {"role": "assistant", "content": "Something else entirely."},
            {"role": "user", "content": "Again."},
        ]
        cached = self._chat(url, second)
        self.assertGreaterEqual(cached, n_system)
        # The user checkpoint holds the whole first prompt but its last token.
        self.assertEqual(cached, len(prompt) - 1)

        # A new user turn under the same system prompt reuses the system part.
        other = [first[0], {"role": "user", "content": "Different question?"}]
        self.assertGreaterEqual(self._chat(url, other), n_system)
        return prompt_cache, provider

    def test_agent_loop_with_tool_messages(self):
        # Agent requests end with a tool result, not a user message. The
        # stored answer is rendered differently next time, so only a
        # checkpoint at the end of the prompt can serve the next request.
        provider, prompt_cache, url = self._serve()
        tokenizer = provider.tokenizer
        messages = [
            {"role": "system", "content": self.SYSTEM},
            {"role": "user", "content": "List the files."},
            {"role": "assistant", "content": "Calling list_files."},
            {"role": "tool", "content": "a.py b.py c.py"},
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        with self.assertLogs(level="INFO") as logs:
            self.assertEqual(self._chat(url, messages), 0)
            messages += [
                {"role": "assistant", "content": "Calling read_file a.py."},
                {"role": "tool", "content": "print('hi')"},
            ]
            cached = self._chat(url, messages)
        self.assertGreaterEqual(cached, len(prompt) - 1)
        self.assertEqual(prompt_cache.stats_by_type()["system"]["n_sequences"], 1)
        text = "\n".join(logs.output)
        self.assertIn(f"Prompt cache: reused 0 of {len(prompt)} tokens", text)
        self.assertIn(f"Prompt cache: reused {cached} of", text)
        self.assertIn("Prompt cache: diverged from a cached prompt at token", text)

    def test_junction_checkpoint(self):
        for batchable in (False, True):
            with self.subTest(batchable=batchable):
                self._check_junction(batchable)

    def _check_junction(self, batchable):
        # A client changes the middle of the prompt on each request (e.g. a
        # timestamp). The second request saves a checkpoint where it
        # diverged from the first; the third reuses it.
        provider, prompt_cache, url = self._serve(batchable=batchable)
        tokenizer = provider.tokenizer
        filler = "You are a helpful assistant. " * 45

        def messages(stamp):
            system = f"{filler}Time: {stamp}.\n{filler}"
            return [
                {"role": "system", "content": system},
                {"role": "user", "content": "Hello!"},
            ]

        prompts = [
            tokenizer.apply_chat_template(messages(t), add_generation_prompt=True)
            for t in ("alpha", "bravo")
        ]
        k = next(i for i, (a, b) in enumerate(zip(*prompts)) if a != b)
        self.assertGreater(k, 256)

        self.assertEqual(self._chat(url, messages("alpha")), 0)
        with self.assertLogs(level="INFO") as logs:
            self.assertEqual(self._chat(url, messages("bravo")), 0)
        self.assertIn(
            f"Prompt cache: checkpoint at the junction, token {k}",
            "\n".join(logs.output),
        )
        self.assertGreaterEqual(self._chat(url, messages("charlie")), k)

    def test_checkpoints_are_reused(self):
        self._check_reuse()

    def test_checkpoints_with_quantized_kv(self):
        prompt_cache, provider = self._check_reuse(kv_bits=4)
        tokens = provider.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": "Hi"},
            ],
            add_generation_prompt=True,
        )
        cache, rest = prompt_cache.fetch_nearest_cache(provider.model_key, tokens)
        self.assertIsNotNone(cache)
        self.assertLess(len(rest), len(tokens))
        for c in cache:
            self.assertIsInstance(c, QuantizedRotatingKVCache)

    def test_checkpoint_hit_matches_cold_run(self):
        # The second turn resumes from the user checkpoint. A cold run with
        # the prefill split at the same points gives the same tokens.
        for kv_bits in (None, 4):
            with self.subTest(kv_bits=kv_bits):
                provider, _, url = self._serve(kv_bits)
                tokenizer, model = provider.tokenizer, provider.model
                first = [
                    {"role": "system", "content": self.SYSTEM},
                    {"role": "user", "content": "Hello!"},
                ]
                second = first + [
                    {"role": "assistant", "content": "Something else entirely."},
                    {"role": "user", "content": "Again."},
                ]
                self._chat(url, first)
                cached, text = self._chat(url, second, max_tokens=12, content=True)
                self.assertGreater(cached, 0)

                prompt = tokenizer.apply_chat_template(
                    second, add_generation_prompt=True
                )
                sys_tokens = tokenizer.apply_chat_template(
                    first[:1] + [{"role": "user", "content": ""}]
                )
                n_system = next(
                    i for i, (a, b) in enumerate(zip(sys_tokens, prompt)) if a != b
                )
                cache = make_prompt_cache(model)
                done = 0
                for end in (n_system, cached):
                    model(mx.array(prompt[done:end])[None], cache=cache)
                    maybe_quantize_kv_cache(cache, 0, 64, kv_bits)
                    done = end
                tokens = []
                for t, _ in generate_step(
                    mx.array(prompt[cached:]),
                    model,
                    max_tokens=12,
                    prompt_cache=cache,
                    kv_bits=kv_bits,
                    kv_group_size=64,
                    quantized_kv_start=0,
                ):
                    if t in tokenizer.eos_token_ids:
                        break
                    tokens.append(t)
                self.assertEqual(text, tokenizer.decode(tokens))

    def test_penalties_see_the_cached_prompt(self):
        # A repetition penalty counts the prompt tokens held by the cache
        # (checkpoint or cache hit) as in a cold run.
        provider, _, url = self._serve()
        tokenizer, model = provider.tokenizer, provider.model
        first = [
            {"role": "system", "content": self.SYSTEM},
            {"role": "user", "content": "Say hello hello hello."},
        ]
        second = first + [
            {"role": "assistant", "content": "hello hello hello"},
            {"role": "user", "content": "Again, hello hello hello."},
        ]
        extra = dict(repetition_penalty=1.5, repetition_context_size=200)
        self._chat(url, first, **extra)
        cached, text = self._chat(url, second, 16, True, **extra)
        self.assertGreater(cached, 0)

        prompt = tokenizer.apply_chat_template(second, add_generation_prompt=True)
        tokens = []
        for t, _ in generate_step(
            mx.array(prompt),
            model,
            max_tokens=16,
            logits_processors=make_logits_processors(
                repetition_penalty=1.5, repetition_context_size=200
            ),
        ):
            if t in tokenizer.eos_token_ids:
                break
            tokens.append(t)
        self.assertEqual(text, tokenizer.decode(tokens))

    def test_prompt_cache_bytes(self):
        # --prompt-cache-bytes holds in the single path. A checkpoint that
        # can't fit next to the live cache is not copied.
        for cap in (1, 1 << 30):
            with self.subTest(cap=cap):
                provider, prompt_cache, url = self._serve(prompt_cache_bytes=cap)
                first = [
                    {"role": "system", "content": self.SYSTEM},
                    {"role": "user", "content": "Hello!"},
                ]
                self._chat(url, first)
                self.assertLessEqual(prompt_cache.nbytes, cap)
                stats = prompt_cache.stats_by_type()
                n = 0 if cap == 1 else 1
                self.assertEqual(stats["system"]["n_sequences"], n)
                self.assertEqual(stats["assistant"]["n_sequences"], n)

    def test_prompt_cache_bytes_batched(self):
        # The batched path trims after its checkpoint and final inserts too.
        provider, prompt_cache, url = self._serve(prompt_cache_bytes=1, batchable=True)
        first = [
            {"role": "system", "content": self.SYSTEM},
            {"role": "user", "content": "Hello!"},
        ]
        self._chat(url, first)
        self._chat(url, first + [{"role": "assistant", "content": "Hi."}])
        self.assertLessEqual(prompt_cache.nbytes, 1)

    def test_checkpoint_room(self):
        gb = 1 << 30
        # 10 GB in use, 1 GB copy, 0.5 GB prefill scratch, 19 GB limit.
        self.assertLessEqual(checkpoint_room(gb, 10 * gb, 19 * gb, gb // 2, 0.9), 0)
        # 16.5 GB in use: 0.9 * 19 GB = 17.1 GB is exceeded by 0.9 GB.
        short = checkpoint_room(gb, 16 * gb + gb // 2, 19 * gb, gb // 2, 0.9)
        self.assertEqual(short, 18 * gb - int(0.9 * 19 * gb))
        self.assertGreater(short, 0)
        # A larger headroom fraction leaves room.
        self.assertLessEqual(
            checkpoint_room(gb, 16 * gb + gb // 2, 19 * gb, gb // 2, 1.0), 0
        )

    def test_memory_flags(self):
        from mlx_lm.server import SERVER_DEFAULTS, make_parser

        args = make_parser().parse_args([])
        for name, value in SERVER_DEFAULTS.items():
            self.assertEqual(getattr(args, name), value, name)
        args = make_parser().parse_args(
            [
                "--memory-headroom-fraction", "0.8",
                "--oom-retry-step-divisor", "4",
                "--junction-min-gap-tokens", "64",
                "--min-prefill-step", "32",
                "--prefill-score-bytes", "2",
                "--prefill-step-warn-below", "256",
            ]
        )  # fmt: skip
        gen = self._generator(prefill_step_size=2048)
        gen.model_provider.cli_args = args
        self.assertEqual(gen._opt("memory_headroom_fraction"), 0.8)
        self.assertEqual(gen._opt("junction_min_gap_tokens"), 64)
        budget = gen._prefill_memory_budget(args.oom_retry_step_divisor)
        self.assertEqual(budget.nbytes, (512 << 20) // 4)
        self.assertEqual((budget.min_step, budget.score_bytes), (32, 2))
        # Headroom reaches the room check.
        limit = mx.device_info()["max_recommended_working_set_size"]
        with mock.patch(
            "mlx_lm.server.mx.get_active_memory", return_value=int(0.85 * limit)
        ):
            self.assertIsNotNone(gen._memory_shortfall(0, None))
            self.assertGreater(gen._memory_shortfall(0, None), 0)
            args.memory_headroom_fraction = 0.9
            self.assertLessEqual(gen._memory_shortfall(0, None), 0)

    def test_memory_flags_are_validated(self):
        from mlx_lm.server import make_parser

        bad = [
            ("--oom-retry-step-divisor", "0"),
            ("--min-prefill-step", "0"),
            ("--prefill-score-bytes", "0"),
            ("--prefill-memory-mb", "-1"),
            ("--junction-min-gap-tokens", "-1"),
            ("--prefill-step-warn-below", "-5"),
            ("--memory-headroom-fraction", "0"),
            ("--memory-headroom-fraction", "1.5"),
            ("--memory-headroom-fraction", "abc"),
            ("--min-prefill-step", "2.5"),
            ("--buffer-cache-mb", "-1"),
        ]
        for flag, value in bad:
            with self.subTest(flag=flag, value=value):
                stderr = mock.patch("sys.stderr", new_callable=io.StringIO)
                with stderr as err, self.assertRaises(SystemExit):
                    make_parser().parse_args([flag, value])
                self.assertIn(flag, err.getvalue())
        ok = make_parser().parse_args(
            ["--memory-headroom-fraction", "1", "--prefill-memory-mb", "0"]
        )
        self.assertEqual(
            (ok.memory_headroom_fraction, ok.prefill_memory_mb), (1.0, 0)
        )
        # Unset by default: MLX's own cache limit stays.
        self.assertIsNone(make_parser().parse_args([]).buffer_cache_mb)
        self.assertEqual(make_parser().parse_args(["--buffer-cache-mb", "512"]).buffer_cache_mb, 512)

    def test_foreign_cli_args_are_clamped(self):
        gen = self._generator(prefill_step_size=128)
        args = gen.model_provider.cli_args
        args.oom_retry_step_divisor = 0
        args.min_prefill_step = -3
        args.memory_headroom_fraction = 2.0
        args.junction_min_gap_tokens = -10
        self.assertEqual(gen._opt("oom_retry_step_divisor"), 1)
        self.assertEqual(gen._opt("min_prefill_step"), 1)
        self.assertEqual(gen._opt("memory_headroom_fraction"), 1.0)
        self.assertEqual(gen._opt("junction_min_gap_tokens"), 0)
        args.memory_headroom_fraction = -1
        self.assertEqual(gen._opt("memory_headroom_fraction"), 0.9)
        self.assertEqual(gen._prefill_memory_budget(0).nbytes, 512 << 20)

    def test_retry_step_divisor_reaches_the_prefill(self):
        # The retry's step is the configured step divided by the flag.
        gen = self._generator(prefill_step_size=128)
        gen.model_provider.cli_args.oom_retry_step_divisor = 4
        gen.model_provider.cli_args.prefill_memory_mb = 0
        seen = []
        ctx = GenerationContext(None, None, None, None, None, None)
        cache = [RotatingKVCache(max_size=16) for _ in range(2)]
        gen._prefill_checkpoints(
            _tiny_llama(), None, cache, 2, list(range(200)), 0,
            [list(range(100)), list(range(100, 200))], ["system", "user"],
            mx.default_stream(mx.default_device()),
            lambda p, t: seen.append(p), ctx, step_divisor=4,
        )  # fmt: skip
        steps = [b - a for a, b in zip(seen, seen[1:]) if b > a]
        self.assertEqual(max(steps), 32)

    def test_junction_gap_flag(self):
        gen = self._generator(prefill_step_size=64)
        gen.prompt_cache.last_divergence = (100, 300)
        segs, types_ = [list(range(300))], ["user"]
        self.assertEqual(gen._junction_split(segs, types_, 0, 300), (segs, types_))
        gen.model_provider.cli_args.junction_min_gap_tokens = 50
        segs2, types2 = gen._junction_split(segs, types_, 0, 300)
        self.assertEqual([len(x) for x in segs2], [100, 200])

    def test_checkpoint_skipped_without_memory_room(self):
        provider, prompt_cache, url = self._serve()
        messages = [
            {"role": "system", "content": self.SYSTEM},
            {"role": "user", "content": "Hello!"},
        ]
        limit = mx.device_info()["max_recommended_working_set_size"]
        with mock.patch("mlx_lm.server.mx.get_active_memory", return_value=limit):
            with self.assertLogs(level="WARNING") as logs:
                self._chat(url, messages)
        text = "\n".join(logs.output)
        self.assertEqual(text.count("checkpoint skipped: no memory room"), 1)
        stats = prompt_cache.stats_by_type()
        self.assertEqual(stats["system"]["n_sequences"], 0)
        self.assertEqual(stats["assistant"]["n_sequences"], 1)

    OOM = (
        "[METAL] Command buffer execution failed: Insufficient Memory "
        "(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)"
    )

    def _failing_eval(self, fail_at, message=OOM):
        """mx.eval that raises on the given calls made by _prefill_checkpoints
        (1-based); other evals run normally."""
        real_eval = mx.eval
        calls = {"n": 0}

        def fake_eval(*args):
            if _caller(1) == "_prefill_checkpoints":
                calls["n"] += 1
                if calls["n"] in fail_at:
                    raise RuntimeError(message)
            return real_eval(*args)

        return mock.patch("mlx_lm.server.mx.eval", side_effect=fake_eval)

    def _messages(self, system=None):
        return [
            {"role": "system", "content": system or self.SYSTEM},
            {"role": "user", "content": "Hello!"},
        ]

    def test_metal_oom_signature(self):
        from mlx_lm.server import _is_metal_oom

        self.assertTrue(_is_metal_oom(RuntimeError(self.OOM)))
        self.assertTrue(
            _is_metal_oom(
                RuntimeError(
                    "[METAL] Command buffer execution failed: Insufficient Memory"
                )
            )
        )
        self.assertFalse(_is_metal_oom(RuntimeError("out of memory")))
        self.assertFalse(_is_metal_oom(MemoryError(self.OOM)))

    def test_prefill_oom_is_retried_once(self):
        _, _, cold_url = self._serve()
        _, expected = self._chat(cold_url, self._messages(), 8, True)

        provider, prompt_cache, url = self._serve()
        other = self._messages("An unrelated system prompt. " * 12)
        self._chat(url, other)
        n_before = len(prompt_cache)
        self.assertGreater(n_before, 0)

        # The retry sees no memory room: stored caches go first.
        limit = mx.device_info()["max_recommended_working_set_size"]
        real_active = mx.get_active_memory

        def active():
            caller = _caller(2)
            return limit if caller == "_free_memory_after_oom" else real_active()

        memory = mock.patch("mlx_lm.server.mx.get_active_memory", side_effect=active)
        with self.assertLogs(level="WARNING") as logs, self._failing_eval({1}):
            with memory:
                _, text = self._chat(url, self._messages(), 8, True)
        self.assertIn("retrying once", "\n".join(logs.output))
        self.assertEqual(text, expected)
        # The unrelated entries were evicted; this request's are stored.
        tokens = provider.tokenizer.apply_chat_template(
            other, add_generation_prompt=True
        )
        _, rest = prompt_cache.fetch_nearest_cache(provider.model_key, tokens)
        self.assertEqual(len(rest), len(tokens))
        self.assertEqual(prompt_cache.stats_by_type()["system"]["n_sequences"], 1)

    def test_other_runtime_error_is_not_retried(self):
        provider, prompt_cache, url = self._serve()
        self._chat(url, self._messages("An unrelated system prompt. " * 12))
        n_before, bytes_before = len(prompt_cache), prompt_cache.nbytes
        with self.assertNoLogs(level="WARNING"), self._failing_eval({1}, "boom"):
            try:
                response = requests.post(
                    f"{url}/v1/chat/completions",
                    json={
                        "model": "chat_model",
                        "max_tokens": 4,
                        "messages": self._messages(),
                    },
                )
                self.assertNotEqual(response.status_code, 200)
            except requests.ConnectionError:
                pass
        self.assertEqual(len(prompt_cache), n_before)
        self.assertEqual(prompt_cache.nbytes, bytes_before)

    def test_progress_after_an_oom_retry_only_moves_forward(self):
        provider, prompt_cache, url = self._serve()
        provider.cli_args.prefill_step_size = 16
        # Fail the third chunk: the retry starts again from 0.
        with self._failing_eval({3}):
            response = requests.post(
                f"{url}/v1/chat/completions",
                json={
                    "model": "chat_model",
                    "max_tokens": 4,
                    "stream": True,
                    "messages": self._messages(),
                },
            )
            lines = [l.decode() for l in response.iter_lines() if l]
        keepalives = [l for l in lines if l.startswith(": keepalive")]
        done = [int(l.split()[2].split("/")[0]) for l in keepalives]
        self.assertGreater(len(done), 2)
        self.assertEqual(done, sorted(done))

    def test_prefill_oom_twice_fails_the_request(self):
        provider, prompt_cache, url = self._serve()
        messages = [
            {"role": "system", "content": self.SYSTEM},
            {"role": "user", "content": "Hello!"},
        ]
        with self._failing_eval({1, 2}):
            try:
                response = requests.post(
                    f"{url}/v1/chat/completions",
                    json={"model": "chat_model", "max_tokens": 4, "messages": messages},
                )
                self.assertNotEqual(response.status_code, 200)
            except requests.ConnectionError:
                pass
        # The server keeps serving.
        self._chat(url, messages)

    def test_trimmable_cache_skips_checkpoints(self):
        self.assertTrue(stays_trimmable([KVCache(), CacheList(KVCache())]))
        self.assertFalse(stays_trimmable([KVCache(), RotatingKVCache(max_size=16)]))
        self.assertFalse(stays_trimmable([CacheList(RotatingKVCache(max_size=16))]))

        gen = self._generator(prefill_step_size=4)
        n = gen._prefill_checkpoints(
            _tiny_llama(),
            None,
            [KVCache(), KVCache()],
            2,
            list(range(40)),
            0,
            [list(range(20)), list(range(20, 40))],
            ["system", "user"],
            mx.default_stream(mx.default_device()),
            lambda *_: None,
            GenerationContext(None, None, None, None, None, None),
        )
        self.assertEqual(n, 0)
        self.assertEqual(len(gen.prompt_cache), 0)

    def test_stop_during_checkpoint_prefill(self):
        gen = self._generator(prefill_step_size=4)
        ctx = GenerationContext(None, None, None, None, None, None)

        def progress(done, total):
            if done > 0:
                ctx.stop()

        cache = [RotatingKVCache(max_size=16) for _ in range(2)]
        n = gen._prefill_checkpoints(
            _tiny_llama(),
            None,
            cache,
            2,
            list(range(40)),
            0,
            [list(range(20)), list(range(20, 40))],
            ["system", "user"],
            mx.default_stream(mx.default_device()),
            progress,
            ctx,
        )
        self.assertIsNone(n)
        self.assertEqual(len(gen.prompt_cache), 0)
        self.assertEqual(cache[0].offset, 4)

    def _generator(self, prefill_step_size):
        gen = object.__new__(ResponseGenerator)
        gen.model_provider = types.SimpleNamespace(
            model_key="tiny",
            cli_args=types.SimpleNamespace(
                prefill_step_size=prefill_step_size,
                prefill_memory_mb=512,
                kv_bits=None,
                kv_group_size=64,
                quantized_kv_start=0,
            ),
        )
        gen.prompt_cache = LRUPromptCache()
        gen._is_distributed = False
        return gen


def _caller(depth):
    """Name of the function `depth` calls up from a mock's side_effect,
    not counting unittest.mock frames."""
    f = sys._getframe(2)
    while depth:
        if not f.f_code.co_filename.endswith("mock.py"):
            depth -= 1
            if not depth:
                break
        f = f.f_back
    return f.f_code.co_name


def _tiny_llama():
    from mlx_lm.models import llama

    mx.random.seed(0)
    model = llama.Model(
        llama.ModelArgs(
            model_type="llama",
            hidden_size=32,
            num_hidden_layers=2,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            vocab_size=64,
            rope_theta=10000.0,
            tie_word_embeddings=True,
        )
    )
    mx.eval(model.parameters())
    return model


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
    @staticmethod
    def _hybrid(n, window=8, quantized=False):
        """A sliding-window layer and a growing KV layer over tokens 0..n-1,
        each position's keys/values its own index."""
        kv = mx.arange(n, dtype=mx.float32).reshape(1, 1, n, 1) * mx.ones((1, 1, 1, 64))
        rotating, full = RotatingKVCache(max_size=window), KVCache()
        rotating.update_and_fetch(kv, kv)
        full.update_and_fetch(kv, kv)
        if quantized:
            full = full.to_quantized(group_size=64, bits=8)
        mx.eval(full.state, rotating.state)
        return [rotating, full]

    @staticmethod
    def _kv_bytes(n, quantized=False):
        """A growing layer's n positions, unpadded."""
        layer = TestLRUPromptCache._hybrid(n, quantized=quantized)[1]
        k, v = layer.keys, layer.values
        arrays = (*k, *v) if quantized else (k, v)
        return sum(x[..., :n, :].nbytes for x in arrays)

    @staticmethod
    def _positions(layer):
        if isinstance(layer, QuantizedKVCache):
            k = mx.dequantize(*layer.keys, group_size=layer.group_size, bits=layer.bits)
        else:
            k = layer.keys
        return k[0, 0, : layer.offset, 0].tolist()

    def _windows(self, *ns):
        # Stored cut to the window.
        def stored(w):
            return w.nbytes * min(w.max_size, w.keys.shape[2]) // w.keys.shape[2]

        return sum(stored(self._hybrid(n)[0]) for n in ns)

    def _chain(self):
        # A system and a user checkpoint, then the answered prompt.
        cache = LRUPromptCache()
        model = "m"
        cache.insert_cache(model, list(range(16)), self._hybrid(16), cache_type="system")
        cache.insert_cache(model, list(range(32)), self._hybrid(32), cache_type="user")
        cache.insert_cache(model, list(range(48)), self._hybrid(48))
        return cache, model

    def test_entries_share_the_growing_layers(self):
        cache, model = self._chain()
        self.assertEqual(len(cache), 3)
        # The growing layer's 48 positions are stored once.
        self.assertEqual(cache.nbytes, self._windows(16, 32, 48) + self._kv_bytes(48))
        # A fetch copies an entry whole.
        self.assertEqual(
            cache.entry_nbytes((model, list(range(32)))),
            sum(c.nbytes for c in self._hybrid(32)),
        )
        stats = cache.stats_by_type()
        self.assertEqual(stats["system"]["n_bytes"], self._windows(16) + self._kv_bytes(16))

        # Each one comes back whole, its own positions only.
        for n in (16, 32, 48):
            c, rest = cache.fetch_nearest_cache(model, list(range(n)) + [99])
            self.assertEqual(rest, [99])
            self.assertIsInstance(c[1], KVCache)
            self.assertEqual(c[1].offset, n)
            self.assertEqual(self._positions(c[1]), list(range(n)))
            # A copy: extending it leaves the stored chunks alone.
            c[1].update_and_fetch(mx.zeros((1, 1, 4, 64)), mx.zeros((1, 1, 4, 64)))
            c[1].keys[..., :2, :] = -1
        c, _ = cache.fetch_nearest_cache(model, list(range(48)) + [99])
        self.assertEqual(self._positions(c[1]), list(range(48)))

    def test_the_next_turn_shares_the_checkpoints(self):
        # Two answers to the same document and question: siblings in the
        # trie, the part above them stored once.
        cache, model = self._chain()
        cache.insert_cache(model, list(range(32)) + [7] * 16, self._hybrid(48))
        self.assertEqual(
            cache.nbytes, self._windows(16, 32, 48, 48) + self._kv_bytes(48) + self._kv_bytes(16)
        )
        c, _ = cache.fetch_nearest_cache(model, list(range(32)) + [7] * 16 + [99])
        self.assertEqual(self._positions(c[1]), list(range(48)))

    def test_evicting_an_entry_keeps_the_chunks_others_use(self):
        cache, model = self._chain()
        # The answered prompt goes first (assistant entries are evicted
        # before checkpoints): only its own positions are freed.
        cache.trim_to(n_sequences=2)
        self.assertEqual(len(cache), 2)
        self.assertEqual(cache.nbytes, self._windows(16, 32) + self._kv_bytes(32))
        for n in (16, 32):
            c, _ = cache.fetch_nearest_cache(model, list(range(n)) + [99])
            self.assertEqual(c[1].offset, n)
            self.assertEqual(self._positions(c[1]), list(range(n)))
        # The system checkpoint before the user one: the user one keeps
        # its first positions.
        cache.trim_to(n_sequences=1, keep=(model, list(range(32))))
        self.assertEqual(cache.nbytes, self._windows(32) + self._kv_bytes(32))
        c, _ = cache.fetch_nearest_cache(model, list(range(32)) + [99])
        self.assertEqual(self._positions(c[1]), list(range(32)))
        cache.trim_to(n_sequences=0)
        self.assertEqual(cache.nbytes, 0)
        self.assertEqual(cache._chunks, {})

    def test_replacing_an_entry(self):
        cache, model = self._chain()
        before = cache.nbytes
        cache.insert_cache(model, list(range(32)), self._hybrid(32), cache_type="user")
        self.assertEqual(len(cache), 3)
        self.assertEqual(cache.nbytes, before + self._kv_bytes(16))
        for n in (16, 32, 48):
            c, _ = cache.fetch_nearest_cache(model, list(range(n)) + [99])
            self.assertEqual(self._positions(c[1]), list(range(n)))

    def test_a_live_cache_is_copied(self):
        cache = LRUPromptCache()
        live = self._hybrid(16)
        cache.insert_cache("m", list(range(16)), live, cache_type="system", copy=True)
        # The live cache goes on.
        kv = mx.full((1, 1, 4, 64), -1.0)
        live[1].update_and_fetch(kv, kv)
        live[0].update_and_fetch(kv, kv)
        c, _ = cache.fetch_nearest_cache("m", list(range(16)) + [99])
        self.assertEqual(self._positions(c[1]), list(range(16)))
        self.assertEqual(c[0].offset, 16)

    def test_insert_nbytes(self):
        cache, model = self._chain()
        add = cache.insert_nbytes(model, list(range(64)), self._hybrid(64))
        # The window as stored, and the 16 new positions, unpadded.
        self.assertEqual(add, self._windows(64) + self._kv_bytes(64) // 4)
        cache.insert_cache(model, list(range(64)), self._hybrid(64))
        self.assertEqual(cache.nbytes, self._windows(16, 32, 48, 64) + self._kv_bytes(64))

    def test_layers_of_another_kind_are_not_shared(self):
        # A checkpoint before --quantized-kv-start and a quantized answer.
        cache = LRUPromptCache()
        cache.insert_cache("m", list(range(16)), self._hybrid(16), cache_type="system")
        cache.insert_cache("m", list(range(48)), self._hybrid(48, quantized=True))
        self.assertEqual(
            cache.nbytes,
            self._windows(16, 48) + self._kv_bytes(16) + self._kv_bytes(48, quantized=True),
        )
        # Both quantized: shared.
        cache = LRUPromptCache()
        cache.insert_cache("m", list(range(16)), self._hybrid(16, quantized=True), cache_type="system")
        cache.insert_cache("m", list(range(48)), self._hybrid(48, quantized=True))
        self.assertEqual(cache.nbytes, self._windows(16, 48) + self._kv_bytes(48, quantized=True))
        for n in (16, 48):
            c, _ = cache.fetch_nearest_cache("m", list(range(n)) + [99])
            self.assertIsInstance(c[1], QuantizedKVCache)
            self.assertEqual((c[1].offset, c[1].bits, c[1].group_size), (n, 8, 64))
            self.assertEqual(self._positions(c[1]), list(range(n)))

    def test_an_exact_hit_and_its_prefill_leave_the_entry_alone(self):
        # The fetched copy of a one-chunk entry is trimmed by its last token,
        # which the prefill then writes into its buffer.
        cache = LRUPromptCache()
        cache.insert_cache("m", list(range(16)), self._hybrid(16)[1:])
        for _ in range(2):
            c, rest = cache.fetch_nearest_cache("m", list(range(16)))
            self.assertEqual(rest, [15])
            kv = mx.full((1, 1, 1, 64), -1.0)
            c[0].update_and_fetch(kv, kv)
            mx.eval(c[0].keys)
            self.assertEqual(self._positions(c[0])[-1], -1.0)
        c, _ = cache.fetch_nearest_cache("m", list(range(16)) + [99])
        self.assertEqual(self._positions(c[0]), list(range(16)))

    def test_a_stored_window_is_cut_to_what_it_uses(self):
        # A chunked prefill leaves window + chunk positions in the buffer;
        # the stored copy keeps the window, and the next updates see the
        # same keys either way.
        for keep in (0, 2):
            live = RotatingKVCache(max_size=16, keep=keep)
            for n in (20, 30):
                kv = mx.random.normal((1, 2, n, 8))
                live.update_and_fetch(kv, kv)
            self.assertGreater(live.keys.shape[2], 16)
            cache = LRUPromptCache()
            cache.insert_cache("m", list(range(50)), [live], copy=True)
            self.assertEqual(cache.nbytes, live.nbytes * 16 // live.keys.shape[2])
            stored, _ = cache.fetch_nearest_cache("m", list(range(50)) + [1])
            stored = stored[0]
            for n in (5, 1, 1, 7, 1):
                kv = mx.random.normal((1, 2, n, 8))
                a, _ = live.update_and_fetch(kv, kv)
                b, _ = stored.update_and_fetch(kv, kv)
                self.assertEqual(a.shape, b.shape)
                self.assertTrue(mx.array_equal(a, b).item())
                self.assertEqual((live.offset, live._idx), (stored.offset, stored._idx))
                self.assertTrue(
                    mx.array_equal(live.make_mask(1, window_size=8), stored.make_mask(1, window_size=8)).item()
                    if live.make_mask(1, window_size=8) is not None
                    else stored.make_mask(1, window_size=8) is None
                )

    def test_a_diverged_prompt_shares_nothing(self):
        cache = LRUPromptCache()
        cache.insert_cache("m", list(range(16)), self._hybrid(16), cache_type="system")
        cache.insert_cache("m", list(range(8)) + [99] * 24, self._hybrid(32))
        self.assertEqual(cache.nbytes, self._windows(16, 32) + self._kv_bytes(16) + self._kv_bytes(32))

    def test_trim_to_keeps_an_entry(self):
        cache = LRUPromptCache()
        cache.insert_cache("m", [1], [MockCache("aaaa")])
        cache.insert_cache("m", [2], [MockCache("bbbb")])
        cache.insert_cache("m", [3], [MockCache("cccc")])
        cache.trim_to(n_bytes=4, keep=("m", [1]))
        self.assertEqual(len(cache), 1)
        # An exact hit leaves the last token to process.
        self.assertEqual(cache.fetch_nearest_cache("m", [1])[1], [1])
        self.assertEqual(cache.last_fetched, ("m", [1]))
        self.assertEqual(cache.entry_nbytes(("m", [1])), 4)

    def test_exact_hit_on_a_cache_that_cannot_trim(self):
        # A hybrid model's SSM state can't drop its last token: an exact
        # hit falls back to the nearest shorter entry instead.
        cache = LRUPromptCache()
        cache.insert_cache("m", [1, 2], [MockCache("short", is_trimmable=False)])
        cache.insert_cache("m", [1, 2, 3, 4], [MockCache("full", is_trimmable=False)])
        c, rest = cache.fetch_nearest_cache("m", [1, 2, 3, 4])
        self.assertEqual(c, [MockCache("short")])
        self.assertEqual(rest, [3, 4])
        # With no shorter entry: nothing to reuse.
        cache = LRUPromptCache()
        cache.insert_cache("m", [1, 2], [MockCache("only", is_trimmable=False)])
        c, rest = cache.fetch_nearest_cache("m", [1, 2])
        self.assertIsNone(c)
        self.assertEqual(rest, [1, 2])

    def test_one_token_prefix_is_reused(self):
        cache = LRUPromptCache()
        cache.insert_cache("m", [7], [MockCache("one", is_trimmable=False)])
        c, rest = cache.fetch_nearest_cache("m", [7, 8, 9])
        self.assertEqual(c, [MockCache("one")])
        self.assertEqual(rest, [8, 9])

    def test_sliding_window_keeps_prefixes(self):
        # A sliding window below its size is trimmable only until it is
        # full: its prefix entries must stay.
        def rotating(offset):
            c = RotatingKVCache(max_size=16)
            c.update_and_fetch(mx.zeros((1, 1, offset, 8)), mx.zeros((1, 1, offset, 8)))
            return [c]

        cache = LRUPromptCache()
        cache.insert_cache("m", [1, 2, 3], rotating(3), cache_type="system")
        cache.insert_cache("m", [1, 2, 3, 4, 5], rotating(5))
        self.assertEqual(len(cache), 2)
        c, rest = cache.fetch_nearest_cache("m", [1, 2, 3, 9])
        self.assertEqual(rest, [9])

        # A cache that stays trimmable drops its prefixes, as before.
        cache = LRUPromptCache()
        cache.insert_cache("m", [1, 2, 3], [MockCache("a")], cache_type="system")
        cache.insert_cache("m", [1, 2, 3, 4, 5], [MockCache("b")])
        self.assertEqual(len(cache), 1)

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
        self.assertEqual(t, [2])  # exact: its last token left to process
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
        self.assertEqual(t, [3])
        c, t = cache.fetch_nearest_cache(model, [3, 4])
        self.assertEqual(c, [MockCache("test3")])
        self.assertEqual(t, [4])

        cache.insert_cache(model, [4, 5], [MockCache("test4")], cache_type="user")
        c, t = cache.fetch_nearest_cache(model, [2, 3])
        self.assertEqual(c, None)
        self.assertEqual(t, [2, 3])
        c, t = cache.fetch_nearest_cache(model, [3, 4])
        self.assertEqual(c, [MockCache("test3")])
        self.assertEqual(t, [4])
        c, t = cache.fetch_nearest_cache(model, [4, 5])
        self.assertEqual(c, [MockCache("test4")])
        self.assertEqual(t, [5])

        cache.insert_cache(model, [5, 6], [MockCache("test5")])
        cache.insert_cache(model, [6, 7], [MockCache("test6")])
        c, t = cache.fetch_nearest_cache(model, [5, 6])
        self.assertEqual(c, None)
        self.assertEqual(t, [5, 6])
        c, t = cache.fetch_nearest_cache(model, [6, 7])
        self.assertEqual(c, [MockCache("test6")])
        self.assertEqual(t, [7])
        c, t = cache.fetch_nearest_cache(model, [4, 5])
        self.assertEqual(c, [MockCache("test4")])
        self.assertEqual(t, [5])

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
        load.return_value = (mock.Mock(), tokenizer)
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
