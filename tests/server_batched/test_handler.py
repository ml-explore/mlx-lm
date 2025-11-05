# ABOUTME: Validates handler helper delegating to continuous batching runtime.
# ABOUTME: Ensures runtime hook executes only when configured for streaming.

import io
import types
import unittest
from unittest import mock

from .util import ensure_mlx_stub

ensure_mlx_stub()

from mlx_lm.generate import GenerationResponse
from mlx_lm.server import APIHandler, PromptCache
from mlx_lm.server_batched.handler import maybe_handle_continuous_batching


class FakeResponse:
    __slots__ = (
        "text",
        "token",
        "logprobs",
        "finish_reason",
        "prompt_tokens",
        "prompt_tps",
        "generation_tokens",
        "generation_tps",
        "peak_memory",
        "from_draft",
    )

    def __init__(self, text, *, finish_reason=None):
        self.text = text
        self.token = 1
        self.logprobs = None
        self.finish_reason = finish_reason
        self.prompt_tokens = 0
        self.prompt_tps = 0.0
        self.generation_tokens = 0
        self.generation_tps = 0.0
        self.peak_memory = 0.0
        self.from_draft = False


class DummyRuntime:
    def __init__(self):
        self.calls = []

    def submit_request(self, *args, **kwargs):
        self.calls.append((args, kwargs))

        def generator():
            yield FakeResponse("hello")
            yield FakeResponse("", finish_reason="stop")

        return ("req-1", generator())


class HandlerRuntimeTests(unittest.TestCase):
    def make_handler(self, runtime=None, stream=True):
        handler = types.SimpleNamespace()
        handler.batch_runtime = runtime
        handler.stream = stream
        handler.request_id = "req-0"
        handler.max_tokens = 4
        return handler

    def test_helper_invokes_runtime_when_enabled(self):
        runtime = DummyRuntime()
        handler = self.make_handler(runtime=runtime, stream=True)

        result = maybe_handle_continuous_batching(
            handler,
            prompt_tokens=[1, 2],
            stop_id_sequences=[],
            sampler_settings={"temp": 0.0},
            stopping_settings={"eos_token_id": 42},
            logit_bias=None,
            repetition_penalty=None,
            repetition_context_size=None,
        )

        self.assertIsNotNone(result)
        request_id, generator = result
        self.assertEqual(request_id, "req-1")
        events = list(generator)
        self.assertEqual(len(events), 2)
        self.assertEqual(runtime.calls[0][1]["max_new_tokens"], 4)

    def test_helper_skips_when_runtime_absent(self):
        handler = self.make_handler(runtime=None, stream=True)
        result = maybe_handle_continuous_batching(
            handler,
            prompt_tokens=[1],
            stop_id_sequences=[],
            sampler_settings={},
            stopping_settings={},
            logit_bias=None,
            repetition_penalty=None,
            repetition_context_size=None,
        )
        self.assertIsNone(result)

    def test_helper_skips_when_stream_disabled(self):
        runtime = DummyRuntime()
        handler = self.make_handler(runtime=runtime, stream=False)
        result = maybe_handle_continuous_batching(
            handler,
            prompt_tokens=[1],
            stop_id_sequences=[],
            sampler_settings={},
            stopping_settings={},
            logit_bias=None,
            repetition_penalty=None,
            repetition_context_size=None,
        )
        self.assertIsNone(result)


class DummyTokenizer:
    has_tool_calling = False
    vocab_size = 8
    tool_call_start = "<tool-call>"
    tool_call_end = "</tool-call>"
    eos_token_id = 0

    def encode(self, prompt, add_special_tokens=True):
        if isinstance(prompt, str):
            return [1] * max(len(prompt), 1)
        return [1]

    def decode(self, tokens):
        return "X"

    @property
    def detokenizer(self):
        detok = types.SimpleNamespace()
        detok.add_token = lambda *_: None
        detok.finalize = lambda: None
        detok.last_segment = "X"
        return detok


class FakeVector(list):
    def __getitem__(self, item):
        result = super().__getitem__(item)
        if isinstance(item, slice):
            return FakeVector(result)
        return result

    def tolist(self):
        return list(self)


class FakeLogprobsArray:
    def __init__(self, value=0.0):
        self.value = value

    def __getitem__(self, token):
        if isinstance(token, (list, tuple, FakeVector)):
            return FakeVector([self.value] * len(token))
        if hasattr(token, "tolist"):
            data = token.tolist()
            return FakeVector([self.value] * len(data))
        return types.SimpleNamespace(item=lambda: self.value)

    def __iter__(self):
        yield self.value

    def __len__(self):
        return 1

    def tolist(self):
        return [self.value]

    def __neg__(self):
        return self


class HandlerFallbackTests(unittest.TestCase):
    def setUp(self):
        self._patchers = []
        self._logprob_value = 0.0

        def start_patch(target, new):
            patcher = mock.patch(target, new=new)
            self._patchers.append(patcher)
            patcher.start()

        def fake_argpartition(arr, kth, axis=-1):
            size = max(int(kth) + 1, 1)
            return FakeVector(list(range(size)))

        def fake_take_along_axis(arr, indices, axis=-1):
            length = len(indices) if hasattr(indices, "__len__") else 1
            return FakeVector([self._logprob_value] * length)

        start_patch("mlx.core.argpartition", fake_argpartition)
        start_patch("mlx.core.take_along_axis", fake_take_along_axis)

    def tearDown(self):
        while self._patchers:
            self._patchers.pop().stop()

    def make_handler(self, *, streaming=True, logprobs=-1, tools=None):
        handler = APIHandler.__new__(APIHandler)
        handler.stream = streaming
        handler.body = {"messages": [{"role": "user", "content": "Hi"}]}
        if tools is not None:
            handler.body["tools"] = tools
        handler.max_tokens = 4
        handler.temperature = 0.0
        handler.top_p = 1.0
        handler.min_p = 0.0
        handler.top_k = 0
        handler.xtc_probability = 0.0
        handler.xtc_threshold = 0.0
        handler.logit_bias = None
        handler.logprobs = logprobs
        handler.repetition_penalty = None
        handler.repetition_context_size = None
        handler.num_draft_tokens = 0
        handler.stream_options = {"include_usage": False}
        handler.tokenizer = DummyTokenizer()
        handler.prompt_cache = PromptCache()
        handler.prompt_cache.cache = []
        handler.prompt_cache.tokens = []
        handler.model_provider = types.SimpleNamespace(
            model_key=("repo", None, None),
            draft_model=None,
            cli_args=types.SimpleNamespace(chat_template_args={}),
        )
        handler.model = object()
        handler.request_id = "req"
        handler.object_type = "chat.completion.chunk"
        handler.wfile = io.BytesIO()
        handler.end_headers = lambda: None
        handler.send_header = lambda *_, **__: None
        handler.generate_response = lambda *_, **__: {}
        handler.completion_usage_response = lambda *_, **__: {}
        handler.get_prompt_cache = lambda prompt: prompt
        handler._ensure_batch_runtime = lambda: handler.batch_runtime
        handler.batch_runtime = object()
        handler.system_fingerprint = "fingerprint"
        self._logprob_value = 0.0
        return handler

    def test_requests_without_tool_call_use_runtime(self):
        handler = self.make_handler()
        handler.tokenizer.has_tool_calling = False
        responses = [
            GenerationResponse(
                text="hello",
                token=1,
                logprobs=FakeLogprobsArray(),
                from_draft=False,
                prompt_tokens=4,
                prompt_tps=0.0,
                generation_tokens=1,
                generation_tps=0.0,
                peak_memory=0.0,
                finish_reason="stop",
            )
        ]
        with mock.patch("mlx_lm.server.stream_generate", return_value=iter([])) as mock_stream:
            with mock.patch(
                "mlx_lm.server.maybe_handle_continuous_batching",
                return_value=("req-batch", iter(responses)),
            ) as mock_batch:
                handler.handle_completion([1, 2, 3], [])
        self.assertTrue(mock_batch.called)
        self.assertFalse(mock_stream.called)

    def test_tool_call_requests_fall_back(self):
        handler = self.make_handler(tools=[{"type": "function"}])
        handler.tokenizer.has_tool_calling = True
        with mock.patch(
            "mlx_lm.server.stream_generate",
            return_value=iter(
                [
                    GenerationResponse(
                        text="legacy",
                        token=1,
                        logprobs=FakeLogprobsArray(),
                        from_draft=False,
                        prompt_tokens=2,
                        prompt_tps=0.0,
                        generation_tokens=1,
                        generation_tps=0.0,
                        peak_memory=0.0,
                        finish_reason="stop",
                    )
                ]
            ),
        ) as mock_stream:
            with mock.patch(
                "mlx_lm.server.maybe_handle_continuous_batching",
                return_value=("req-batch", iter([])),
            ) as mock_batch:
                handler.handle_completion([1, 2], [])
        self.assertFalse(mock_batch.called)
        self.assertTrue(mock_stream.called)

    def test_logprobs_requests_fall_back(self):
        handler = self.make_handler(logprobs=5)
        with mock.patch(
            "mlx_lm.server.stream_generate",
            return_value=iter(
                [
                    GenerationResponse(
                        text="legacy",
                        token=1,
                        logprobs=FakeLogprobsArray(),
                        from_draft=False,
                        prompt_tokens=2,
                        prompt_tps=0.0,
                        generation_tokens=1,
                        generation_tps=0.0,
                        peak_memory=0.0,
                        finish_reason="stop",
                    )
                ]
            ),
        ) as mock_stream:
            with mock.patch(
                "mlx_lm.server.maybe_handle_continuous_batching",
                return_value=("req-batch", iter([])),
            ) as mock_batch:
                handler.handle_completion([1, 2], [])
        self.assertFalse(mock_batch.called)
        self.assertTrue(mock_stream.called)


if __name__ == "__main__":
    unittest.main()
