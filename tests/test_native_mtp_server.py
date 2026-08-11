# Copyright © 2026 Apple Inc.
"""Focused server-contract coverage for request-local native Qwen MTP."""

import http.client
import json
import threading
from http.server import HTTPServer
from types import SimpleNamespace

import mlx.core as mx
import pytest

import mlx_lm.server as server


def _args(*, mtp=True, seed=17, xtc_probability=0.0, max_tokens=4, top_k=3):
    return server.GenerationArguments(
        model=server.ModelDescription("model", "default_model", None),
        sampling=server.SamplingArguments(
            temperature=0.7,
            top_p=0.8,
            top_k=top_k,
            min_p=0.05,
            xtc_probability=xtc_probability,
            xtc_threshold=0.1,
        ),
        logits=server.LogitsProcessorArguments(None, 0, 20, 0, 20, 0, 20),
        stop_words=[],
        max_tokens=max_tokens,
        num_draft_tokens=2,
        logprobs=False,
        top_logprobs=0,
        seed=seed,
        chat_template_kwargs=None,
        mtp=mtp,
    )


class _PromptCache:
    def __init__(self):
        self.fetches = 0
        self.inserts = 0

    def fetch_nearest_cache(self, _key, prompt):
        self.fetches += 1
        return None, prompt

    def insert_cache(self, *_args, **_kwargs):
        self.inserts += 1


class _Queue:
    def __init__(self, *, cancel=False):
        self.items = []
        self.cancel = cancel

    def put(self, item):
        self.items.append(item)
        if isinstance(item, server.GenerationContext):
            self.ctx = item
        elif self.cancel and isinstance(item, server.Response):
            self.ctx.stop()


def _validation_handler(body):
    handler = server.APIHandler.__new__(server.APIHandler)
    handler.body = body
    handler.response_generator = SimpleNamespace(
        cli_args=SimpleNamespace(draft_model=None)
    )
    handler.stream = False
    handler.mtp = body.get("mtp", False)
    handler.max_tokens = 4
    handler.temperature = 0.7
    handler.top_p = 0.8
    handler.top_k = 3
    handler.min_p = 0.0
    handler.num_draft_tokens = 2
    handler.repetition_penalty = 0.0
    handler.repetition_context_size = 20
    handler.presence_penalty = 0.0
    handler.presence_context_size = 20
    handler.frequency_penalty = 0.0
    handler.frequency_context_size = 20
    handler.logprobs = False
    handler.top_logprobs = -1
    handler.xtc_probability = body.get("xtc_probability", 0.0)
    handler.xtc_threshold = 0.1
    handler.requested_model = "model"
    handler.adapter = None
    handler.seed = 17
    handler.logit_bias = None
    return handler


def _response_generator(prompt_cache, *, draft_model=None):
    generator = server.ResponseGenerator.__new__(server.ResponseGenerator)
    generator.model_provider = SimpleNamespace(
        model=SimpleNamespace(
            mtp_capability=SimpleNamespace(supported=True, reason="supported"),
            vocab_size=7,
        ),
        tokenizer=SimpleNamespace(
            has_thinking=False,
            has_tool_calling=False,
            tool_parser=lambda *_: {},
            convert_ids_to_tokens=lambda ids: [str(i) for i in ids],
            encode=lambda _text: [],
            eos_token_ids=set(),
        ),
        draft_model=draft_model,
        model_key=("model", None, None),
        is_batchable=True,
        cli_args=SimpleNamespace(prefill_step_size=8),
    )
    generator.prompt_cache = prompt_cache
    generator._is_distributed = False
    generator._tokenize = lambda *_: ([1, 2], [], [], "normal")
    stop_matcher = SimpleNamespace(make_state=lambda: None, _trie={})
    generator._make_state_machine = lambda *_: (stop_matcher, object())
    generator._log_cache_stats = lambda: None
    return generator


def _completion_stream(closed, *, error=None):
    try:
        if error is not None:
            raise error
        yield SimpleNamespace(
            text="ok",
            token=2,
            logprobs=mx.array([-2.0, -1.0, 0.0]),
            finish_reason="length",
            mtp_drafts=5,
            mtp_accepted=3,
            mtp_bypass_reason=None,
        )
    finally:
        closed.append(True)


def test_native_sampling_maps_public_fields_and_rejects_xtc():
    args = _args()
    config = server._make_native_mtp_sampling_config(args.sampling, args.seed)
    assert config.temperature == 0.7
    assert config.top_p == 0.8
    assert config.top_k == 3
    assert config.min_p == 0.05
    assert config.seed == 17

    xtc = _args(xtc_probability=0.1)
    with pytest.raises(ValueError, match="native_mtp_xtc_unsupported"):
        server._make_native_mtp_sampling_config(xtc.sampling, xtc.seed)


def test_native_mtp_is_never_admitted_to_batching():
    generator = server.ResponseGenerator.__new__(server.ResponseGenerator)
    generator.model_provider = SimpleNamespace(is_batchable=True)
    assert not generator._is_batchable(_args(mtp=True, seed=None))
    assert generator._is_batchable(_args(mtp=False, seed=None))


@pytest.mark.parametrize(
    ("body", "message"),
    (
        (
            {"mtp": True, "draft_model": "assistant"},
            "native_mtp_external_draft_unsupported",
        ),
        (
            {"mtp": True, "prompt_cache": True},
            "native_mtp_prefix_reuse_unsupported",
        ),
        (
            {"mtp": True, "xtc_probability": 0.1},
            "native_mtp_xtc_unsupported",
        ),
    ),
)
def test_native_request_contract_rejects_incompatible_fields(body, message):
    with pytest.raises(ValueError, match=message):
        _validation_handler(body).validate_model_parameters()


def test_single_native_mtp_disables_prompt_cache_and_propagates_metadata(monkeypatch):
    prompt_cache = _PromptCache()
    generator = _response_generator(prompt_cache)
    queue = _Queue()
    captured = {}
    closed = []

    monkeypatch.setattr(
        server.StopSequenceMatcher,
        "match",
        staticmethod(lambda state, _trie, _token: (state, False)),
    )

    def fake_stream_generate(**kwargs):
        captured.update(kwargs)
        return _completion_stream(closed)

    monkeypatch.setattr(server, "stream_generate", fake_stream_generate)
    generator._serve_single((queue, object(), _args()))

    response = next(item for item in queue.items if isinstance(item, server.Response))
    assert captured["mtp"] is True
    assert captured["prompt_cache"] is None
    assert captured["draft_model"] is None
    assert captured["sampler"] is None
    assert captured["mtp_sampling_config"].seed == 17
    assert response.finish_reason == "length"
    assert (response.mtp_drafts, response.mtp_accepted) == (5, 3)
    assert prompt_cache.fetches == prompt_cache.inserts == 0
    assert closed == [True]


def test_non_mtp_single_path_preserves_cache_and_dispatch_shape(monkeypatch):
    prompt_cache = _PromptCache()
    generator = _response_generator(prompt_cache)
    queue = _Queue()
    captured = {}
    closed = []

    monkeypatch.setattr(server, "make_prompt_cache", lambda _model: [object()])
    monkeypatch.setattr(
        server.StopSequenceMatcher,
        "match",
        staticmethod(lambda state, _trie, _token: (state, False)),
    )

    def fake_stream_generate(**kwargs):
        captured.update(kwargs)
        return _completion_stream(closed)

    monkeypatch.setattr(server, "stream_generate", fake_stream_generate)
    generator._serve_single((queue, object(), _args(mtp=False, seed=None)))

    assert "mtp" not in captured
    assert "mtp_sampling_config" not in captured
    assert captured["prompt_cache"] is not None
    assert prompt_cache.fetches == prompt_cache.inserts == 1
    assert closed == [True]


def test_external_draft_fails_before_generation(monkeypatch):
    generator = _response_generator(_PromptCache(), draft_model=object())
    queue = _Queue()
    monkeypatch.setattr(
        server,
        "stream_generate",
        lambda **_kwargs: pytest.fail("generation must not start"),
    )
    generator._serve_single((queue, object(), _args()))
    error = next(item for item in queue.items if isinstance(item, Exception))
    assert "native_mtp_external_draft_unsupported" in str(error)


def test_missing_native_capability_fails_closed_before_generation(monkeypatch):
    generator = _response_generator(_PromptCache())
    generator.model_provider.model = object()
    queue = _Queue()
    monkeypatch.setattr(
        server,
        "stream_generate",
        lambda **_kwargs: pytest.fail("generation must not start"),
    )
    generator._serve_single((queue, object(), _args()))
    error = next(item for item in queue.items if isinstance(item, Exception))
    assert "native_mtp_model_capability_missing" in str(error)


@pytest.mark.parametrize(
    ("args", "prompt", "message"),
    (
        (
            _args(max_tokens=0),
            [1, 2],
            "native_mtp_max_tokens_must_be_positive",
        ),
        (_args(), [], "native_mtp_prompt_must_be_nonempty"),
        (
            _args(top_k=7),
            [1, 2],
            "native MTP top_k must be smaller than vocabulary size",
        ),
    ),
)
def test_native_inputs_are_validated_before_context_or_generation(
    monkeypatch, args, prompt, message
):
    generator = _response_generator(_PromptCache())
    generator._tokenize = lambda *_: (prompt, [], [], "normal")
    queue = _Queue()
    monkeypatch.setattr(
        server,
        "stream_generate",
        lambda **_kwargs: pytest.fail("generation must not start"),
    )
    generator._serve_single((queue, object(), args))
    error = next(item for item in queue.items if isinstance(item, Exception))
    assert message in str(error)
    assert not any(isinstance(item, server.GenerationContext) for item in queue.items)


@pytest.mark.parametrize("error", (None, RuntimeError("synthetic generation error")))
def test_cancellation_and_errors_close_the_underlying_generator(monkeypatch, error):
    generator = _response_generator(_PromptCache())
    queue = _Queue(cancel=error is None)
    closed = []
    monkeypatch.setattr(
        server.StopSequenceMatcher,
        "match",
        staticmethod(lambda state, _trie, _token: (state, False)),
    )
    monkeypatch.setattr(
        server,
        "stream_generate",
        lambda **_kwargs: _completion_stream(closed, error=error),
    )

    generator._serve_single((queue, object(), _args()))
    assert closed == [True]
    if error is not None:
        assert any(isinstance(item, RuntimeError) for item in queue.items)


@pytest.mark.parametrize("stream", (False, True))
def test_terminal_response_metadata_is_exposed_for_stream_and_nonstream(stream):
    handler = server.APIHandler.__new__(server.APIHandler)
    handler.request_id = "request"
    handler.system_fingerprint = "fingerprint"
    handler.object_type = "chat.completion.chunk" if stream else "chat.completion"
    handler.requested_model = "model"
    handler.created = 1
    handler.stream = stream

    response = handler.generate_response(
        "ok",
        "stop",
        None if stream else 2,
        None if stream else 1,
        mtp_drafts=8,
        mtp_accepted=6,
        mtp_bypass_reason="native_mtp_model_capability_missing",
    )
    assert response["choices"][0]["finish_reason"] == "stop"
    assert response["generation_metadata"] == {
        "mtp_drafts": 8,
        "mtp_accepted": 6,
        "mtp_bypass_reason": "native_mtp_model_capability_missing",
    }


class _WireResponseGenerator:
    def __init__(self, *, supported=True):
        self.model = SimpleNamespace(
            mtp_capability=SimpleNamespace(
                supported=supported,
                reason="native_mtp_weights_not_loaded",
            ),
            vocab_size=7,
        )
        self.cli_args = SimpleNamespace(
            num_draft_tokens=2,
            max_tokens=4,
            temp=0.7,
            top_p=0.8,
            top_k=3,
            min_p=0.0,
            draft_model=None,
            allowed_origins=[],
        )

    def generate(self, request, args, progress_callback=None):
        if args.mtp:
            server._make_native_mtp_sampling_config(args.sampling, args.seed)
            server._validate_native_mtp_model(self.model, args.sampling)
            if not request.prompt:
                raise server.NativeMTPRequestError("native_mtp_prompt_must_be_nonempty")
        text_sm = server.TextStateMachine({"normal": []})
        ctx = server.GenerationContext(
            has_tool_calling=False,
            has_thinking=False,
            tool_parser=lambda *_: {},
            text_sm=text_sm,
            initial_state="normal",
            prompt=[1, 2],
            prompt_cache_count=0,
        )

        def responses():
            yield server.Response(
                "ok",
                2,
                -0.1,
                "length",
                (),
                mtp_drafts=5,
                mtp_accepted=3,
                mtp_bypass_reason=None,
            )

        return ctx, responses()


def _wire_request(body, *, supported=True):
    response_generator = _WireResponseGenerator(supported=supported)
    httpd = HTTPServer(
        ("127.0.0.1", 0),
        lambda *args, **kwargs: server.APIHandler(
            response_generator,
            *args,
            system_fingerprint="test-fingerprint",
            **kwargs,
        ),
    )
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=5)
    try:
        connection.request(
            "POST",
            "/v1/completions",
            body=json.dumps(body),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        payload = response.read().decode()
        return response.status, payload
    finally:
        connection.close()
        httpd.shutdown()
        httpd.server_close()
        thread.join()


@pytest.mark.parametrize(
    ("body", "status", "message"),
    (
        ({"prompt": "hi", "mtp": "yes"}, 422, "mtp must be of type bool"),
        (
            {"prompt": "hi", "mtp": True, "xtc_probability": 0.1},
            422,
            "native_mtp_xtc_unsupported",
        ),
        (
            {"prompt": "hi", "mtp": True, "draft_model": "assistant"},
            422,
            "native_mtp_external_draft_unsupported",
        ),
        (
            {"prompt": "hi", "mtp": True, "prompt_cache": True},
            422,
            "native_mtp_prefix_reuse_unsupported",
        ),
        (
            {"prompt": "hi", "mtp": True, "max_tokens": 0},
            422,
            "native_mtp_max_tokens_must_be_positive",
        ),
        (
            {"prompt": "hi", "mtp": True, "stream": True, "max_tokens": 0},
            422,
            "native_mtp_max_tokens_must_be_positive",
        ),
        (
            {"prompt": "hi", "mtp": True, "max_tokens": True},
            422,
            "native_mtp_max_tokens_must_be_positive",
        ),
        (
            {"prompt": "", "mtp": True},
            422,
            "native_mtp_prompt_must_be_nonempty",
        ),
        (
            {"prompt": "", "mtp": True, "stream": True},
            422,
            "native_mtp_prompt_must_be_nonempty",
        ),
        (
            {"prompt": "hi", "mtp": True, "top_k": 7},
            422,
            "native MTP top_k must be smaller than vocabulary size",
        ),
        (
            {"prompt": "hi", "mtp": True, "stream": True, "top_k": 7},
            422,
            "native MTP top_k must be smaller than vocabulary size",
        ),
        ({"mtp": True}, 400, "Request did not contain a prompt"),
    ),
)
def test_wire_validation_errors_are_structured_responses(body, status, message):
    actual_status, payload = _wire_request(body)
    assert actual_status == status
    error = json.loads(payload)["error"]
    assert error["type"] == "invalid_request_error"
    assert message in error["message"]


def test_wire_unsupported_capability_is_unprocessable_not_404():
    status, payload = _wire_request({"prompt": "hi", "mtp": True}, supported=False)
    assert status == 422
    error = json.loads(payload)["error"]
    assert error["code"] == "native_mtp_weights_not_loaded"


@pytest.mark.parametrize("stream", (False, True))
def test_wire_terminal_metadata_for_stream_and_nonstream(stream):
    status, payload = _wire_request(
        {"prompt": "hi", "mtp": True, "stream": stream, "max_tokens": 2}
    )
    assert status == 200
    if stream:
        packets = [
            json.loads(line.removeprefix("data: "))
            for line in payload.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        response = packets[-1]
    else:
        response = json.loads(payload)
    assert response["choices"][0]["finish_reason"] == "length"
    assert response["generation_metadata"] == {
        "mtp_drafts": 5,
        "mtp_accepted": 3,
        "mtp_bypass_reason": None,
    }
