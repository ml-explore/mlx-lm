# Copyright © 2026 Apple Inc.

import asyncio
import json
import math
import pathlib
import unittest
from collections.abc import AsyncIterator
from typing import Any
from unittest import mock

import httpx
import jsonschema
import openai

from mlx_lm import tokenizer_utils
from mlx_lm.generate import TextStateMachine
from mlx_lm.server_v2 import settings
from mlx_lm.server_v2.api import app
from mlx_lm.server_v2.api import openai as openai_api
from mlx_lm.server_v2.api.openai import inputs, outputs, schema
from mlx_lm.server_v2.chat import rendering
from mlx_lm.server_v2.engine import core, errors, protocol
from tests.server_v2 import helpers

SCHEMAS = json.loads(
    pathlib.Path(__file__).with_name("openai_schemas.json").read_text()
)
# The OpenAI schemas of a response and of a stream chunk, by path.
PATH_SCHEMAS = {
    "/v1/chat/completions": (
        "CreateChatCompletionResponse",
        "CreateChatCompletionStreamResponse",
    ),
    "/v1/completions": ("CreateCompletionResponse", "CreateCompletionResponse"),
    "/v1/models": ("ListModelsResponse", None),
}
MESSAGES = [{"role": "user", "content": "Hello!"}]
TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
USAGE = protocol.Usage(prompt_tokens=5, cached_tokens=2, completion_tokens=3)


def check_schema(body: dict, name: str) -> None:
    body_schema = {**SCHEMAS, "$ref": f"#/components/schemas/{name}"}
    jsonschema.validate(body, body_schema, cls=jsonschema.Draft202012Validator)


async def check_response(response: httpx.Response) -> None:
    await response.aread()
    path = response.request.url.path
    if response.status_code >= 400:
        check_schema(response.json(), "ErrorResponse")
    elif path.startswith("/v1/models/"):
        check_schema(response.json(), "Model")
    elif path in PATH_SCHEMAS:
        name, chunk_name = PATH_SCHEMAS[path]
        if not response.headers["content-type"].startswith("text/event-stream"):
            return check_schema(response.json(), name)
        for line in response.text.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                chunk = json.loads(line.removeprefix("data: "))
                check_schema(chunk, "ErrorResponse" if "error" in chunk else chunk_name)


def tokens(content: str = "", **delta: Any) -> protocol.TokensEvent:
    delta = TextStateMachine.Delta(content=content, **delta)
    return protocol.TokensEvent(token_ids=(1,), delta=delta)


def finished(
    reason: protocol.FinishReason = protocol.FinishReason.EOS,
) -> protocol.FinishedEvent:
    return protocol.FinishedEvent(reason=reason, usage=USAGE)


def sse_events(response: httpx.Response) -> list[str]:
    """The data of each server-sent event; the last one is "[DONE]"."""
    events = [e for e in response.text.split("\n\n") if e.startswith("data: ")]
    return [e.removeprefix("data: ") for e in events]


def _merge_chunks(chunks: list[str]) -> tuple[dict, str, list[int]]:
    """The message, finish reason and tool call indexes of a chat stream."""
    message, indexes = {}, []
    for chunk in chunks[:-1]:
        (choice,) = json.loads(chunk)["choices"]
        finish_reason = choice["finish_reason"]
        for key, value in choice["delta"].items():
            if key != "tool_calls":
                message[key] = message.get(key, "") + value
                continue
            for call in value:
                indexes.append(call.pop("index"))
                message.setdefault(key, []).append(call)
    return message, finish_reason, indexes


async def _updates(
    batches: list[list[protocol.Event]],
) -> AsyncIterator[protocol.Update]:
    for events in batches:
        yield protocol.Update.from_events(events)


class TestOpenAI(unittest.IsolatedAsyncioTestCase):

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = helpers.load_chat_tokenizer()
        cls.model, cls.model_tokenizer = helpers.load_model()

    async def asyncSetUp(self):
        self.serve(helpers.StubEngine(tokens("Hi!"), finished()))

    def serve(
        self,
        engine: core.Engine | helpers.StubEngine,
        tokenizer: tokenizer_utils.TokenizerWrapper | None = None,
        **config,
    ) -> None:
        self.engine = engine
        tokenizer = tokenizer or self.tokenizer
        config = settings.ServerConfig(model_id="org/model", **config)
        state = app.ServerState(engine=engine, tokenizer=tokenizer, config=config)
        self.app = app.make_app(state, openai_api.router)
        self.http = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app),
            base_url="http://test",
            event_hooks={"response": [check_response]},
        )
        self.addAsyncCleanup(self.http.aclose)
        self.client = openai.AsyncOpenAI(
            base_url="http://test/v1",
            api_key="none",
            http_client=self.http,
            max_retries=0,
            _strict_response_validation=True,
        )

    async def post(self, path: str, body: dict) -> httpx.Response:
        headers = {"content-type": "application/json"}
        return await self.http.post(path, content=json.dumps(body), headers=headers)

    async def test_chat(self):
        self.serve(
            helpers.StubEngine(tokens("Hi!"), finished(protocol.FinishReason.LENGTH))
        )
        response = await self.client.chat.completions.create(
            model="org/model", messages=MESSAGES
        )
        self.assertEqual(response.model, "org/model")
        (choice,) = response.choices
        self.assertEqual(
            (choice.message.content, choice.finish_reason), ("Hi!", "length")
        )
        usage = response.usage
        self.assertEqual((usage.prompt_tokens, usage.total_tokens), (5, 8))
        self.assertEqual(usage.prompt_tokens_details.cached_tokens, 2)
        request = self.engine.requests[0]
        expected = rendering.from_messages(
            self.tokenizer, MESSAGES, params=request.params, parse_tool_calls=False
        )
        self.assertEqual(request, expected)
        self.assertTrue(request.checkpoints)

        body = {
            "messages": MESSAGES,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        response = await self.post("/v1/chat/completions", body)
        self.assertEqual(response.headers["cache-control"], "no-cache")
        *chunks, done = sse_events(response)
        self.assertEqual(done, "[DONE]")
        chunks = [json.loads(c) for c in chunks]
        deltas = [
            (c["choices"][0]["delta"], c["choices"][0]["finish_reason"])
            for c in chunks[:-1]
        ]
        expected = [
            ({"role": "assistant", "content": ""}, None),
            ({"content": "Hi!"}, None),
            ({}, "length"),
        ]
        self.assertEqual(deltas, expected)
        self.assertEqual([c["usage"] for c in chunks[:-1]], [None] * 3)
        self.assertEqual(chunks[-1]["choices"], [])
        self.assertEqual(chunks[-1]["usage"]["completion_tokens"], 3)

        # Usage only when asked for.
        body = {"messages": MESSAGES, "stream": True}
        for options in ({}, {"stream_options": {"include_usage": False}}):
            response = await self.post("/v1/chat/completions", {**body, **options})
            self.assertNotIn('"usage"', response.text)

        # OpenAI has no cancelled reason.
        self.serve(helpers.StubEngine(finished(protocol.FinishReason.CANCELLED)))
        response = await self.post("/chat/completions", {"messages": MESSAGES})
        choice = response.json()["choices"][0]
        self.assertEqual(
            (choice["message"]["content"], choice["finish_reason"]), (None, "stop")
        )

    async def test_chat_with_engine(self):
        engine = core.Engine(self.model, self.model_tokenizer)
        await engine.start()
        self.addAsyncCleanup(engine.close)
        self.serve(engine, self.model_tokenizer)
        kwargs = {"model": "org/model", "messages": MESSAGES, "max_tokens": 8}
        kwargs.update(logprobs=True, top_logprobs=2)
        response = await self.client.chat.completions.create(**kwargs)
        (choice,) = response.choices
        self.assertEqual(choice.finish_reason, "length")
        self.assertTrue(choice.message.content)
        self.assertEqual(len(choice.logprobs.content), 8)
        self.assertEqual(len(choice.logprobs.content[0].top_logprobs), 2)

        stream = await self.client.chat.completions.create(
            **kwargs, stream=True, stream_options={"include_usage": True}
        )
        content, num_logprobs, usage = "", 0, None
        async for chunk in stream:
            if chunk.choices:
                content += chunk.choices[0].delta.content or ""
            if chunk.choices and chunk.choices[0].logprobs:
                num_logprobs += len(chunk.choices[0].logprobs.content)
            usage = chunk.usage or usage
        self.assertEqual(content, choice.message.content)
        self.assertEqual((usage.completion_tokens, num_logprobs), (8, 8))
        self.assertGreater(usage.prompt_tokens_details.cached_tokens, 0)

        response = await self.client.completions.create(
            model="org/model", prompt="One, two,", max_tokens=4, logprobs=1
        )
        logprobs = response.choices[0].logprobs
        self.assertIn(logprobs.tokens[0], logprobs.top_logprobs[0])
        # One per generated token, the end token too.
        self.assertEqual(len(logprobs.tokens), response.usage.completion_tokens)

    async def test_completion(self):
        response = await self.client.completions.create(
            model="org/model", prompt="One, two,", max_tokens=4
        )
        self.assertEqual(response.choices[0].text, "Hi!")
        self.assertIsNone(response.choices[0].logprobs)
        self.assertEqual(response.usage.completion_tokens, 3)
        # A completion is plain text: the engine does not parse it.
        request = self.engine.requests[0]
        self.assertEqual(request.tokens, tuple(self.tokenizer.encode("One, two,")))
        self.assertIsNone(request.text_state)

        body = {
            "prompt": "One",
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        *chunks, done = sse_events(await self.post("/v1/completions", body))
        chunk, usage = [json.loads(c) for c in chunks]
        self.assertEqual(done, "[DONE]")
        self.assertEqual(chunk["choices"][0]["text"], "Hi!")
        self.assertEqual(chunk["choices"][0]["finish_reason"], "stop")
        self.assertEqual((usage["choices"], usage["usage"]["total_tokens"]), ([], 8))

    async def test_logprobs(self):
        # JSON has no -inf, and a token inside a character has no bytes.
        cut, plain = 94, 1000
        sampled = protocol.TokenLogprob(
            token=cut, logprob=-math.inf, top=((plain, -2.0),)
        )
        event = tokens("Hi")
        event = protocol.TokensEvent(
            token_ids=(cut,), delta=event.delta, logprobs=(sampled,)
        )
        self.serve(helpers.StubEngine(event, finished()))
        body = {"messages": MESSAGES, "logprobs": True}
        response = await self.post("/v1/chat/completions", body)
        (entry,) = response.json()["choices"][0]["logprobs"]["content"]
        self.assertEqual((entry["logprob"], entry["bytes"]), (-9999.0, None))
        self.assertEqual(entry["top_logprobs"][0]["logprob"], -2.0)

        response = await self.post("/v1/completions", {"prompt": "One", "logprobs": 1})
        logprobs = response.json()["choices"][0]["logprobs"]
        decode = self.tokenizer.decode
        top = {decode([plain]): -2.0, decode([cut]): -9999.0}
        # The OpenAI API lists the sampled token too, if it is not in the top.
        self.assertEqual(logprobs["top_logprobs"], [top])

    async def test_stream_equals_response(self):
        call = '{"name": "f", "id": "abc", "arguments": {}}'
        events = [
            tokens(reasoning="Hm."),
            tokens("Hi", tool_calls=(call,)),
            tokens(" there", tool_calls=(call,)),
            tokens("."),
            finished(),
        ]
        call = {
            "id": "abc",
            "type": "function",
            "function": {"name": "f", "arguments": "{}"},
        }
        message = {
            "role": "assistant",
            "content": "Hi there.",
            "reasoning": "Hm.",
            "reasoning_content": "Hm.",
            "tool_calls": [call, call],
        }
        chat = {
            "messages": MESSAGES,
            "tools": TOOLS,
            "chat_template_kwargs": {"think": True},
        }
        completion = {"prompt": "One"}
        for batches in [[events], [[e] for e in events], [events[:2], [], events[2:]]]:
            first = len(self.engine.requests)
            with mock.patch.object(
                self.engine,
                "updates",
                side_effect=lambda handle, **kwargs: _updates(batches),
            ):
                response = await self.post("/v1/chat/completions", chat)
                stream = await self.post(
                    "/v1/chat/completions", {**chat, "stream": True}
                )
                text = await self.post("/v1/completions", completion)
                text_stream = await self.post(
                    "/v1/completions", {**completion, "stream": True}
                )
                single = await self.post(
                    "/v1/chat/completions", {**chat, "parallel_tool_calls": False}
                )
            self.assertEqual(self.engine.requests[first].text_state, "reasoning")
            (choice,) = response.json()["choices"]
            self.assertEqual(
                (choice["message"], choice["finish_reason"]), (message, "tool_calls")
            )
            self.assertEqual(
                _merge_chunks(sse_events(stream)), (message, "tool_calls", [0, 1])
            )
            self.assertEqual(text.json()["choices"][0]["text"], "Hi there.")
            # Only the first call goes out, whatever the batches.
            self.assertEqual(
                single.json()["choices"][0]["message"]["tool_calls"], [call]
            )
            chunks = [
                json.loads(c)["choices"][0]["text"]
                for c in sse_events(text_stream)[:-1]
            ]
            self.assertEqual("".join(chunks), "Hi there.")
            for response in (stream, text_stream):
                self.assertEqual(response.text.count(": keepalive"), batches.count([]))

    async def test_tool_calls(self):
        call = '{"name": "f", "arguments": {"a": "\\u00e9"}}'
        first = tokens("Hi.", tool_calls=(call, call))
        self.serve(helpers.StubEngine(first, tokens(" Bye."), finished()))
        kwargs = {"model": "org/model", "messages": MESSAGES, "tools": TOOLS}
        response = await self.client.chat.completions.create(**kwargs)
        (choice,) = response.choices
        self.assertEqual(
            (choice.message.content, choice.finish_reason), ("Hi. Bye.", "tool_calls")
        )
        arguments = [c.function.arguments for c in choice.message.tool_calls]
        # Not escaped, as in mlx_lm.server.
        self.assertEqual(arguments, ['{"a": "\u00e9"}'] * 2)

        # With tool_choice "none", the template shows the tools, but a call is text.
        with mock.patch.object(
            rendering, "from_messages", wraps=rendering.from_messages
        ) as from_messages:
            none = await self.client.chat.completions.create(
                **kwargs, tool_choice="none"
            )
        self.assertEqual(from_messages.call_args.kwargs["tools"], TOOLS)
        self.assertFalse(self.engine.requests[-1].parse_tool_calls)
        (choice,) = none.choices
        text = f"<tool_call>{call}</tool_call>"
        self.assertEqual(
            (choice.message.content, choice.finish_reason),
            ("Hi." + text + text + " Bye.", "stop"),
        )

        # Only the first call goes out; the text after it stays.
        single = await self.client.chat.completions.create(
            **kwargs, parallel_tool_calls=False
        )
        (choice,) = single.choices
        self.assertEqual(
            (choice.message.content, choice.finish_reason), ("Hi. Bye.", "tool_calls")
        )
        self.assertEqual(len(choice.message.tool_calls), 1)

        # A call does not turn a cut-off answer into a tool call answer.
        self.serve(helpers.StubEngine(first, finished(protocol.FinishReason.LENGTH)))
        cut = await self.client.chat.completions.create(**kwargs)
        self.assertEqual(cut.choices[0].finish_reason, "length")

    async def test_params(self):
        defaults = {"temperature": 0.5, "top_k": 5}
        self.serve(self.engine, default_params=settings.GenerationParams(**defaults))
        sampling = {
            "top_k": 3,
            "min_p": 0.1,
            "xtc_probability": 0.5,
            "xtc_threshold": 0.2,
            "repetition_penalty": 1.1,
            "repetition_context_size": 5,
            "presence_penalty": 0.5,
            "presence_context_size": 5,
            "frequency_penalty": 0.5,
            "frequency_context_size": 5,
        }
        cases = [
            ({}, {}),
            (sampling, sampling),
            ({"temperature": 0.0, "top_p": 0.5}, {"temperature": 0.0, "top_p": 0.5}),
            ({"top_k": -1}, {"top_k": 0}),
            ({"stop": "x"}, {"stop": ("x",)}),
            ({"stop": ["x", "y"]}, {"stop": ("x", "y")}),
            ({"logit_bias": {"5": 1.5}}, {"logit_bias": {5: 1.5}}),
            ({"max_tokens": 3, "max_completion_tokens": 7}, {"max_tokens": 7}),
            ({"seed": 1, "temperature": 0.0, "n": 1}, {"temperature": 0.0}),
            # Sampled token only.
            ({"logprobs": True}, {"logprobs": 0}),
            # Fields that the server does not know are ignored.
            ({"user": "u", "metadata": {}, "response_format": {"type": "text"}}, {}),
        ]
        for body, params in cases:
            with self.subTest(**body):
                response = await self.post(
                    "/v1/chat/completions", {"messages": MESSAGES, **body}
                )
                self.assertEqual(response.status_code, 200)
                expected = settings.GenerationParams(**{**defaults, **params})
                self.assertEqual(self.engine.requests[-1].params, expected)

    async def test_completion_params(self):
        body = {"prompt": "One", "max_tokens": 3, "logprobs": 2, "stop": "xy"}
        response = await self.post("/v1/completions", {**body, "top_k": 7})
        self.assertEqual(response.status_code, 200)
        expected = settings.GenerationParams(
            max_tokens=3, logprobs=2, stop=("xy",), top_k=7
        )
        self.assertEqual(self.engine.requests[-1].params, expected)

    async def test_reasoning_effort(self):
        self.serve(self.engine, template_kwargs={"think": False})
        low, high = {"reasoning_effort": "low"}, {"reasoning_effort": "high"}
        cases = [
            ({}, {"think": False}),
            (low, {"think": False, **low}),
            ({**low, "chat_template_kwargs": high}, {"think": False, **high}),
        ]
        for body, expected in cases:
            with self.subTest(**body):
                with mock.patch.object(
                    rendering, "from_messages", wraps=rendering.from_messages
                ) as from_messages:
                    body = {"messages": MESSAGES, **body}
                    await self.post("/v1/chat/completions", body)
                self.assertEqual(
                    from_messages.call_args.kwargs["template_kwargs"], expected
                )

    async def test_bad_requests(self):
        self.serve(
            self.engine, default_params=settings.GenerationParams(temperature=0.5)
        )
        unsupported = "unsupported_value"
        chat = [
            ({"n": 2}, "n", unsupported),
            # The default temperature is not 0.
            ({"seed": 1}, "seed", unsupported),
            ({"stop": 5}, "stop", None),
            ({"stop": ["x"] * 17}, "stop", None),
            ({"logit_bias": {"a": 1}}, "logit_bias", None),
            ({"logit_bias": {"5": 101}}, "logit_bias", None),
            ({"presence_penalty": 3}, "presence_penalty", None),
            ({"messages": []}, "messages", None),
            (
                {"messages": [{"role": "user", "content": [{"type": "image_url"}]}]},
                "messages",
                unsupported,
            ),
            (
                {"response_format": {"type": "json_object"}},
                "response_format",
                unsupported,
            ),
            ({"top_logprobs": 2}, "top_logprobs", None),
            ({"logprobs": True, "top_logprobs": 21}, "top_logprobs", None),
            ({"tool_choice": "required"}, "tool_choice", unsupported),
            ({"tools": [*TOOLS, {"type": "custom"}]}, "tools", unsupported),
            ({"frequency_penalty": 3}, "frequency_penalty", None),
            ({"chat_template_kwargs": {"max_length": 4}}, "chat_template_kwargs", None),
            ({"chat_template_kwargs": {"tools": []}}, "chat_template_kwargs", None),
        ]
        completion = [
            ({"prompt": ["One"]}, "prompt", unsupported),
            ({"prompt": [1, 2]}, "prompt", unsupported),
            ({"prompt": [[1, 2]]}, "prompt", unsupported),
            ({"prompt": 5}, "prompt", None),
            ({"echo": True}, "echo", unsupported),
            ({"suffix": "x"}, "suffix", unsupported),
            ({"best_of": 2}, "best_of", unsupported),
            ({"logprobs": 6}, "logprobs", None),
            ({"prompt": "\ud800"}, None, None),
        ]
        for path, base, cases in (
            ("/v1/chat/completions", {"messages": MESSAGES}, chat),
            ("/v1/completions", {"prompt": "One"}, completion),
        ):
            for body, param, code in cases:
                with self.subTest(path=path, **body):
                    response = await self.post(path, {**base, **body})
                    error = response.json()["error"]
                    self.assertEqual(response.status_code, 400)
                    self.assertEqual((error["param"], error["code"]), (param, code))
                    self.assertEqual(error["type"], "invalid_request_error")
        # The message names the temperature that the server default sets.
        response = await self.post("/v1/completions", {"prompt": "One", "seed": 1})
        self.assertIn("temperature is 0.5", response.json()["error"]["message"])
        # Without the JSON content type, the body is not JSON.
        body = json.dumps({"prompt": "One"})
        response = await self.http.post("/v1/completions", content=body)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.engine.requests, [])

    async def test_failure(self):
        self.serve(
            helpers.StubEngine(
                tokens("Hi"), protocol.FailedEvent(error=errors.EngineError("Injected"))
            )
        )
        response = await self.post("/v1/chat/completions", {"messages": MESSAGES})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.headers["x-should-retry"], "false")
        # The stream fails after it starts.
        for path, body in (
            ("/v1/chat/completions", {"messages": MESSAGES}),
            ("/v1/completions", {"prompt": "One"}),
        ):
            response = await self.post(path, {**body, "stream": True})
            *_, error, done = sse_events(response)
            self.assertEqual(
                (json.loads(error)["error"]["type"], done), ("server_error", "[DONE]")
            )

        closed = errors.UnavailableError(core.EngineStatus.CLOSED)
        self.serve(helpers.StubEngine(protocol.FailedEvent(error=closed)))
        response = await self.post("/v1/completions", {"prompt": "One"})
        self.assertEqual(response.status_code, 503)

        # The engine refuses at submit: a stream gets a plain 503, not an event.
        self.engine.status = core.EngineStatus.CLOSED
        for body in ({"messages": MESSAGES}, {"messages": MESSAGES, "stream": True}):
            response = await self.post("/v1/chat/completions", body)
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.headers["x-should-retry"], "false")

    async def test_models(self):
        models = await self.client.models.list()
        self.assertEqual([m.id for m in models.data], ["org/model"])
        model = await self.client.models.retrieve("org/model")
        self.assertEqual((model.id, model.max_model_len), ("org/model", 4096))
        self.assertEqual(models.data, [model])
        with self.assertRaises(openai.NotFoundError):
            await self.client.models.retrieve("other")
        with self.assertRaises(openai.NotFoundError) as e:
            await self.client.chat.completions.create(model="other", messages=MESSAGES)
        self.assertEqual(e.exception.code, "model_not_found")
        with self.assertRaises(openai.NotFoundError):
            await self.client.completions.create(model="other", prompt="Hi")

        self.engine.config = settings.EngineConfig()
        for path in ("/v1/models/org/model", "/v1/models"):
            response = await self.http.get(path)
            self.assertEqual(response.status_code, 200)
            self.assertNotIn("max_model_len", response.text)

    def test_template_message(self):
        def tool_call(arguments: str | dict) -> dict:
            function = {"name": "f", "arguments": arguments}
            return {"id": "a", "type": "function", "function": function}

        parts = [{"type": "text", "text": "Hi"}, {"type": "text"}]
        calls = [tool_call('{"a": 1}'), tool_call({"a": 2}), tool_call("")]
        assistant = {"role": "assistant", "content": None, "reasoning": "Hmm."}
        messages = [
            {"role": "developer", "content": "Be brief."},
            {"role": "user", "content": parts},
            {**assistant, "tool_calls": calls},
            {"role": "assistant", "content": "a", "tool_calls": []},
            {"role": "assistant", "content": "b", "tool_calls": None},
        ]
        # Templates read the arguments as a mapping.
        calls = [tool_call({"a": 1}), tool_call({"a": 2}), tool_call({})]
        assistant = {"role": "assistant", "content": "", "reasoning_content": "Hmm."}
        expected = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
            {**assistant, "tool_calls": calls},
            # Llama 3 templates fail on an empty list.
            {"role": "assistant", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]
        messages = [schema.RequestMessage.model_validate(m) for m in messages]
        self.assertEqual([inputs.template_message(m) for m in messages], expected)

    async def test_bad_messages(self):
        tool_calls = [
            "x",
            ["x"],
            [{"id": "a"}],
            [{"function": {"name": "f", "arguments": "{"}}],
            [{"function": {"name": "f", "arguments": "[" * 10**5}}],
            *[
                [{"function": {"name": "f", "arguments": arguments}}]
                for arguments in ("null", "[1]", "5", '"s"', None, 5, [1])
            ],
        ]
        chats = [
            [{"content": "Hi"}],
            [{"role": "user", "content": 5}],
            [{"role": "user", "content": ["Hi"]}],
            [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}],
            [{"role": "user", "content": [{"type": "text", "text": 5}]}],
            *[[{"role": "assistant", "tool_calls": tc}] for tc in tool_calls],
        ]
        for messages in chats:
            with self.subTest(messages=messages):
                body = {"messages": messages}
                response = await self.post("/v1/chat/completions", body)
                self.assertEqual(response.status_code, 400)

    def test_logprob_is_finite(self):
        for logprob, expected in [
            (-1.5, -1.5),
            (float("-inf"), -9999.0),
            (-1e9, -9999.0),
            (float("nan"), -9999.0),
        ]:
            entry = outputs._logprob_fields(self.model_tokenizer, 5, logprob)
            self.assertEqual(entry["logprob"], expected)

    async def test_disconnect_cancels(self):
        for stream in (True, False):
            engine = helpers.StubEngine()
            self.serve(engine)
            body = json.dumps({"messages": MESSAGES, "stream": stream}).encode()
            messages = [{"type": "http.request", "body": body}]
            scope = {
                "type": "http",
                "method": "POST",
                "path": "/v1/chat/completions",
                "query_string": b"",
                "headers": [(b"content-type", b"application/json")],
            }

            async def receive() -> dict:
                if messages:
                    return messages.pop()
                # The client goes away once the engine has the request.
                await engine.submitted.wait()
                return {"type": "http.disconnect"}

            async def send(message: dict) -> None:
                pass

            await self.app(scope, receive, send)
            await asyncio.wait_for(engine.cancel_called.wait(), timeout=10)
            self.assertEqual(engine.cancelled, [0])


if __name__ == "__main__":
    unittest.main()
