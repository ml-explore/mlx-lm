# Copyright © 2026 Apple Inc.

import json
import pathlib
import unittest
from collections.abc import AsyncIterator
from typing import Any
from unittest import mock

import httpx
import jsonschema

from mlx_lm import tokenizer_utils
from mlx_lm.generate import TextStateMachine
from mlx_lm.server_v2 import settings
from mlx_lm.server_v2.api import anthropic, app, openai
from mlx_lm.server_v2.api.anthropic import inputs, schema
from mlx_lm.server_v2.chat import rendering
from mlx_lm.server_v2.engine import core, errors, protocol
from tests.server_v2 import helpers

SCHEMAS = json.loads(
    pathlib.Path(__file__).with_name("anthropic_schemas.json").read_text()
)
MESSAGES = [{"role": "user", "content": "Hello!"}]
TOOLS = [{"name": "f", "input_schema": {"type": "object"}}]
USAGE = protocol.Usage(prompt_tokens=5, cached_tokens=2, completion_tokens=3)


def check_schema(body: dict, name: str) -> None:
    body_schema = {**SCHEMAS, "$ref": f"#/components/schemas/{name}"}
    jsonschema.validate(body, body_schema, cls=jsonschema.Draft202012Validator)


async def check_response(response: httpx.Response) -> None:
    """Check an Anthropic response against the types of the Anthropic SDK."""
    await response.aread()
    path = response.request.url.path
    if not path.startswith("/v1/messages"):
        return
    if response.status_code >= 400:
        check_schema(response.json(), "ErrorResponse")
    elif path.endswith("/count_tokens"):
        check_schema(response.json(), "MessageTokensCount")
    elif not response.headers["content-type"].startswith("text/event-stream"):
        check_schema(response.json(), "Message")
    else:
        for event in sse_events(response):
            if event["type"] == "error":
                check_schema(event, "ErrorResponse")
            elif event["type"] != "ping":
                check_schema(event, "RawMessageStreamEvent")


def tokens(content: str = "", **delta: Any) -> protocol.TokensEvent:
    delta = TextStateMachine.Delta(content=content, **delta)
    return protocol.TokensEvent(token_ids=(1,), delta=delta)


def finished(
    reason: protocol.FinishReason = protocol.FinishReason.EOS, **kwargs: Any
) -> protocol.FinishedEvent:
    return protocol.FinishedEvent(reason=reason, usage=USAGE, **kwargs)


def sse_events(response: httpx.Response) -> list[dict]:
    """The data of each server-sent event; the event name must be its type."""
    events = []
    for text in response.text.split("\n\n")[:-1]:
        name, data = text.split("\n")
        event = json.loads(data.removeprefix("data: "))
        if name != f"event: {event['type']}":
            raise AssertionError(f"The event {name!r} has the type {event['type']}.")
        events.append(event)
    return events


def _fold(events: list[dict]) -> dict:
    """The message of a stream, built as the Anthropic SDK builds it."""
    message = None
    for event in events:
        match event["type"]:
            case "message_start":
                message = event["message"]
            case "content_block_start":
                if event["index"] != len(message["content"]):
                    raise AssertionError(f"A block starts at index {event['index']}.")
                message["content"].append(event["content_block"])
            case "content_block_delta":
                block, delta = message["content"][event["index"]], event["delta"]
                if delta["type"] == "input_json_delta":
                    block["input"] = json.loads(delta["partial_json"])
                else:
                    key = delta["type"].removesuffix("_delta")
                    block[key] += delta[key]
            case "message_delta":
                message.update(event["delta"])
                message["usage"] = event["usage"]
    return message


async def _updates(
    batches: list[list[protocol.Event]],
) -> AsyncIterator[protocol.Update]:
    for events in batches:
        yield protocol.Update.from_events(events)


class TestAnthropic(unittest.IsolatedAsyncioTestCase):

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
        self.app = app.make_app(state, openai.router, anthropic.router)
        self.http = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app),
            base_url="http://test",
            event_hooks={"response": [check_response]},
        )
        self.addAsyncCleanup(self.http.aclose)

    async def post(self, body: dict, path: str = "/v1/messages") -> httpx.Response:
        return await self.http.post(path, json={"max_tokens": 8, **body})

    async def test_message(self):
        call = '{"name": "f", "id": "abc", "arguments": {"a": 1}}'
        events = [
            tokens(reasoning="Hm."),
            tokens(reasoning=" Ok."),
            tokens("Hi", tool_calls=(call, call)),
            tokens(" there"),
            tokens("."),
            finished(),
        ]
        tool_use = {"type": "tool_use", "id": "abc", "name": "f", "input": {"a": 1}}
        content = [
            {"type": "thinking", "thinking": "Hm. Ok.", "signature": ""},
            {"type": "text", "text": "Hi"},
            tool_use,
            tool_use,
            {"type": "text", "text": " there."},
        ]
        usage = {
            "input_tokens": 3,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 2,
            "output_tokens": 3,
        }
        start, delta, stop = (
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
        )
        # The first and the last block come in two events, so they have two deltas.
        two, one = [start, delta, delta, stop], [start, delta, stop]
        blocks = two + one * 3 + two
        event_types = ["message_start", *blocks, "message_delta", "message_stop"]
        body = {"messages": MESSAGES, "tools": TOOLS}
        single = {"type": "auto", "disable_parallel_tool_use": True}
        for batches in [[events], [[e] for e in events], [events[:2], events[2:]]]:
            with mock.patch.object(
                self.engine,
                "updates",
                side_effect=lambda handle: _updates(batches),
            ):
                response = (await self.post(body)).json()
                stream = await self.post({**body, "stream": True})
                one_call = (await self.post({**body, "tool_choice": single})).json()
            self.assertTrue(response.pop("id").startswith("msg_"))
            self.assertEqual(
                response,
                {
                    "type": "message",
                    "role": "assistant",
                    "model": "org/model",
                    "content": content,
                    "stop_reason": "tool_use",
                    "stop_sequence": None,
                    "usage": usage,
                },
            )
            stream_events = sse_events(stream)
            self.assertEqual([e["type"] for e in stream_events], event_types)
            message = _fold(stream_events)
            del message["id"]
            self.assertEqual(message, response)
            # Only the first call goes out; the text after it stays.
            self.assertEqual(one_call["content"], content[:3] + content[4:])

    async def test_stop_reasons(self):
        text = tokens("Hi")
        call = tokens("Hi", tool_calls=('{"name": "f", "arguments": {}}',))
        stop = finished(protocol.FinishReason.STOP_STRING, stop_string="##")
        length = finished(protocol.FinishReason.LENGTH)
        cases = [
            (text, stop, "stop_sequence", "##"),
            # The client must run the calls, also after a stop sequence.
            (call, stop, "tool_use", None),
            # A call does not turn a cut-off reply into a tool_use reply.
            (call, length, "max_tokens", None),
        ]
        body = {"messages": MESSAGES, "tools": TOOLS}
        for content, end, reason, stop_sequence in cases:
            with self.subTest(reason=end.reason, calls=content.delta.tool_calls):
                self.serve(helpers.StubEngine(content, end))
                response = (await self.post(body)).json()
                stream = await self.post({**body, "stream": True})
                expected = {"stop_reason": reason, "stop_sequence": stop_sequence}
                self.assertEqual({k: response[k] for k in expected}, expected)
                self.assertEqual(sse_events(stream)[-2]["delta"], expected)

    async def test_request(self):
        self.serve(self.engine, template_kwargs={"x": 1})
        image = {"type": "image", "source": {"type": "base64", "data": "AA=="}}
        document = {"type": "document", "source": {"type": "text", "data": "D"}}
        body = {
            "model": "any",
            "metadata": {"user_id": "u"},
            "system": [
                {"type": "text", "text": "x-anthropic-billing-header: cch=1;"},
                {"type": "text", "text": "Be"},
                {"type": "text", "text": "it."},
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Weather?", "cache_control": {}},
                        image,
                    ],
                },
                {"role": "system", "content": "Note."},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Hm.", "signature": "s"},
                        {"type": "redacted_thinking", "data": "d"},
                        {"type": "text", "text": "I look."},
                        {"type": "tool_use", "id": "a", "name": "f", "input": {}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "a",
                            "content": [{"type": "text", "text": "Sun"}, document],
                        },
                        {"type": "text", "text": "Thanks."},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "b", "name": "f", "input": {}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "b", "content": "Rain"},
                    ],
                },
            ],
            "tools": [{**TOOLS[0], "description": "Weather"}],
            "thinking": {"type": "disabled"},
            "output_config": {"effort": "low"},
            "chat_template_kwargs": {"x": 2, "reasoning_effort": "high"},
            "temperature": 0.5,
            "top_p": 0.9,
            "top_k": 4,
            "stop_sequences": ["x"],
        }
        calls = [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "f", "arguments": {}},
            }
            for call_id in ("a", "b")
        ]
        # Tool results come first; only tool results add no message of their own.
        # Images and documents of older messages are notes.
        messages = [
            {"role": "system", "content": "Be\nit."},
            {
                "role": "user",
                "content": "Weather?\n[Image omitted: the server does not support images.]",
            },
            {"role": "system", "content": "Note."},
            {
                "role": "assistant",
                "content": "I look.",
                "reasoning_content": "Hm.",
                "tool_calls": calls[:1],
            },
            {
                "role": "tool",
                "tool_call_id": "a",
                "content": "Sun\n[Document omitted: the server does not support documents.]",
            },
            {"role": "user", "content": "Thanks."},
            {"role": "assistant", "content": "", "tool_calls": calls[1:]},
            {"role": "tool", "tool_call_id": "b", "content": "Rain"},
        ]
        function = {
            "name": "f",
            "description": "Weather",
            "parameters": TOOLS[0]["input_schema"],
        }
        with mock.patch.object(
            rendering, "from_messages", wraps=rendering.from_messages
        ) as from_messages:
            response = await self.post(body)
            count = await self.post(body, "/v1/messages/count_tokens")
        self.assertEqual(response.status_code, 200)
        for args in from_messages.call_args_list:
            self.assertEqual(args.args[1], messages)
            self.assertEqual(
                args.kwargs["tools"], [{"type": "function", "function": function}]
            )
            kwargs = {"x": 2, "enable_thinking": False, "reasoning_effort": "high"}
            self.assertEqual(args.kwargs["template_kwargs"], kwargs)
        request = self.engine.requests[-1]
        self.assertEqual(count.json(), {"input_tokens": len(request.tokens)})
        for thinking in ("adaptive", "enabled"):
            fields = {
                "messages": MESSAGES,
                "system": "Be brief.",
                "thinking": {"type": thinking, "budget_tokens": 1024},
                "output_config": {"effort": "low"},
            }
            body = schema.CountTokensRequest.model_validate(fields)
            kwargs = {"enable_thinking": True, "reasoning_effort": "low"}
            self.assertEqual(
                inputs.template_kwargs(body, tokenizer=self.tokenizer), kwargs
            )
            system = {"role": "system", "content": "Be brief."}
            self.assertEqual(inputs.template_messages(body)[0], system)
        params = {"temperature": 0.5, "top_p": 0.9, "top_k": 4, "stop": ("x",)}
        expected = settings.GenerationParams(max_tokens=8, **params)
        self.assertEqual((request.params, request.parse_tool_calls), (expected, True))

        # With tool_choice "none", the template shows the tools, but a call is text.
        call = '{"name": "f", "arguments": {}}'
        self.serve(helpers.StubEngine(tokens(tool_calls=(call,)), finished()))
        body = {"messages": MESSAGES, "tools": TOOLS, "tool_choice": {"type": "none"}}
        response = (await self.post(body)).json()
        self.assertFalse(self.engine.requests[-1].parse_tool_calls)
        await self.post({"messages": MESSAGES})
        self.assertFalse(self.engine.requests[-1].parse_tool_calls)
        text = {"type": "text", "text": f"<tool_call>{call}</tool_call>"}
        self.assertEqual(
            (response["content"], response["stop_reason"]), ([text], "end_turn")
        )

        # Counting needs no tool call parser.
        self.serve(self.engine, helpers.load_tokenizer())
        body = {"messages": MESSAGES, "tools": TOOLS}
        count = await self.post(body, "/v1/messages/count_tokens")
        self.assertEqual(count.status_code, 200)
        self.assertEqual((await self.post(body)).status_code, 400)

        # As in Qwen3.8: an effort the template refuses is left out.
        template = (
            "{% if reasoning_effort|default('low') != 'low' %}"
            "{{ raise_exception('Unexpected reasoning effort.') }}"
            "{% endif %}{{ messages[0].content }}"
        )
        self.serve(self.engine, helpers.load_tokenizer(chat_template=template))
        body = {"messages": MESSAGES, "output_config": {"effort": "high"}}
        count = await self.post(body, "/v1/messages/count_tokens")
        self.assertEqual(count.status_code, 200)

    async def test_bad_requests(self):
        image = {"type": "image", "source": {"type": "base64", "data": "AA=="}}
        tool_result = {"type": "tool_result", "tool_use_id": "a", "content": [image]}
        user = {"role": "user", "content": [image]}
        system = {"role": "system", "content": "Note."}
        images = [{"messages": [user]}, {"messages": [user, system]}]
        prompt = [
            *images,
            {"messages": []},
            {"tools": [{**TOOLS[0], "type": "web_search_20250305"}]},
            {"chat_template_kwargs": {"tools": []}},
            {"output_config": {"effort": "minimal"}},
        ]
        # count_tokens takes the first four.
        generation = [
            {"messages": [*MESSAGES, {"role": "assistant", "content": "Hi"}]},
            {"tool_choice": {"type": "any"}},
            {"tool_choice": {"type": "tool", "name": "f"}},
            {"output_config": {"format": {"type": "json_schema", "schema": {}}}},
            {"stop_sequences": ["x"] * 17},
        ]
        error = {
            "type": "error",
            "error": {"type": "invalid_request_error", "message": mock.ANY},
        }
        for path, cases in (
            ("/v1/messages", prompt + generation),
            ("/v1/messages/count_tokens", prompt),
        ):
            for case in cases:
                with self.subTest(path=path, **case):
                    response = await self.post({"messages": MESSAGES, **case}, path)
                    self.assertEqual(
                        (response.status_code, response.json()), (400, error)
                    )
        for case in generation[:4]:
            body = {"messages": MESSAGES, **case}
            response = await self.post(body, "/v1/messages/count_tokens")
            self.assertEqual(response.status_code, 200)
        # The message names the Anthropic field.
        response = await self.post({"messages": MESSAGES, **generation[-1]})
        self.assertTrue(
            response.json()["error"]["message"].startswith("stop_sequences")
        )
        bad_block = {"messages": [{"role": "user", "content": [{"type": "text"}]}]}
        message = (await self.post(bad_block)).json()["error"]["message"]
        self.assertEqual(message, "messages.0.content.0.text.text: Field required")
        for case in images:
            response = await self.post(case)
            message = response.json()["error"]["message"]
            self.assertEqual(message, "Images and documents are not supported.")
        tool_image = {"messages": [{"role": "user", "content": [tool_result]}]}
        self.assertEqual((await self.post(tool_image)).status_code, 200)
        self.assertEqual((await self.http.head("/api/hello")).status_code, 200)

    async def test_failure(self):
        failed = protocol.FailedEvent(error=errors.EngineError("Injected"))
        self.serve(helpers.StubEngine(tokens("Hi"), failed))
        response = await self.post({"messages": MESSAGES})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.headers["x-should-retry"], "false")
        self.assertEqual(response.json()["error"]["type"], "api_error")
        # The stream fails after it starts.
        stream = await self.post({"messages": MESSAGES, "stream": True})
        start, *_, error = sse_events(stream)
        self.assertEqual(
            (start["type"], error["error"]["type"]), ("message_start", "api_error")
        )

    async def test_context_length(self):
        length = finished(protocol.FinishReason.LENGTH)
        self.serve(helpers.StubEngine(tokens("Hi"), length))
        count = await self.post({"messages": MESSAGES}, "/v1/messages/count_tokens")
        num_tokens = count.json()["input_tokens"]
        response = await self.post({"messages": MESSAGES, "max_tokens": 5000})
        self.assertEqual(
            response.json()["stop_reason"], "model_context_window_exceeded"
        )
        self.assertEqual(self.engine.requests[-1].params.max_tokens, 4096 - num_tokens)

        long = [{"role": "user", "content": "Hello! " * 4096}]
        response = await self.post({"messages": long})
        self.assertRegex(
            response.json()["error"]["message"],
            r"^prompt is too long: \d+ tokens > 4095 maximum$",
        )

    async def test_messages_with_engine(self):
        engine = core.Engine(self.model, self.model_tokenizer)
        await engine.start()
        self.addAsyncCleanup(engine.close)
        self.serve(engine, self.model_tokenizer)
        body = {"messages": MESSAGES}
        chat = (await self.post(body, "/v1/chat/completions")).json()
        message = (await self.post(body)).json()
        # The same tokens as the OpenAI API.
        text = chat["choices"][0]["message"]["content"]
        self.assertEqual(message["content"], [{"type": "text", "text": text}])
        usage = message["usage"]
        num_prompt = usage["input_tokens"] + usage["cache_read_input_tokens"]
        self.assertEqual(num_prompt, chat["usage"]["prompt_tokens"])


if __name__ == "__main__":
    unittest.main()
