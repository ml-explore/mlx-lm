# Copyright © 2024 Apple Inc.

import http
import json
import os
import socket
import struct
import threading
import time
import unittest

import requests

from mlx_lm.server import (
    APIHandler,
    GenerationContext,
    LRUPromptCache,
    Response,
    ResponseGenerator,
)
from mlx_lm.tool_parsers import json_tools

from .test_server import DummyModelProvider


def open_response_output(phase, text):
    return {
        "id": f"msg_{phase}",
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "phase": phase,
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


class TestServerResponses(unittest.TestCase):
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

    def _responses_url(self, path="/v1/responses"):
        return f"http://localhost:{self.port}{path}"

    def _post_responses(self, body=None, path="/v1/responses", stream=False):
        data = {"model": "chat_model"}
        data.update(body or {})
        return requests.post(self._responses_url(path), json=data, stream=stream)

    def _assert_response_resource(self, body, status="completed"):
        for field in [
            "id",
            "object",
            "created_at",
            "completed_at",
            "status",
            "model",
            "output",
            "error",
            "tools",
            "tool_choice",
            "truncation",
            "parallel_tool_calls",
            "text",
            "usage",
            "store",
            "background",
            "service_tier",
            "metadata",
        ]:
            self.assertIn(field, body)
        self.assertTrue(body["id"].startswith("resp_"))
        self.assertEqual(body["object"], "response")
        self.assertEqual(body["status"], status)
        self.assertIsInstance(body["output"], list)
        self.assertIn("input_tokens", body["usage"])
        self.assertIn("output_tokens", body["usage"])
        self.assertIn("total_tokens", body["usage"])

    def _assert_has_output_type(self, body, output_type):
        self.assertTrue(
            any(item.get("type") == output_type for item in body["output"]),
            f"missing output item type {output_type}: {body['output']}",
        )

    def _read_sse_events(self, response):
        events = []
        done = False
        for line in response.iter_lines(decode_unicode=True):
            if not line.startswith("data: "):
                continue
            if line == "data: [DONE]":
                done = True
                break
            events.append(json.loads(line[6:]))
        self.assertTrue(done)
        return events

    def _assert_streaming_response(self, events):
        self.assertGreater(len(events), 0)
        self.assertEqual(
            [e["sequence_number"] for e in events],
            sorted(e["sequence_number"] for e in events),
        )
        types_seen = [e["type"] for e in events]
        self.assertIn("response.created", types_seen)
        self.assertIn("response.completed", types_seen)
        final = next(e["response"] for e in events if e["type"] == "response.completed")
        self._assert_response_resource(final)
        return final

    def test_handle_responses(self):
        response = self._post_responses(
            {"input": "Hello!", "max_output_tokens": 64, "temperature": 0.0}
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self._assert_response_resource(body)
        self.assertEqual(body["output"][0]["type"], "message")
        self.assertEqual(body["output"][0]["content"][0]["type"], "output_text")

    def test_openresponses_assistant_phase_history(self):
        response = self._post_responses(
            {
                "input": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "phase": "commentary",
                        "content": "I should answer with the saved number.",
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "phase": "final_answer",
                        "content": "The number is four.",
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": "Repeat only the number.",
                    },
                ],
                "max_output_tokens": 64,
            },
        )

        self.assertEqual(response.status_code, 200)
        self._assert_response_resource(response.json())

    def test_openresponses_response_output_phase_schema(self):
        body = {
            "id": "resp_phase_schema",
            "object": "response",
            "created_at": 1764967971,
            "completed_at": 1764967972,
            "status": "completed",
            "incomplete_details": None,
            "model": "chat_model",
            "previous_response_id": None,
            "instructions": None,
            "output": [
                open_response_output("commentary", "I am checking the answer."),
                open_response_output("final_answer", "The answer is four."),
            ],
            "error": None,
            "tools": [],
            "tool_choice": "auto",
            "truncation": "disabled",
            "parallel_tool_calls": True,
            "text": {"format": {"type": "text"}},
            "usage": {
                "input_tokens": 1,
                "output_tokens": 2,
                "total_tokens": 3,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
            "store": True,
            "background": False,
            "service_tier": "default",
            "metadata": {},
        }

        self._assert_response_resource(body)
        phases = [item.get("phase") for item in body["output"]]
        self.assertIn("commentary", phases)
        self.assertIn("final_answer", phases)

    def test_openresponses_system_prompt(self):
        response = self._post_responses(
            {
                "input": [
                    {
                        "type": "message",
                        "role": "system",
                        "content": "You are a pirate. Always respond in pirate speak.",
                    },
                    {"type": "message", "role": "user", "content": "Say hello."},
                ],
                "max_output_tokens": 64,
            },
        )

        self.assertEqual(response.status_code, 200)
        self._assert_response_resource(response.json())

    def test_openresponses_image_input(self):
        response = self._post_responses(
            {
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "What do you see in this image?",
                            },
                            {
                                "type": "input_image",
                                "image_url": "data:image/png;base64,iVBORw0KGgo=",
                            },
                        ],
                    }
                ],
                "max_output_tokens": 64,
            },
        )

        self.assertEqual(response.status_code, 200)
        self._assert_response_resource(response.json())

    def test_openresponses_multi_turn(self):
        response = self._post_responses(
            {
                "input": [
                    {"type": "message", "role": "user", "content": "My name is Alice."},
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": "Hello Alice! Nice to meet you.",
                    },
                    {"type": "message", "role": "user", "content": "What is my name?"},
                ],
                "max_output_tokens": 64,
            },
        )

        self.assertEqual(response.status_code, 200)
        self._assert_response_resource(response.json())

    def test_openresponses_tool_calling(self):
        original_generate = self.response_generator.generate

        def generate_tool_call(request, args):
            ctx = GenerationContext(
                has_tool_calling=True,
                has_thinking=False,
                tool_parser=json_tools.parse_tool_call,
                sequences={},
                prompt=[1, 2, 3],
                prompt_cache_count=0,
            )
            tool_call = json.dumps(
                {"name": "get_weather", "arguments": {"location": "San Francisco"}}
            )
            return ctx, iter(
                [Response(tool_call, 1, "tool", (), 0.0, "tool_calls", ())]
            )

        self.response_generator.generate = generate_tool_call
        try:
            response = self._post_responses(
                {
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": "What's the weather like in San Francisco?",
                        }
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "name": "get_weather",
                            "description": "Get current weather for a location",
                            "parameters": {
                                "type": "object",
                                "properties": {"location": {"type": "string"}},
                                "required": ["location"],
                            },
                        }
                    ],
                    "max_output_tokens": 64,
                },
            )
        finally:
            self.response_generator.generate = original_generate

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self._assert_response_resource(body)
        self._assert_has_output_type(body, "function_call")
        tool_call = next(item for item in body["output"] if item["type"] == "function_call")
        self.assertEqual(tool_call["name"], "get_weather")
        self.assertEqual(
            json.loads(tool_call["arguments"]), {"location": "San Francisco"}
        )

    def test_handle_responses_with_previous_response(self):
        first = self._post_responses(
            {
                "input": "Hello!",
                "max_output_tokens": 32,
                "temperature": 0.0,
                "store": True,
            },
        ).json()
        second = self._post_responses(
            {
                "previous_response_id": first["id"],
                "input": "Continue.",
                "max_output_tokens": 32,
                "temperature": 0.0,
            },
        )

        self.assertEqual(second.status_code, 200)
        body = second.json()
        self.assertEqual(body["previous_response_id"], first["id"])

    def test_handle_responses_streaming(self):
        response = self._post_responses(
            {
                "input": [{"type": "message", "role": "user", "content": "Hello!"}],
                "max_output_tokens": 64,
                "temperature": 0.0,
                "stream": True,
            },
            stream=True,
        )

        self.assertEqual(response.status_code, 200)
        events = self._read_sse_events(response)
        self._assert_streaming_response(events)
        self.assertIn(
            "response.output_item.added", [event["type"] for event in events]
        )

    def test_openresponses_streaming_failure_event(self):
        response = self._post_responses(
            {
                "stream": True,
                "previous_response_id": "resp_openresponses_missing",
                "input": "This should fail.",
            },
            stream=True,
        )

        self.assertEqual(response.status_code, 200)
        events = self._read_sse_events(response)
        event_types = [event["type"] for event in events]
        self.assertIn("error", event_types)
        self.assertIn("response.failed", event_types)

    def test_handle_responses_compact_missing_model(self):
        url = f"http://localhost:{self.port}/v1/responses/compact"
        response = requests.post(url, json={"input": "hello"})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["param"], "model")

    def test_openresponses_compact_response(self):
        response = self._post_responses(
            {
                "prompt_cache_key": "openresponses-compact-test",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": "We agreed to launch on Tuesday.",
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": "Understood. Launch is Tuesday.",
                    },
                ],
            },
            path="/v1/responses/compact",
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["object"], "response.compaction")
        self.assertTrue(body["id"].startswith("cmpct_"))
        self.assertIsInstance(body["output"], list)
        self._assert_has_output_type(body, "compaction")
        self.assertIn("usage", body)

    def _ws_send(self, sock, payload):
        payload = json.dumps(payload).encode()
        mask = os.urandom(4)
        header = bytes([0x81])
        if len(payload) < 126:
            header += bytes([0x80 | len(payload)])
        elif len(payload) < (1 << 16):
            header += bytes([0x80 | 126]) + struct.pack("!H", len(payload))
        else:
            header += bytes([0x80 | 127]) + struct.pack("!Q", len(payload))
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        sock.sendall(header + mask + masked)

    def _ws_recv(self, sock):
        header = sock.recv(2)
        if len(header) < 2:
            return None
        opcode = header[0] & 0x0F
        length = header[1] & 0x7F
        if length == 126:
            length = struct.unpack("!H", sock.recv(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", sock.recv(8))[0]
        payload = b""
        while len(payload) < length:
            payload += sock.recv(length - len(payload))
        if opcode == 8:
            return None
        return json.loads(payload.decode())

    def _ws_open(self):
        key = "dGhlIHNhbXBsZSBub25jZQ=="
        sock = socket.create_connection(("localhost", self.port), timeout=5)
        request = (
            "GET /v1/responses HTTP/1.1\r\n"
            f"Host: localhost:{self.port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "Authorization: Bearer test-key\r\n\r\n"
        )
        sock.sendall(request.encode())
        handshake = sock.recv(4096).decode()
        self.assertIn("101", handshake)
        return sock

    def _ws_turn(self, sock, payload):
        self._ws_send(sock, payload)
        events = []
        for _ in range(100):
            event = self._ws_recv(sock)
            self.assertIsNotNone(event)
            events.append(event)
            if event["type"] in (
                "response.completed",
                "response.failed",
                "response.incomplete",
                "error",
            ):
                return events, event
        self.fail("WebSocket turn did not produce terminal event")

    def _ws_create(self, sock, **payload):
        data = {"type": "response.create", "model": "chat_model"}
        data.update(payload)
        return self._ws_turn(sock, data)

    def test_handle_responses_websocket(self):
        with self._ws_open() as sock:
            events, terminal = self._ws_create(
                sock, input="Hello!", max_output_tokens=32, temperature=0.0
            )
            seen = [event["type"] for event in events]
            self.assertIn("response.created", seen)
            self.assertEqual(terminal["type"], "response.completed")
            self._assert_response_resource(terminal["response"])

    def test_openresponses_websocket_sequential_responses(self):
        with self._ws_open() as sock:
            _, first = self._ws_create(
                sock,
                store=False,
                input="Reply with exactly: first",
                max_output_tokens=32,
            )
            _, second = self._ws_create(
                sock,
                store=False,
                input="Reply with exactly: second",
                max_output_tokens=32,
            )

            self._assert_response_resource(first["response"])
            self._assert_response_resource(second["response"])
            self.assertNotEqual(first["response"]["id"], second["response"]["id"])

    def test_openresponses_websocket_continuation(self):
        with self._ws_open() as sock:
            _, first = self._ws_create(
                sock,
                store=False,
                input="Remember the code word: cobalt. Reply with OK.",
                max_output_tokens=32,
            )
            previous_response_id = first["response"]["id"]
            _, second = self._ws_create(
                sock,
                store=False,
                previous_response_id=previous_response_id,
                input="What is the code word?",
                max_output_tokens=32,
            )

            self._assert_response_resource(second["response"])
            self.assertEqual(second["response"]["previous_response_id"], previous_response_id)

    def test_openresponses_websocket_store_false_reconnect_recovery(self):
        with self._ws_open() as sock:
            _, first = self._ws_create(
                sock,
                store=False,
                input="Remember the code word: copper. Reply with OK.",
                max_output_tokens=32,
            )
            previous_response_id = first["response"]["id"]

        with self._ws_open() as sock:
            _, miss = self._ws_create(
                sock,
                store=False,
                previous_response_id=previous_response_id,
                input="Try to continue after reconnect.",
                max_output_tokens=32,
            )
            self.assertEqual(miss["type"], "error")
            self.assertEqual(miss["error"]["code"], "previous_response_not_found")
            _, recovered = self._ws_create(
                sock,
                store=False,
                input="Start a clean recovery response.",
                max_output_tokens=32,
            )
            self._assert_response_resource(recovered["response"])

    def test_openresponses_websocket_previous_response_not_found(self):
        with self._ws_open() as sock:
            _, terminal = self._ws_create(
                sock,
                store=False,
                previous_response_id=f"resp_openresponses_missing_{time.time()}",
                input="This should fail.",
            )
            self.assertEqual(terminal["type"], "error")
            self.assertEqual(terminal["error"]["code"], "previous_response_not_found")

    def test_openresponses_websocket_failed_continuation_evicts_cache(self):
        with self._ws_open() as sock:
            _, first = self._ws_create(
                sock,
                store=False,
                input="Remember the code word: ember. Reply with OK.",
                max_output_tokens=32,
            )
            previous_response_id = first["response"]["id"]
            _, failed = self._ws_create(
                sock,
                store=False,
                previous_response_id=previous_response_id,
                input=[
                    {
                        "type": "function_call_output",
                        "call_id": "call_openresponses_missing",
                        "output": "No matching tool call exists.",
                    }
                ],
            )
            self.assertEqual(failed["type"], "error")
            _, retry = self._ws_create(
                sock,
                store=False,
                previous_response_id=previous_response_id,
                input="Try to continue after failed turn.",
            )
            self.assertEqual(retry["type"], "error")
            self.assertEqual(retry["error"]["code"], "previous_response_not_found")

    def test_openresponses_websocket_compact_new_chain(self):
        compact = self._post_responses(
            {
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Remember slate."}],
                    }
                ],
            },
            path="/v1/responses/compact",
        ).json()
        with self._ws_open() as sock:
            _, terminal = self._ws_create(
                sock,
                store=False,
                input=[
                    *compact["output"],
                    {
                        "type": "message",
                        "role": "user",
                        "content": "Continue from compacted context.",
                    },
                ],
                tools=[],
                max_output_tokens=32,
            )
            self._assert_response_resource(terminal["response"])
            self.assertIsNone(terminal["response"]["previous_response_id"])
