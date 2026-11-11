# Copyright © 2026 Apple Inc.

import asyncio
import gc
import json
import threading
import unittest
import weakref
from typing import Any
from unittest import mock

import fastapi
import httpx
import pydantic

from mlx_lm.server_v2 import settings
from mlx_lm.server_v2.api import app, http
from mlx_lm.server_v2.engine import core, errors, protocol
from tests.server_v2 import helpers

DISCONNECT = {"type": "http.disconnect"}


class Body(pydantic.BaseModel):
    x: int


class _TestRoute(http.ErrorRoute):
    @staticmethod
    def error_body(error: http.APIError) -> dict[str, Any]:
        return {"status": error.status, "message": str(error), "param": error.param}


class TestApp(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.engine = helpers.StubEngine()
        config = settings.ServerConfig(model_id="m")
        self.state = app.ServerState(engine=self.engine, tokenizer=None, config=config)
        router = fastapi.APIRouter(route_class=_TestRoute)

        @router.post("/echo")
        async def echo(body: Body) -> dict:
            return body.model_dump()

        @router.get("/fail")
        async def fail() -> None:
            raise RuntimeError("Injected failure")

        transport = httpx.ASGITransport(app=app.make_app(self.state, router))
        self.http = httpx.AsyncClient(transport=transport, base_url="http://test")
        self.addAsyncCleanup(self.http.aclose)

    def test_api_error(self):
        cases = [
            (errors.ContextLengthExceededError("x"), 400, "context_length_exceeded"),
            (errors.InvalidRequestError("x"), 400, None),
            (errors.UnavailableError(core.EngineStatus.CLOSED), 503, None),
            (errors.EngineError("x"), 500, None),
        ]
        # The engine logs its own failures.
        with self.assertNoLogs(level="ERROR"):
            for error, status, code in cases:
                e = http.APIError.from_exception(error)
                self.assertEqual((e.status, e.code), (status, code))
        # A ValueError of the server is not the fault of the client.
        with self.assertLogs(level="ERROR"):
            e = http.APIError.from_exception(ValueError("x"))
        self.assertEqual(e.status, 500)
        with self.assertLogs(level="ERROR"):
            e = http.APIError.from_exception(RuntimeError("A secret"))
        self.assertEqual(str(e), "The server failed to process the request.")
        self.assertIs(http.APIError.from_exception(e), e)

        response = http.APIError(405, "x").to_response({"a": 1})
        self.assertEqual(json.loads(response.body), {"a": 1})
        self.assertNotIn("x-should-retry", response.headers)
        response = http.APIError(502, "x").to_response({})
        self.assertEqual(response.headers["x-should-retry"], "false")

    async def test_handlers(self):
        failed = "The server failed to process the request."
        bad_json = "The request body is not valid JSON."
        not_json = "The request body must be a JSON object, sent as application/json."
        unparsed = "There was an error parsing the body"
        typed = {"content-type": "application/json"}
        # Not UTF-8: FastAPI raises its own 400.
        not_utf8 = {"content": b"\xff", "headers": typed}
        cases = [
            ("GET", "/fail", {}, 500, failed, None),
            ("GET", "/nope", {}, 404, "Not Found", None),
            ("GET", "/echo", {}, 405, "Method Not Allowed", None),
            ("POST", "/echo", {"json": {"x": "a"}}, 400, "x: ", "x"),
            ("POST", "/echo", {"json": [1]}, 400, not_json, None),
            ("POST", "/echo", {"content": b'{"x": 1}'}, 400, not_json, None),
            ("POST", "/echo", {"content": b"{", "headers": typed}, 400, bad_json, None),
            ("POST", "/echo", not_utf8, 400, unparsed, None),
        ]
        for method, path, kwargs, status, message, param in cases:
            with self.subTest(method=method, path=path, **kwargs):
                response = await self.http.request(method, path, **kwargs)
                error = response.json()
                self.assertEqual((error["status"], error["param"]), (status, param))
                self.assertEqual(response.status_code, status)
                self.assertTrue(error["message"].startswith(message))
                if status == 405:
                    self.assertEqual(response.headers["allow"], "POST")

    async def test_health(self):
        cases = [
            (core.EngineStatus.OK, 0.0, 200, "ok"),
            (core.EngineStatus.OK, 61.0, 503, "stalled"),
            (core.EngineStatus.CLOSED, 0.0, 503, "closed"),
            (core.EngineStatus.CLOSED, 61.0, 503, "closed"),
            (core.EngineStatus.FAILED, 0.0, 503, "failed"),
        ]
        for status, stalled_for, code, name in cases:
            self.engine.status = status
            self.engine.stalled_for = stalled_for
            response = await self.http.get("/health")
            self.assertEqual(
                (response.status_code, response.json()), (code, {"status": name})
            )
            head = await self.http.head("/health")
            self.assertEqual(head.status_code, code)

    async def test_prompt_queue(self):
        # Two threads render two prompts at once.
        barrier = threading.Barrier(2, timeout=10)
        prompts = app.PromptQueue(num_threads=2)
        both = await asyncio.gather(
            prompts.run(barrier.wait), prompts.run(barrier.wait)
        )
        self.assertEqual(sorted(both), [0, 1])

        prompts = app.PromptQueue(num_threads=1)
        with self.assertRaises(ValueError):
            await prompts.run(int, "x")
        async with prompts._semaphore:
            waiting = asyncio.ensure_future(prompts.run(len, "abc"))
            await asyncio.sleep(0)
            self.assertFalse(waiting.done())
        self.assertEqual(await waiting, 3)

        body = mock.Mock()
        body_ref = weakref.ref(body)
        gone = mock.Mock(receive=mock.AsyncMock(return_value=DISCONNECT))
        # Another prompt runs, so this one waits.
        async with prompts._semaphore:
            with self.assertRaises(http.APIError) as e:
                run = prompts.run(print, body)
                await asyncio.wait_for(http.while_connected(gone, run), timeout=10)
            self.assertEqual(e.exception.status, 499)
            # The loop cancels the waiting prompt on its next turn.
            await asyncio.sleep(0)
            del body, e
            gc.collect()
            self.assertIsNone(body_ref())

    async def test_while_connected_leaves_no_task(self):
        async def wait_forever() -> dict:
            await asyncio.Event().wait()

        req = mock.Mock(receive=wait_forever)
        self.assertEqual(await http.while_connected(req, asyncio.sleep(0, "x")), "x")
        await asyncio.sleep(0)
        self.assertEqual(asyncio.all_tasks(), {asyncio.current_task()})

    async def test_event_stream_keepalive(self):
        closed = asyncio.Event()

        async def events():
            try:
                yield "a"
                await asyncio.Event().wait()
            finally:
                closed.set()

        response = http.event_stream(events(), keepalive_event="k", keepalive=0.01)
        stream = response.body_iterator
        self.assertEqual([await anext(stream) for _ in range(3)], ["a", "k", "k"])
        await stream.aclose()
        await asyncio.wait_for(closed.wait(), timeout=10)


class TestLifespan(unittest.TestCase):

    def test_lifespan(self):
        end = protocol.FinishedEvent(
            reason=protocol.FinishReason.EOS,
            usage=protocol.Usage(prompt_tokens=1, cached_tokens=0, completion_tokens=0),
        )
        engine = helpers.StubEngine(end)
        config = settings.ServerConfig(model_id="m")
        state = app.ServerState(engine=engine, tokenizer=None, config=config)
        asgi_app = app.make_app(state, fastapi.APIRouter(route_class=_TestRoute))

        async def serve() -> list[protocol.Update]:
            async with asgi_app.router.lifespan_context(asgi_app):
                handle = state.engine.submit(protocol.Request([1]))
                stream = state.engine.updates(handle)
                return [update async for update in stream]

        self.assertEqual(asyncio.run(serve()), [protocol.Update([], end)])
        self.assertTrue(engine.started)
        self.assertEqual(engine.status, core.EngineStatus.CLOSED)


if __name__ == "__main__":
    unittest.main()
