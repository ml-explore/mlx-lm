# Copyright © 2026 Apple Inc.

import abc
import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any, Self, TypeVar

import fastapi
import pydantic
from fastapi import responses
from starlette import exceptions as starlette_exceptions

from mlx_lm.server_v2.engine import errors

__all__ = [
    "APIError",
    "ErrorRoute",
    "answer_http_error",
    "event_stream",
    "log_request",
    "while_connected",
]

_logger = logging.getLogger(__name__)

_T = TypeVar("_T")


class APIError(Exception):
    """An error with its HTTP status; an API sends it in its own format."""

    def __init__(
        self,
        status: int,
        message: str,
        *,
        param: str | None = None,
        code: str | None = None,
        headers: Mapping[str, str] | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.param = param
        self.code = code
        # E.g. the "allow" header of a 405.
        self.headers = {} if headers is None else headers

    @classmethod
    def from_exception(cls, e: Exception) -> Self:
        """The error for any exception that a request raised.

        Logs an exception that is not an API, HTTP or engine error, with its traceback.
        """
        if isinstance(e, APIError):
            return e
        if isinstance(e, fastapi.exceptions.RequestValidationError):
            return cls.from_validation_error(e)
        # E.g. a 400 for a body that FastAPI cannot parse.
        if isinstance(e, starlette_exceptions.HTTPException):
            return cls(e.status_code, str(e.detail), headers=e.headers)

        # Before InvalidRequestError: it is one with a code.
        if isinstance(e, errors.ContextLengthExceededError):
            return cls(400, str(e), code="context_length_exceeded")
        if isinstance(e, errors.InvalidRequestError):
            return cls(400, str(e))
        if isinstance(e, errors.UnavailableError):
            return cls(503, str(e))

        # The engine logs its own failures.
        if not isinstance(e, errors.EngineError):
            _logger.error("A request failed.", exc_info=e)
        return cls(500, "The server failed to process the request.")

    @classmethod
    def from_validation_error(
        cls, e: fastapi.exceptions.RequestValidationError | pydantic.ValidationError
    ) -> Self:
        """The 400 for the first error of a request body or of a value in it."""
        error, *_ = e.errors()
        if error["type"] == "json_invalid":
            return cls(400, "The request body is not valid JSON.")

        # E.g. "stop.str": the field, then a union member.
        loc = [str(p) for p in error["loc"]]
        # The errors of a request body start at "body".
        if isinstance(e, fastapi.exceptions.RequestValidationError):
            loc = loc[1:]

        if not loc:
            message = (
                "The request body must be a JSON object, sent as application/json."
            )
            return cls(400, message)

        return cls(400, f"{'.'.join(loc)}: {error['msg']}", param=loc[0])

    def to_response(self, body: dict[str, Any]) -> responses.JSONResponse:
        """A response with ``body``, the error in the format of an API."""
        headers = dict(self.headers)
        if self.status >= 500:
            # The SDKs retry a 5xx, and here a retry fails too.
            headers["x-should-retry"] = "false"
        return responses.JSONResponse(body, self.status, headers=headers)


class ErrorRoute(fastapi.routing.APIRoute, abc.ABC):
    """A route that answers its errors in the format of its API."""

    @staticmethod
    @abc.abstractmethod
    def error_body(error: APIError) -> dict[str, Any]:
        """The response body of ``error``."""

    def get_route_handler(
        self,
    ) -> Callable[[fastapi.Request], Awaitable[responses.Response]]:
        handler = super().get_route_handler()

        async def route_handler(req: fastapi.Request) -> responses.Response:
            try:
                return await handler(req)
            except Exception as e:
                error = APIError.from_exception(e)
                return error.to_response(self.error_body(error))

        return route_handler


async def answer_http_error(
    req: fastapi.Request,
    e: starlette_exceptions.HTTPException,
    *,
    route_class: type[ErrorRoute],
) -> responses.JSONResponse:
    """Answer a 404 or a 405 outside the routes, in the format of ``route_class``."""
    error = APIError.from_exception(e)
    body = route_class.error_body(error)
    return error.to_response(body)


def log_request(body: pydantic.BaseModel) -> None:
    """Log the body at the DEBUG level: it holds the prompt."""
    if _logger.isEnabledFor(logging.DEBUG):
        _logger.debug("Request body: %s", body.model_dump_json(exclude_none=True))


def event_stream(
    events: AsyncIterator[str], *, keepalive_event: str, keepalive: float
) -> responses.StreamingResponse:
    """Server-sent events that no cache may store, as in mlx_lm.server."""
    headers = {"cache-control": "no-cache"}
    return responses.StreamingResponse(
        _with_keepalive(events, keepalive_event=keepalive_event, keepalive=keepalive),
        media_type="text/event-stream",
        headers=headers,
    )


async def _with_keepalive(
    events: AsyncIterator[str], *, keepalive_event: str, keepalive: float
) -> AsyncIterator[str]:
    """The events, and ``keepalive_event`` after ``keepalive`` seconds without one."""
    next_event = asyncio.ensure_future(anext(events))
    try:
        while True:
            done, _ = await asyncio.wait({next_event}, timeout=keepalive)
            if not done:
                yield keepalive_event
                continue
            try:
                event = next_event.result()
            except StopAsyncIteration:
                return
            yield event
            next_event = asyncio.ensure_future(anext(events))
    finally:
        # Cancel the request now, not when the GC frees the events. A running
        # generator cannot close, so close the events once the cancel lands.
        next_event.cancel()
        next_event.add_done_callback(lambda _: asyncio.ensure_future(events.aclose()))


async def while_connected(req: fastapi.Request, awaitable: Awaitable[_T]) -> _T:
    """Await, but cancel it and raise a 499 if the client disconnects first."""
    task = asyncio.ensure_future(awaitable)

    async def gone() -> None:
        while (await req.receive())["type"] != "http.disconnect":
            pass

    disconnect = asyncio.ensure_future(gone())
    try:
        await asyncio.wait({task, disconnect}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        disconnect.cancel()
        task.cancel()

    if not task.done():
        raise APIError(499, "The client disconnected.")
    return task.result()
