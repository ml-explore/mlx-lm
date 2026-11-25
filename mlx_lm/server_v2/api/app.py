# Copyright © 2026 Apple Inc.

import asyncio
import contextlib
import dataclasses
import functools
import time
from collections.abc import AsyncIterator, Callable
from concurrent import futures
from typing import Annotated, Any, TypeVar

import fastapi
from fastapi import responses
from starlette import exceptions as starlette_exceptions

from mlx_lm import tokenizer_utils
from mlx_lm.server_v2 import settings
from mlx_lm.server_v2.api import http
from mlx_lm.server_v2.engine import core

__all__ = ["PromptQueue", "ServerState", "ServerStateDep", "make_app"]


_T = TypeVar("_T")


class PromptQueue:
    """Render prompts on a few threads, in the order they come."""

    def __init__(self, num_threads: int = 4):
        self._semaphore = asyncio.Semaphore(num_threads)
        self._executor = futures.ThreadPoolExecutor(num_threads, "mlx-lm-prompt")

    async def run(self, fn: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
        """Run ``fn`` on a prompt thread."""
        # Wait here, not in the pool queue: a cancelled wait frees the body.
        await self._semaphore.acquire()

        loop = asyncio.get_running_loop()
        future = self._executor.submit(fn, *args, **kwargs)
        future.add_done_callback(
            lambda _: loop.call_soon_threadsafe(self._semaphore.release)
        )
        return await asyncio.wrap_future(future)


@dataclasses.dataclass
class ServerState:
    """What the routes share: the engine, the tokenizer and the config."""

    engine: core.Engine
    tokenizer: tokenizer_utils.TokenizerWrapper
    config: settings.ServerConfig
    created: int = dataclasses.field(
        init=False, default_factory=lambda: int(time.time())
    )
    prompts: PromptQueue = dataclasses.field(init=False, default_factory=PromptQueue)

    @property
    def is_stalled(self) -> bool:
        """Whether the engine is OK but took no step for ``stall_timeout``.

        False while idle, and for an engine that closed or failed.
        """
        if self.engine.status != core.EngineStatus.OK:
            return False
        return self.engine.seconds_since_step() > self.config.stall_timeout


def _get_state(req: fastapi.Request) -> ServerState:
    return req.app.state.server


ServerStateDep = Annotated[ServerState, fastapi.Depends(_get_state)]


_health = fastapi.APIRouter()


@_health.api_route("/health", methods=["GET", "HEAD"])
async def health(state: ServerStateDep) -> responses.JSONResponse:
    status = state.engine.status
    if state.is_stalled:
        status = "stalled"
    return responses.JSONResponse(
        {"status": status}, 200 if status == core.EngineStatus.OK else 503
    )


def make_app(
    state: ServerState, router: fastapi.APIRouter, *routers: fastapi.APIRouter
) -> fastapi.FastAPI:
    """The app with the routes of one API or more.

    A 404 or 405 outside the routes has the format of the first API.
    """
    app = fastapi.FastAPI(
        lifespan=_lifespan, docs_url=None, redoc_url=None, openapi_url=None
    )

    app.state.server = state
    app.include_router(_health)
    for api_router in (router, *routers):
        app.include_router(api_router)

    answer = functools.partial(http.answer_http_error, route_class=router.route_class)
    app.add_exception_handler(starlette_exceptions.HTTPException, answer)
    return app


@contextlib.asynccontextmanager
async def _lifespan(app: fastapi.FastAPI) -> AsyncIterator[None]:
    engine = app.state.server.engine
    await engine.start()
    try:
        yield
    finally:
        await engine.close()
