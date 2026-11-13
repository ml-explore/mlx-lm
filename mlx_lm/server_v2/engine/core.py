# Copyright © 2026 Apple Inc.

import asyncio
import collections
import contextlib
import copy
import dataclasses
import enum
import functools
import itertools
import logging
import time
import weakref
from collections.abc import AsyncIterator, Sequence
from concurrent import futures
from typing import Any, assert_never, get_args

import mlx.core as mx
import mlx.nn as nn

from mlx_lm import sample_utils, tokenizer_utils
from mlx_lm.generate import TextState, TextStateMachine
from mlx_lm.models import cache
from mlx_lm.server_v2 import settings
from mlx_lm.server_v2.engine import distributed, errors, generations, protocol

__all__ = ["Engine", "EngineStatus"]

_logger = logging.getLogger(__name__)


class EngineStatus(enum.StrEnum):
    """NEW until the warm-up ends, then OK until a close or a failure."""

    NEW = "new"
    OK = "ok"
    CLOSED = "closed"
    FAILED = "failed"


@dataclasses.dataclass(frozen=True, eq=False, kw_only=True)
class _Submit:
    """Queue a checked request."""

    handle: int
    request: protocol.Request
    # The token limit: the requested one, or the default within the context.
    max_tokens: int


@dataclasses.dataclass(frozen=True)
class _Cancel:
    """End a request where it is."""

    handle: int


@dataclasses.dataclass(frozen=True)
class _Close:
    """Stop the worker after the commands before it."""


# A message from the loop thread to the worker thread.
_Command = _Submit | _Cancel | _Close

# The events of a step, each with the handle of its request.
_StepEvents = list[tuple[int, protocol.Event]]


class Engine:
    """Generate for many requests at once.

    The engine runs on the event loop: ``start`` makes its step task. The task runs
    each step of the ``_Worker`` on one thread that owns all MLX work, so the loop
    stays free while the GPU works. The two threads share only the command queue
    and the events that a step returns.

    With a group, each rank runs an engine on its part of the model. Rank 0 takes
    the requests and shares its commands every step; all ranks then run the same
    steps.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: tokenizer_utils.TokenizerWrapper,
        *,
        config: settings.EngineConfig = settings.EngineConfig(),
        group: mx.distributed.Group | None = None,
    ):
        self._model = model
        self.config = config
        self.error: BaseException | None = None

        # Set by the warm-up before the OK status; read-only after it.
        self._vocab_size = 0
        self._handles = itertools.count()
        self._status = EngineStatus.NEW
        self._task: asyncio.Task | None = None
        self._ready = asyncio.Event()
        self._wakeup = asyncio.Event()
        # The queue of each request that waits for its updates, by handle.
        self._queues: dict[int, asyncio.Queue[list[protocol.Event]]] = {}
        # The requests that have no end event yet, by handle.
        self._open: set[int] = set()
        # None while idle: the time idle is not a stall.
        self._last_step: float | None = None

        # The loop thread adds the commands, and the worker thread takes them.
        self._commands: collections.deque[_Command] = collections.deque()
        self._executor = futures.ThreadPoolExecutor(1, "mlx-lm-engine")
        self._worker = _Worker(
            model, tokenizer, config=config, group=group, commands=self._commands
        )

    async def start(self) -> None:
        """Warm up the model on the worker thread, then serve.

        Raises the warm-up error, and ``RuntimeError`` on a second start.
        """
        if self._task is not None:
            raise RuntimeError("An engine can start only once.")

        # Not only parameters: the worker thread cannot evaluate, e.g. RoPE
        mx.eval(self._model.state)

        self._task = asyncio.create_task(self._run())
        await self._ready.wait()
        if self.status == EngineStatus.FAILED:
            raise self.error

    def submit(self, request: protocol.Request) -> int:
        """Check and queue a request; return its handle.

        Raises ``InvalidRequestError`` for a bad request, ``UnavailableError`` when the
        engine is not OK, and ``RuntimeError`` on a rank other than 0.
        """
        if self._worker.rank != 0:
            raise RuntimeError("Only rank 0 takes requests.")
        if (status := self._status) != EngineStatus.OK:
            raise errors.UnavailableError(status)
        _check_request(request, vocab_size=self._vocab_size)

        max_tokens = _context_max_tokens(
            request,
            default=self.config.max_tokens,
            context_length=self.config.context_length,
        )

        handle = next(self._handles)
        submission = _Submit(handle=handle, request=request, max_tokens=max_tokens)
        self._queues[handle] = asyncio.Queue()
        self._open.add(handle)
        self._add_command(submission)
        return handle

    def updates(self, handle: int) -> AsyncIterator[protocol.Update]:
        """The updates of a request, each with its new events.

        They end with the request; closing or freeing them cancels it. Reading raises
        ``EngineError`` if the request fails, ``UnavailableError`` if the engine stops.
        """
        stream = self._updates(handle)
        loop = asyncio.get_running_loop()

        def leave_on_loop(*, request_handle: int) -> None:
            # The GC can call this on any thread, also after the loop closed. A closed
            # loop raises RuntimeError, and then no request is left to cancel.
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(self._leave, request_handle)

        # A generator that never started has no finally: leave when it is freed.
        weakref.finalize(stream, leave_on_loop, request_handle=handle)
        return stream

    def cancel(self, handle: int) -> None:
        """Stop a request; it ends with ``FinishReason.CANCELLED``."""
        # An engine that stops ends all its requests itself.
        if self._status == EngineStatus.OK:
            self._add_command(_Cancel(handle))

    async def close(self) -> None:
        """End all requests with an error and stop the engine."""
        # A started engine stays NEW until its warm-up ends.
        is_open = self._status in (EngineStatus.NEW, EngineStatus.OK)
        if self._task is not None and is_open:
            self._status = EngineStatus.CLOSED
            # The last command: submit and cancel need the OK status.
            self._add_command(_Close())

        await self.join()

    async def join(self) -> None:
        """Wait until the engine stops, e.g. on a rank without a server."""
        if self._task is not None:
            # A caller that stops waiting must not cancel the engine.
            await asyncio.shield(self._task)

    @property
    def status(self) -> EngineStatus:
        """Where the engine is in its life."""
        return self._status

    def seconds_since_step(self) -> float:
        """The time since the last step; zero while idle."""
        if (last_step := self._last_step) is None:
            return 0.0
        return time.monotonic() - last_step

    def _add_command(self, command: _Command) -> None:
        """Queue a command for the worker thread, and wake the step task."""
        self._commands.append(command)
        self._wakeup.set()

    async def _updates(self, handle: int) -> AsyncIterator[protocol.Update]:
        """The updates of a request. Raises the error of a failed request."""
        queue = self._queues[handle]
        ended = False
        try:
            while True:
                update = protocol.Update.from_events(await queue.get())
                ended = update.end is not None
                yield update
                if ended:
                    return
        finally:
            # Also when the client disconnects: a request that ended needs no cancel.
            self._leave(handle, cancel=not ended)

    def _leave(self, handle: int, *, cancel: bool = True) -> None:
        """Drop the queue of a request, and cancel it if asked; safe to repeat."""
        if self._queues.pop(handle, None) is not None and cancel:
            self.cancel(handle)

    async def _run(self) -> None:
        """Warm up, serve until a close or a failure, then shut down."""
        loop = asyncio.get_running_loop()
        try:
            self._vocab_size = await loop.run_in_executor(
                self._executor, self._worker.warm_up
            )
            # A close during the warm-up stays.
            if self._status == EngineStatus.NEW:
                self._status = EngineStatus.OK
        except asyncio.CancelledError:
            self._status = EngineStatus.CLOSED
            self._executor.shutdown(wait=False)
            return
        except BaseException as e:
            self._fail(e)
            self._executor.shutdown(wait=False)
            return
        finally:
            self._ready.set()

        try:
            await self._steps()
            # Rank 0's close also closes the other ranks.
            self._status = EngineStatus.CLOSED
        except asyncio.CancelledError:
            self._status = EngineStatus.CLOSED
        except BaseException as e:
            _logger.exception("The engine failed.")
            # The traceback frames hold the arrays of the failed step.
            self._fail(e.with_traceback(None))

        try:
            events = await loop.run_in_executor(
                self._executor, self._worker.shut_down, self._status
            )
        except Exception:
            _logger.exception("The engine failed to shut down.")
            events = []

        self._dispatch(events)
        # End the requests that are left, e.g. one whose command a failed step lost.
        failures = []
        for handle in self._open:
            error = errors.UnavailableError(self._status)
            failures.append((handle, protocol.FailedEvent(error=error)))
        self._dispatch(failures)
        self._executor.shutdown()

    async def _steps(self) -> None:
        """Run the steps until a close, and send the events after each."""
        loop = asyncio.get_running_loop()
        idle = True
        while True:
            # The other ranks wait for rank 0 in the next step.
            if idle and self._worker.rank == 0:
                await self._wait_for_command()
            advance = loop.run_in_executor(self._executor, self._worker.advance)
            events, idle, closed = await advance

            self._last_step = time.monotonic()
            self._dispatch(events)
            if closed:
                return

    async def _wait_for_command(self) -> None:
        """Wait for a command. Idle time is not a stall."""
        self._last_step = None
        while not self._commands:
            self._wakeup.clear()
            await self._wakeup.wait()
        self._last_step = time.monotonic()

    def _fail(self, error: BaseException) -> None:
        """Set the FAILED status and its error."""
        self._status = EngineStatus.FAILED
        self.error = error

    def _dispatch(self, events: _StepEvents) -> None:
        """Put the events of a step in the queues of their requests."""
        by_handle = {}
        for handle, event in events:
            by_handle.setdefault(handle, []).append(event)

        for handle, request_events in by_handle.items():
            last = request_events[-1]
            if isinstance(last, protocol.FinishedEvent | protocol.FailedEvent):
                self._open.discard(handle)
            # A request that has left has no queue.
            if (queue := self._queues.get(handle)) is not None:
                queue.put_nowait(request_events)


class _Worker:
    """The state and the steps of an engine: all of it runs on the worker thread.

    The loop thread reads only ``rank``, and adds the commands to the queue.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: tokenizer_utils.TokenizerWrapper,
        *,
        config: settings.EngineConfig,
        group: mx.distributed.Group | None,
        commands: collections.deque[_Command],
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._config = config

        self._group = group
        self.rank = 0 if group is None else group.rank()
        self._commands = commands

        self._machine = TextStateMachine.from_tokenizer(tokenizer)
        self._no_tool_calls_machine = TextStateMachine.from_tokenizer(
            tokenizer, tool_calls=False
        )
        self._eos_token_ids = frozenset(tokenizer.eos_token_ids)
        self._xtc_special_tokens = (*tokenizer.encode("\n"), *tokenizer.eos_token_ids)

        # Whether every layer's cache can drop its last tokens.
        self._trimmable = False

        # Every request from its submit to its end, by handle.
        self._generations: dict[int, generations.Generation] = {}
        self._waiting: collections.deque[generations.Generation] = collections.deque()
        # One request at a time.
        self._prefilling: generations.Generation | None = None

        self._decoding: generations.Decoding | None = None
        self._cache_store: cache.LRUPromptCache | None = None

        # The events of this step; the step returns them.
        self._events: _StepEvents = []

    def warm_up(self) -> int:
        """Warm up the model, and make the batch and the cache.

        Return the vocabulary size.
        """
        distributed.sync_seed(self._group)
        vocab_size, trimmable = _probe_model(self._model)
        self._trimmable = _sync_trimmable(trimmable, group=self._group)

        self._cache_store = cache.LRUPromptCache(
            max_size=self._config.prompt_cache_size,
            max_bytes=self._config.prompt_cache_bytes,
        )
        self._decoding = generations.Decoding(
            self._model, max_rows=self._config.decode_concurrency
        )

        return vocab_size

    def advance(self) -> tuple[_StepEvents, bool, bool]:
        """Handle the commands and run a step.

        Return the events of the step, whether the engine is idle, and whether a
        close came.
        """
        if self._process_commands():
            return self._take_events(), True, True

        self._step()
        idle = self._is_idle
        if idle and not self._commands:
            # Nothing is left to run: free the memory once the queued work ends.
            mx.synchronize()
            mx.clear_cache()

        return self._take_events(), idle, False

    @property
    def _is_idle(self) -> bool:
        """Whether there is no work: no request, and no row to remove."""
        return not self._generations and self._decoding.is_empty

    def shut_down(self, status: EngineStatus) -> _StepEvents:
        """End the requests that are left, and free the memory; return the events."""
        for command in self._take_queued_commands():
            if isinstance(command, _Submit):
                self._submit(command)

        for generation in list(self._generations.values()):
            failure = protocol.FailedEvent(error=errors.UnavailableError(status))
            self._finish(generation, event=failure)

        # The caller can keep the engine after close.
        self._free_batch_and_cache()
        return self._take_events()

    def _free_batch_and_cache(self) -> None:
        """Close and free the decode batch and the cache store."""
        if self._decoding is not None:
            try:
                self._decoding.close()
            except Exception:
                _logger.exception("Closing the decode batch failed.")

        self._decoding = None
        self._cache_store = None
        mx.clear_cache()

    def _process_commands(self) -> bool:
        """Handle this step's commands. True after a close."""
        for command in self._next_commands():
            match command:
                case _Submit():
                    self._submit(command)
                case _Cancel(handle=handle):
                    self._cancel(handle)
                case _Close():
                    return True
                case _:
                    assert_never(command)

        return False

    def _next_commands(self) -> list[_Command]:
        """This step's commands: rank 0 takes its queue, and every rank gets them."""
        commands = []
        if self.rank == 0:
            commands = self._take_queued_commands()
        return distributed.share(self._group, commands)

    def _take_queued_commands(self) -> list[_Command]:
        """Take the commands in the queue now, without waiting for more."""
        # Only this thread takes from the queue, so these commands are there.
        return [self._commands.popleft() for _ in range(len(self._commands))]

    def _submit(self, submission: _Submit) -> None:
        """Make the generation of a new request, track it, and queue it last."""
        generation = self._new_generation(submission)
        self._generations[submission.handle] = generation
        self._waiting.append(generation)

    def _new_generation(self, submission: _Submit) -> generations.Generation:
        """The generation of a new request, before its prefill."""
        request = submission.request
        machine = self._machine
        if not request.parse_tool_calls:
            machine = self._no_tool_calls_machine
        text_reader, stop_reader = _text_readers(machine, request)

        return generations.Generation(
            handle=submission.handle,
            request=request,
            max_tokens=submission.max_tokens,
            detokenizer=self._tokenizer.detokenizer,
            text_reader=text_reader,
            stop_reader=stop_reader,
            eos_token_ids=self._eos_token_ids,
        )

    def _cancel(self, handle: int) -> None:
        """End the request with this handle, if it has not ended already.

        A prefill keeps the cache it computed, as a decode row does.
        """
        if (generation := self._generations.get(handle)) is None:
            return

        if generation is self._prefilling and generation.num_computed > 0:
            self._store_cache(generation.prompt_cache, generation.prefilled_tokens)

        event = protocol.FinishedEvent(
            reason=protocol.FinishReason.CANCELLED, usage=generation.usage
        )
        self._finish(generation, event=event)

    def _step(self) -> None:
        """One step: remove the ended rows, prefill, then decode one token each."""
        # First: it frees memory, and stores the caches the prefill looks up.
        for prompt_cache, all_tokens in self._decoding.remove_dropped():
            self._store_cache(prompt_cache, all_tokens)

        self._prefill_step()

        for generation, token, logprobs in self._decoding.decode():
            self._read_token(generation, token, logprobs=logprobs)

    def _prefill_step(self) -> None:
        """Prefill up to ``prefill_step_size`` prompt tokens, in arrival order.

        One request at a time: the next starts when the current one decodes, if a
        row is free.
        """
        tokens_left = self._config.prefill_step_size
        while tokens_left > 0:
            if self._prefilling is None:
                if not self._waiting or self._decoding.is_full:
                    return
                self._start_prefill(self._waiting.popleft())
            tokens_left -= self._prefill_request(self._prefilling, tokens_left)

    def _prefill_request(
        self, generation: generations.Generation, num_tokens: int
    ) -> int:
        """Prefill up to ``num_tokens``; return how many. A full prompt decodes."""
        num_done = 0
        while num_done < num_tokens and not generation.is_prefilled:
            num_done += self._prefill_chunk(generation, num_tokens - num_done)
        if generation.is_prefilled:
            self._start_decode(generation)
        return num_done

    def _start_prefill(self, generation: generations.Generation) -> None:
        """Move a waiting request into the prefill, with a cache to fill."""
        self._prefilling = generation
        prompt_cache, num_cached = self._fetch_cache(generation.request)
        generation.prompt_cache = prompt_cache
        generation.num_cached = num_cached
        generation.num_prefilled = num_cached

    def _fetch_cache(self, request: protocol.Request) -> tuple[list[Any], int]:
        """A cache with the longest cached prefix of the prompt, and its length."""
        tokens = request.tokens
        # The last token goes to the decode batch.
        earlier_tokens = list(tokens[:-1])
        prompt_cache, rest = self._cache_store.fetch_nearest_cache(
            model=None, tokens=earlier_tokens
        )
        if prompt_cache is None:
            prompt_cache = cache.make_prompt_cache(self._model)

        num_cached = len(tokens) - 1 - len(rest)
        return prompt_cache, num_cached

    def _prefill_chunk(
        self, generation: generations.Generation, num_tokens: int
    ) -> int:
        """Prefill at most ``num_tokens`` more prompt tokens, and return how many."""
        request = generation.request
        start = generation.num_prefilled
        checkpoints = _checkpoints_to_store(request, trimmable=self._trimmable)

        # The last token goes to the decode batch.
        chunk_end = _chunk_end(
            start,
            num_tokens=num_tokens,
            prompt_end=len(request.tokens) - 1,
            checkpoints=checkpoints,
        )

        chunk = mx.array(request.tokens[start:chunk_end])[None]
        self._model(chunk, cache=generation.prompt_cache)
        # Do not wait: the next decode step can queue behind the chunk.
        mx.async_eval([c.state for c in generation.prompt_cache])
        # Each chunk copies the growing cache; free the old copies.
        mx.clear_cache()

        generation.num_prefilled = chunk_end
        for p, kind in checkpoints:
            if p == chunk_end:
                self._store_checkpoint(generation, kind)

        return chunk_end - start

    def _store_checkpoint(
        self, generation: generations.Generation, kind: protocol.CheckpointKind
    ) -> None:
        """Store a copy of the prompt cache at the current prefill position."""
        checkpoint = copy.deepcopy(generation.prompt_cache)
        self._store_cache(checkpoint, generation.prefilled_tokens, cache_type=kind)

    def _store_cache(
        self,
        prompt_cache: list[Any],
        tokens: list[int],
        *,
        cache_type: str = "assistant",
    ) -> None:
        """Keep a cache in the cache store, at the same size on every rank."""
        # A pipeline rank holds only its layers: every rank counts the largest.
        nbytes = sum(c.nbytes for c in prompt_cache)
        nbytes = distributed.all_max(self._group, nbytes)

        self._cache_store.insert_cache(
            model=None,
            tokens=tokens,
            prompt_cache=prompt_cache,
            cache_type=cache_type,
            nbytes=nbytes,
        )

    def _start_decode(self, generation: generations.Generation) -> None:
        """Move a prefilled request into the decode batch."""
        sampler, processors = _sampler_and_processors(
            generation.request.params,
            xtc_special_tokens=self._xtc_special_tokens,
        )

        # After the add: if it fails, the request is still in prefill.
        self._decoding.add(generation, sampler=sampler, logits_processors=processors)
        self._prefilling = None

    def _read_token(
        self,
        generation: generations.Generation,
        token: int,
        *,
        logprobs: tuple[protocol.TokenLogprob, ...],
    ) -> None:
        """Send the text of a new token; end the request if the token ends it."""
        try:
            delta, reason = generation.read_token(token)
        except Exception as e:
            _logger.exception("Request %d failed.", generation.handle)
            # The traceback frames hold the arrays of the failed step.
            self._finish(generation, event=_failure(e.with_traceback(None)))
            return

        event = protocol.TokensEvent(token_ids=(token,), delta=delta, logprobs=logprobs)
        self._add_event(generation, event)

        if reason is not None:
            event = protocol.FinishedEvent(
                reason=reason,
                usage=generation.usage,
                stop_string=generation.stop_string,
            )
            self._finish(generation, event=event)

    def _add_event(
        self, generation: generations.Generation, event: protocol.Event
    ) -> None:
        """Keep an event for the end of the step; only rank 0 sends events."""
        if self.rank == 0:
            self._events.append((generation.handle, event))

    def _take_events(self) -> _StepEvents:
        """Take the events of this step."""
        events, self._events = self._events, []
        return events

    def _finish(
        self,
        generation: generations.Generation,
        *,
        event: protocol.FinishedEvent | protocol.FailedEvent,
    ) -> None:
        """End a request where it is, and send its last event."""
        if generation is self._prefilling:
            self._prefilling = None
        elif generation in self._decoding:
            # The next step removes its row, and keeps its cache.
            self._decoding.drop(generation)
        else:
            self._waiting.remove(generation)

        del self._generations[generation.handle]
        self._add_event(generation, event)


def _check_request(request: protocol.Request, *, vocab_size: int) -> None:
    """Check what GenerationParams cannot: the prompt, the vocabulary, the state."""
    tokens = request.tokens
    params = request.params
    text_state = request.text_state

    # A bool is an int, but MLX makes a bool array of it.
    if len(tokens) == 0 or not all(type(t) is int for t in tokens):
        raise errors.InvalidRequestError("The prompt must be a non-empty list of ints.")
    if min(tokens) < 0 or max(tokens) >= vocab_size:
        raise errors.InvalidRequestError(
            "The prompt has a token id that is not in the vocabulary."
        )

    if params.top_k >= vocab_size:
        raise errors.InvalidRequestError(f"top_k must be less than {vocab_size}.")
    if params.logit_bias and max(params.logit_bias) >= vocab_size:
        raise errors.InvalidRequestError(
            "logit_bias has a token id that is not in the vocabulary."
        )

    # The engine thread makes the parser: a bad state there would fail the engine.
    if text_state is not None and text_state not in get_args(TextState):
        raise errors.InvalidRequestError(f"Unknown text state {text_state!r}.")


def _context_max_tokens(
    request: protocol.Request, *, default: int, context_length: int | None
) -> int:
    """The new tokens that fit in the context. Raises if the request does not fit."""
    requested = request.params.max_tokens
    num_tokens = len(request.tokens)
    max_tokens = requested or default

    if context_length is None:
        return max_tokens

    if requested is None:
        # Without a limit, a request may fill the context.
        max_tokens = min(max_tokens, context_length - num_tokens)

    if (need := num_tokens + max(max_tokens, 1)) > context_length:
        raise errors.ContextLengthExceededError(
            f"The context length is {context_length} tokens, but the request needs "
            f"{need} ({num_tokens} in the prompt)."
        )

    return max_tokens


def _probe_model(model: nn.Module) -> tuple[int, bool]:
    """Run the model once; return its vocabulary size and whether its cache trims."""
    prompt_cache = cache.make_prompt_cache(model)
    if names := _unbatchable(prompt_cache):
        raise ValueError(f"The cache of this model cannot batch: {names}.")

    logits = model(mx.zeros((1, cache.KVCache.step), mx.uint32), cache=prompt_cache)
    mx.eval(logits)
    vocab_size = logits.shape[-1]
    return vocab_size, _all_trimmable(prompt_cache)


def _unbatchable(prompt_cache: list[Any]) -> set[str]:
    """The names of the cache classes that cannot join a batch."""
    return {type(c).__name__ for c in prompt_cache if not hasattr(c, "merge")}


def _all_trimmable(prompt_cache: list[Any]) -> bool:
    """Whether every layer's cache can drop its last tokens, at any length."""
    # A rotating cache trims only until it wraps, so it counts as not trimmable.
    # A CacheList trims when all its caches do.
    layers = []
    for c in prompt_cache:
        layers += c.caches if isinstance(c, cache.CacheList) else [c]

    return all(type(c) is cache.KVCache for c in layers)


def _sync_trimmable(trimmable: bool, *, group: mx.distributed.Group | None) -> bool:
    """Whether the caches trim; the ranks must agree, or their cache stores differ."""
    if distributed.all_min(group, trimmable) != distributed.all_max(group, trimmable):
        raise ValueError(
            "Some ranks can trim their caches and others cannot. Use tensor "
            "parallelism for this model."
        )
    return trimmable


def _text_readers(
    machine: TextStateMachine, request: protocol.Request
) -> tuple[TextStateMachine.Reader | None, TextStateMachine.Reader | None]:
    """The parser of the generated text, and the finder of its stop strings."""
    # Without a state the text is plain: it is all content.
    text_reader = None
    if request.text_state is not None:
        text_reader = machine.reader(state=request.text_state)

    stop_reader = None
    if request.params.stop:
        stop_reader = TextStateMachine.from_stop_words(request.params.stop).reader()

    return text_reader, stop_reader


def _checkpoints_to_store(
    request: protocol.Request, *, trimmable: bool
) -> tuple[tuple[int, protocol.CheckpointKind], ...]:
    """The checkpoints where the prefill stores a copy of the prompt cache."""
    # A trimmable cache's final entry also serves the user checkpoints.
    if not trimmable:
        user = protocol.CheckpointKind.USER
        checkpoints = list(request.checkpoints)
        # Without a user checkpoint, keep the prompt end for prompts that start with it.
        if all(kind != user for _, kind in checkpoints):
            prompt_end = len(request.tokens) - 1
            checkpoints.append((prompt_end, user))
        return tuple(checkpoints)

    system = protocol.CheckpointKind.SYSTEM
    return tuple((p, kind) for p, kind in request.checkpoints if kind == system)


def _chunk_end(
    start: int,
    *,
    num_tokens: int,
    prompt_end: int,
    checkpoints: Sequence[tuple[int, protocol.CheckpointKind]],
) -> int:
    """The chunk end: ``start + num_tokens``, the prompt end or the next checkpoint."""
    next_checkpoints = [p for p, _ in checkpoints if p > start]
    return min(start + num_tokens, prompt_end, *next_checkpoints)


def _sampler_and_processors(
    params: settings.GenerationParams, *, xtc_special_tokens: tuple[int, ...]
) -> tuple[sample_utils.Sampler, list[sample_utils.LogitsProcessor]]:
    """The sampler and the logits processors of a request's parameters."""
    sampler = _cached_sampler(
        temperature=params.temperature,
        top_p=params.top_p,
        min_p=params.min_p,
        top_k=params.top_k,
        xtc_probability=params.xtc_probability,
        xtc_threshold=params.xtc_threshold,
        xtc_special_tokens=xtc_special_tokens,
    )

    # Processors can make arrays, so they belong on the engine thread.
    processors = sample_utils.make_logits_processors(
        logit_bias=params.logit_bias,
        repetition_penalty=params.repetition_penalty,
        repetition_context_size=params.repetition_context_size,
        presence_penalty=params.presence_penalty,
        presence_context_size=params.presence_context_size,
        frequency_penalty=params.frequency_penalty,
        frequency_context_size=params.frequency_context_size,
    )

    return sampler, processors


# TODO(michalk8): Equal settings must share one sampler object, so the batch samples
# them in one call; each new value compiles again, and its graph is never freed.
# Pass them as arrays and vmap.
@functools.lru_cache(maxsize=256)
def _cached_sampler(
    *,
    temperature: float,
    top_p: float,
    min_p: float,
    top_k: int,
    xtc_probability: float,
    xtc_threshold: float,
    xtc_special_tokens: tuple[int, ...],
) -> sample_utils.Sampler:
    """One sampler per setting, so the batch samples equal requests in one call."""
    return sample_utils.make_sampler(
        temperature,
        top_p=top_p,
        min_p=min_p,
        top_k=top_k,
        xtc_probability=xtc_probability,
        xtc_threshold=xtc_threshold,
        xtc_special_tokens=list(xtc_special_tokens),
    )


def _failure(error: Exception) -> protocol.FailedEvent:
    """The event of a request that failed inside the engine."""
    failure = errors.EngineError(f"{type(error).__name__}: {error}")
    failure.__cause__ = error
    return protocol.FailedEvent(error=failure)
