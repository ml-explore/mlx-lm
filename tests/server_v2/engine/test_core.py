# Copyright © 2026 Apple Inc.

import asyncio
import gc
import threading
import unittest
from collections.abc import AsyncIterator, Sequence
from unittest import mock

import mlx.core as mx

from mlx_lm import utils
from mlx_lm.generate import TextStateMachine
from mlx_lm.server_v2 import settings
from mlx_lm.server_v2.engine import core, errors, generations, protocol
from tests.server_v2 import helpers

Delta = TextStateMachine.Delta
LENGTH = protocol.FinishReason.LENGTH
STOP_STRING = protocol.FinishReason.STOP_STRING


class FlakyModel:
    """A model that raises on the given calls."""

    def __init__(self, model):
        self.model = model
        self.calls = 0
        self.fail_at = set()

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.calls in self.fail_at:
            raise RuntimeError("Injected failure")
        return self.model(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.model, name)


class TestEngine(unittest.IsolatedAsyncioTestCase):

    @classmethod
    def setUpClass(cls):
        model, cls.tokenizer = helpers.load_model()
        cls.model = FlakyModel(model)
        messages = [{"role": "user", "content": "Count from 1 to 50."}]
        cls.prompt = cls.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )

    def setUp(self):
        self.model.fail_at = set()

    async def engine(self, *, start: bool = True, **config) -> core.Engine:
        config = settings.EngineConfig(**config)
        engine = core.Engine(self.model, self.tokenizer, config=config)
        self.addAsyncCleanup(engine.close)
        if start:
            await engine.start()
        return engine

    def submit(
        self,
        engine: core.Engine,
        tokens: list[int] | None = None,
        checkpoints: tuple = (),
        state: str | None = None,
        **params,
    ) -> AsyncIterator[protocol.Update]:
        tokens = self.prompt if tokens is None else tokens
        params = settings.GenerationParams(**params)
        request = protocol.Request(
            tokens, params=params, checkpoints=checkpoints, text_state=state
        )
        return engine.updates(engine.submit(request))

    async def token_ids(self, stream: AsyncIterator[protocol.Update]) -> list[int]:
        """The token ids of the stream, read to its end."""
        ids = []
        async with asyncio.timeout(delay=60):
            async for update in stream:
                for event in update.events:
                    ids += event.token_ids
        return ids

    async def collect(self, stream: AsyncIterator[protocol.Update]):
        """The text of the stream, and the end that closes it."""
        text = ""
        async with asyncio.timeout(delay=60):
            async for update in stream:
                text += "".join(event.delta.content for event in update.events)
                if update.end is not None:
                    return text, update.end

    async def test_lifecycle(self):
        engine = await self.engine(start=False)
        probe_model = core._probe_model
        statuses = []

        def probe_and_record(model):
            statuses.append(engine.status)
            return probe_model(model)

        self.enterContext(mock.patch.object(core, "_probe_model", probe_and_record))
        await engine.close()
        await engine.start()
        statuses.append(engine.status)
        with self.assertRaises(RuntimeError):
            await engine.start()
        await engine.close()
        statuses.append(engine.status)
        status = core.EngineStatus
        self.assertEqual(statuses, [status.NEW, status.OK, status.CLOSED])
        # The caller can keep the engine, so close frees its caches.
        self.assertIsNone(engine._worker._decoding)
        self.assertIsNone(engine._worker._cache_store)
        with self.assertRaises(errors.UnavailableError):
            self.submit(engine)
        engine.cancel(0)
        self.assertEqual(len(engine._commands), 0)

        engine = await self.engine(start=False)
        self.model.fail_at = {self.model.calls + 1}
        with self.assertRaisesRegex(RuntimeError, "Injected"):
            await engine.start()
        self.assertEqual(engine.status, core.EngineStatus.FAILED)

    async def test_generate(self):
        engine = await self.engine()
        text, end = await self.collect(self.submit(engine, max_tokens=10))
        self.assertEqual(engine._open, set())
        self.assertEqual(end.reason, protocol.FinishReason.LENGTH)
        self.assertEqual(end.usage.prompt_tokens, len(self.prompt))
        self.assertEqual(end.usage.completion_tokens, 10)
        self.assertTrue(text)

        cached, end = await self.collect(self.submit(engine, max_tokens=10))
        self.assertEqual(cached, text)
        self.assertGreater(end.usage.cached_tokens, 0)

        engine = await self.engine(prefill_step_size=5)
        stream = self.submit(engine, max_tokens=10)
        self.assertEqual((await self.collect(stream))[0], text)

    async def test_logprobs(self):
        engine = await self.engine()
        async for update in self.submit(engine, max_tokens=4, logprobs=2):
            for event in update.events:
                (token,) = event.token_ids
                (logprobs,) = event.logprobs
                first, _ = logprobs.top
                # Greedy decoding takes the most likely token.
                self.assertEqual(first, (token, logprobs.logprob))
        async for update in self.submit(engine, max_tokens=1):
            for event in update.events:
                self.assertEqual(event.logprobs, ())
        # Zero most likely tokens: the sampled token only.
        async for update in self.submit(engine, max_tokens=1, logprobs=0):
            for event in update.events:
                (logprobs,) = event.logprobs
                self.assertEqual(logprobs.top, ())

    async def test_stop_string(self):
        engine = await self.engine()
        text, _ = await self.collect(self.submit(engine, max_tokens=30))
        stop_string = text[len(text) // 2 :][:3]
        stream = self.submit(engine, max_tokens=30, stop=(stop_string,))
        cut, end = await self.collect(stream)
        self.assertEqual((end.reason, end.stop_string), (STOP_STRING, stop_string))
        self.assertEqual(cut, text[: text.index(stop_string)])

    async def test_submit_checks_the_request(self):
        engine = await self.engine(context_length=len(self.prompt) + 3)
        size = engine._vocab_size
        cases = [
            {"tokens": []},
            {"tokens": [0.5]},
            {"tokens": [-1]},
            {"tokens": [True, True]},
            {"tokens": [size]},
            {"top_k": size},
            {"logit_bias": {size: 1.0}},
            {"state": "other"},
        ]
        for kwargs in cases:
            with (
                self.subTest(**kwargs),
                self.assertRaises(errors.InvalidRequestError),
            ):
                self.submit(engine, **kwargs)
        too_long = [{"max_tokens": 4}, {"tokens": self.prompt + [0] * 3}]
        for kwargs in too_long:
            with self.assertRaises(errors.ContextLengthExceededError):
                self.submit(engine, **kwargs)
        # Without max_tokens, a request generates up to the context length.
        _, end = await self.collect(self.submit(engine))
        self.assertEqual(end.reason, protocol.FinishReason.LENGTH)
        self.assertEqual(end.usage.completion_tokens, 3)

    async def test_updates_that_never_start_are_released(self):
        engine = await self.engine()
        stream = self.submit(engine, max_tokens=1000)
        self.assertEqual(len(engine._queues), 1)
        del stream
        gc.collect()
        async with asyncio.timeout(delay=60):
            while engine._queues or engine._worker._generations:
                await asyncio.sleep(0.01)

    async def test_join_does_not_cancel_the_engine(self):
        engine = await self.engine()
        with self.assertRaises(TimeoutError):
            await asyncio.wait_for(engine.join(), timeout=0.05)
        self.assertEqual((engine.status, engine.error), (core.EngineStatus.OK, None))
        _, end = await self.collect(self.submit(engine, max_tokens=2))
        self.assertEqual(end.reason, protocol.FinishReason.LENGTH)

    async def test_cancelled_task_closes_the_engine(self):
        engine = await self.engine()
        running = self.submit(engine, max_tokens=1000)
        await anext(running)
        engine._task.cancel()
        with self.assertNoLogs(level="ERROR"):
            await engine.join()
        status = core.EngineStatus.CLOSED
        self.assertEqual((engine.status, engine.error), (status, None))
        with self.assertRaisesRegex(errors.UnavailableError, "closed"):
            await self.collect(running)

    async def test_cancel(self):
        engine = await self.engine(decode_concurrency=1)
        running = self.submit(engine, max_tokens=1000)
        waiting = self.submit(engine, max_tokens=10)
        await anext(running)
        # The handles count from 0.
        engine.cancel(1)
        engine.cancel(0)
        for stream in (running, waiting):
            _, end = await self.collect(stream)
            self.assertEqual(end.reason, protocol.FinishReason.CANCELLED)
        empty = protocol.Usage(
            prompt_tokens=len(self.prompt), cached_tokens=0, completion_tokens=0
        )
        self.assertEqual(end.usage, empty)

        # Closing the stream cancels too: else the request holds the only row for hours.
        no_eos = {token: -100.0 for token in self.tokenizer.eos_token_ids}
        closing = self.submit(engine, max_tokens=10**6, logit_bias=no_eos)
        await anext(closing)
        await closing.aclose()
        async with asyncio.timeout(delay=10):
            await self.token_ids(self.submit(engine, max_tokens=1))
        self.assertEqual(engine._worker._generations, {})
        self.assertNotIn(2, engine._queues)

    async def test_finished_request_is_not_cancelled(self):
        engine = await self.engine()
        add_command = mock.Mock(wraps=engine._add_command)
        engine._add_command = add_command
        stream = self.submit(engine, max_tokens=2)
        await self.collect(stream)
        await stream.aclose()
        commands = [type(call.args[0]) for call in add_command.call_args_list]
        self.assertEqual(commands, [core._Submit])

    async def test_cancel_and_retry_in_one_step(self):
        engine = await self.engine()
        running = self.submit(engine, max_tokens=1000)
        await anext(running)
        # No await between them, so one step gets both commands.
        engine.cancel(0)
        retry = self.submit(engine, max_tokens=1)
        _, end = await self.collect(retry)
        # The step stores the cancelled cache before the retry looks it up.
        self.assertGreater(end.usage.cached_tokens, 0)

    async def test_decode_concurrency(self):
        engine = await self.engine(decode_concurrency=1)
        first = self.submit(engine, max_tokens=1000)
        second = self.submit(engine, max_tokens=1)
        await anext(first)
        self.assertEqual(len(engine._worker._waiting), 1)
        await second.aclose()

    async def test_sampling_params_reach_the_sampler(self):
        engine = await self.engine()
        greedy = await self.token_ids(self.submit(engine, max_tokens=8))
        hot = self.submit(engine, max_tokens=8, temperature=5.0, top_k=1)
        # With top_k 1, any temperature takes the most likely token.
        self.assertEqual(await self.token_ids(hot), greedy)
        # A token that the model does not pick, made certain.
        target = max(greedy) + 1
        biased = self.submit(engine, max_tokens=3, logit_bias={target: 100.0})
        self.assertEqual(await self.token_ids(biased), [target] * 3)
        # The greedy text repeats tokens; a heavy penalty stops that.
        self.assertLess(len(set(greedy)), len(greedy))
        penalized = self.submit(engine, max_tokens=8, repetition_penalty=100.0)
        ids = await self.token_ids(penalized)
        self.assertEqual(len(set(ids)), len(ids))

    async def test_tool_calls_follow_the_request(self):
        tokenizer = helpers.load_chat_tokenizer()
        engine = core.Engine(self.model, tokenizer)
        text = '<tool_call>{"a": 1}</tool_call>'
        tokens = tokenizer.encode(text)
        expected = {True: Delta(tool_calls=('{"a": 1}',)), False: Delta(content=text)}
        for tool_calls, delta in expected.items():
            request = protocol.Request(
                [0], text_state="normal", parse_tool_calls=tool_calls
            )
            submission = core._Submit(handle=0, request=request, max_tokens=len(tokens))
            generation = engine._worker._new_generation(submission)
            got = Delta()
            for token in tokens:
                got += generation.read_token(token)[0]
            self.assertEqual(got, delta)

    async def test_group_of_one(self):
        engine = await self.engine(max_tokens=10)
        text, _ = await self.collect(self.submit(engine))
        # The commands and the sizes go through the group calls and the pickle.
        group = mx.distributed.init()
        grouped = core.Engine(self.model, self.tokenizer, group=group)
        self.addAsyncCleanup(grouped.close)
        await grouped.start()
        stream = self.submit(grouped, max_tokens=10)
        self.assertEqual((await self.collect(stream))[0], text)

    async def test_cancel_during_prefill(self):
        engine = await self.engine(prefill_step_size=2)
        prefill_chunk = engine._worker._prefill_chunk

        def cancel_then_prefill(generation, num_tokens):
            # On the worker thread: add the command, as the loop would.
            command = core._Cancel(generation.handle)
            engine._commands.append(command)
            return prefill_chunk(generation, num_tokens)

        engine._worker._prefill_chunk = cancel_then_prefill
        _, end = await self.collect(self.submit(engine, max_tokens=10))
        self.assertEqual(end.reason, protocol.FinishReason.CANCELLED)
        self.assertEqual(end.usage.completion_tokens, 0)
        # The prefill keeps its cache: the same prompt goes on from there.
        engine._worker._prefill_chunk = prefill_chunk
        _, end = await self.collect(self.submit(engine, max_tokens=1))
        self.assertGreater(end.usage.cached_tokens, 0)

    async def test_requests_decode_in_one_batch(self):
        engine = await self.engine()
        generator = engine._worker._decoding._generator
        generator._make_batch = mock.Mock(wraps=generator._make_batch)
        # No await between the submits, so the three requests wait together.
        streams = [self.submit(engine, max_tokens=2) for _ in range(3)]
        await asyncio.gather(*(self.collect(stream) for stream in streams))
        self.assertIn(mock.call(3), generator._make_batch.call_args_list)

    async def test_prefill_stores_checkpoints(self):
        system = (5, protocol.CheckpointKind.SYSTEM)
        user = (9, protocol.CheckpointKind.USER)
        prompt_end = (len(self.prompt) - 1, protocol.CheckpointKind.USER)
        cases = [
            # The final entry of a trimmable cache serves the user checkpoint.
            (True, (system, user), (system,)),
            (False, (system, user), (system, user)),
            # Without a user checkpoint, the prompt end is kept.
            (False, (system,), (system, prompt_end)),
        ]
        for trimmable, checkpoints, stored in cases:
            engine = await self.engine()
            engine._worker._trimmable = trimmable
            insert = mock.Mock(wraps=engine._worker._cache_store.insert_cache)
            engine._worker._cache_store.insert_cache = insert
            stream = self.submit(engine, checkpoints=checkpoints, max_tokens=1)
            await self.collect(stream)
            # The other insert is of the finished request.
            calls = [
                c
                for c in insert.call_args_list
                if c.kwargs["cache_type"] != "assistant"
            ]
            found = [(len(c.kwargs["tokens"]), c.kwargs["cache_type"]) for c in calls]
            self.assertEqual(found, list(stored))
            # Each is a copy: the prefill does not move a stored cache on.
            offsets = [c.kwargs["prompt_cache"][0].offset for c in calls]
            self.assertEqual(offsets, [n for n, _ in stored])

    async def test_close_ends_the_requests(self):
        engine = await self.engine()
        running = self.submit(engine, max_tokens=1000)
        await anext(running)
        # A request that comes right before the close is ended too.
        waiting = self.submit(engine, max_tokens=10)
        with self.assertNoLogs(level="ERROR"):
            await engine.close()
        # A cut-off answer must not look complete.
        for stream in (running, waiting):
            with self.assertRaisesRegex(errors.UnavailableError, "closed"):
                await self.collect(stream)
        with self.assertRaises(errors.UnavailableError):
            self.submit(engine)

    async def test_failure_ends_a_request_that_was_queued_late(self):
        engine = await self.engine()
        running = self.submit(engine, max_tokens=1000)
        await anext(running)
        stuck, release = threading.Event(), threading.Event()

        def failing_step():
            stuck.set()
            release.wait(timeout=60)
            raise RuntimeError("Injected")

        engine._worker._step = failing_step
        await asyncio.to_thread(stuck.wait, timeout=60)
        # The step took its commands already: this one is left in the queue.
        late = self.submit(engine, max_tokens=1)
        release.set()
        with self.assertLogs(level="ERROR"):
            with self.assertRaises(errors.UnavailableError):
                await self.collect(late)

    async def test_stall_and_idle(self):
        engine = await self.engine(keepalive=0.01)
        stuck, release = threading.Event(), threading.Event()
        step = engine._worker._step

        def blocking_step():
            step()
            stuck.set()
            release.wait(timeout=60)

        engine._worker._step = blocking_step
        request = protocol.Request(self.prompt)
        stream = engine.updates(engine.submit(request))
        await asyncio.to_thread(stuck.wait, timeout=60)
        # The loop is free while the step blocks, and it sees the stall.
        self.assertGreater(engine.seconds_since_step(), 0)
        self.assertTrue((await anext(stream)).is_empty)
        release.set()
        engine._worker._step = step
        await self.collect(stream)
        # Idle time is not a stall.
        async with asyncio.timeout(delay=10):
            while engine.seconds_since_step() != 0.0:
                await asyncio.sleep(0.01)

    async def test_failure_stops_the_engine(self):
        engine = await self.engine(decode_concurrency=1)
        self.model.fail_at = {self.model.calls + 2}
        with self.assertLogs(level="ERROR"):
            running = self.submit(engine, max_tokens=10)
            waiting = self.submit(engine, max_tokens=10)
            for stream in (running, waiting):
                with self.assertRaises(errors.UnavailableError):
                    await self.collect(stream)
            await engine.join()
        self.assertEqual(engine.status, core.EngineStatus.FAILED)
        self.assertIsInstance(engine.error, RuntimeError)
        with self.assertRaises(errors.UnavailableError):
            self.submit(engine)

    async def test_lost_commands_end_with_the_failure(self):
        engine = await self.engine()
        new_generation = core._Worker._new_generation
        submissions = []

        def fail_second(worker, submission):
            submissions.append(submission)
            if len(submissions) == 2:
                raise RuntimeError("Injected failure")
            return new_generation(worker, submission)

        self.enterContext(
            mock.patch.object(core._Worker, "_new_generation", fail_second)
        )
        # No await between them, so one step takes all three commands.
        streams = [self.submit(engine, max_tokens=5) for _ in range(3)]
        with self.assertLogs(level="ERROR"):
            for stream in streams:
                with self.assertRaises(errors.UnavailableError):
                    async with asyncio.timeout(delay=10):
                        await self.collect(stream)

    async def test_failed_shutdown_ends_the_requests(self):
        engine = await self.engine()
        streams = [self.submit(engine, max_tokens=1000) for _ in range(2)]
        for stream in streams:
            await anext(stream)
        shut_down = engine._worker.shut_down

        def failing_shut_down(status):
            # The events are lost, but the updates close on their thread.
            shut_down(status)
            raise RuntimeError("Injected")

        engine._worker.shut_down = failing_shut_down
        raised = []
        with self.assertLogs(level="ERROR"):
            await engine.close()
            for stream in streams:
                with self.assertRaises(errors.UnavailableError) as caught:
                    await self.collect(stream)
                raised.append(caught.exception)
        # Each request raises its own error.
        self.assertIsNot(*raised)

    async def test_failed_decode_ends_only_its_request(self):
        engine = await self.engine()
        read_token = generations.Generation.read_token

        def broken_read(generation, token):
            if generation.max_tokens == 5:
                raise RuntimeError("Injected decode failure")
            return read_token(generation, token)

        with mock.patch.object(generations.Generation, "read_token", broken_read):
            with self.assertLogs(level="ERROR"):
                # No await between the submits, so both decode together.
                good = self.submit(engine, max_tokens=20)
                broken = self.submit(engine, max_tokens=5)
                with self.assertRaises(errors.EngineError):
                    await self.collect(broken)
                _, end = await self.collect(good)
        self.assertEqual(end.reason, LENGTH)


class TestReadToken(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = utils.load_tokenizer("mlx-community/Qwen3-4B-4bit")

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(
        self,
        tokens: list[int],
        *,
        state: str | None = "normal",
        eos: Sequence[int] = (),
        stop: tuple[str, ...] = (),
    ) -> tuple[Delta, protocol.FinishReason]:
        tokenizer = self.tokenizer
        machine = TextStateMachine.from_tokenizer(tokenizer)
        params = settings.GenerationParams(stop=stop)
        request = protocol.Request([0], text_state=state, params=params)
        reader, stop_reader = core._text_readers(machine, request)
        # The limit ends the request at its last token.
        generation = generations.Generation(
            handle=0,
            request=protocol.Request([0]),
            max_tokens=len(tokens),
            detokenizer=tokenizer.detokenizer,
            text_reader=reader,
            stop_reader=stop_reader,
            eos_token_ids=frozenset(eos),
        )
        delta = Delta()
        for token in tokens:
            new, reason = generation.read_token(token)
            delta += new
            if reason is not None:
                return delta, reason

    def test_eos(self):
        (eos,) = self.tokenizer.eos_token_ids
        tokens = self.encode("Hi") + [eos]
        expected = (Delta(content="Hi"), protocol.FinishReason.EOS)
        # The eos text is not in the updates, and eos wins over the limit.
        self.assertEqual(self.decode(tokens, eos=[eos]), expected)
        longer = tokens + self.encode(" more")
        self.assertEqual(self.decode(longer, eos=[eos]), expected)

    def test_end_releases_a_held_stop_prefix(self):
        # "a" can start the stop string "ab", so it waits; the limit ends the text.
        expected = (Delta(content="xa"), LENGTH)
        self.assertEqual(self.decode(self.encode("xa"), stop=("ab",)), expected)
        expected = (Delta(content="x"), STOP_STRING)
        self.assertEqual(self.decode(self.encode("xab"), stop=("ab",)), expected)

    def test_split_utf8_at_a_marker(self):
        # "🦜" is two tokens: it and a lone space wait for more, but stay reasoning.
        (space,) = self.encode(" ")
        tokens = self.encode("a🦜") + [space] + self.encode("</think>\n\n🦜b")
        self.assertEqual(
            self.decode(tokens, state="reasoning"),
            (Delta(content="🦜b", reasoning="a🦜 "), LENGTH),
        )


if __name__ == "__main__":
    unittest.main()
