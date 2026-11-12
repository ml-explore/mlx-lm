# Copyright © 2026 Apple Inc.

import argparse
import asyncio
import contextlib
import faulthandler
import io
import json
import logging
import pathlib
import socket
import tempfile
import unittest
from unittest import mock

import mlx.core as mx

from mlx_lm.server_v2 import cli, settings
from mlx_lm.server_v2.engine import core

FLAGS = {"temp": None, "top_p": None, "top_k": None, "min_p": None}


class Server:
    """A uvicorn.Server that does not bind: it serves until ``should_exit``."""

    def __init__(self):
        self.should_exit = False

    async def serve(self) -> None:
        # Ends on its own, so that a broken monitor fails and does not hang.
        for _ in range(100):
            if self.should_exit:
                return
            await asyncio.sleep(0)


def interrupt(main: object) -> None:
    """asyncio.run as on Ctrl-C."""
    main.close()
    raise KeyboardInterrupt


class TestCLI(unittest.TestCase):

    def test_defaults(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        path = pathlib.Path(directory)
        args = argparse.Namespace(**FLAGS)
        self.assertEqual(cli._default_params(args, path), settings.GenerationParams())
        config = {"temperature": 0.6, "top_p": 0.9, "top_k": 20, "min_p": None}
        config["repetition_penalty"] = 1.05
        # Greedy, unless the config samples.
        (path / "generation_config.json").write_text(json.dumps(config))
        self.assertEqual(cli._default_params(args, path), settings.GenerationParams())
        config = {**config, "do_sample": True, "max_length": 10}
        (path / "generation_config.json").write_text(json.dumps(config))
        expected = settings.GenerationParams(
            temperature=0.6, top_p=0.9, top_k=20, repetition_penalty=1.05
        )
        self.assertEqual(cli._default_params(args, path), expected)
        args.temp = 0.0
        args.min_p = 0.1
        expected = settings.GenerationParams(
            temperature=0.0, top_p=0.9, top_k=20, min_p=0.1, repetition_penalty=1.05
        )
        self.assertEqual(cli._default_params(args, path), expected)
        # Some configs turn top-k off with -1.
        config = {**config, "top_k": -1}
        (path / "generation_config.json").write_text(json.dumps(config))
        params = cli._default_params(args, path)
        self.assertEqual((params.top_p, params.top_k), (0.9, 0))
        # Sampling without a temperature, or with a null one, uses 1.0.
        config = {"do_sample": True, "temperature": None}
        (path / "generation_config.json").write_text(json.dumps(config))
        args = argparse.Namespace(**FLAGS)
        self.assertEqual(cli._default_params(args, path).temperature, 1.0)

    def test_bad_generation_config(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        path = pathlib.Path(directory)
        args = argparse.Namespace(**FLAGS)
        args.temp = 0.2
        texts = [
            '{"do_sample": true, "top_p": 0.0}',
            '{"do_sample": true, "top_k": "20"}',
            "{",
            "[1]",
            "null",
        ]
        for text in texts:
            with self.subTest(text=text):
                (path / "generation_config.json").write_text(text)
                with self.assertLogs(level="WARNING"):
                    params = cli._default_params(args, path)
                self.assertEqual(params, settings.GenerationParams(temperature=0.2))

    def test_config_value(self):
        key = "max_position_embeddings"
        self.assertEqual(cli._config_value({key: 8, "text_config": {key: 9}}, key), 8)
        self.assertEqual(
            cli._config_value({key: None, "text_config": {key: 9}}, key), 9
        )
        self.assertIsNone(cli._config_value({}, key))

    def test_chat_template_args(self):
        cases = [
            ("[1]", "must be a JSON object"),
            ('{"tokenize": false}', "['tokenize']"),
        ]
        for value, message in cases:
            stderr = io.StringIO()
            with (
                self.assertRaises(SystemExit) as e,
                contextlib.redirect_stderr(stderr),
            ):
                cli.main(["--model", "m", "--chat-template-args", value])
            self.assertEqual(e.exception.code, 2)
            self.assertIn(message, stderr.getvalue())

    def test_bad_flags_exit_before_the_load(self):
        load = self.enterContext(mock.patch.object(cli, "_load"))
        busy = self.enterContext(socket.create_server(("127.0.0.1", 0)))
        busy_port = str(busy.getsockname()[1])
        cases = [
            (["--decode-concurrency", "0"], "decode_concurrency"),
            (["--max-tokens", "0"], "max_tokens"),
            (["--temp", "-1"], "temperature"),
            (["--top-p", "0"], "top_p"),
            (["--port", busy_port], f"127.0.0.1:{busy_port}"),
        ]
        for flags, name in cases:
            stderr = io.StringIO()
            with (
                self.subTest(flags=flags),
                self.assertRaises(SystemExit) as e,
                contextlib.redirect_stderr(stderr),
            ):
                cli.main(["--model", "m", *flags])
            self.assertEqual(e.exception.code, 2)
            self.assertIn(name, stderr.getvalue())
        load.assert_not_called()

    def test_main(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        path = pathlib.Path(directory)
        loaded = (None, None, {"max_position_embeddings": 64})
        self.enterContext(mock.patch.object(cli.utils, "_download", return_value=path))
        load = mock.patch.object(cli.utils, "load", return_value=loaded)
        load = self.enterContext(load)
        engine_class = self.enterContext(mock.patch.object(cli.core, "Engine"))
        server_class = self.enterContext(mock.patch.object(cli.uvicorn, "Server"))
        self.enterContext(mock.patch.object(cli.socket, "create_server"))
        self.enterContext(mock.patch.object(asyncio, "run", interrupt))
        self.enterContext(mock.patch.object(logging, "basicConfig"))
        set_cache_limit = self.enterContext(mock.patch.object(mx, "set_cache_limit"))
        argv = ["--model", "m", "--decode-concurrency", "4"]
        argv += ["--temp", "0.5", "--chat-template-args", '{"x": 1}']
        argv += ["--max-tokens", "100", "--prefill-step-size", "64"]
        argv += ["--prompt-cache-size", "3", "--prompt-cache-bytes", "1GB"]
        argv += ["--cache-limit", "2GB", "--chat-template", "T"]
        argv += ["--tool-parser", "json_tools", "--port", "9000"]
        argv += ["--allowed-origins", "http://a, http://b"]
        # Ctrl-C ends with 0, unless the engine failed.
        for error, code in ((None, 0), (RuntimeError("Injected"), 1)):
            engine_class.return_value.error = error
            with self.assertRaises(SystemExit) as e:
                cli.main(argv)
            self.assertEqual(e.exception.code, code)
        expected = settings.EngineConfig(
            context_length=64,
            max_tokens=100,
            decode_concurrency=4,
            prefill_step_size=64,
            prompt_cache_size=3,
            prompt_cache_bytes=10**9,
        )
        self.assertEqual(engine_class.call_args.kwargs["config"], expected)
        set_cache_limit.assert_called_with(2 * 10**9)
        tokenizer_config = load.call_args.kwargs["tokenizer_config"]
        self.assertEqual(tokenizer_config["chat_template"], "T")
        self.assertEqual(tokenizer_config["tool_parser_type"], "json_tools")
        uvicorn_config = server_class.call_args.args[0]
        self.assertEqual(uvicorn_config.port, 9000)
        self.assertEqual(uvicorn_config.timeout_graceful_shutdown, 30)
        cors = uvicorn_config.app
        self.assertEqual(cors.allow_origins, ["http://a", "http://b"])
        config = cors.app.state.server.config
        self.assertEqual(
            (config.default_params.temperature, config.template_kwargs), (0.5, {"x": 1})
        )
        self.assertEqual(config.model_id, "m")


class TestServe(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        wait = asyncio.wait

        async def wait_a_turn(tasks: set, timeout: float) -> tuple:
            return await wait(tasks, timeout=0)

        # Each check of the monitor waits a turn, not a second.
        self.enterContext(mock.patch.object(asyncio, "wait", wait_a_turn))
        self.server = Server()
        self.state = mock.Mock(is_stalled=False)
        self.state.engine.status = core.EngineStatus.OK
        self.state.engine.close = mock.AsyncMock()
        self.state.config.stall_timeout = 60

    async def serve(self) -> None:
        await cli._serve(self.server, self.state)

    async def test_failed_engine_stops_the_server(self):
        self.state.engine.status = core.EngineStatus.FAILED
        with self.assertLogs(level="ERROR"):
            await self.serve()
        self.assertTrue(self.server.should_exit)

    async def test_shutdown_closes_the_engine(self):
        self.server.should_exit = True
        await self.serve()
        self.state.engine.close.assert_awaited_once()

    async def test_stall_dumps_once(self):
        stalls = iter([True, True, True])

        def stalled() -> bool:
            stall = next(stalls, None)
            if stall is None:
                self.server.should_exit = True
                return False
            return stall

        type(self.state).is_stalled = mock.PropertyMock(side_effect=stalled)
        with (
            mock.patch.object(faulthandler, "dump_traceback") as dump,
            self.assertLogs(level="ERROR"),
        ):
            await self.serve()
        dump.assert_called_once_with(all_threads=True)


if __name__ == "__main__":
    unittest.main()
