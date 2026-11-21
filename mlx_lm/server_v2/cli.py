# Copyright © 2026 Apple Inc.

import argparse
import asyncio
import dataclasses
import faulthandler
import json
import logging
import pathlib
import socket
import sys
from typing import Any

import mlx.core as mx
import pydantic
import transformers
import uvicorn
from fastapi.middleware import cors

from mlx_lm import utils
from mlx_lm.server_v2 import settings
from mlx_lm.server_v2.api import anthropic, app, openai
from mlx_lm.server_v2.chat import rendering
from mlx_lm.server_v2.engine import core

__all__ = ["main"]

_logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """Check the flags, load the model, then serve until a shutdown or a failure."""
    parser = _parser()
    args = parser.parse_args(argv)
    if not isinstance(args.chat_template_args, dict):
        parser.error("--chat-template-args must be a JSON object.")
    if reserved := rendering.reserved_kwargs(args.chat_template_args):
        parser.error(f"--chat-template-args cannot set {reserved}.")

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    group = mx.distributed.init()
    group = group if group.size() > 1 else None
    if group is not None and args.adapter_path:
        parser.error("--adapter-path is not supported when distributed.")

    # Before the load: a bad flag must not wait for the model.
    try:
        engine_config = _engine_config(args)
        settings.GenerationParams(**_sampling_flags(args))
    except pydantic.ValidationError as e:
        error, *_ = e.errors()
        parser.error(f"{error['loc'][0]}: {error['msg']}")

    # Before the load: a busy port must not wait for the model.
    if group is None or group.rank() == 0:
        try:
            socket.create_server((args.host, args.port)).close()
        except OSError as e:
            parser.error(f"cannot listen on {args.host}:{args.port}: {e.strerror}")

    model, tokenizer, config, model_path = _load(args, group=group)
    default_params = _default_params(args, model_path)
    _logger.info("Default parameters: %s", default_params)

    context_length = _context_length(config, model_path)
    engine_config = dataclasses.replace(engine_config, context_length=context_length)
    mx.set_cache_limit(args.cache_limit)
    engine = core.Engine(model, tokenizer, config=engine_config, group=group)

    if group is not None and group.rank() != 0:
        # The other ranks have no server: rank 0 sends the requests and the close.
        asyncio.run(_run_engine_only(engine))
    else:
        server_config = settings.ServerConfig(
            model_id=args.model,
            default_params=default_params,
            template_kwargs=args.chat_template_args,
        )
        state = app.ServerState(
            engine=engine, tokenizer=tokenizer, config=server_config
        )
        _run_server(args, state)

    sys.exit(1 if engine.error is not None else 0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve one model over OpenAI and Anthropic APIs (experimental)."
    )
    parser.add_argument("--model", required=True, help="Path or Hugging Face repo")
    parser.add_argument("--adapter-path", help="Path to trained adapter weights")

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for the HTTP server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the HTTP server (default: 8080)",
    )
    parser.add_argument(
        "--allowed-origins",
        type=lambda x: [origin.strip() for origin in x.split(",")],
        help="Comma-separated origins for CORS (default: none)",
    )

    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Run custom model and tokenizer code from the model repository",
    )
    parser.add_argument("--chat-template", help="A chat template for the tokenizer")
    parser.add_argument(
        "--chat-template-args",
        type=json.loads,
        default={},
        help='JSON arguments for the chat template, e.g. {"enable_thinking": false}',
    )
    parser.add_argument("--tool-parser", help="A module name in mlx_lm.tool_parsers")

    parser.add_argument(
        "--temp", type=float, help="Default temperature (else generation_config.json)"
    )
    parser.add_argument("--top-p", type=float, help="Default top-p")
    parser.add_argument("--top-k", type=int, help="Default top-k")
    parser.add_argument("--min-p", type=float, help="Default min-p")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Default maximum number of generated tokens (default: 8192)",
    )

    parser.add_argument(
        "--decode-concurrency",
        type=int,
        default=32,
        help="Maximum number of requests that decode together (default: 32)",
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=512,
        help="Prompt tokens per prefill step (default: 512)",
    )
    parser.add_argument(
        "--prompt-cache-size",
        type=int,
        default=10,
        help="Maximum number of entries in the prompt cache (default: 10)",
    )
    parser.add_argument(
        "--prompt-cache-bytes",
        type=utils._parse_size,
        default=1 << 63,
        help="Maximum size of the prompt cache, e.g. 8GB (default: no limit)",
    )
    parser.add_argument(
        "--cache-limit",
        type=utils._parse_size,
        default=10**9,
        help="Freed memory that MLX keeps for reuse, e.g. 4GB (default: 1GB)",
    )

    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="When distributed, split the layers over the ranks, not the tensors",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )

    return parser


def _engine_config(args: argparse.Namespace) -> settings.EngineConfig:
    """The engine config of the flags; the model gives the context length."""
    return settings.EngineConfig(
        max_tokens=args.max_tokens,
        decode_concurrency=args.decode_concurrency,
        prefill_step_size=args.prefill_step_size,
        prompt_cache_size=args.prompt_cache_size,
        prompt_cache_bytes=args.prompt_cache_bytes,
    )


# The config keys of the context length, in the order of vLLM's list.
_CONTEXT_KEYS = (
    "max_position_embeddings",
    "n_positions",
    "max_seq_len",
    "seq_length",
    "max_sequence_length",
    "max_seq_length",
    "seq_len",
)


def _context_length(config: dict[str, Any], model_path: pathlib.Path) -> int | None:
    """The context length of the config or its text config.

    Without a key, the transformers default applies.
    """
    configs = (config, config.get("text_config") or {})
    for key in _CONTEXT_KEYS:
        for values in configs:
            if (value := values.get(key)) is not None:
                return value
    try:
        # Without remote code: else transformers asks whether to run it.
        hf_config = transformers.AutoConfig.from_pretrained(
            model_path, trust_remote_code=False
        )
    except Exception:
        return None
    return getattr(hf_config.get_text_config(), "max_position_embeddings", None)


def _default_params(
    args: argparse.Namespace, model_path: pathlib.Path
) -> settings.GenerationParams:
    """The sampling defaults: the flags, else generation_config.json."""
    flags = _sampling_flags(args)
    path = model_path / "generation_config.json"
    try:
        values = _file_params(path)
        values.update(flags)
        return settings.GenerationParams(**values)
    except ValueError as e:
        # A bad file does not stop the server; main() checked the flags.
        _logger.warning("The server ignores the sampling values of %s: %s", path, e)
        return settings.GenerationParams(**flags)


def _file_params(path: pathlib.Path) -> dict[str, Any]:
    """The sampling values of generation_config.json, if the file samples."""
    config = json.loads(path.read_text()) if path.exists() else {}
    if not isinstance(config, dict):
        raise ValueError("The file is not a JSON object.")
    if not config.get("do_sample"):
        return {}

    temperature = config.get("temperature")
    values = {
        # Sampling without a temperature uses 1.0.
        "temperature": 1.0 if temperature is None else temperature,
        "top_p": config.get("top_p"),
        "top_k": config.get("top_k"),
        "min_p": config.get("min_p"),
        "repetition_penalty": config.get("repetition_penalty"),
    }
    # Some configs turn top-k off with -1; here that is 0.
    if values["top_k"] == -1:
        values["top_k"] = 0
    return {k: v for k, v in values.items() if v is not None}


def _sampling_flags(args: argparse.Namespace) -> dict[str, float | int]:
    """The sampling flags that the user set."""
    flags = {
        "temperature": args.temp,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
    }
    return {k: v for k, v in flags.items() if v is not None}


def _load(
    args: argparse.Namespace, *, group: mx.distributed.Group | None
) -> tuple[Any, Any, dict[str, Any], pathlib.Path]:
    """The model, the tokenizer, the config and the model path; a rank's part."""
    tokenizer_config = {"trust_remote_code": args.trust_remote_code}
    if args.chat_template:
        tokenizer_config["chat_template"] = args.chat_template
    if args.tool_parser:
        tokenizer_config["tool_parser_type"] = args.tool_parser

    if group is None:
        model_path = utils._download(args.model)
        model, tokenizer, config = utils.load(
            str(model_path),
            tokenizer_config=tokenizer_config,
            adapter_path=args.adapter_path,
            return_config=True,
            trust_remote_code=args.trust_remote_code,
        )
        return model, tokenizer, config, model_path

    model, tokenizer, config = utils.sharded_load(
        args.model,
        pipeline_group=group if args.pipeline else None,
        tensor_group=None if args.pipeline else group,
        return_config=True,
        tokenizer_config=tokenizer_config,
        trust_remote_code=args.trust_remote_code,
    )
    # The rank downloads only its weights; the configs are all there.
    model_path = utils._download(args.model, allow_patterns=["*.json"])
    return model, tokenizer, config, model_path


async def _run_engine_only(engine: core.Engine) -> None:
    """Run only the engine, as a rank without a server, until rank 0 closes it."""
    await engine.start()
    await engine.join()


def _run_server(args: argparse.Namespace, state: app.ServerState) -> None:
    """Serve HTTP until a shutdown signal or an engine failure."""
    asgi_app = app.make_app(state, openai.router, anthropic.router)
    if args.allowed_origins:
        asgi_app = cors.CORSMiddleware(
            asgi_app,
            allow_origins=args.allowed_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    _logger.warning(
        "The server is experimental and not recommended for production: it only "
        "implements basic security checks."
    )

    server = uvicorn.Server(
        uvicorn.Config(
            asgi_app,
            host=args.host,
            port=args.port,
            log_level=args.log_level.lower(),
            timeout_graceful_shutdown=30,
        )
    )

    try:
        asyncio.run(_serve(server, state))
    except KeyboardInterrupt:
        # uvicorn raises Ctrl-C again after its graceful shutdown.
        pass


async def _serve(server: uvicorn.Server, state: app.ServerState) -> None:
    """Run the server; stop it when the engine fails, and report stalls.

    A shutdown closes the engine at once: the requests in flight end with an
    error, so uvicorn has none to cancel.
    """
    task = asyncio.create_task(server.serve())
    stalled = False
    while not task.done():
        await asyncio.wait({task}, timeout=1.0)
        if state.engine.status == core.EngineStatus.FAILED and not server.should_exit:
            _logger.error("The engine failed, stopping the server.")
            server.should_exit = True

        if server.should_exit and state.engine.status == core.EngineStatus.OK:
            await state.engine.close()

        was_stalled = stalled
        stalled = state.is_stalled
        if stalled and not was_stalled:
            timeout = state.config.stall_timeout
            _logger.error("The engine made no progress for %s s.", timeout)
            faulthandler.dump_traceback(all_threads=True)

    await task
