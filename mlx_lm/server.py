# Copyright © 2023 Apple Inc.

import argparse
from collections import deque
import gc
import json
import logging
import pickle
import platform
import socket
import time
import uuid
import warnings
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Empty as QueueEmpty
from queue import Queue
from threading import Thread
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)
from urllib.parse import unquote

import mlx.core as mx
from huggingface_hub import scan_cache_dir

from . import anthropic_api
from ._version import __version__
from .generate import (
    DEFAULT_QUANTIZED_KV_START,
    BatchGenerator,
    TextStateMachine,
    _model_supports_nemotron_h_mtp,
    _mtp_language_model,
    kv_cache_quantized,
    DEFAULT_PREFILL_SCORE_BYTES,
    MIN_PREFILL_STEP,
    PrefillBudget,
    make_prefill_step,
    make_stop_sequences,
    make_text_state_machine,
    maybe_quantize_kv_cache,
    stream_generate,
    wired_limit,
)
from .models.cache import (
    LRUPromptCache,
    can_trim_prompt_cache,
    make_prompt_cache,
    stays_trimmable,
    trim_prompt_cache,
)

DEFAULT_PREFILL_MEMORY_MB = 512

# Defaults of the server's memory / checkpoint options (see their --help).
# Also used when a cli_args object lacks one (tests, embedders).
SERVER_DEFAULTS = {
    "prefill_memory_mb": DEFAULT_PREFILL_MEMORY_MB,
    "min_prefill_step": MIN_PREFILL_STEP,
    "prefill_score_bytes": DEFAULT_PREFILL_SCORE_BYTES,
    "prefill_step_warn_below": 128,
    "memory_headroom_fraction": 0.9,
    "oom_retry_step_divisor": 2,
    "junction_min_gap_tokens": 256,
    # A speculative (MTP / draft model) single request moves into the batch
    # when another request arrives (see Handoff).
    "batch_handoff": True,
}
# Smallest valid value of the integer options above.
_OPT_MIN = {
    "prefill_memory_mb": 0,
    "min_prefill_step": 1,
    "prefill_score_bytes": 1,
    "prefill_step_warn_below": 0,
    "oom_retry_step_divisor": 1,
    "junction_min_gap_tokens": 0,
}


def _int_at_least(lo):
    def parse(value):
        try:
            n = int(value)
        except ValueError:
            n = None
        if n is None or n < lo:
            raise argparse.ArgumentTypeError(
                f"must be an integer >= {lo}, got {value!r}"
            )
        return n

    return parse


def _fraction(value):
    try:
        x = float(value)
    except ValueError:
        x = None
    if x is None or not 0 < x <= 1:
        raise argparse.ArgumentTypeError(
            f"must be a number in (0, 1], got {value!r}"
        )
    return x
from .multimodal import extract_images, load_image_inputs
from .multimodal import vision_spans as multimodal_vision_spans
from .sample_utils import greedy_sampler, make_logits_processors, make_sampler
from .utils import (
    _parse_size,
    load,
    maybe_set_recommended_wired_limit,
    sharded_load,
)


def _is_metal_oom(e):
    """A Metal command buffer that failed for lack of GPU memory. MLX raises
    it as RuntimeError: "[METAL] Command buffer execution failed:
    Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)"."""
    msg = str(e)
    return isinstance(e, RuntimeError) and (
        "kIOGPUCommandBufferCallbackErrorOutOfMemory" in msg
        or ("[METAL]" in msg and "Insufficient Memory" in msg)
    )


def checkpoint_room(live, active, limit, prefill_budget, headroom_fraction):
    """Bytes missing to copy a live cache of `live` bytes and still run the
    next prefill chunk (`prefill_budget`) within `headroom_fraction` of
    `limit` (Metal's recommended working set; the rest covers allocations
    not counted here, e.g. activations besides the attention scores), with
    `active` bytes in use. 0 or less: it fits."""
    return active + live + prefill_budget - int(headroom_fraction * limit)


def get_system_fingerprint():
    gpu_arch = mx.device_info()["architecture"]
    return f"{__version__}-{mx.__version__}-{platform.platform()}-{gpu_arch}"


class ToolCallFormatter:
    def __init__(self, tool_parser, tools, streaming=False):
        self._idx = 0
        self._tool_parser = tool_parser
        self._tools = tools
        self._streaming = streaming

    def _format(self, tc):
        tc_id = tc.pop("id", None) or str(uuid.uuid4())
        tc["arguments"] = json.dumps(tc["arguments"], ensure_ascii=False)
        out = {
            "function": tc,
            "type": "function",
            "id": tc_id,
        }
        if self._streaming:
            out["index"] = self._idx
            self._idx += 1
        return out

    def __call__(self, tool_calls):
        if not tool_calls:
            return []

        result = []
        for tool_text in tool_calls:
            try:
                parsed = self._tool_parser(tool_text, self._tools)
            except (ValueError, json.JSONDecodeError) as e:
                logging.warning(
                    f"Failed to parse tool call ({type(e).__name__}: {e}) — "
                    f"tool text was likely truncated mid-generation."
                )
                continue
            if not isinstance(parsed, list):
                parsed = [parsed]
            result.extend(self._format(tc) for tc in parsed)
        return result


def convert_chat(messages: List[dict], role_mapping: Optional[dict] = None):
    default_role_mapping = {
        "system_prompt": (
            "A chat between a curious user and an artificial intelligence "
            "assistant. The assistant follows the given rules no matter what."
        ),
        "system": "ASSISTANT's RULE: ",
        "user": "USER: ",
        "assistant": "ASSISTANT: ",
        "stop": "\n",
    }
    role_mapping = role_mapping or default_role_mapping

    prompt = ""
    for line in messages:
        role_prefix = role_mapping.get(line["role"], "")
        stop = role_mapping.get("stop", "")
        content = line.get("content", "")
        prompt += f"{role_prefix}{content}{stop}"

    prompt += role_mapping.get("assistant", "")
    return prompt.rstrip()


def process_message_content(messages):
    """
    Convert message content to a format suitable for `apply_chat_template`.

    The function operates on messages in place. It converts the 'content' field
    to a string instead of a list of text fragments.

    Args:
        message_list (list): A list of dictionaries, where each dictionary may
          have a 'content' key containing a list of dictionaries with 'type' and
          'text' keys.

    Raises:
        ValueError: If the 'content' type is not supported or if 'text' is missing.

    """
    for message in messages:
        content = message.get("content")
        if isinstance(content, list) and any(
            fragment.get("type") == "image" for fragment in content
        ):
            # Image request (image_url parts were already extracted and
            # rewritten to {"type": "image"} by multimodal.extract_images):
            # leave the list for the chat template, which renders one image
            # placeholder token per image part.
            pass
        elif isinstance(content, list):
            text_fragments = [
                fragment["text"] for fragment in content if fragment["type"] == "text"
            ]
            if len(text_fragments) != len(content):
                raise ValueError("Only 'text' content type is supported.")
            message["content"] = "".join(text_fragments)
        elif content is None:
            message["content"] = ""

        if tool_calls := message.get("tool_calls"):
            for tool_call in tool_calls:
                if func := tool_call.get("function"):
                    if args := func.get("arguments"):
                        func["arguments"] = json.loads(args)


@dataclass
class ModelDescription:
    model: str
    draft: str
    adapter: str


@dataclass
class SamplingArguments:
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    xtc_probability: float
    xtc_threshold: float


@dataclass
class LogitsProcessorArguments:
    logit_bias: Optional[Dict[int, float]]
    repetition_penalty: float
    repetition_context_size: int
    presence_penalty: float
    presence_context_size: int
    frequency_penalty: float
    frequency_context_size: int


@dataclass
class GenerationArguments:
    model: ModelDescription
    sampling: SamplingArguments
    logits: LogitsProcessorArguments

    stop_words: List[str]

    max_tokens: int
    num_draft_tokens: int
    logprobs: bool
    top_logprobs: int
    seed: Optional[int]
    chat_template_kwargs: Optional[Dict[str, Any]]
    # Reasoning tokens at most before the answer (see handle_completion);
    # None: as long as the model takes.
    thinking_budget: Optional[int] = None


@dataclass
class CompletionRequest:
    request_type: Literal["chat", "text"]

    prompt: str

    messages: List[Any]
    tools: Optional[List[Any]]
    role_mapping: Optional[Dict[str, Any]]

    # Raw bytes of any images attached to the chat messages, in order (see
    # mlx_lm/multimodal.py). None/empty for text-only requests.
    images: Optional[List[bytes]] = None
    # The prompt as tokens, used as is (no template): a request carried on
    # after its thinking budget ran out.
    prompt_tokens: Optional[List[int]] = None


# Put before the model's close of its reasoning when its thinking budget
# is spent: a thought cut off mid-sentence otherwise reads on into the answer.
THINKING_BUDGET_NOTE = "\n\nI have to answer now.\n"


def _now() -> float:
    """The clock of the per-request statistics (a test replaces it)."""
    return time.monotonic()


class _ResumedStopSequences:
    """Stop sequences whose matcher starts where a handed-off request's
    left off (BatchGenerator asks each sequence's for a matcher once)."""

    def __init__(self, stop_sequences, tokens):
        self._stop_sequences = stop_sequences
        self._tokens = list(tokens)

    def matcher(self):
        m = self._stop_sequences.matcher()
        for t in self._tokens:
            m.advance(t)
        return m


@dataclass
class Handoff:
    """A speculative single request stopped for the batch (another request
    arrived): what it needs to go on there. ``cache`` holds the model's
    layers for ``tokens`` -- the prompt and every token already sent."""

    rqueue: Any
    args: Any
    ctx: Any
    cache: List[Any]
    tokens: List[int]
    remaining: int


@dataclass
class GenerationContext:
    has_tool_calling: bool
    has_thinking: bool
    tool_parser: Callable[[str, Any], Dict]

    text_sm: TextStateMachine
    initial_state: str

    prompt: List[int]
    prompt_cache_count: int = -1
    # The model that serves this request (for the next turn's warm).
    model_key: Any = None

    _should_stop: bool = False

    def stop(self):
        self._should_stop = True


@dataclass
class Response:
    text: str
    token: int
    logprob: float
    finish_reason: Optional[str]
    top_tokens: Tuple[Dict[str, Any]]
    # Drafted (speculative decoding) and accepted, not sampled one by one.
    from_draft: bool = False
    # When the generation thread made it: the statistics' decode time, free
    # of the handler's writes to a slow client.
    made_at: float = field(default_factory=lambda: _now())


class TimeBudget:
    def __init__(self, budget=0.5, iterations=25, sync_frequency=10):
        self._is_distributed = mx.distributed.init().size() > 1
        self._budget = budget
        self._iterations = iterations
        self._sync_frequency = sync_frequency
        self._start = None
        self._current_iterations = None
        self._loops = 0
        self._time_spent = 0

    def __iter__(self):
        self._start = time.time()
        self._current_iterations = 0
        return self

    def __next__(self):
        if not self._is_distributed:
            if time.time() - self._start > self._budget:
                raise StopIteration()
            return None

        self._current_iterations += 1
        if self._current_iterations <= self._iterations:
            return None

        self._loops += 1
        self._time_spent += time.time() - self._start
        if self._loops % self._sync_frequency == 0:
            loop_time = mx.distributed.all_sum(self._time_spent).item()
            avg_loop_time = loop_time / (
                mx.distributed.init().size() * self._sync_frequency
            )
            factor = self._budget / avg_loop_time
            self._iterations = max(round(self._iterations * factor), 1)
            self._loops = 0
            self._time_spent = 0
        raise StopIteration()


class ModelProvider:
    def __init__(self, cli_args: argparse.Namespace):
        """Load models on demand and persist them across the whole process."""
        self.cli_args = cli_args
        self.model_key = None
        self.model = None
        self.tokenizer = None
        self.draft_model = None
        self.is_batchable = False
        self.supports_mtp = False
        self.head_mtp = False
        self.image_inputs = None

        group = mx.distributed.init()
        self.pipeline_group = group if group.size() > 1 and cli_args.pipeline else None
        self.tensor_group = (
            group if group.size() > 1 and not cli_args.pipeline else None
        )
        self.is_distributed = group.size() > 1

        # Maps model and adapter paths the actual paths to be used. Used to
        # map 'default_model' to the provided model by cli argument but could
        # be used for more in the future.
        self._model_map = {}
        self._adapter_map = {}
        self._draft_model_map = {}
        self._model_map["default_model"] = self.cli_args.model
        self._adapter_map["default_model"] = self.cli_args.adapter_path
        self._draft_model_map["default_model"] = self.cli_args.draft_model
        # getattr: a ModelProvider built from another parser's Namespace
        # (tests, embedders) has no model_alias.
        for _alias in (getattr(self.cli_args, "model_alias", None) or []):
            self._model_map[_alias] = self.cli_args.model
            self._adapter_map[_alias] = self.cli_args.adapter_path
            self._draft_model_map[_alias] = self.cli_args.draft_model

        # Build the tokenizer config for later use in load
        self._tokenizer_config = {"trust_remote_code": cli_args.trust_remote_code}
        if cli_args.chat_template:
            self._tokenizer_config["chat_template"] = cli_args.chat_template

    def reset(self) -> None:
        self.model_key = None
        self.model = None
        self.tokenizer = None
        self.draft_model = None
        self.is_batchable = False
        self.supports_mtp = False
        self.head_mtp = False
        self.image_inputs = None

    def _load(self, model_path, adapter_path=None, draft_model_path=None):
        if self.is_distributed and (
            adapter_path is not None or draft_model_path is not None
        ):
            raise ValueError(
                "Loading with adapters or draft models not supported in distributed mode"
            )

        # Remove the old model if it exists. Dropping the refs returns the
        # weights to MLX's buffer pool, not the OS; clear that pool so a
        # later load of a different model does not keep every previous one.
        self.reset()
        gc.collect()
        mx.clear_cache()

        # Load the model and tokenizer
        if self.is_distributed:
            model, tokenizer = sharded_load(
                model_path,
                pipeline_group=self.pipeline_group,
                tensor_group=self.tensor_group,
                tokenizer_config=self._tokenizer_config,
                trust_remote_code=self.cli_args.trust_remote_code,
            )
        else:
            model, tokenizer = load(
                model_path,
                adapter_path=adapter_path,
                tokenizer_config=self._tokenizer_config,
                model_config={
                    "mmap_lookup_tables": getattr(
                        self.cli_args, "mmap_lookup_tables", False
                    ),
                    "lazy_towers": getattr(self.cli_args, "lazy_towers", False),
                },
                trust_remote_code=self.cli_args.trust_remote_code,
            )

        # Use the default chat template if needed
        if self.cli_args.use_default_chat_template:
            if tokenizer.chat_template is None:
                tokenizer.chat_template = tokenizer.default_chat_template

        # Load the draft model for speculative decoding
        draft_model = None
        if draft_model_path is not None:
            draft_model, draft_tokenizer = load(draft_model_path)
            if draft_tokenizer.vocab_size != tokenizer.vocab_size:
                logging.warning(
                    "Draft model tokenizer does not match model tokenizer. "
                    "Speculative decoding may not work as expected."
                )

        # Compute batchability: of the model itself. A draft model doesn't
        # stop it -- a request runs with the drafter while it's alone and in
        # the batch without it otherwise (_is_batchable, Handoff).
        is_batchable = all(hasattr(c, "merge") for c in make_prompt_cache(model))
        # A Nemotron-H MTP head only ever gets used via _serve_single's
        # stream_generate call, which auto-dispatches to it for a single
        # unbatched sequence -- BatchGenerator (the batchable path) is
        # generic multi-sequence decoding with no knowledge of mtp_step at
        # all. Every cache type here (including Mamba's ArraysCache)
        # supports merge(), so without a check every request would go
        # through BatchGenerator and never reach MTP (confirmed live:
        # identical decode tok/s with and without the MTP fork). The check
        # is per request (_is_batchable): only one MTP would serve -- greedy,
        # no logits processors -- is kept out of batching; a sampled or
        # penalized request never uses MTP and batches as before.
        # A Qwen3.5 MTP head (mtp_generate_step) serves any sampler.
        head_mtp = _mtp_language_model(model) is not None
        supports_mtp = head_mtp or _model_supports_nemotron_h_mtp(model)

        # Update the member variables
        self.model_key = (model_path, adapter_path, draft_model_path)
        self.model = model
        self.tokenizer = tokenizer
        self.draft_model = draft_model
        self.is_batchable = is_batchable
        self.supports_mtp = supports_mtp
        self.head_mtp = head_mtp

        # Image preprocessing for vision-capable models (None otherwise).
        # Built at load time so a malformed processor config fails loudly
        # here, not on the first image request.
        self.image_inputs = load_image_inputs(model, model_path)

    def load_default(self):
        if self._model_map["default_model"] is not None:
            self.load("default_model", None, "default_model")

    def load(self, model_path, adapter_path=None, draft_model_path=None):
        adapter_path = self._adapter_map.get(model_path, adapter_path)
        model_path = self._model_map.get(model_path, model_path)
        draft_model_path = self._draft_model_map.get(draft_model_path, draft_model_path)

        model_key = (model_path, adapter_path, draft_model_path)
        if self.model_key != model_key:
            self._load(*model_key)

        return self.model, self.tokenizer


def _make_sampler(args, tokenizer):
    return make_sampler(
        args.sampling.temperature,
        top_p=args.sampling.top_p,
        top_k=args.sampling.top_k,
        min_p=args.sampling.min_p,
        xtc_probability=args.sampling.xtc_probability,
        xtc_threshold=args.sampling.xtc_threshold,
        xtc_special_tokens=tokenizer.encode("\n") + list(tokenizer.eos_token_ids),
    )


def _make_logits_processors(args):
    return make_logits_processors(
        args.logits.logit_bias,
        args.logits.repetition_penalty,
        args.logits.repetition_context_size,
        args.logits.presence_penalty,
        args.logits.presence_context_size,
        args.logits.frequency_penalty,
        args.logits.frequency_context_size,
    )


def _format_top_logprobs(logprobs, top_n, tokenizer) -> Tuple[Dict[str, Any]]:
    """Returns info dicts for the top `top_n` tokens from `logprobs`"""
    if top_n <= 0:
        return ()
    sorted_indices = mx.argpartition(-logprobs, kth=top_n - 1)
    top_indices = sorted_indices[:top_n].tolist()
    top_probs = logprobs[top_indices].tolist()
    txts = tokenizer.convert_ids_to_tokens(top_indices)
    return tuple(
        {"id": i, "token": s, "logprob": g}
        for i, s, g in zip(top_indices, txts, top_probs)
    )


class ResponseGenerator:
    def __init__(self, model_provider: ModelProvider, prompt_cache: LRUPromptCache):
        self.model_provider = model_provider
        self.prompt_cache = prompt_cache
        self.requests = Queue()
        # Next turns to prefill while idle (warm_next_turn): the latest few.
        self._warm_jobs = deque(maxlen=2)
        self._state_machine_cache = {}

        self._time_budget = TimeBudget()
        self._is_distributed = mx.distributed.init().size() > 1
        self._rank = mx.distributed.init().rank()
        self._stop = False
        self._generation_failed = False
        self._generation_thread = Thread(target=self._run_generate)
        self._generation_thread.start()

    def _run_generate(self):
        try:
            self._generate()
        except Exception as e:
            logging.exception(f"mlx_lm.server generation thread died: {e}")
            self._generation_failed = True

    def generation_available(self):
        return self._generation_thread.is_alive() and not self._generation_failed

    def stop_and_join(self):
        self._stop = True
        self._generation_thread.join()

    def join(self):
        self._generation_thread.join()

    @property
    def is_healthy(self):
        return self.generation_available()

    def _junction_split(self, segments, segment_types, n_cached, n_prompt):
        """Split the segments where the nearest cached prompt diverged from
        this one (a "user" checkpoint there), when the reused cache stops
        short of it. A later request that changes the same place (e.g. a
        timestamp) reuses the common part."""
        div = self.prompt_cache.last_divergence
        if div is None:
            return segments, segment_types
        # A junction checkpoint needs this many tokens to the reused prefix,
        # to the other checkpoints and to the prompt end.
        k, gap = div[0], self._opt("junction_min_gap_tokens")
        ends, end = [], 0
        for seg in segments:
            end += len(seg)
            ends.append(end)
        if k - n_cached < gap or n_prompt - 1 - k < gap:
            return segments, segment_types
        if any(abs(k - e) < gap for e in ends[:-1]):
            return segments, segment_types
        i = next(i for i, e in enumerate(ends) if e > k)
        cut = k - (ends[i] - len(segments[i]))
        logging.info(f"Prompt cache: checkpoint at the junction, token {k}")
        segments = (
            segments[:i] + [segments[i][:cut], segments[i][cut:]] + segments[i + 1 :]
        )
        segment_types = segment_types[:i] + ["user"] + segment_types[i:]
        return segments, segment_types

    def _trim_to_cap(self, batch_generator):
        """Keep the stored caches and the active batch's live caches within
        --prompt-cache-bytes."""
        total = getattr(self.cli_args, "prompt_cache_bytes", None)
        if total is not None:
            active = batch_generator.prompt_cache_nbytes
            self.prompt_cache.trim_to(n_bytes=total - active)

    def _log_cache_reuse(self, n_reused, n_prompt):
        logging.info(f"Prompt cache: reused {n_reused} of {n_prompt} tokens")
        # A client that changes an earlier part of the prompt shows here.
        div = self.prompt_cache.last_divergence
        if div is not None:
            logging.info(
                f"Prompt cache: diverged from a cached prompt at token {div[0]} "
                f"(of {div[1]})"
            )

    def _log_cache_stats(self):
        n_sequences = len(self.prompt_cache)
        n_bytes = self.prompt_cache.nbytes
        logging.info(f"Prompt Cache: {n_sequences} sequences, {n_bytes / 1e9:.2f} GB")
        for cache_type, stats in self.prompt_cache.stats_by_type().items():
            n_sequences = stats["n_sequences"]
            n_bytes = stats["n_bytes"]
            logging.info(
                f"- {cache_type}: {n_sequences} sequences, {n_bytes / 1e9:.2f} GB"
            )

    def _next_request(self, timeout=None):
        request = None
        if not self._is_distributed or self._rank == 0:
            try:
                if timeout is not None:
                    request = self.requests.get(timeout=timeout)
                else:
                    request = self.requests.get_nowait()
            except QueueEmpty:
                pass
        return self._share_request(request)

    def _share_object(self, obj):
        if not self._is_distributed:
            return obj

        if self._rank == 0:
            if obj is None:
                mx.eval(mx.distributed.all_sum(0))
                return None
            data = mx.array(pickle.dumps(obj))
            mx.eval(mx.distributed.all_sum(data.size))
            mx.eval(mx.distributed.all_sum(data))
            return obj
        else:
            size = mx.distributed.all_sum(0).item()
            if size == 0:
                return None
            data = mx.zeros(size, dtype=mx.uint8)
            data = mx.distributed.all_sum(data)
            return pickle.loads(data)

    def _share_request(self, request):
        if not self._is_distributed:
            return request

        shareable = request[1:] if request is not None else None
        shareable = self._share_object(shareable)
        if shareable is None:
            return None

        rq = request[0] if request is not None else Queue()
        return rq, *shareable

    def _tokenize(self, tokenizer, request, args):
        """Tokenize a request and split the prompt into segments.

        Returns a tuple

          * prompt - Full list of tokens
          * segments - A list of lists of tokens. Up to 3 segments that
            correspond to system prompt, context, thinking tail.
          * segment_types - A string per segment indicating if the segment is a
            system prompt or a user prompt or nothing special.
          * initial state - A string that contains the initial state of the
            state machine (normal or thinking depending on whether we have tail
            or not)
        """
        if request.prompt_tokens is not None:
            prompt = list(request.prompt_tokens)
            return prompt, [prompt], ["assistant"], "normal"
        if request.request_type == "chat":
            messages = request.messages
            tools = request.tools
            role_mapping = request.role_mapping

            if tokenizer.has_chat_template:
                process_message_content(messages)
                if tools and not tokenizer.has_tool_calling:
                    logging.warning(
                        "Received tools but model does not support tool calling. "
                        "If you think this is an error, file an issue here: "
                        "https://github.com/ml-explore/mlx-lm/issues"
                    )

                chat_template_args = self.model_provider.cli_args.chat_template_args
                if args.chat_template_kwargs:
                    chat_template_args = chat_template_args.copy()
                    chat_template_args.update(args.chat_template_kwargs)
                template_kwargs = dict(
                    tools=tools,
                    tokenize=True,
                    **chat_template_args,
                )
                prompt = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    **template_kwargs,
                )
            else:
                prompt = tokenizer.encode(convert_chat(messages, role_mapping))
                return prompt, [prompt], ["assistant"], "normal"
        else:
            prompt = tokenizer.encode(request.prompt)
            return prompt, [prompt], ["assistant"], "normal"

        # If we are here it means we have a chat request so we need to search
        # for segments for better cache management.

        # Choose the initial state among only reasoning or normal
        initial_state = "normal"
        if tokenizer.has_thinking:
            think_start = tokenizer.rfind_think_start(prompt)
            think_end = tokenizer.rfind_think_end(prompt)
            if think_start > think_end:
                initial_state = "reasoning"

        segments = []
        segment_types = []

        # Find where the system prompt ends and add it as a segment.
        num_system = 0
        sys_end = 0
        for m in messages:
            if m["role"] == "system":
                num_system += 1
            else:
                break
        if num_system > 0:
            sys_tokens = tokenizer.apply_chat_template(
                messages[:num_system] + [{"role": "user", "content": ""}],
                add_generation_prompt=False,
                **template_kwargs,
            )
            for i, (a, b) in enumerate(zip(sys_tokens, prompt)):
                if a != b:
                    sys_end = i
                    break
            if sys_end > 0 and sys_end < len(prompt):
                segments.append(prompt[:sys_end])
                segment_types.append("system")

        # Find a tail segment that contains thinking tokens (small up to 11
        # tokens)
        tail_start = len(prompt)
        if tokenizer.has_thinking:
            think_start = tokenizer.rfind_think_start(prompt, start=tail_start - 11)
            if think_start >= 0:
                tail_start = think_start

        # Finalize the segments and return
        if sys_end < tail_start:
            segments.append(prompt[sys_end:tail_start])
            segment_types.append("user")
        if tail_start < len(prompt):
            segments.append(prompt[tail_start:])
            segment_types.append("assistant")
        if not segments:
            segments = [prompt]
            segment_types = ["assistant"]

        return prompt, segments, segment_types, initial_state

    def _make_state_machine(self, model_key, tokenizer, stop_words):
        """Make (and cache) a StopSequences and TextStateMachine."""
        cache_key = (model_key, tuple(stop_words))
        rs = self._state_machine_cache.get(cache_key)
        if rs is not None:
            return rs

        stop_sequences = make_stop_sequences(tokenizer, stop_words)
        text_sm = make_text_state_machine(tokenizer, stop_words)

        if len(self._state_machine_cache) > 100:
            self._state_machine_cache.clear()
        self._state_machine_cache[cache_key] = (stop_sequences, text_sm)

        return stop_sequences, text_sm

    def _uses_mtp(self, args):
        """Whether stream_generate would serve this request with the
        model's MTP head (the same conditions it checks)."""
        if not getattr(self.model_provider, "supports_mtp", False) or getattr(self.cli_args, "max_kv_size", None) is not None:
            return False
        if getattr(self.model_provider, "head_mtp", False):
            return args.num_draft_tokens != 0 and not _make_logits_processors(args)
        sampler = _make_sampler(args, self.model_provider.tokenizer)
        return sampler is greedy_sampler and not _make_logits_processors(args)

    def _uses_speculation(self, args, request=None):
        """Whether the single path would serve this request speculatively
        (an MTP head or the draft model)."""
        if getattr(request, "images", None):
            return False
        return self._uses_mtp(args) or getattr(self.model_provider, "draft_model", None) is not None

    def _is_batchable(self, args, request=None, alone=True):
        """Whether the request runs in the batch. A speculative one (MTP,
        draft model) runs on its own while it's `alone` -- the fastest for
        one stream -- and in the batch without speculation when others run
        or wait."""
        # Image requests need their input embeddings built up front (see
        # _serve_single); BatchGenerator only takes token segments.
        has_images = bool(getattr(request, "images", None))
        return (
            self.model_provider.is_batchable
            and args.seed is None
            and not has_images
            and not (alone and self._uses_speculation(args, request))
        )

    def _others_waiting(self):
        # Distributed: only rank 0 holds the queue, and every rank must take
        # the same path -- no switching by load there.
        if self._is_distributed:
            return False
        return not self.requests.empty()

    def _main_slots(self, cache):
        """A stored cache as the batch takes it: the model's layers only (a
        speculative single request stores the draft model's beside them)."""
        if cache is None:
            return None
        n_main = len(make_prompt_cache(self.model_provider.model))
        return cache[:n_main] if len(cache) > n_main else cache

    def _insert_handoff(self, batch_generator, batch_results, handoff, tokenizer):
        """A handed-off request into the batch, without speculation. A cache
        that can be trimmed gives back its last token, fed again to start
        the batch's next step; one that can't (a recurrent layer) goes on
        from the nearest stored entry, the rest fed again."""
        model_key = self.model_provider.model_key
        tokens = handoff.tokens
        cache = handoff.cache
        if can_trim_prompt_cache(cache):
            trim_prompt_cache(cache, 1)
            history, rest = tokens[:-1], tokens[-1:]
        else:
            # The entry the request's own prefill stored (system / user
            # checkpoints), else from scratch.
            self.prompt_cache.insert_cache(model_key, tokens[:], cache, cache_type="assistant")
            cache, left = self.prompt_cache.fetch_nearest_cache(model_key, tokens[:-1])
            cache = self._main_slots(cache)
            n = len(tokens) - 1 - len(left) if cache is not None else 0
            history, rest = tokens[:n], tokens[n:]
        stop_sequences, _ = self._make_state_machine(model_key, tokenizer, handoff.args.stop_words)
        sent = tokens[len(handoff.ctx.prompt):]
        # Its stop sequences go on from what it sent: one split by the move
        # still stops it.
        stop_sequences = _ResumedStopSequences(stop_sequences, sent)
        # What it sent is the detokenizer's history (only that: the single
        # path's detokenizer saw no prompt): a word split at the move keeps
        # its spacing.
        detokenizer = tokenizer.detokenizer
        for t in sent[-8:]:
            detokenizer.add_token(t)
        _ = detokenizer.last_segment
        (uid,) = batch_generator.insert_segments(
            segments=[[rest]],
            max_tokens=[handoff.remaining],
            caches=[cache],
            all_tokens=[history],
            samplers=[_make_sampler(handoff.args, tokenizer)],
            logits_processors=[_make_logits_processors(handoff.args)],
            stop_sequences=[stop_sequences],
        )
        # For tests: how many moved, and the text the last one had sent.
        self.handoffs = getattr(self, "handoffs", 0) + 1
        self.last_handoff_sent = tokenizer.decode(tokens[len(handoff.ctx.prompt):])
        batch_results[uid] = {
            "ctx": handoff.ctx,
            "rqueue": handoff.rqueue,
            "detokenizer": detokenizer,
            "segment_types": [],
            "top_logprobs": handoff.args.top_logprobs,
            "handoff": True,
        }

    def _generate(self):
        # Local thread stream that we 'll pass to the BatchGenerator to make
        # sure that all generation runs in the same stream as the
        # synchronization messages.
        generation_stream = mx.default_stream(mx.default_device())

        # Load the default model if it is given
        self.model_provider.load_default()

        current_model = None
        current_tokenizer = None
        current_model_key = None
        batch_generator = None
        drain_batch = False
        batch_results = {}

        unprocessed_requests = []

        def get_next_request(timeout=None):
            if unprocessed_requests:
                return unprocessed_requests.pop()
            else:
                return self._next_request(timeout)

        if self._is_distributed:
            seed = mx.distributed.all_sum(mx.random.state[0]).view(mx.uint64).item()
            mx.random.seed(seed)

        while not self._stop:
            request = None
            if not drain_batch:
                timeout = (
                    None
                    if (batch_generator is not None and len(batch_results) > 0)
                    else 0.1
                )
                request = get_next_request(timeout=timeout)

            # Idle: prefill a finished request's next turn (warm_next_turn).
            if (
                request is None
                and self._warm_jobs
                and not drain_batch
                and (batch_generator is None or len(batch_results) == 0)
            ):
                self._warm_next(self._warm_jobs.popleft(), generation_stream)
                continue

            # We got a request
            if request is not None:
                rqueue, request, args = request

                # Can it be added to the current batch? A speculative one
                # alone (the batch empty, nobody waiting) doesn't join: the
                # batch drains and it runs on its own.
                alone = len(batch_results) == 0 and not unprocessed_requests and not self._others_waiting()
                if (
                    batch_generator is not None
                    and current_model == args.model
                    and self._is_batchable(args, request, alone=alone)
                ):
                    try:
                        prompt, segments, segment_types, initial_state = self._tokenize(
                            current_tokenizer, request, args
                        )
                    except Exception as e:
                        rqueue.put(e)
                        continue

                    stop_sequences, text_sm = self._make_state_machine(
                        self.model_provider.model_key,
                        current_tokenizer,
                        args.stop_words,
                    )

                    self._log_cache_stats()
                    cache, rest = self.prompt_cache.fetch_nearest_cache(
                        current_model_key, prompt
                    )
                    # Stored by a speculative single request: without the
                    # draft model's slots.
                    cache = self._main_slots(cache)
                    prompt_cache_count = len(prompt) - len(rest)
                    self._log_cache_reuse(prompt_cache_count, len(prompt))
                    segments, segment_types = self._junction_split(
                        segments, segment_types, prompt_cache_count, len(prompt)
                    )
                    N = prompt_cache_count
                    while N > 0:
                        if N >= len(segments[0]):
                            N -= len(segments.pop(0))
                            segment_types.pop(0)
                        else:
                            segments[0] = segments[0][N:]
                            break

                    ctx = GenerationContext(
                        has_tool_calling=current_tokenizer.has_tool_calling,
                        has_thinking=current_tokenizer.has_thinking,
                        tool_parser=current_tokenizer.tool_parser,
                        text_sm=text_sm,
                        initial_state=initial_state,
                        prompt=prompt,
                        prompt_cache_count=prompt_cache_count,
                        model_key=self.model_provider.model_key,
                    )
                    rqueue.put(ctx)

                    (uid,) = batch_generator.insert_segments(
                        segments=[segments],
                        max_tokens=[args.max_tokens],
                        caches=[cache],
                        all_tokens=[prompt[:prompt_cache_count]],
                        samplers=[_make_sampler(args, current_tokenizer)],
                        logits_processors=[_make_logits_processors(args)],
                        stop_sequences=[stop_sequences],
                    )
                    batch_results[uid] = {
                        "ctx": ctx,
                        "rqueue": rqueue,
                        "detokenizer": current_tokenizer.detokenizer,
                        "segment_types": segment_types[::-1],
                        "top_logprobs": args.top_logprobs,
                    }
                    # just making sure we don't leave a reference around
                    del cache

                    self._trim_to_cap(batch_generator)
                    continue

                # No batch generator. Load the model and if it's not
                # batchable serve sequential, o/w make a batch generaotr and
                # serve batched
                elif batch_generator is None:
                    try:
                        model, tokenizer = self.model_provider.load(
                            args.model.model, args.model.adapter, args.model.draft
                        )
                    except Exception as e:
                        rqueue.put(e)
                        continue

                    alone = not unprocessed_requests and not self._others_waiting()
                    handoff = None
                    if not self._is_batchable(args, request, alone=alone):
                        handoff = self._serve_single((rqueue, request, args), generation_stream)
                        if handoff is None:
                            continue

                    current_model = args.model
                    current_tokenizer = tokenizer
                    current_model_key = self.model_provider.model_key
                    batch_results = {}
                    batch_generator = BatchGenerator(
                        model,
                        completion_batch_size=self.cli_args.decode_concurrency,
                        prefill_batch_size=self.cli_args.prompt_concurrency,
                        prefill_step_size=self.cli_args.prefill_step_size,
                        stream=generation_stream,
                        # The batch quantizes its KV cache as the unbatched
                        # path does (from the first token).
                        kv_bits=self.cli_args.kv_bits,
                        kv_group_size=self.cli_args.kv_group_size,
                    )
                    if handoff is not None:
                        # The speculative request goes on in the batch; the
                        # one waiting joins it next. Its own cache isn't
                        # kept past this (the batch has what it needs).
                        self._insert_handoff(batch_generator, batch_results, handoff, tokenizer)
                        handoff = None
                    else:
                        unprocessed_requests.append((rqueue, request, args))
                    continue

                # We have a batch but this request cannot be added to the
                # batch so drain it to process the request.
                else:
                    drain_batch = True
                    unprocessed_requests.append((rqueue, request, args))
                    continue

            # No request so serve from the current batch
            elif batch_generator is not None:
                if len(batch_results) == 0:
                    if drain_batch:
                        current_model = None
                        current_tokenizer = None
                        current_model_key = None
                        batch_generator.close()
                        batch_generator = None
                        drain_batch = False
                    continue

                uids_to_remove = []
                for _ in self._time_budget:
                    prompt_responses, gen_responses = batch_generator.next()
                    if not prompt_responses and not gen_responses:
                        break

                    # Progress report for prompt processing
                    for r in prompt_responses:
                        result = batch_results[r.uid]
                        # A handed-off request is past its prompt: its
                        # catching up isn't prompt progress to its client.
                        if not result.get("handoff"):
                            result["rqueue"].put(r.progress)
                        if result["ctx"]._should_stop:
                            uids_to_remove.append(r.uid)

                    # Save the caches at end of segments
                    eos_ids = [
                        r.uid
                        for r in prompt_responses
                        if r.end_of_segment
                        and not r.end_of_prompt
                        and batch_results[r.uid]["segment_types"]
                    ]
                    caches = batch_generator.extract_cache(eos_ids)
                    for uid, (cache, cache_key) in caches.items():
                        self.prompt_cache.insert_cache(
                            self.model_provider.model_key,
                            cache_key[:],
                            cache,
                            cache_type=batch_results[uid]["segment_types"].pop(),
                        )
                    if caches:
                        self._trim_to_cap(batch_generator)
                    del caches

                    for r in gen_responses:
                        result = batch_results[r.uid]

                        # Don't decode the final stop token
                        if r.finish_reason == "stop":
                            result["detokenizer"].finalize()
                            text = result["detokenizer"].last_segment
                        elif r.finish_reason == "length":
                            result["detokenizer"].add_token(r.token)
                            result["detokenizer"].finalize()
                            text = result["detokenizer"].last_segment
                        else:
                            result["detokenizer"].add_token(r.token)
                            text = result["detokenizer"].last_segment

                        result["rqueue"].put(
                            Response(
                                text,
                                r.token,
                                r.logprobs[r.token].item(),
                                r.finish_reason,
                                _format_top_logprobs(
                                    r.logprobs,
                                    result["top_logprobs"],
                                    current_tokenizer,
                                ),
                            )
                        )

                        if r.finish_reason is not None:
                            result["rqueue"].put(None)
                            self.prompt_cache.insert_cache(
                                current_model_key,
                                r.all_tokens[:],
                                r.prompt_cache,
                                cache_type="assistant",
                            )
                            self._trim_to_cap(batch_generator)
                            del batch_results[r.uid]

                        if result["ctx"]._should_stop:
                            uids_to_remove.append(r.uid)

                uids_to_remove = self._share_object(uids_to_remove)
                if uids_to_remove:
                    batch_generator.remove(uids_to_remove)
                    for uid in uids_to_remove:
                        # It may have already been removed during
                        # generation
                        batch_results.pop(uid, None)

        # Make sure the model and prompt cache are destroyed in the generation
        # thread under same stream.
        self.model_provider.reset()
        del self.prompt_cache
        gc.collect()

    def _serve_single(self, request, stream):
        rqueue, request, args = request

        # Define the progress callback
        # Progress only moves forward: a retry after an OOM starts again
        # below what the client has already seen.
        sent = [-1]

        def progress(tokens_processed, tokens_total):
            if tokens_processed > sent[0] or tokens_processed == tokens_total:
                sent[0] = max(sent[0], tokens_processed)
                rqueue.put((tokens_processed, tokens_total))

        try:
            # Load the model and tokenizer
            model = self.model_provider.model
            tokenizer = self.model_provider.tokenizer
            draft_model = self.model_provider.draft_model

            image_inputs = None
            if request.images:
                image_inputs = self.model_provider.image_inputs
                if image_inputs is None:
                    raise ValueError(
                        "This model does not accept image inputs (no vision "
                        "tower / image processor). Use a vision-capable model "
                        "or remove the image."
                    )

            # Prepare the prompt and state machine
            prompt, segments, segment_types, initial_state = self._tokenize(
                tokenizer, request, args
            )

            # Image requests: expand each image placeholder into its real
            # soft-token span and fuse the image features into the prompt
            # embeddings once, up front. `cache_prompt` keys the prompt cache
            # (image positions hashed by image content -- see
            # multimodal.ImageInputs.build).
            input_embeddings = None
            cache_prompt = prompt
            if image_inputs is not None:
                prompt, input_embeddings, cache_prompt = image_inputs.build(
                    prompt, request.images
                )
            stop_sequences, text_sm = self._make_state_machine(
                self.model_provider.model_key,
                tokenizer,
                args.stop_words,
            )

            # Start the generation context
            ctx = GenerationContext(
                has_thinking=tokenizer.has_thinking,
                has_tool_calling=tokenizer.has_tool_calling,
                tool_parser=tokenizer.tool_parser,
                text_sm=text_sm,
                initial_state=initial_state,
                prompt=prompt,
                model_key=self.model_provider.model_key,
            )
            rqueue.put(ctx)

            # Seed if requested
            if args.seed is not None:
                mx.random.seed(args.seed)

            # Make the sampler and logit processor
            sampler = _make_sampler(args, tokenizer)
            logits_processors = _make_logits_processors(args)

            use_draft = (
                self.model_provider.draft_model is not None
                and input_embeddings is None
            )
            n_main = len(make_prompt_cache(model))
            prompt_embeddings = input_embeddings
            # A Metal OOM in the checkpoint prefill frees stored caches and
            # starts once more from a fresh cache fetch, with half the prefill
            # step (the failed cache is lost). Not covered: the prefill inside
            # stream_generate (caches that stay trimmable, images, the tail
            # after the last checkpoint).
            for attempt in range(2):
                input_embeddings = prompt_embeddings
                seg, seg_types = segments, segment_types
                # Load the KV cache
                self._log_cache_stats()
                cache, rest = self.prompt_cache.fetch_nearest_cache(
                    self.model_provider.model_key, cache_prompt
                )
                reused = self.prompt_cache.last_fetched
                # Image requests run without the draft model (it can't consume
                # the fused image embeddings), so their caches have no draft
                # slots: match the slots to this request.
                cache, rest = self._match_draft_slots(
                    cache, rest, cache_prompt, use_draft
                )
                if cache is None:
                    reused = None
                ctx.prompt_cache_count = len(cache_prompt) - len(rest)
                self._log_cache_reuse(ctx.prompt_cache_count, len(cache_prompt))
                cache_key = cache_prompt[:]
                if input_embeddings is not None:
                    # `rest` is a slice of the cache key (pseudo ids at image
                    # positions); feed the real ids + embeddings for that tail.
                    n_cached = ctx.prompt_cache_count
                    rest = prompt[n_cached:]
                    input_embeddings = input_embeddings[n_cached:]
                if cache is None:
                    cache = make_prompt_cache(self.model_provider.model)
                    if use_draft:
                        cache += make_prompt_cache(self.model_provider.draft_model)

                self._log_prefill_step(
                    model, len(cache_prompt), ctx.prompt_cache_count
                )

                # Prefill up to the end of the system and user segments and
                # save a checkpoint at each, as the batched path does. A
                # non-trimmable cache (e.g. a sliding window) can't be cut back
                # from the whole sequence when the next turn diverges in the
                # last answer.
                n_prefilled = 0
                if input_embeddings is None:
                    seg, seg_types = self._junction_split(
                        seg, seg_types, ctx.prompt_cache_count, len(prompt)
                    )
                    try:
                        n_prefilled = self._prefill_checkpoints(
                            model,
                            draft_model if use_draft else None,
                            cache,
                            n_main,
                            prompt,
                            ctx.prompt_cache_count,
                            seg,
                            seg_types,
                            stream,
                            progress,
                            ctx,
                            step_divisor=self._opt("oom_retry_step_divisor")
                            ** attempt,
                        )
                    except RuntimeError as e:
                        if attempt or self._is_distributed or not _is_metal_oom(e):
                            raise
                        oom = str(e)
                    else:
                        if n_prefilled is None:
                            rqueue.put(None)
                            return
                        rest = rest[n_prefilled:]
                        break
                    # Out of the except block: the exception and its traceback
                    # (frames holding the failed cache) are gone, so its
                    # buffers can be freed.
                    cache = None
                    self._free_memory_after_oom(oom, reused)
                else:
                    break

            def generate_progress(processed, total):
                progress(n_prefilled + processed, n_prefilled + total)

            # Process the prompt and generate tokens
            # Own matcher: the automaton is cached across requests.
            stop_matcher = stop_sequences.matcher()
            # An image's tokens attend to each other in both directions on
            # models that want it (Gemma 4 26B/12B): the model is told where
            # they are for this prefill, and nothing else sees it.
            set_vision_spans = getattr(model, "set_vision_spans", None)
            if input_embeddings is not None and set_vision_spans is not None:
                set_vision_spans(multimodal_vision_spans(cache_prompt))
            # Qwen3.5: mRoPE positions for this image request only.
            set_media_positions = getattr(model, "set_media_positions", None)
            if input_embeddings is not None and set_media_positions is not None:
                set_media_positions(getattr(image_inputs, "media_positions", None))
            # Moves into the batch when another request arrives (Handoff):
            # only a speculative text request, which is why it's here alone.
            handoff_ok = (
                self._opt("batch_handoff")
                and self.model_provider.is_batchable
                and not self._is_distributed
                and input_embeddings is None
                and args.seed is None
                and (use_draft or self._uses_mtp(args))
            )
            handoff = None
            n_out = 0
            try:
                steps = stream_generate(
                    model=model,
                    stream=stream,
                    tokenizer=tokenizer,
                    prompt=rest,
                    max_tokens=args.max_tokens,
                    sampler=sampler,
                    logits_processors=logits_processors,
                    prompt_cache=cache,
                    draft_model=draft_model if use_draft else None,
                    num_draft_tokens=args.num_draft_tokens,
                    input_embeddings=input_embeddings,
                    prompt_progress_callback=generate_progress,
                    token_history=prompt[: ctx.prompt_cache_count + n_prefilled],
                    prefill_step_size=self.cli_args.prefill_step_size,
                    prefill_memory_budget=self._prefill_memory_budget(),
                    kv_bits=self.cli_args.kv_bits,
                    kv_group_size=self.cli_args.kv_group_size,
                    quantized_kv_start=self.cli_args.quantized_kv_start,
                )
                for gen in steps:
                    finish_reason = gen.finish_reason

                    # Token-level stop word detection
                    if stop_matcher.advance(gen.token):
                        finish_reason = "stop"

                    rqueue.put(
                        Response(
                            gen.text,
                            gen.token,
                            gen.logprobs[gen.token].item(),
                            finish_reason,
                            _format_top_logprobs(
                                gen.logprobs, args.top_logprobs, tokenizer
                            ),
                            from_draft=gen.from_draft,
                        )
                    )
                    cache_key.append(gen.token)

                    if ctx._should_stop:
                        if self._is_distributed:
                            raise NotImplementedError()
                        break

                    if finish_reason is not None:
                        break

                    n_out += 1
                    if handoff_ok and n_out < args.max_tokens and self._others_waiting():
                        handoff = Handoff(rqueue, args, ctx, None, cache_key, args.max_tokens - n_out)
                        break
            finally:
                # Closed here, not when collected: the cache then holds the
                # prompt and every token sent (what Handoff and the stored
                # entry rely on).
                if "steps" in locals():
                    steps.close()
                if set_vision_spans is not None:
                    set_vision_spans(None)
                if set_media_positions is not None:
                    set_media_positions(None)

            if handoff is not None:
                handoff.cache = cache[:n_main]
                return handoff

            rqueue.put(None)

            # Save the KV cache again
            self.prompt_cache.insert_cache(
                self.model_provider.model_key, cache_key, cache
            )
            cap = getattr(self.cli_args, "prompt_cache_bytes", None)
            if cap is not None:
                self.prompt_cache.trim_to(n_bytes=cap)

        except Exception as e:
            rqueue.put(e)

    def warm_next_turn(self, model_key, request, args, content, tool_calls, reasoning="", echoes_reasoning=False):
        """After a chat request: the conversation with this answer, as the
        client will send it back, gets prefilled while the server is idle
        and stored as a cache entry. A model whose cache can't be trimmed
        back (an SSM / delta-rule layer) otherwise reuses only up to where
        the answer starts: the client's copy of it (its reasoning only if
        the client sends that back, the tool calls re-rendered) differs
        from what was generated. Whether it does is a guess (an Anthropic
        client, or reasoning in its earlier answers); a wrong one leaves
        the entry unused, as does a next request with other template
        options or tools."""
        if not getattr(self.cli_args, "warm_next_turn", True) or self._is_distributed:
            return
        if request.request_type != "chat" or getattr(request, "images", None):
            return
        message = self._answer_as_sent_back(request, content, tool_calls, reasoning, echoes_reasoning)
        self._warm_jobs.append((model_key, request, args, message))

    @staticmethod
    def _answer_as_sent_back(request, content, tool_calls, reasoning="", echoes_reasoning=False):
        """The answer as the client's next request carries it."""
        message = {"role": "assistant", "content": content}
        # A client that sends the reasoning back (Anthropic clients send
        # their thinking blocks; others show it in earlier turns) gets it
        # rendered by a thinking template: the next turn has it too.
        echoes_reasoning = echoes_reasoning or any(
            m.get("role") == "assistant" and m.get("reasoning_content") for m in request.messages
        )
        if reasoning and echoes_reasoning:
            message["reasoning_content"] = reasoning
        if tool_calls:
            message["tool_calls"] = tool_calls
        return message

    def _next_turn_tokens(self, tokenizer, request, args, message):
        """The tokens the client's next request starts with: the common
        prefix of renders of the conversation + this answer + a next
        message (tool results after tool calls, else a user message) with
        different contents -- whatever the template does to earlier turns."""
        chat_template_args = self.model_provider.cli_args.chat_template_args
        if args.chat_template_kwargs:
            chat_template_args = {**chat_template_args, **args.chat_template_kwargs}

        def render(filler, role):
            messages = json.loads(json.dumps(request.messages)) + [json.loads(json.dumps(message))]
            for m in messages:
                content = m.get("content")
                if isinstance(content, list):
                    m["content"] = "".join(f.get("text", "") for f in content if f.get("type") == "text")
                elif content is None:
                    m["content"] = ""
                for tc in m.get("tool_calls") or []:
                    func = tc.get("function") or {}
                    # As the request's own are (process_message_content):
                    # an empty string stays one.
                    if isinstance(func.get("arguments"), str) and func["arguments"]:
                        func["arguments"] = json.loads(func["arguments"])
            if role == "tool":
                messages += [
                    {"role": "tool", "tool_call_id": tc.get("id", ""), "content": filler}
                    for tc in message["tool_calls"]
                ]
            else:
                messages.append({"role": "user", "content": filler})
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tools=request.tools,
                tokenize=True,
                **chat_template_args,
            )

        # Next messages that start with different bytes, so no token of
        # theirs is shared ("\u2581" and "\u2582" share a byte-level token:
        # the warmed entry ran one token into the message, and the next
        # request, which can't trim it back, never reused it): a letter, a
        # digit, CJK, whitespace and punctuation (both can merge with the
        # template's text before the message), JSON, and nothing (a template
        # can skip a wrapper for empty content).
        fillers = ["a1", "7", "\u4e2d", " x", "\nx", "?", '{"a": 0}', "[1]", '"q"', ""]
        # After tool calls: their results, a structured one too (a template
        # can render it unquoted). Not a user message instead: Qwen3.5 keeps
        # the reasoning of the turns since the last user message, so the
        # call renders differently before one -- the entry would end before
        # the answer for the common case, the results.
        if message.get("tool_calls"):
            probes = [(f, "tool") for f in fillers] + [({"a": 0}, "tool")]
        else:
            probes = [(f, "user") for f in fillers]
        renders, error = [], None
        for filler, role in probes:
            try:
                renders.append(render(filler, role))
            except Exception as e:
                # A template that rejects one (a structured result): the
                # others still apply.
                error = e
        if not renders:
            raise error
        n = 0
        while all(n < len(r) for r in renders) and all(r[n] == renders[0][n] for r in renders):
            n += 1
        # Text before the message can still merge with a message no probe
        # starts like: end the entry at a special token (never merged) when
        # one is near, else a token earlier.
        special = set(getattr(tokenizer, "all_special_ids", None) or [])
        special |= set((getattr(tokenizer, "added_tokens_encoder", None) or {}).values())
        cut = next((i + 1 for i in range(n - 1, max(n - 16, 0) - 1, -1) if renders[0][i] in special), None)
        return list(renders[0][: cut if cut is not None else max(n - 1, 0)])

    def _warm_next(self, job, stream):
        # Best effort: a failure (e.g. out of memory) loses this entry, never
        # the generation thread.
        try:
            self._warm_next_job(job, stream)
        except Exception as e:
            logging.warning(f"Prompt cache: next turn not warmed: {e!r}")
        finally:
            mx.clear_cache()

    def _warm_next_job(self, job, stream):
        key, request, args, message = job
        model = self.model_provider.model
        if model is None or self.model_provider.draft_model is not None:
            return
        if key != self.model_provider.model_key:
            return   # another model is loaded now
        try:
            if stays_trimmable(make_prompt_cache(model)):
                return   # a trimmable cache reuses a longer entry anyway
            tokens = self._next_turn_tokens(self.model_provider.tokenizer, request, args, message)
        except Exception as e:
            logging.debug(f"Prompt cache: next turn not warmed: {e!r}")
            return
        cache, rest = self.prompt_cache.fetch_nearest_cache(key, tokens)
        reused = len(tokens) - len(rest)
        limit = getattr(self.cli_args, "warm_next_turn_max_tokens", 16384)
        if cache is None or reused == 0 or not rest or len(rest) > limit:
            return
        # Only into free memory: stored entries aren't evicted for a guess.
        nbytes = sum(c.nbytes for c in cache if c is not None)
        short = self._memory_shortfall(nbytes, self._prefill_memory_budget())
        if short is not None and short > 0:
            logging.info("Prompt cache: next turn not warmed (no free memory)")
            return
        prefill_step = make_prefill_step(
            model, self.cli_args.prefill_step_size, self._prefill_memory_budget()
        )
        with wired_limit(model, [stream]), mx.stream(stream):
            done = 0
            while done < len(rest):
                if not self.requests.empty():
                    logging.info("Prompt cache: next turn not warmed (a request came in)")
                    return
                # The cache grows each chunk: still only into free memory.
                nbytes = sum(c.nbytes for c in cache if c is not None)
                short = self._memory_shortfall(nbytes, self._prefill_memory_budget())
                if short is not None and short > 0:
                    logging.info("Prompt cache: next turn not warmed (no free memory)")
                    return
                step = prefill_step(reused + done, kv_cache_quantized(cache))
                chunk = rest[done : done + step]
                model(mx.array(chunk)[None], cache=cache)
                maybe_quantize_kv_cache(
                    cache,
                    quantized_kv_start=self.cli_args.quantized_kv_start,
                    kv_group_size=self.cli_args.kv_group_size,
                    kv_bits=self.cli_args.kv_bits,
                )
                mx.eval([c.state for c in cache if c is not None])
                done += len(chunk)
                mx.clear_cache()
        self.prompt_cache.insert_cache(key, tokens, cache, cache_type="user")
        cap = getattr(self.cli_args, "prompt_cache_bytes", None)
        if cap is not None:
            self.prompt_cache.trim_to(n_bytes=cap)
        logging.info(f"Prompt cache: warmed the next turn: {len(rest)} tokens on top of {reused}")

    def _memory_shortfall(self, nbytes, budget):
        """checkpoint_room with the current memory in use, or None if the
        device reports no working-set limit."""
        limit = mx.device_info().get("max_recommended_working_set_size")
        if not limit:
            return None
        return checkpoint_room(
            nbytes,
            mx.get_active_memory(),
            limit,
            budget.nbytes if budget else 0,
            self._opt("memory_headroom_fraction"),
        )

    def _free_stored(self, short, keep=None):
        self.prompt_cache.trim_to(
            n_bytes=max(0, self.prompt_cache.nbytes - short), keep=keep
        )
        gc.collect()
        mx.synchronize()
        mx.clear_cache()

    def _make_memory_room(self, nbytes, divisor=1):
        """Evict stored caches until a copy of `nbytes` and a prefill chunk
        (the budget divided by `divisor`, as the current attempt runs) fit
        in GPU memory. False if they can't."""
        budget = self._prefill_memory_budget(divisor)
        short = self._memory_shortfall(nbytes, budget)
        if short is None or short <= 0:
            return True
        if self.prompt_cache.nbytes > 0:
            self._free_stored(short)
            short = self._memory_shortfall(nbytes, budget)
        return short <= 0

    def _free_memory_after_oom(self, msg, reused):
        """Evict stored caches (not `reused`, the entry this request starts
        from) for the shortfall of the retry: the reused cache again plus
        the prefill budget the retry runs with."""
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
        n_bytes = self.prompt_cache.nbytes
        live = self.prompt_cache.entry_nbytes(reused)
        budget = self._prefill_memory_budget(self._opt("oom_retry_step_divisor"))
        short = self._memory_shortfall(live, budget)
        if short is not None and short > 0:
            self._free_stored(short, keep=reused)
        logging.warning(
            f"Prefill ran out of memory ({msg}); prompt cache trimmed from "
            f"{n_bytes / 1e9:.2f} to {self.prompt_cache.nbytes / 1e9:.2f} GB, "
            "retrying once"
        )

    def _opt(self, name):
        """A server option; out-of-range values (cli_args not from
        make_parser) are clamped to the valid range."""
        value = getattr(self.cli_args, name, SERVER_DEFAULTS[name])
        if name in _OPT_MIN:
            return max(_OPT_MIN[name], int(value))
        if name == "memory_headroom_fraction":
            return min(1.0, value) if value > 0 else SERVER_DEFAULTS[name]
        return value

    def _prefill_memory_budget(self, divisor=1):
        """The PrefillBudget from the server options (None if disabled);
        `divisor` scales it down for an OOM retry."""
        divisor = max(1, int(divisor))
        mb = self._opt("prefill_memory_mb")
        if not mb:
            return None
        return PrefillBudget(
            nbytes=(mb << 20) // divisor,
            min_step=self._opt("min_prefill_step"),
            score_bytes=self._opt("prefill_score_bytes"),
        )

    def _log_prefill_step(self, model, n_prompt, n_cached):
        budget = self._prefill_memory_budget()
        max_step = self.cli_args.prefill_step_size
        if budget is None or n_prompt - n_cached <= 1:
            return
        step = make_prefill_step(model, max_step, budget)
        kv_bits = self.cli_args.kv_bits
        start = self.cli_args.quantized_kv_start
        first = step(n_cached, kv_bits is not None and n_cached >= start)
        last = step(n_prompt, kv_bits is not None and n_prompt >= start)
        if last < self._opt("prefill_step_warn_below"):
            logging.warning(
                f"Prefill step down to {last} at {n_prompt} tokens: prefill "
                "is slow. A larger --prefill-memory-mb gives larger steps."
            )
        elif last < max_step:
            logging.info(
                f"Prefill step {first} at {n_cached} tokens, {last} at "
                f"{n_prompt} tokens (--prefill-memory-mb "
                f"{self._opt('prefill_memory_mb')})"
            )

    def _prefill_checkpoints(
        self,
        model,
        draft_model,
        cache,
        n_main,
        prompt,
        n_cached,
        segments,
        segment_types,
        stream,
        progress,
        ctx,
        step_divisor=1,
    ):
        """Prefill `prompt[n_cached:]` up to the end of each system / user
        segment and insert a copy of the cache there. The last prompt token
        is left for generation. Return the number of tokens prefilled, or
        None if the request was stopped."""
        # Such a cache is trimmed back from the full sequence on the next
        # turn, and insert_cache drops its prefixes anyway.
        if stays_trimmable(cache):
            return 0
        ends = []
        end = 0
        for seg, seg_type in zip(segments, segment_types):
            end += len(seg)
            if seg_type in ("system", "user"):
                ends.append((min(end, len(prompt) - 1), seg_type))
        ends = [(e, t) for e, t in ends if e > n_cached]
        if not ends:
            return 0

        cli_args = self.cli_args
        quantize = lambda: maybe_quantize_kv_cache(
            cache, cli_args.quantized_kv_start, cli_args.kv_group_size, cli_args.kv_bits
        )
        # A retry after an OOM runs smaller chunks (step_divisor > 1).
        max_step = cli_args.prefill_step_size
        min_step = min(self._opt("min_prefill_step"), max_step)
        prefill_step = make_prefill_step(
            model,
            max(min_step, max_step // step_divisor),
            self._prefill_memory_budget(step_divisor),
        )
        total = len(prompt) - n_cached
        done = n_cached
        skipped = False
        with wired_limit(model, [stream]), mx.stream(stream):
            progress(0, total)
            for end, seg_type in ends:
                while done < end:
                    if ctx._should_stop and not self._is_distributed:
                        return None
                    step = prefill_step(done, kv_cache_quantized(cache))
                    n = min(step, end - done)
                    tokens = mx.array(prompt[done : done + n])[None]
                    model(tokens, cache=cache[:n_main])
                    if draft_model is not None and len(cache) > n_main:
                        draft_model(tokens, cache=cache[n_main:])
                    # Same state generate_step would have at this offset.
                    quantize()
                    mx.eval([c.state for c in cache if c is not None])
                    done += n
                    progress(done - n_cached, total)
                    mx.clear_cache()
                # The live cache and the checkpoint count against
                # --prompt-cache-bytes: make room before it's stored. It
                # copies only what the entries above it don't hold (the
                # positions past the last checkpoint, the sliding windows).
                key = self.model_provider.model_key
                nbytes = sum(c.nbytes for c in cache if c is not None)
                added = lambda: self.prompt_cache.insert_nbytes(key, prompt[:end], cache)
                cap = getattr(cli_args, "prompt_cache_bytes", None)
                if cap is not None:
                    if nbytes + added() > cap:
                        continue
                    self.prompt_cache.trim_to(n_bytes=cap - nbytes - added())
                    # An evicted checkpoint above it leaves more to copy.
                    if self.prompt_cache.nbytes + nbytes + added() > cap:
                        continue
                # The copy and the next chunk must fit in GPU memory too:
                # older entries go first, else no checkpoint.
                if not self._make_memory_room(added(), step_divisor):
                    if not skipped:
                        logging.warning(
                            "Prompt cache: checkpoint skipped: no memory room"
                        )
                    skipped = True
                    continue
                # Copied and evaluated now, so the next room check sees its
                # memory.
                self.prompt_cache.insert_cache(
                    key, prompt[:end], cache, cache_type=seg_type, copy=True
                )
        return done - n_cached

    def _match_draft_slots(self, cache, rest, cache_prompt, use_draft):
        """A cached prompt's slots for this request: a cache without draft
        slots can't serve a draft request (recomputed), and one with them
        loses them for a request without the draft (it wouldn't advance
        them)."""
        draft = self.model_provider.draft_model
        # A drafter without a cache of its own (the Gemma 4 assistant) has no
        # slots to match.
        if cache is None or draft is None or not make_prompt_cache(draft):
            return cache, rest
        n_main = len(make_prompt_cache(self.model_provider.model))
        if use_draft and len(cache) == n_main:
            return None, cache_prompt
        if not use_draft and len(cache) > n_main:
            return cache[:n_main], rest
        return cache, rest

    def _await_response(self, response_queue):
        # Wherever the request was when the thread died, nothing more will be
        # put on its queue, so give up rather than block forever.
        while True:
            try:
                return response_queue.get(timeout=1.0)
            except QueueEmpty:
                if not self.generation_available():
                    raise RuntimeError("generation thread died") from None

    def generate(
        self,
        request: CompletionRequest,
        generation_args: GenerationArguments,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ):
        response_queue = Queue()
        if not self.generation_available():
            raise RuntimeError("generation thread died")
        self.requests.put((response_queue, request, generation_args))

        def _inner():
            while True:
                response = self._await_response(response_queue)
                if response is None:
                    break
                if isinstance(response, Exception):
                    raise response
                if isinstance(response, tuple):
                    if progress_callback is not None:
                        progress_callback(*response)
                    continue
                yield response

        ctx = self._await_response(response_queue)
        if isinstance(ctx, Exception):
            raise ctx

        return ctx, _inner()

    @property
    def cli_args(self):
        return self.model_provider.cli_args


class APIHandler(BaseHTTPRequestHandler):
    anthropic: Optional[anthropic_api.MessageStream] = None

    def __init__(
        self,
        response_generator: ResponseGenerator,
        *args,
        system_fingerprint: Optional[str] = None,
        **kwargs,
    ):
        """
        Create static request specific metadata
        """
        self.created = int(time.time())
        self.response_generator = response_generator
        self.system_fingerprint = system_fingerprint or get_system_fingerprint()
        super().__init__(*args, **kwargs)

    def _set_cors_headers(self):
        allowed_origins = self.response_generator.cli_args.allowed_origins
        origin = self.headers.get("Origin")
        if "*" in allowed_origins:
            self.send_header("Access-Control-Allow-Origin", "*")
        elif origin in allowed_origins:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "*")
        self.send_header("Access-Control-Allow-Headers", "*")

    def _set_completion_headers(self, status_code: int = 200):
        self.send_response(status_code)
        self.send_header("Content-type", "application/json")
        self._set_cors_headers()

    def _set_stream_headers(self, status_code: int = 200):
        self.send_response(status_code)
        self.send_header("Content-type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self._set_cors_headers()

    def do_OPTIONS(self):
        self._set_completion_headers(204)
        self.end_headers()

    def do_POST(self):
        """
        Respond to a POST request from a client.
        """
        request_factories = {
            "/v1/completions": self.handle_text_completions,
            "/v1/chat/completions": self.handle_chat_completions,
            "/chat/completions": self.handle_chat_completions,
            "/v1/messages": self.handle_chat_completions,
        }

        # Without the query: Anthropic's beta clients (Claude Code, the SDK's
        # client.beta.messages) post to /v1/messages?beta=true.
        self.route = self.path.split("?", 1)[0]
        if self.route not in request_factories:
            self._set_completion_headers(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
            return

        # Fetch and parse request body
        content_length = self.headers.get("Content-Length")
        if content_length is None:
            self._set_completion_headers(411)
            self.end_headers()
            self.wfile.write(
                self._error(411, "Content-Length header is required")
            )
            return
        try:
            content_length = int(content_length)
        except ValueError:
            self._set_completion_headers(400)
            self.end_headers()
            self.wfile.write(
                self._error(400, "Invalid Content-Length header")
            )
            return
        raw_body = self.rfile.read(content_length)
        try:
            self.body = json.loads(raw_body.decode())
        except json.JSONDecodeError as e:
            logging.error(f"JSONDecodeError: {e} - Raw body: {raw_body.decode()}")
            self._set_completion_headers(400)
            self.end_headers()
            self.wfile.write(
                self._error(400, f"Invalid JSON in request body: {e}")
            )
            return

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            debug_body = json.dumps(self.body, indent="\t")
            logging.debug(f"Incoming Request Body: {debug_body}")
        if not isinstance(self.body, dict):
            debug_body = json.dumps(self.body, indent="\t")
            logging.error(f"Invalid Request Body: {debug_body}")
            self._set_completion_headers(400)
            self.end_headers()
            self.wfile.write(
                self._error(400, "Request should be a JSON dictionary")
            )
            return

        # Anthropic Messages API: run as a chat request, answer in its format.
        self.anthropic = None
        if self.route == "/v1/messages":
            try:
                self.body = anthropic_api.to_chat_request(self.body)
            except (ValueError, KeyError, TypeError, AttributeError) as e:
                self._set_completion_headers(400)
                self.end_headers()
                self.wfile.write(
                    json.dumps(anthropic_api.error_body(400, str(e))).encode()
                )
                return
            self.anthropic = anthropic_api.MessageStream(
                self.body.get("model", "default_model")
            )

        # Extract request parameters from the body
        self.stream = self.body.get("stream", False)
        self.stream_options = self.body.get("stream_options", None)
        self.requested_model = self.body.get("model", "default_model")
        self.requested_draft_model = self.body.get("draft_model", "default_model")
        self.num_draft_tokens = self.body.get(
            "num_draft_tokens", self.response_generator.cli_args.num_draft_tokens
        )
        self.adapter = self.body.get("adapters", None)
        self.max_tokens = self.body.get("max_completion_tokens", None)
        if self.max_tokens is None:
            self.max_tokens = self.body.get(
                "max_tokens", self.response_generator.cli_args.max_tokens
            )
        self.temperature = self.body.get(
            "temperature", self.response_generator.cli_args.temp
        )
        self.top_p = self.body.get("top_p", self.response_generator.cli_args.top_p)
        self.top_k = self.body.get("top_k", self.response_generator.cli_args.top_k)
        self.min_p = self.body.get("min_p", self.response_generator.cli_args.min_p)
        self.repetition_penalty = self.body.get("repetition_penalty", 0.0)
        self.repetition_context_size = self.body.get("repetition_context_size", 20)
        self.presence_penalty = self.body.get("presence_penalty", 0.0)
        self.presence_context_size = self.body.get("presence_context_size", 20)
        self.frequency_penalty = self.body.get("frequency_penalty", 0.0)
        self.frequency_context_size = self.body.get("frequency_context_size", 20)
        self.xtc_probability = self.body.get("xtc_probability", 0.0)
        self.xtc_threshold = self.body.get("xtc_threshold", 0.1)
        self.logit_bias = self.body.get("logit_bias", None)
        self.logprobs = self.body.get("logprobs", False)
        self.top_logprobs = self.body.get("top_logprobs", -1)
        self.seed = self.body.get("seed", None)
        self.chat_template_kwargs = self.body.get("chat_template_kwargs")
        self.thinking_budget = self.body.get("thinking_budget", None)
        try:
            self.validate_model_parameters()
        except ValueError as e:
            self._set_completion_headers(400)
            self.end_headers()
            self.wfile.write(self._error(400, str(e)))
            return

        # Get stop sequences
        stop_words = self.body.get("stop")
        stop_words = stop_words or []
        stop_words = [stop_words] if isinstance(stop_words, str) else stop_words

        # Create the completion request
        try:
            request = request_factories[self.route]()
        except ValueError as e:
            # e.g. a malformed / unfetchable image_url -- report it instead of
            # dropping the connection.
            self._set_completion_headers(400)
            self.end_headers()
            self.wfile.write(self._error(400, str(e)))
            return
        self.handle_completion(request, stop_words)

    def _validate(
        self,
        name,
        expected_type,
        min_val=None,
        max_val=None,
        optional=False,
        whitelist=None,
    ):
        value = getattr(self, name)
        if optional and value is None:
            return
        if not isinstance(value, expected_type):
            try:
                allowed = tuple(et.__name__ for et in expected_type)
            except TypeError:
                allowed = expected_type.__name__
            raise ValueError(f"{name} must be of type {allowed}")
        if whitelist is not None and value in whitelist:
            return
        if min_val is not None and value < min_val:
            raise ValueError(f"{name} must be at least {min_val}")
        if max_val is not None and value > max_val:
            raise ValueError(f"{name} must be at most {max_val}")

    def validate_model_parameters(self):
        """Validate that the passed model parameters have correct types and values."""
        self._validate("stream", bool)
        self._validate("max_tokens", int, min_val=0)
        if self.thinking_budget is not None and (not isinstance(self.thinking_budget, int)
                                                 or isinstance(self.thinking_budget, bool) or self.thinking_budget < 0):
            raise ValueError("thinking_budget must be a non-negative integer")
        self._validate("temperature", (float, int), min_val=0)
        self._validate("top_p", (float, int), min_val=0, max_val=1)
        self._validate("top_k", int, min_val=0)
        self._validate("min_p", (float, int), min_val=0, max_val=1)
        self._validate("num_draft_tokens", int, min_val=0)
        self._validate("repetition_penalty", (float, int), min_val=0)
        self._validate("repetition_context_size", int, min_val=0)
        self._validate("presence_penalty", (float, int))
        self._validate("presence_context_size", int, min_val=0)
        self._validate("frequency_penalty", (float, int))
        self._validate("frequency_context_size", int, min_val=0)
        self._validate("logprobs", bool)
        self._validate("top_logprobs", int, min_val=0, max_val=11, whitelist=[-1])
        self._validate("xtc_probability", float, min_val=0, max_val=1)
        self._validate("xtc_threshold", float, min_val=0, max_val=1)
        self._validate("requested_model", str)
        self._validate("adapter", str, optional=True)
        self._validate("seed", int, optional=True)
        self._validate("logit_bias", dict, optional=True)

        if self.logit_bias is not None:
            try:
                self.logit_bias = {int(k): float(v) for k, v in self.logit_bias.items()}
            except ValueError as e:
                raise ValueError("logit_bias must be a dict of int to float") from e

    def generate_response(
        self,
        text: str,
        finish_reason: Union[Literal["length", "stop"], None],
        prompt_token_count: Optional[int] = None,
        completion_token_count: Optional[int] = None,
        prompt_cache_count: Optional[int] = None,
        token_logprobs: Optional[List[float]] = None,
        top_tokens: Optional[List[Tuple[Dict[str, Any]]]] = None,
        tokens: Optional[List[int]] = None,
        tool_calls: Optional[List[str]] = None,
        reasoning_text: Optional[str] = None,
    ) -> dict:
        """
        Generate a single response packet based on response type (stream or
        not), completion type and parameters.

        Args:
            text (str): Text generated by model
            finish_reason (Union[Literal["length", "stop"], None]): The reason the
              response is being sent: "length", "stop" or `None`.
            prompt_token_count (Optional[int]): The number of tokens in the prompt,
              used to populate the "usage" field (not used when stream).
            completion_token_count (Optional[int]): The number of tokens in the
              response, used to populate the "usage" field (not used when stream).
            prompt_cache_count (Optional[int]): The portion of prompt_token_count
              that was found in the cache when servicing the request.
            token_logprobs (Optional[List[float]]): The log probabilities per token,
              in token order.
            top_tokens (Optional[List[Tuple[Dict[str, Any]]]]): List of outputs from
              _format_top_logprobs, giving info on the top N tokens at each token position.
            tokens (Optional[List[int]]): List of tokens to return with logprobs structure
            tool_calls (Optional[List[str]]): List of tool calls.
            reasoning_text (Optional[str]): The reasoning text generated by the model.

        Returns:
            dict: A dictionary containing the response, in the same format as
              OpenAI's API.
        """
        token_logprobs = token_logprobs or []
        top_logprobs = top_tokens or []
        tool_calls = tool_calls or []

        # Static response
        response = {
            "id": self.request_id,
            "system_fingerprint": self.system_fingerprint,
            "object": self.object_type,
            "model": self.requested_model,
            "created": self.created,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                },
            ],
        }

        if top_logprobs:
            response["choices"][0]["logprobs"] = {
                "content": [
                    dict(i[0], top_logprobs=i) if i else {} for i in top_logprobs
                ]
            }
        elif token_logprobs:
            response["choices"][0]["logprobs"] = {
                "content": [
                    dict(id=i, logprob=g) for i, g in zip(tokens, token_logprobs)
                ]
            }

        if not self.stream:
            if not (
                isinstance(prompt_token_count, int)
                and isinstance(completion_token_count, int)
            ):
                raise ValueError(
                    "Response type is complete, but token counts not provided"
                )

            response["usage"] = {
                "prompt_tokens": prompt_token_count,
                "completion_tokens": completion_token_count,
                "total_tokens": prompt_token_count + completion_token_count,
            }
            if prompt_cache_count is not None and prompt_cache_count >= 0:
                response["usage"]["prompt_tokens_details"] = {
                    "cached_tokens": prompt_cache_count,
                }

        choice = response["choices"][0]

        # Add dynamic response
        if self.object_type.startswith("chat.completion"):
            key_name = "delta" if self.stream else "message"
            choice[key_name] = {"role": "assistant"}
            if not self.stream:
                # The schema requires "content" field to be present
                choice[key_name]["content"] = text if text else None
            elif text:
                choice[key_name]["content"] = text
            if reasoning_text:
                choice[key_name]["reasoning"] = reasoning_text
            if tool_calls:
                choice[key_name]["tool_calls"] = tool_calls
        elif self.object_type == "text_completion":
            choice.update(text=text)
        else:
            raise ValueError(f"Unsupported response type: {self.object_type}")

        return response

    def _matched_stop(self, finish_reason, tokens, stop_words) -> Optional[str]:
        """The stop word the generation ended on, if any (not an EOS token)."""
        if finish_reason != "stop":
            return None
        tokenizer = self.response_generator.model_provider.tokenizer
        for word in stop_words:
            ids = tokenizer.encode(word, add_special_tokens=False)
            if ids and tokens[-len(ids) :] == ids:
                return word
        return None

    def _error(self, status: int, message: str) -> bytes:
        if getattr(self, "route", None) == "/v1/messages":
            return json.dumps(anthropic_api.error_body(status, message)).encode()
        return json.dumps({"error": message}).encode()

    def _as_client_has_it(self, call: dict) -> dict:
        """A sent tool call as the client will send it back."""
        call = {k: v for k, v in call.items() if k != "index"}
        if self.anthropic is not None:
            call["id"] = anthropic_api.tool_use_id(call.get("id"))
        return call

    def _write_chunk(self, resp: dict):
        if self.anthropic is not None:
            self.wfile.write(self.anthropic.chunk(resp))
        else:
            self.wfile.write(f"data: {json.dumps(resp)}\n\n".encode())

    def handle_completion(self, request: CompletionRequest, stop_words: List[str]):
        """
        Generate a response to a prompt and send it to the client in a single batch.

        Args:
            prompt (List[int]): The tokenized prompt.
            stop_words (List[str]): A list of stop words
        """
        args = GenerationArguments(
            model=ModelDescription(
                model=self.requested_model,
                draft=self.requested_draft_model,
                adapter=self.adapter,
            ),
            sampling=SamplingArguments(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                xtc_probability=self.xtc_probability,
                xtc_threshold=self.xtc_threshold,
            ),
            logits=LogitsProcessorArguments(
                logit_bias=self.logit_bias,
                repetition_penalty=self.repetition_penalty,
                repetition_context_size=self.repetition_context_size,
                presence_penalty=self.presence_penalty,
                presence_context_size=self.presence_context_size,
                frequency_penalty=self.frequency_penalty,
                frequency_context_size=self.frequency_context_size,
            ),
            stop_words=stop_words,
            max_tokens=self.max_tokens,
            num_draft_tokens=self.num_draft_tokens,
            logprobs=self.logprobs,
            top_logprobs=self.top_logprobs,
            seed=self.seed,
            chat_template_kwargs=self.chat_template_kwargs,
            thinking_budget=self.thinking_budget,
        )

        # Keep connection allive during long prompt processing (and also log
        # the progress)
        def keepalive_callback(processed, total):
            logging.info(f"Prompt processing progress: {processed}/{total}")
            if self.stream:
                msg = f": keepalive {processed}/{total}\n\n".encode()
                self.wfile.write(msg)
                self.wfile.flush()

        # Create the token generator
        started = _now()
        try:
            ctx, response = self.response_generator.generate(
                request,
                args,
                progress_callback=keepalive_callback,
            )
        except Exception as e:
            self._set_completion_headers(404)
            self.end_headers()
            self.wfile.write(self._error(404, str(e)))
            return

        # Prepare the headers
        if self.stream:
            self._set_stream_headers(200)
            self.end_headers()
            if self.anthropic is not None:
                self.wfile.write(
                    self.anthropic.start(len(ctx.prompt), ctx.prompt_cache_count)
                )
                self.wfile.flush()
            logging.debug("Starting stream:")
        else:
            self._set_completion_headers(200)
            logging.debug("Starting completion:")

        # The request's own: its prompt and cache, for the usage and stats
        # (a thinking budget's carry-on has a longer prompt of its own).
        first_ctx = ctx

        # Tool call formatter
        tool_formatter = ToolCallFormatter(ctx.tool_parser, request.tools, self.stream)

        # Initialize the text state machine
        sm_state = ctx.text_sm.make_state(ctx.initial_state)

        # Variables to save the generated text, tokens, logprobs, tools etc
        prev_state = ctx.initial_state
        finish_reason = "stop"
        reasoning_text = ""
        made_tool_call = False
        tool_text = ""
        tool_calls = []
        text = ""
        tokens = []
        token_logprobs = []
        top_tokens = []
        # A tool call is held back until it closes: a long one (a big
        # argument) sends nothing for minutes, so keep the stream alive.
        tool_keepalive_at = time.monotonic()
        # The whole answer (the stream sends and resets text / tool_calls).
        answer_text = ""
        answer_reasoning = ""
        answer_calls = []   # as sent to the client, with their ids

        def send_calls(calls):
            formatted = tool_formatter(calls)
            answer_calls.extend(formatted)
            return formatted
        first_token_at = last_token_at = None
        n_tokens = n_drafted = 0

        try:
            # A thinking budget: the tokens in the reasoning state counted;
            # at the budget the reasoning is closed by the server. Not for
            # an image request (its prompt needs its images' features) or a
            # model without thinking tokens.
            # The model's tokenizer as the request started (the provider may
            # load another model meanwhile: then no budget is applied).
            provider = self.response_generator.model_provider
            tokenizer = provider.tokenizer
            budget = args.thinking_budget
            if budget is not None and (request.images or not getattr(tokenizer, "has_thinking", False)
                                       or ctx.model_key != provider.model_key):
                budget = None
            thought = 0
            budget_spent = False
            while True:
                switched = False
                for gen in response:
                    # The budget spent before this token: it's dropped and
                    # the reasoning closed (budget 0 closes it at once) --
                    # not halfway through a close the model is writing
                    # itself (a partial match held in the buffer).
                    if (budget is not None and not budget_spent and prev_state == "reasoning"
                            and thought >= budget and not sm_state[3]):
                        budget_spent = True
                        switched = True
                        break
                    logging.debug(gen.text)
                    last_token_at = getattr(gen, "made_at", None) or _now()
                    if first_token_at is None:
                        first_token_at = last_token_at
                    n_tokens += 1
                    n_drafted += bool(getattr(gen, "from_draft", False))

                    # Advance the text state machine to strip control sequences
                    if gen.finish_reason == "stop":
                        sm_state, current_state = TextStateMachine.discard(sm_state)
                        clean_text = ""
                    elif gen.finish_reason == "length":
                        sm_state, clean_text, current_state = TextStateMachine.step(
                            sm_state, gen.text
                        )
                        sm_state, flushed, current_state = TextStateMachine.flush(sm_state)
                        clean_text += flushed
                    else:
                        sm_state, clean_text, current_state = TextStateMachine.step(
                            sm_state, gen.text
                        )

                    # Collect the clean text by state: reasoning, tool, or normal
                    if current_state == "reasoning":
                        reasoning_text += clean_text
                        answer_reasoning += clean_text
                    elif current_state == "tool":
                        tool_text += clean_text
                    elif current_state == "normal":
                        if prev_state == "tool":
                            tool_calls.append(tool_text)
                            tool_text = ""
                            made_tool_call = True
                        text += clean_text
                        answer_text += clean_text

                    # Add the tokens and logprobs to the vars.
                    tokens.append(gen.token)
                    if args.logprobs:
                        token_logprobs.append(gen.logprob)
                    if args.top_logprobs > 0:
                        top_tokens.append(gen.top_tokens)

                    if (
                        self.stream
                        and current_state != "tool"
                        and (text or tool_calls or reasoning_text)
                    ):
                        resp = self.generate_response(
                            text,
                            None,
                            tool_calls=send_calls(tool_calls),
                            reasoning_text=reasoning_text,
                        )
                        self._write_chunk(resp)
                        self.wfile.flush()
                        reasoning_text = ""
                        text = ""
                        tool_calls = []
                    elif self.stream and current_state == "tool":
                        now = time.monotonic()
                        if now - tool_keepalive_at >= 5:
                            tool_keepalive_at = now
                            self.wfile.write(b": keepalive tool\n\n")
                            self.wfile.flush()

                    if gen.finish_reason is not None:
                        finish_reason = gen.finish_reason

                    # Reasoning tokens toward the thinking budget.
                    if budget is not None and current_state == "reasoning":
                        thought += 1

                    prev_state = current_state
                if not switched:
                    break
                ctx.stop()
                close = THINKING_BUDGET_NOTE + tokenizer.think_end + "\n\n"
                closing = tokenizer.encode(close, add_special_tokens=False)
                logging.info(f"Thinking budget of {args.thinking_budget} tokens spent: the answer goes on.")
                # Through the state machine as if generated, the note and
                # then the close: the reasoning ends for the client too.
                for piece in (THINKING_BUDGET_NOTE, tokenizer.think_end + "\n\n"):
                    sm_state, clean_text, current_state = TextStateMachine.step(sm_state, piece)
                    if current_state == "reasoning":
                        reasoning_text += clean_text
                        answer_reasoning += clean_text
                    elif current_state == "normal":
                        text += clean_text
                        answer_text += clean_text
                prev_state = current_state
                tokens.extend(closing)
                # Not the model's: no logprobs of their own (the lists stay
                # aligned with the tokens).
                if args.logprobs:
                    token_logprobs.extend([0.0] * len(closing))
                if args.top_logprobs > 0:
                    top_tokens.extend([()] * len(closing))
                n_tokens += len(closing)
                # The close used what was left: the answer ends there.
                if len(tokens) >= args.max_tokens:
                    finish_reason = "length"
                    break
                carried = CompletionRequest(
                    request_type=request.request_type, prompt=request.prompt, messages=request.messages,
                    tools=request.tools, role_mapping=request.role_mapping,
                    prompt_tokens=list(ctx.prompt) + tokens,
                )
                left = args.max_tokens - len(tokens)
                # A seed goes on rather than repeating the first part's draws.
                seed = None if args.seed is None else args.seed + 1
                ctx, response = self.response_generator.generate(
                    carried, replace(args, max_tokens=left, thinking_budget=None, seed=seed),
                    progress_callback=keepalive_callback,
                )

            if prev_state == "tool" and tool_text:
                tool_calls.append(tool_text)
                made_tool_call = True

            if finish_reason == "stop" and made_tool_call:
                finish_reason = "tool_calls"

            if self.stream:
                resp = self.generate_response(
                    text,
                    finish_reason,
                    tool_calls=send_calls(tool_calls),
                    reasoning_text=reasoning_text,
                )
                self._write_chunk(resp)
                self.wfile.flush()
                if self.anthropic is not None:
                    self.wfile.write(
                        self.anthropic.stop(
                            len(tokens), self._matched_stop(finish_reason, tokens, stop_words)
                        )
                    )
                elif (
                    self.stream_options is not None
                    and self.stream_options["include_usage"]
                ):
                    resp = self.completion_usage_response(
                        len(first_ctx.prompt),
                        len(tokens),
                        first_ctx.prompt_cache_count,
                    )
                    self._write_chunk(resp)
                    self.wfile.flush()
                if self.anthropic is None:
                    self.wfile.write("data: [DONE]\n\n".encode())
                self.wfile.flush()
            else:
                resp = self.generate_response(
                    text,
                    finish_reason,
                    len(first_ctx.prompt),
                    len(tokens),
                    first_ctx.prompt_cache_count,
                    token_logprobs=token_logprobs,
                    top_tokens=top_tokens,
                    tokens=tokens,
                    reasoning_text=reasoning_text,
                    tool_calls=send_calls(tool_calls),
                )
                if self.anthropic is not None:
                    resp = anthropic_api.to_message(
                        resp,
                        self.anthropic.model,
                        self._matched_stop(finish_reason, tokens, stop_words),
                    )
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    response_debug = json.dumps(resp, indent="\t")
                    logging.debug(f"Outgoing Response: {response_debug}")

                response_json = json.dumps(resp).encode()
                self.send_header("Content-Length", str(len(response_json)))
                self.end_headers()
                self.wfile.write(response_json)
                self.wfile.flush()
            if finish_reason in ("stop", "tool_calls"):
                self.response_generator.warm_next_turn(
                    ctx.model_key,
                    request,
                    args,
                    answer_text,
                    [self._as_client_has_it(c) for c in answer_calls],
                    reasoning=answer_reasoning,
                    echoes_reasoning=self.anthropic is not None,
                )
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            # Client disconnected mid-stream (Ctrl-C, timeout, navigated
            # away) -- the generation already did its useful work, so just
            # stop cleanly instead of crashing the request thread.
            logging.debug(f"Client disconnected mid-response, stopping generation: {e!r}")
        finally:
            ctx.stop()
            # One line per request that produced a token, for clients that
            # keep speed statistics (LLMTray): the first token's wait covers
            # the queue and the prompt's uncached part; decoding ends at the
            # last token, not when the client has the response.
            if first_token_at is not None:
                logging.info(
                    f"Request stats: prompt={len(first_ctx.prompt)} "
                    f"cached={max(first_ctx.prompt_cache_count, 0)} "
                    f"first_token_s={first_token_at - started:.3f} "
                    f"tokens={n_tokens} decode_s={last_token_at - first_token_at:.3f} "
                    f"drafted={n_drafted}"
                )

    def completion_usage_response(
        self,
        prompt_token_count: Optional[int] = None,
        completion_token_count: Optional[int] = None,
        prompt_cache_count: Optional[int] = None,
    ):
        response = {
            "id": self.request_id,
            "system_fingerprint": self.system_fingerprint,
            "object": "chat.completion",
            "model": self.requested_model,
            "created": self.created,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_token_count,
                "completion_tokens": completion_token_count,
                "total_tokens": prompt_token_count + completion_token_count,
            },
        }
        if prompt_cache_count is not None and prompt_cache_count >= 0:
            response["usage"]["prompt_tokens_details"] = {
                "cached_tokens": prompt_cache_count,
            }
        return response

    def handle_chat_completions(self) -> CompletionRequest:
        """
        Handle a chat completion request.

        Returns:
            mx.array: A mx.array of the tokenized prompt from the request body
        """
        body = self.body
        assert "messages" in body, "Request did not contain messages"

        # Determine response type
        self.request_id = f"chatcmpl-{uuid.uuid4()}"
        self.object_type = "chat.completion.chunk" if self.stream else "chat.completion"

        messages = body["messages"]
        images = extract_images(messages)

        return CompletionRequest(
            "chat",
            "",
            messages,
            body.get("tools") or None,
            body.get("role_mapping"),
            images=images or None,
        )

    def handle_text_completions(self) -> CompletionRequest:
        """
        Handle a text completion request.

        Returns:
            mx.array: A mx.array of the tokenized prompt from the request body
        """
        # Determine response type
        self.request_id = f"cmpl-{uuid.uuid4()}"
        self.object_type = "text_completion"
        assert "prompt" in self.body, "Request did not contain a prompt"
        return CompletionRequest(
            "text",
            self.body["prompt"],
            [],
            None,
            None,
        )

    def do_GET(self):
        """
        Respond to a GET request from a client.
        """
        path = self.path.split("?", 1)[0]
        if any(path == p or path.startswith(p + "/") for p in ("/v1/models", "/api/v0/models")):
            # /api/v0/models is LM Studio's own REST convention -- some
            # clients probe that shape by default. Same OpenAI-shaped
            # response as /v1/models, not LM Studio's richer schema, but
            # enough to stop 404s and hand back a usable model list.
            self.handle_models_request()
        elif self.path == "/health":
            self.handle_health_check()
        else:
            self._set_completion_headers(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def handle_health_check(self):
        """
        Handle a GET request for the /health endpoint.
        """
        is_healthy = self.response_generator.is_healthy
        status_code = 200 if is_healthy else 503
        status = "ok" if is_healthy else "unavailable"

        self._set_completion_headers(status_code)
        self.end_headers()

        self.wfile.write(json.dumps({"status": status}).encode())
        self.wfile.flush()

    def handle_models_request(self):
        """
        Handle a GET request for the /v1/models endpoint.
        """
        self._set_completion_headers(200)
        self.end_headers()

        files = ["config.json", "model.safetensors.index.json", "tokenizer_config.json"]

        # /v1/models[/<id>] and LM Studio's /api/v0/models[/<id>]: the id is
        # what follows the prefix (splitting on "/" took "models" itself as
        # the id for /api/v0/models, so it always listed nothing).
        path = unquote(self.path.split("?", 1)[0])
        prefix = "/api/v0/models" if path.startswith("/api/v0/models") else "/v1/models"
        filter_repo_id = path[len(prefix):].strip("/") or None

        def probably_mlx_lm(repo):
            if repo.repo_type != "model":
                return False
            if "main" not in repo.refs:
                return False
            if filter_repo_id is not None and repo.repo_id != filter_repo_id:
                return False
            file_names = {f.file_path.name for f in repo.refs["main"].files}
            return all(f in file_names for f in files)

        # Scan the cache directory for downloaded mlx models
        hf_cache_info = scan_cache_dir()
        downloaded_models = [
            repo for repo in hf_cache_info.repos if probably_mlx_lm(repo)
        ]

        # Create a list of available models
        models = [
            {
                "id": repo.repo_id,
                "object": "model",
                "created": self.created,
            }
            for repo in downloaded_models
        ]

        cli_args = self.response_generator.cli_args
        if cli_args.model:
            model_path = Path(cli_args.model)
            if model_path.exists():
                model_id = str(model_path.resolve())
                # The id is an absolute path: its leading "/" was stripped
                # from the filter with the separator.
                if filter_repo_id in (None, model_id.strip("/")):
                    models.append(
                        {
                            "id": model_id,
                            "object": "model",
                            "created": self.created,
                        }
                    )
        # --model-alias names are models a client can ask for too.
        for alias in getattr(cli_args, "model_alias", None) or []:
            if filter_repo_id in (None, alias):
                models.append({"id": alias, "object": "model", "created": self.created})

        response = {"object": "list", "data": models}

        response_json = json.dumps(response).encode()
        self.wfile.write(response_json)
        self.wfile.flush()


def _run_http_server(
    host: str,
    port: int,
    response_generator,
    server_class=ThreadingHTTPServer,
    handler_class=APIHandler,
):
    server_address = (host, port)
    infos = socket.getaddrinfo(
        *server_address, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
    )
    server_class.address_family, _, _, _, server_address = next(iter(infos))
    httpd = server_class(
        server_address,
        lambda *args, **kwargs: handler_class(
            response_generator,
            system_fingerprint=get_system_fingerprint(),
            *args,
            **kwargs,
        ),
    )
    warnings.warn(
        "mlx_lm.server is not recommended for production as "
        "it only implements basic security checks."
    )
    logging.info(f"Starting httpd at {host} on port {port}...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()
        response_generator.stop_and_join()


def run(
    host: str,
    port: int,
    model_provider: ModelProvider,
    server_class=ThreadingHTTPServer,
    handler_class=APIHandler,
):
    group = mx.distributed.init()
    prompt_cache = LRUPromptCache(model_provider.cli_args.prompt_cache_size)
    response_generator = ResponseGenerator(model_provider, prompt_cache)
    if group.rank() == 0:
        _run_http_server(host, port, response_generator)
    else:
        response_generator.join()


def make_parser():
    parser = argparse.ArgumentParser(description="MLX Http Server.")
    parser.add_argument(
        "--model",
        type=str,
        help="The path to the MLX model weights, tokenizer, and config",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Optional path for the trained adapter weights and config.",
    )
    parser.add_argument(
        "--host",
        type=str,
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
        type=lambda x: x.split(","),
        default="*",
        help="Allowed origins (default: *)",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        help="A model to be used for speculative decoding.",
        default=None,
    )
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        help="Number of tokens to draft when using speculative decoding.",
        default=3,
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer",
    )
    parser.add_argument(
        "--allow-image-urls",
        action="store_true",
        help="Fetch http(s) image_url parts (off by default: the server would "
        "request any URL a client names, loopback and LAN included). "
        "data: URIs always work.",
    )
    parser.add_argument(
        "--model-alias",
        action="append",
        default=None,
        help="Extra model name a client may request that should route to --model "
        "instead of mlx_lm trying to fetch it from the HF Hub. Repeatable.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default="",
        help="Specify a chat template for the tokenizer",
        required=False,
    )
    parser.add_argument(
        "--use-default-chat-template",
        action="store_true",
        help="Use the default chat template",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Default sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Default nucleus sampling top-p (default: 1.0)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Default top-k sampling (default: 0, disables top-k)",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.0,
        help="Default min-p sampling (default: 0.0, disables min-p)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Default maximum number of tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--chat-template-args",
        type=json.loads,
        help="""A JSON formatted string of arguments for the tokenizer's apply_chat_template, e.g. '{"enable_thinking":false}'""",
        default="{}",
    )
    parser.add_argument(
        "--no-batch-handoff",
        dest="batch_handoff",
        action="store_false",
        help="Keep a speculative (MTP / draft model) request on its own until it "
        "ends, instead of moving it into the batch when another request arrives",
    )
    parser.add_argument(
        "--decode-concurrency",
        type=int,
        default=32,
        help="When a request is batchable then decode that many requests in parallel",
    )
    parser.add_argument(
        "--prompt-concurrency",
        type=int,
        default=8,
        help="When a request is batchable then process that many prompts in parallel",
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=2048,
        help="Step size for prefill processing (default: 2048)",
    )
    parser.add_argument(
        "--prefill-memory-mb",
        type=_int_at_least(0),
        default=SERVER_DEFAULTS["prefill_memory_mb"],
        help="Memory for the attention scores of one prefill step, in MB, "
        "where they are materialized (head dims without a fused kernel, or "
        "a quantized KV cache). The step shrinks as the prompt grows (not "
        "below --min-prefill-step). Single-request path only (draft model, "
        "MTP, --kv-bits, seed, images): batched prefill keeps "
        "--prefill-step-size. 0 keeps --prefill-step-size "
        f"(default: {SERVER_DEFAULTS['prefill_memory_mb']})",
    )
    parser.add_argument(
        "--buffer-cache-mb",
        type=_int_at_least(0),
        default=None,
        help="Cap on MLX's cache of freed buffers, in MB (mx.set_cache_limit). "
        "By default MLX keeps freed buffers up to its memory limit; decoding "
        "with a draft model allocates buffers of a new size every step (the "
        "verify pass rewrites the sliding windows), and on Gemma 4 26B with a "
        "22k-token prompt they piled up to ~4 GB, pushing the process into "
        "swap on a 24 GB Mac. Unset keeps MLX's default.",
    )
    parser.add_argument(
        "--min-prefill-step",
        type=_int_at_least(1),
        default=SERVER_DEFAULTS["min_prefill_step"],
        help="Smallest prefill step --prefill-memory-mb may shrink to "
        f"(default: {SERVER_DEFAULTS['min_prefill_step']})",
    )
    parser.add_argument(
        "--prefill-score-bytes",
        type=_int_at_least(1),
        default=SERVER_DEFAULTS["prefill_score_bytes"],
        help="Bytes per attention score assumed by --prefill-memory-mb "
        "(4: fp32, the safe case; 2: bf16) "
        f"(default: {SERVER_DEFAULTS['prefill_score_bytes']})",
    )
    parser.add_argument(
        "--prefill-step-warn-below",
        type=_int_at_least(0),
        default=SERVER_DEFAULTS["prefill_step_warn_below"],
        help="Log a warning (slow prefill) when the prefill step shrinks "
        f"below this (default: {SERVER_DEFAULTS['prefill_step_warn_below']})",
    )
    parser.add_argument(
        "--memory-headroom-fraction",
        type=_fraction,
        default=SERVER_DEFAULTS["memory_headroom_fraction"],
        help="Fraction of Metal's recommended working set that memory in "
        "use, a prompt-cache checkpoint copy and the prefill budget may "
        "take together; above it, stored caches are evicted, then the "
        "checkpoint is skipped. The rest is headroom for allocations not "
        "counted (default: "
        f"{SERVER_DEFAULTS['memory_headroom_fraction']})",
    )
    parser.add_argument(
        "--oom-retry-step-divisor",
        type=_int_at_least(1),
        default=SERVER_DEFAULTS["oom_retry_step_divisor"],
        help="After a Metal out-of-memory error in the checkpoint prefill, "
        "retry once with the prefill step and --prefill-memory-mb divided "
        f"by this (default: {SERVER_DEFAULTS['oom_retry_step_divisor']})",
    )
    parser.add_argument(
        "--junction-min-gap-tokens",
        type=_int_at_least(0),
        default=SERVER_DEFAULTS["junction_min_gap_tokens"],
        help="A checkpoint where the prompt diverges from a cached one is "
        "saved only this many tokens or more away from the reused part, "
        "other checkpoints and the prompt end (default: "
        f"{SERVER_DEFAULTS['junction_min_gap_tokens']})",
    )
    parser.add_argument(
        "--prompt-cache-size",
        type=int,
        default=10,
        help="Maximum number of distinct KV caches to hold in the prompt cache",
    )
    parser.add_argument(
        "--warm-next-turn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="While idle, prefill a finished chat request's next turn (the "
        "conversation with the answer as the client sends it back) and cache "
        "it. For models whose cache can't be trimmed back (SSM / delta-rule "
        "layers); others skip it. Default: on.",
    )
    parser.add_argument(
        "--warm-next-turn-max-tokens",
        type=int,
        default=16384,
        help="Longest prefill --warm-next-turn does for one turn (default 16384).",
    )
    parser.add_argument(
        "--prompt-cache-bytes",
        type=_parse_size,
        help="Maximum size in bytes of the KV caches",
    )
    parser.add_argument(
        "--kv-bits",
        type=int,
        default=None,
        help="Number of bits for KV cache quantization (e.g., 4 or 8). "
        "Reduces memory usage for long contexts. Disables batching, so "
        "requests are served one at a time. Default: None (full precision)",
    )
    parser.add_argument(
        "--kv-group-size",
        type=int,
        default=64,
        help="Group size for KV cache quantization (default: 64)",
    )
    parser.add_argument(
        "--quantized-kv-start",
        type=int,
        default=DEFAULT_QUANTIZED_KV_START,
        help="Token position to start KV cache quantization "
        f"(default: {DEFAULT_QUANTIZED_KV_START})",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Use pipelining instead of tensor parallelism",
    )
    parser.add_argument(
        "--mmap-lookup-tables",
        action="store_true",
        help="Read lookup-only tables (Gemma 4 per-layer embeddings) by row "
        "from the weights files instead of loading them. Less memory, a "
        "little slower decoding",
    )
    parser.add_argument(
        "--lazy-towers",
        action="store_true",
        help="Load the image and audio towers on their first use",
    )
    return parser


def main():
    args = make_parser().parse_args()
    from . import multimodal

    multimodal.ALLOW_IMAGE_URLS = args.allow_image_urls
    _ = maybe_set_recommended_wired_limit()
    if args.buffer_cache_mb is not None:
        mx.set_cache_limit(args.buffer_cache_mb << 20)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), None),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    run(args.host, args.port, ModelProvider(args))


if __name__ == "__main__":
    print(
        "Calling `python -m mlx_lm.server...` directly is deprecated."
        " Use `mlx_lm.server...` or `python -m mlx_lm server ...` instead."
    )
    main()
