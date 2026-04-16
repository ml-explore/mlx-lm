# Copyright © 2023-2024 Apple Inc.

import argparse
import json
import logging
import pickle
import platform
import socket
import time
import uuid
import warnings
from collections import deque
from dataclasses import dataclass
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
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import mlx.core as mx
from huggingface_hub import scan_cache_dir

from ._version import __version__
from .generate import (
    BatchGenerator,
    SequenceStateMachine,
    generation_stream,
    stream_generate,
)
from .models.cache import (
    LRUPromptCache,
    make_prompt_cache,
)
from .sample_utils import make_logits_processors, make_sampler
from .utils import _parse_size, load, sharded_load


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
            parsed = self._tool_parser(tool_text, self._tools)
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
        if isinstance(content, list):
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


@dataclass
class CompletionRequest:
    request_type: Literal["chat", "text"]

    prompt: str

    messages: List[Any]
    tools: Optional[List[Any]]
    role_mapping: Optional[Dict[str, Any]]


@dataclass
class GenerationContext:
    has_tool_calling: bool
    has_thinking: bool
    tool_parser: Callable[[str, Any], Dict]

    sequences: Dict[Tuple[int], str]

    prompt: List[int]
    prompt_cache_count: int = -1

    _should_stop: bool = False

    def stop(self):
        self._should_stop = True


@dataclass
class Response:
    text: str
    token: int
    state: str
    match: Tuple[int]
    logprob: float
    finish_reason: Optional[str]
    top_tokens: Tuple[Dict[str, Any]]


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
            with mx.stream(generation_stream):
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

        group = mx.distributed.init()
        self.pipeline_group = group if group.size() > 1 and cli_args.pipeline else None
        self.tensor_group = (
            group if group.size() > 1 and not cli_args.pipeline else None
        )
        self.is_distributed = group.size() > 1

        # Preload the default model if it is provided
        self.default_model_map = {}
        if self.cli_args.model is not None:
            self.default_model_map[self.cli_args.model] = "default_model"
            self.load(self.cli_args.model, draft_model_path="default_model")

    # Added in adapter_path to load dynamically
    def load(self, model_path, adapter_path=None, draft_model_path=None):
        model_path = self.default_model_map.get(model_path, model_path)
        if self.model_key == (model_path, adapter_path, draft_model_path):
            return self.model, self.tokenizer

        # Remove the old model if it exists.
        self.model = None
        self.tokenizer = None
        self.model_key = None
        self.draft_model = None

        # Building tokenizer_config
        tokenizer_config = {
            "trust_remote_code": True if self.cli_args.trust_remote_code else None
        }
        if self.cli_args.chat_template:
            tokenizer_config["chat_template"] = self.cli_args.chat_template

        if model_path == "default_model":
            if self.cli_args.model is None:
                raise ValueError(
                    "A model path has to be given as a CLI "
                    "argument or in the HTTP request"
                )
            adapter_path = adapter_path or self.cli_args.adapter_path
            # TODO: Generalize distributed load
            if self.is_distributed:
                model, tokenizer = sharded_load(
                    self.cli_args.model, self.pipeline_group, self.tensor_group
                )
            else:
                model, tokenizer = load(
                    self.cli_args.model,
                    adapter_path=adapter_path,
                    tokenizer_config=tokenizer_config,
                )
        else:
            # TODO: Generalize distributed load
            if self.is_distributed:
                model, tokenizer = sharded_load(
                    model_path, self.pipeline_group, self.tensor_group
                )
            else:
                model, tokenizer = load(
                    model_path,
                    adapter_path=adapter_path,
                    tokenizer_config=tokenizer_config,
                )

        if self.cli_args.use_default_chat_template:
            if tokenizer.chat_template is None:
                tokenizer.chat_template = tokenizer.default_chat_template

        self.model_key = (model_path, adapter_path, draft_model_path)
        self.model = model
        self.tokenizer = tokenizer

        def validate_draft_tokenizer(draft_tokenizer):
            # Check if tokenizers are compatible
            if draft_tokenizer.vocab_size != tokenizer.vocab_size:
                logging.warning(
                    "Draft model tokenizer does not match model tokenizer. "
                    "Speculative decoding may not work as expected."
                )

        # Load draft model if specified
        if (
            draft_model_path == "default_model"
            and self.cli_args.draft_model is not None
        ):
            self.draft_model, draft_tokenizer = load(self.cli_args.draft_model)
            validate_draft_tokenizer(draft_tokenizer)

        elif draft_model_path is not None and draft_model_path != "default_model":
            self.draft_model, draft_tokenizer = load(draft_model_path)
            validate_draft_tokenizer(draft_tokenizer)

        if self.draft_model is None:
            self.is_batchable = all(
                hasattr(c, "merge") for c in make_prompt_cache(self.model)
            )

        return self.model, self.tokenizer


def _make_sampler(args, tokenizer):
    return make_sampler(
        args.sampling.temperature,
        top_p=args.sampling.top_p,
        top_k=args.sampling.top_k,
        min_p=args.sampling.min_p,
        xtc_probability=args.sampling.xtc_probability,
        xtc_threshold=args.sampling.xtc_threshold,
        xtc_special_tokens=[
            tokenizer.eos_token_id,
            tokenizer.encode("\n"),
        ],
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
        self._state_machine_cache = {}

        self._time_budget = TimeBudget()
        self._is_distributed = mx.distributed.init().size() > 1
        self._rank = mx.distributed.init().rank()
        self._stop = False
        self._generation_thread = Thread(target=self._generate)
        self._generation_thread.start()

    def stop_and_join(self):
        self._stop = True
        self._generation_thread.join()

    def join(self):
        self._generation_thread.join()

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

        with mx.stream(generation_stream):
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

        # It is not a user message so no segmentation needed.
        if messages[-1]["role"] != "user":
            return prompt, [prompt], ["assistant"], initial_state

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

    def _make_state_machine(
        self, model_key, tokenizer, stop_words, initial_state="normal"
    ):
        """Make a new SequenceStateMachine or fetch it if we 've made it before.

        Return also a dictionary that maps the token sequences in the state
        machine to their strings.
        """
        cache_key = (model_key, tuple(stop_words), initial_state)
        rs = self._state_machine_cache.get(cache_key)
        if rs is not None:
            return rs

        # Will hold the state machine transitions and the sequences map to
        # strings.
        transitions = {}
        sequences = {}

        # Add all the stop sequences
        common_stops = []
        for t in tokenizer.eos_token_ids:
            sequences[(t,)] = tokenizer.convert_ids_to_tokens(t)
            common_stops.append(((t,), None))
        for w in stop_words:
            t = tuple(tokenizer.encode(w, add_special_tokens=False))
            sequences[t] = w
            common_stops.append((t, None))

        # From normal to stop
        transitions["normal"] = list(common_stops)

        # Reasoning related transitions
        if tokenizer.has_thinking:
            ts = tokenizer.think_start_tokens
            te = tokenizer.think_end_tokens
            transitions["normal"].append((ts, "reasoning"))
            transitions["reasoning"] = [(te, "normal")]
            transitions["reasoning"].extend(common_stops)
            sequences[ts] = tokenizer.think_start
            sequences[te] = tokenizer.think_end

        # Tool calling relating transitions
        if tokenizer.has_tool_calling:
            ts = tokenizer.tool_call_start_tokens
            te = tokenizer.tool_call_end_tokens
            transitions["normal"].append((ts, "tool"))
            transitions["tool"] = [(te, "normal")]
            transitions["tool"].extend(common_stops)
            sequences[ts] = tokenizer.tool_call_start
            sequences[te] = tokenizer.tool_call_end

            # Allow reasoning -> tool transitions for thinking models that
            # emit tool calls without first closing the think block
            # (e.g. Kimi K2.5).
            if tokenizer.has_thinking:
                transitions["reasoning"].append((ts, "tool"))

        sm = SequenceStateMachine(transitions, initial=initial_state)
        if len(self._state_machine_cache) > 100:
            self._state_machine_cache.clear()
        self._state_machine_cache[cache_key] = (sm, sequences)

        return sm, sequences

    def _is_batchable(self, args):
        return self.model_provider.is_batchable and args.seed is None

    def _generate(self):
        current_model = None
        current_sampling = None
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

            # We got a request
            if request is not None:
                rqueue, request, args = request

                # Can it be added to the current batch?
                if (
                    batch_generator is not None
                    and current_model == args.model
                    and self._is_batchable(args)
                ):
                    try:
                        prompt, segments, segment_types, initial_state = self._tokenize(
                            current_tokenizer, request, args
                        )
                    except Exception as e:
                        rqueue.put(e)
                        continue

                    sm, sequences = self._make_state_machine(
                        self.model_provider.model_key,
                        tokenizer,
                        args.stop_words,
                        initial_state,
                    )

                    self._log_cache_stats()
                    cache, rest = self.prompt_cache.fetch_nearest_cache(
                        current_model_key, prompt
                    )
                    prompt_cache_count = len(prompt) - len(rest)
                    N = prompt_cache_count
                    while N > 0:
                        if N >= len(segments[0]):
                            N -= len(segments.pop(0))
                            segment_types.pop(0)
                        else:
                            segments[0] = segments[0][N:]
                            break

                    ctx = GenerationContext(
                        has_tool_calling=tokenizer.has_tool_calling,
                        has_thinking=tokenizer.has_thinking,
                        tool_parser=tokenizer.tool_parser,
                        sequences=sequences,
                        prompt=prompt,
                        prompt_cache_count=prompt_cache_count,
                    )
                    rqueue.put(ctx)

                    (uid,) = batch_generator.insert_segments(
                        segments=[segments],
                        max_tokens=[args.max_tokens],
                        caches=[cache],
                        all_tokens=[prompt[:prompt_cache_count]],
                        samplers=[_make_sampler(args, tokenizer)],
                        logits_processors=[_make_logits_processors(args)],
                        state_machines=[sm],
                    )
                    batch_results[uid] = {
                        "ctx": ctx,
                        "rqueue": rqueue,
                        "detokenizer": tokenizer.detokenizer,
                        "segment_types": segment_types[::-1],
                        "top_logprobs": args.top_logprobs,
                    }
                    # just making sure we don't leave a reference around
                    del cache

                    if self.model_provider.cli_args.prompt_cache_bytes is not None:
                        total = self.model_provider.cli_args.prompt_cache_bytes
                        active = batch_generator.prompt_cache_nbytes
                        self.prompt_cache.trim_to(n_bytes=total - active)
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

                    if not self._is_batchable(args):
                        self._serve_single((rqueue, request, args))
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
                    )
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
                        current_sampling = None
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
                    del caches

                    for r in gen_responses:
                        result = batch_results[r.uid]
                        result["detokenizer"].add_token(r.token)
                        result["rqueue"].put(
                            Response(
                                result["detokenizer"].last_segment,
                                r.token,
                                r.current_state,
                                r.match_sequence,
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
                            del batch_results[r.uid]

                        if result["ctx"]._should_stop:
                            uids_to_remove.append(r.uid)

                uids_to_remove = self._share_object(uids_to_remove)
                if uids_to_remove:
                    with mx.stream(generation_stream):
                        batch_generator.remove(uids_to_remove)
                        for uid in uids_to_remove:
                            # It may have already been removed during
                            # generation
                            batch_results.pop(uid, None)

    def _serve_single(self, request):
        rqueue, request, args = request

        # Define the progress callback
        def progress(tokens_processed, tokens_total):
            rqueue.put((tokens_processed, tokens_total))

        try:
            # Load the model and tokenizer
            model = self.model_provider.model
            tokenizer = self.model_provider.tokenizer
            draft_model = self.model_provider.draft_model

            # Prepare the prompt and state machine
            prompt, _, _, initial_state = self._tokenize(tokenizer, request, args)
            sm, sequences = self._make_state_machine(
                self.model_provider.model_key,
                tokenizer,
                args.stop_words,
                initial_state=initial_state,
            )
            sm_state = sm.make_state()

            # Start the generation context
            ctx = GenerationContext(
                has_thinking=tokenizer.has_thinking,
                has_tool_calling=tokenizer.has_tool_calling,
                tool_parser=tokenizer.tool_parser,
                sequences=sequences,
                prompt=prompt,
            )
            rqueue.put(ctx)

            # Seed if requested
            if args.seed is not None:
                mx.random.seed(args.seed)

            # Make the sampler and logit processor
            sampler = _make_sampler(args, tokenizer)
            logits_processors = _make_logits_processors(args)

            # Load the KV cache
            self._log_cache_stats()
            cache, rest = self.prompt_cache.fetch_nearest_cache(
                self.model_provider.model_key, prompt
            )
            ctx.prompt_cache_count = len(prompt) - len(rest)
            cache_key = prompt[:]
            if cache is None:
                cache = make_prompt_cache(self.model_provider.model)
                if self.model_provider.draft_model is not None:
                    cache += make_prompt_cache(self.model_provider.draft_model)

            # Process the prompt and generate tokens
            for gen in stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=rest,
                max_tokens=args.max_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
                prompt_cache=cache,
                draft_model=draft_model,
                num_draft_tokens=args.num_draft_tokens,
                prompt_progress_callback=progress,
                prefill_step_size=self.cli_args.prefill_step_size,
            ):
                finish_reason = gen.finish_reason
                sm_state, match_sequence, current_state = sm.match(sm_state, gen.token)
                if match_sequence is not None and current_state is None:
                    finish_reason = "stop"
                rqueue.put(
                    Response(
                        gen.text,
                        gen.token,
                        current_state,
                        match_sequence,
                        gen.logprobs[gen.token].item(),
                        finish_reason,
                        _format_top_logprobs(
                            gen.logprobs, args.top_logprobs, tokenizer
                        ),
                    )
                )
                cache_key.append(gen.token)

                if ctx._should_stop:
                    if self._is_distributed:
                        raise NotImplementedError()
                    break

                if finish_reason is not None:
                    break

            rqueue.put(None)

            # Save the KV cache again
            self.prompt_cache.insert_cache(
                self.model_provider.model_key, cache_key, cache
            )

        except Exception as e:
            rqueue.put(e)

    def generate(
        self,
        request: CompletionRequest,
        generation_args: GenerationArguments,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ):
        response_queue = Queue()
        self.requests.put((response_queue, request, generation_args))

        def _inner():
            while True:
                response = response_queue.get()
                if response is None:
                    break
                if isinstance(response, Exception):
                    raise response
                if isinstance(response, tuple):
                    if progress_callback is not None:
                        progress_callback(*response)
                    continue
                yield response

        def _process_control_tokens(ctx, token_stream):
            buffer_size = max(len(s) for s in ctx.sequences)
            buffered_stream = deque()

            for tok in token_stream:
                buffered_stream.append(tok)
                if tok.match is not None:
                    for _ in tok.match:
                        buffered_stream.pop()
                if len(buffered_stream) >= buffer_size:
                    yield buffered_stream.popleft()
            while len(buffered_stream) > 0:
                yield buffered_stream.popleft()

        ctx = response_queue.get()
        if isinstance(ctx, Exception):
            raise ctx

        return ctx, _process_control_tokens(ctx, _inner())

    @property
    def cli_args(self):
        return self.model_provider.cli_args


class APIHandler(BaseHTTPRequestHandler):
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
        }

        if self.path not in request_factories:
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
                json.dumps({"error": "Content-Length header is required"}).encode()
            )
            return
        try:
            content_length = int(content_length)
        except ValueError:
            self._set_completion_headers(400)
            self.end_headers()
            self.wfile.write(
                json.dumps({"error": "Invalid Content-Length header"}).encode()
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
                json.dumps({"error": f"Invalid JSON in request body: {e}"}).encode()
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
                json.dumps({"error": "Request should be a JSON dictionary"}).encode()
            )
            return

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
        self.xtc_threshold = self.body.get("xtc_threshold", 0.0)
        self.logit_bias = self.body.get("logit_bias", None)
        self.logprobs = self.body.get("logprobs", False)
        self.top_logprobs = self.body.get("top_logprobs", -1)
        self.seed = self.body.get("seed", None)
        self.chat_template_kwargs = self.body.get("chat_template_kwargs")
        self.validate_model_parameters()

        # Get stop sequences
        stop_words = self.body.get("stop")
        stop_words = stop_words or []
        stop_words = [stop_words] if isinstance(stop_words, str) else stop_words

        # Create the completion request
        request = request_factories[self.path]()
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
            except ValueError:
                raise ValueError("logit_bias must be a dict of int to float")

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
            if text:
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
        try:
            ctx, response = self.response_generator.generate(
                request,
                args,
                progress_callback=keepalive_callback,
            )
        except Exception as e:
            self._set_completion_headers(404)
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
            return

        # Prepare the headers
        if self.stream:
            self._set_stream_headers(200)
            self.end_headers()
            logging.debug("Starting stream:")
        else:
            self._set_completion_headers(200)
            logging.debug("Starting completion:")

        # Tool call formatter
        tool_formatter = ToolCallFormatter(ctx.tool_parser, request.tools, self.stream)

        # Variables to save the generated text, tokens, logprobs, tools etc
        prev_state = None
        finish_reason = "stop"
        reasoning_text = ""
        made_tool_call = False
        tool_text = ""
        tool_calls = []
        text = ""
        tokens = []
        token_logprobs = []
        top_tokens = []

        try:
            for gen in response:
                logging.debug(gen.text)

                # Collect the text according to our current state and state
                # transitions. Reasoning or tool or normal text.
                if gen.state == "reasoning":
                    reasoning_text += gen.text
                elif gen.state == "tool":
                    tool_text += gen.text
                elif gen.state == "normal":
                    if prev_state == "tool":
                        tool_calls.append(tool_text)
                        tool_text = ""
                        made_tool_call = True
                    text += gen.text

                # Add the tokens and logprobs to the vars.
                tokens.append(gen.token)
                if args.logprobs:
                    token_logprobs.append(gen.logprob)
                if args.top_logprobs > 0:
                    top_tokens.append(gen.top_tokens)

                if (
                    self.stream
                    and gen.state != "tool"
                    and (text or tool_calls or reasoning_text)
                ):
                    resp = self.generate_response(
                        text,
                        None,
                        tool_calls=tool_formatter(tool_calls),
                        reasoning_text=reasoning_text,
                    )
                    self.wfile.write(f"data: {json.dumps(resp)}\n\n".encode())
                    self.wfile.flush()
                    reasoning_text = ""
                    text = ""
                    tool_calls = []

                if gen.finish_reason is not None:
                    finish_reason = gen.finish_reason

                prev_state = gen.state

            if prev_state == "tool" and tool_text:
                tool_calls.append(tool_text)
                made_tool_call = True

            if finish_reason == "stop" and made_tool_call:
                finish_reason = "tool_calls"

            if self.stream:
                resp = self.generate_response(
                    text,
                    finish_reason,
                    tool_calls=tool_formatter(tool_calls),
                    reasoning_text=reasoning_text,
                )
                self.wfile.write(f"data: {json.dumps(resp)}\n\n".encode())
                self.wfile.flush()
                if (
                    self.stream_options is not None
                    and self.stream_options["include_usage"]
                ):
                    resp = self.completion_usage_response(
                        len(ctx.prompt),
                        len(tokens),
                        ctx.prompt_cache_count,
                    )
                    self.wfile.write(f"data: {json.dumps(resp)}\n\n".encode())
                    self.wfile.flush()
                self.wfile.write("data: [DONE]\n\n".encode())
                self.wfile.flush()
            else:
                resp = self.generate_response(
                    text,
                    finish_reason,
                    len(ctx.prompt),
                    len(tokens),
                    ctx.prompt_cache_count,
                    token_logprobs=token_logprobs,
                    top_tokens=top_tokens,
                    tokens=tokens,
                    reasoning_text=reasoning_text,
                    tool_calls=tool_formatter(tool_calls),
                )
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    response_debug = json.dumps(resp, indent="\t")
                    logging.debug(f"Outgoing Response: {response_debug}")

                response_json = json.dumps(resp).encode()
                self.send_header("Content-Length", str(len(response_json)))
                self.end_headers()
                self.wfile.write(response_json)
                self.wfile.flush()
        finally:
            ctx.stop()

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

        return CompletionRequest(
            "chat",
            "",
            body["messages"],
            body.get("tools") or None,
            body.get("role_mapping"),
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
        if self.path.startswith("/v1/models"):
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
        self._set_completion_headers(200)
        self.end_headers()

        self.wfile.write('{"status": "ok"}'.encode())
        self.wfile.flush()

    def handle_models_request(self):
        """
        Handle a GET request for the /v1/models endpoint.
        """
        self._set_completion_headers(200)
        self.end_headers()

        files = ["config.json", "model.safetensors.index.json", "tokenizer_config.json"]

        parts = self.path.split("/")
        filter_repo_id = None
        if len(parts) > 3:
            filter_repo_id = "/".join(parts[3:])

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

        if self.response_generator.cli_args.model:
            model_path = Path(self.response_generator.cli_args.model)
            if model_path.exists():
                model_id = str(model_path.resolve())
                models.append(
                    {
                        "id": model_id,
                        "object": "model",
                        "created": self.created,
                    }
                )

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


def main():
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
        "--prompt-cache-size",
        type=int,
        default=10,
        help="Maximum number of distinct KV caches to hold in the prompt cache",
    )
    parser.add_argument(
        "--prompt-cache-bytes",
        type=_parse_size,
        help="Maximum size in bytes of the KV caches",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Use pipelining instead of tensor parallelism",
    )
    args = parser.parse_args()
    if mx.metal.is_available():
        wired_limit = mx.device_info()["max_recommended_working_set_size"]
        mx.set_wired_limit(wired_limit)

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
