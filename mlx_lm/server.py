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
from dataclasses import dataclass, field
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
from .generate import BatchGenerator, generation_stream, stream_generate
from .models.cache import (
    LRUPromptCache,
    can_trim_prompt_cache,
    make_prompt_cache,
    trim_prompt_cache,
)
from .sample_utils import make_logits_processors, make_sampler
from .utils import _parse_size, load, sharded_load


def _maybe_dequantize_cache(cache):
    """Convert MixedQuantKVCache entries back to KVCache for batch merge.

    Handles both top-level and sub-caches inside CacheList.
    """
    from .models.cache import CacheList
    from .models.mixed_quant_cache import MixedQuantKVCache
    for i, c in enumerate(cache):
        if isinstance(c, MixedQuantKVCache):
            cache[i] = c.to_kvcache()
        elif isinstance(c, CacheList):
            new_subs = list(c.caches)
            changed = False
            for j, sub in enumerate(new_subs):
                if isinstance(sub, MixedQuantKVCache):
                    new_subs[j] = sub.to_kvcache()
                    changed = True
            if changed:
                c.caches = tuple(new_subs)
    return cache


def _maybe_quantize_cache(cache, kv_quant_config, min_tokens=0):
    """Convert KVCache entries to MixedQuantKVCache for LRU storage.

    Handles both top-level KVCache and KVCache sub-caches inside CacheList
    (used by MoE models like GLM-5.1).
    """
    if kv_quant_config is None:
        return cache
    k_bits, v_bits = kv_quant_config
    from .models.cache import KVCache, CacheList
    from .models.mixed_quant_cache import MixedQuantKVCache
    for i, c in enumerate(cache):
        if isinstance(c, MixedQuantKVCache):
            continue
        if isinstance(c, KVCache) and c.offset >= max(min_tokens, 1):
            cache[i] = MixedQuantKVCache.from_kvcache(
                c, k_bits=k_bits, v_bits=v_bits
            )
        elif isinstance(c, CacheList):
            new_subs = list(c.caches)
            changed = False
            for j, sub in enumerate(new_subs):
                if isinstance(sub, KVCache) and sub.offset >= max(min_tokens, 1):
                    new_subs[j] = MixedQuantKVCache.from_kvcache(
                        sub, k_bits=k_bits, v_bits=v_bits
                    )
                    changed = True
            if changed:
                c.caches = tuple(new_subs)
    return cache


def get_system_fingerprint():
    gpu_arch = mx.device_info()["architecture"]
    return f"{__version__}-{mx.__version__}-{platform.platform()}-{gpu_arch}"


class StopCondition(NamedTuple):
    stop_met: bool
    trim_length: int
    trim_text_length: int


def stopping_criteria(
    tokens: List[int],
    eos_token_ids: set,
    stop_id_sequences: List[List[int]],
    stop_words: List[str],
) -> StopCondition:
    """
    Determines whether the token generation should stop based on predefined
    conditions.

    Args:
        tokens (List[int]): The current sequence of generated tokens.
        eos_token_ids (set): The token IDs that represents the
          end-of-sequence. If the last token in ``tokens`` is in the set,
          the generation should stop.
        stop_id_sequences (List[List[[int]]): A list of integer lists, each
          representing a sequence of token IDs. If the end of the `tokens`
          list matches any of these sequences, the generation should stop.
        stop_words (List[str]): The stop words that correspond to the
            ``stop_id_sequences``.

    Returns:
        StopCondition: A named tuple indicating whether the stop condition has
          been met (`stop_met`) and how many tokens should be trimmed from the
          end if it has (`trim_length`) as well as the text that should be
          trimmed.
    """
    if tokens and tokens[-1] in eos_token_ids:
        return StopCondition(stop_met=True, trim_length=0, trim_text_length=0)

    for stop_ids, stop_word in zip(stop_id_sequences, stop_words):
        if len(tokens) >= len(stop_ids):
            if tokens[-len(stop_ids) :] == stop_ids:
                return StopCondition(
                    stop_met=True,
                    trim_length=len(stop_ids),
                    trim_text_length=len(stop_word),
                )

    return StopCondition(stop_met=False, trim_length=0, trim_text_length=0)


def sequence_overlap(s1: Sequence, s2: Sequence) -> bool:
    """
    Checks if a suffix of s1 has overlap with a prefix of s2

    Args:
        s1 (Sequence): The first sequence
        s2 (Sequence): The second sequence

    Returns:
        bool: If the two sequences have overlap
    """
    max_overlap = min(len(s1), len(s2))
    return any(s1[-i:] == s2[:i] for i in range(1, max_overlap + 1))


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
    role_mapping = role_mapping if role_mapping is not None else default_role_mapping

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
        content = message.get("content", None)
        if isinstance(content, list):
            text_fragments = [
                fragment["text"] for fragment in content if fragment["type"] == "text"
            ]
            if len(text_fragments) != len(content):
                raise ValueError("Only 'text' content type is supported.")
            message["content"] = "".join(text_fragments)
        elif content is None:
            message["content"] = ""
        if tool_calls := message.get("tool_calls", False):
            for tool_call in tool_calls:
                if func := tool_call.get("function", False):
                    if args := func.get("arguments", False):
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
    tool_call_start: str
    tool_call_end: str
    tool_parser: Callable[[str, Any], Dict]
    has_thinking: bool
    think_start_id: int
    think_end_id: int
    think_end: str
    eos_token_ids: set
    stop_token_sequences: List[List[int]]
    prompt: List[int]
    prompt_cache_count: int = -1

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
        if self._current_iterations > self._iterations:
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

        if self.draft_model is None and not getattr(self.cli_args, "no_batch", False):
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


def _format_top_logprobs(logprobs, top_logprobs, tokenizer) -> Tuple[Dict[str, Any]]:
    """Returns info dicts for the top `top_logprobs` tokens from `logprobs`"""
    if top_logprobs <= 0:
        return ()
    sorted_indices = mx.argpartition(-logprobs, kth=top_logprobs - 1)
    top_indices = sorted_indices[:top_logprobs].tolist()
    top_logprobs = logprobs[top_indices].tolist()
    txts = tokenizer.convert_ids_to_tokens(top_indices)
    return tuple(
        {"id": i, "token": s, "logprob": g}
        for i, s, g in zip(top_indices, txts, top_logprobs)
    )


class ResponseGenerator:
    def __init__(self, model_provider: ModelProvider, prompt_cache: LRUPromptCache):
        self.model_provider = model_provider
        self.prompt_cache = prompt_cache
        self.requests = Queue()
        self._kv_quant_config = getattr(
            model_provider.cli_args, "kv_quant_config", None
        )
        self._kv_quant_start = getattr(
            model_provider.cli_args, "quantized_kv_start", 0
        )

        self._time_budget = TimeBudget()
        self._is_distributed = mx.distributed.init().size() > 1
        self._rank = mx.distributed.init().rank()
        self._stop = False
        # Generation runs on the main thread (see run()).
        # MLX streams are thread-affine; async_eval requires the
        # stream that was active when the model was loaded.

    def _store_cache(self, model_key, cache_key, cache, **kwargs):
        """Optionally quantize KV cache before inserting into LRU."""
        cache = _maybe_quantize_cache(
            cache, self._kv_quant_config, min_tokens=self._kv_quant_start
        )
        self.prompt_cache.insert_cache(model_key, cache_key, cache, **kwargs)

    def stop_and_join(self):
        self._stop = True

    def join(self):
        pass

    def _log_cache_stats(self):
        ncaches = len(self.prompt_cache)
        nbytes = self.prompt_cache.nbytes
        logging.info(f"KV Caches: {ncaches} seq, {nbytes / 1e9:.2f} GB")

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
                else:
                    data = mx.array(pickle.dumps(obj))
                    mx.eval(mx.distributed.all_sum(data.size))
                    mx.eval(mx.distributed.all_sum(data))
                    return obj
            else:
                size = mx.distributed.all_sum(0).item()
                if size == 0:
                    return None
                else:
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
                return tokenizer.apply_chat_template(
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=True,
                    **chat_template_args,
                )
            else:
                return tokenizer.encode(convert_chat(messages, role_mapping))
        else:
            return tokenizer.encode(request.prompt)

    def _compute_prompt_checkpoint(self, tokenizer, request, prompt):
        if request.request_type != "chat":
            return False, -1
        if request.messages[-1]["role"] != "user":
            return False, -1

        # Save the KV cache at the end of the prompt just before
        # the think start token which will likely be removed in the
        # next turn.
        prompt_checkpoint = -1
        if tokenizer.has_thinking:
            for i in range(1, min(11, len(prompt)) - 1, 1):
                if prompt[-i] == tokenizer.think_start_id:
                    prompt_checkpoint = -i - 1
                    break

        return True, prompt_checkpoint

    def _is_batchable(self, args):
        if not self.model_provider.is_batchable:
            return False
        if args.seed is not None:
            return False

        return True

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

        def progress_callback(info):
            for uid, processed, total in info:
                if uid in batch_results:
                    batch_results[uid]["rqueue"].put((min(processed, total), total))

        def checkpoint_callback(prompts):
            for uid, prompt_end, cache in prompts:
                rs = batch_results[uid]
                if not rs["checkpoint"]:
                    continue
                # cache is a generator -- materialize before checking,
                # extract() can crash on uninitialized sub-caches
                try:
                    cache = list(cache)
                except (TypeError, AttributeError) as e:
                    logging.debug(f"Skipping checkpoint: {e}")
                    continue
                if any(c.empty() for c in cache):
                    continue
                self._store_cache(
                    current_model_key,
                    rs["cache_key"][:-prompt_end],
                    cache,
                    cache_type="user",
                )

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
                        prompt = self._tokenize(current_tokenizer, request, args)
                    except Exception as e:
                        rqueue.put(e)
                        continue

                    ctx = GenerationContext(
                        has_tool_calling=tokenizer.has_tool_calling,
                        tool_call_start=tokenizer.tool_call_start,
                        tool_call_end=tokenizer.tool_call_end,
                        tool_parser=tokenizer.tool_parser,
                        has_thinking=tokenizer.has_thinking,
                        think_start_id=tokenizer.think_start_id,
                        think_end=tokenizer.think_end,
                        think_end_id=tokenizer.think_end_id,
                        eos_token_ids=tokenizer.eos_token_ids,
                        stop_token_sequences=[
                            tokenizer.encode(stop_word, add_special_tokens=False)
                            for stop_word in args.stop_words
                        ],
                        prompt=prompt,
                    )
                    rqueue.put(ctx)

                    self._log_cache_stats()
                    cache, rest = self.prompt_cache.fetch_nearest_cache(
                        current_model_key, prompt
                    )
                    ctx.prompt_cache_count = len(prompt) - len(rest)
                    if cache is None or not cache:
                        cache = make_prompt_cache(self.model_provider.model)
                    elif self._kv_quant_config is not None:
                        # Dequantize for batch merge compatibility
                        cache = _maybe_dequantize_cache(cache)

                    do_checkpoint, checkpoint_position = (
                        self._compute_prompt_checkpoint(tokenizer, request, prompt)
                    )

                    (uid,) = batch_generator.insert(
                        [rest],
                        args.max_tokens,
                        caches=[cache],
                        samplers=[_make_sampler(args, tokenizer)],
                        logits_processors=[_make_logits_processors(args)],
                        prompt_checkpoints=[checkpoint_position],
                    )
                    batch_results[uid] = {
                        "ctx": ctx,
                        "cache_key": prompt[:],
                        "rqueue": rqueue,
                        "detokenizer": tokenizer.detokenizer,
                        "checkpoint": do_checkpoint,
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
                        stop_tokens=tokenizer.eos_token_ids,
                        completion_batch_size=self.cli_args.decode_concurrency,
                        prefill_batch_size=self.cli_args.prompt_concurrency,
                        prefill_step_size=self.cli_args.prefill_step_size,
                        prompt_progress_callback=progress_callback,
                        prompt_checkpoint_callback=checkpoint_callback,
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
                    responses = batch_generator.next()
                    if not responses:
                        break

                    for r in responses:
                        result = batch_results[r.uid]
                        result["cache_key"].append(r.token)
                        if r.finish_reason != "stop":
                            result["detokenizer"].add_token(r.token)

                        result["rqueue"].put(
                            Response(
                                result["detokenizer"].last_segment,
                                r.token,
                                r.logprobs[r.token].item(),
                                r.finish_reason,
                                _format_top_logprobs(
                                    r.logprobs, args.top_logprobs, current_tokenizer
                                ),
                            )
                        )

                        if r.finish_reason is not None:
                            result["rqueue"].put(None)
                            self._store_cache(
                                current_model_key, result["cache_key"], r.prompt_cache
                            )
                            del batch_results[r.uid]

                        if result["ctx"]._should_stop:
                            uids_to_remove.append(r.uid)

                uids_to_remove = self._share_object(uids_to_remove)
                if uids_to_remove:
                    with mx.stream(generation_stream):
                        caches = batch_generator.remove(
                            uids_to_remove, return_prompt_caches=True
                        )
                        for uid, prompt_cache in caches.items():
                            if uid not in batch_results:
                                continue
                            result = batch_results[uid]
                            self._store_cache(
                                current_model_key, result["cache_key"], prompt_cache
                            )
                            del batch_results[uid]

    def _serve_single(self, request):
        rqueue, request, args = request

        # Progress state shared between callback and try block
        _checkpoint_state = {"last": 0, "ctx": None, "cache_key": None, "cache": None}
        _checkpoint_interval = 32768  # save checkpoint every 32K tokens

        def progress(tokens_processed, tokens_total):
            rqueue.put((tokens_processed, tokens_total))
            s = _checkpoint_state
            if s["cache"] is None or tokens_processed <= 0:
                return
            n_cached = s["ctx"].prompt_cache_count + tokens_processed
            # Save at the END of prefill -- cache offset matches key length
            # exactly (no off-by-one from generated tokens).
            # Also save periodic checkpoints during long prefills.
            should_save = (
                tokens_processed == tokens_total  # prefill complete
                or (
                    tokens_processed - s["last"] >= _checkpoint_interval
                    and tokens_processed < tokens_total
                )
            )
            if should_save:
                s["last"] = tokens_processed
                try:
                    self.prompt_cache.insert_cache(
                        self.model_provider.model_key,
                        s["cache_key"][:n_cached],
                        s["cache"],
                        cache_type="user",
                    )
                    logging.info(f"Prefill checkpoint saved: {n_cached} tokens")
                except Exception as e:
                    logging.debug(f"Checkpoint save failed: {e}")

        try:
            # Load the model and tokenizer
            model = self.model_provider.model
            tokenizer = self.model_provider.tokenizer
            draft_model = self.model_provider.draft_model

            # Prepare the prompt
            prompt = self._tokenize(tokenizer, request, args)

            # Start the generation context
            ctx = GenerationContext(
                has_tool_calling=tokenizer.has_tool_calling,
                tool_call_start=tokenizer.tool_call_start,
                tool_call_end=tokenizer.tool_call_end,
                tool_parser=tokenizer.tool_parser,
                has_thinking=tokenizer.has_thinking,
                think_start_id=tokenizer.think_start_id,
                think_end=tokenizer.think_end,
                think_end_id=tokenizer.think_end_id,
                eos_token_ids=tokenizer.eos_token_ids,
                stop_token_sequences=[
                    tokenizer.encode(stop_word, add_special_tokens=False)
                    for stop_word in args.stop_words
                ],
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
            elif self._kv_quant_config is not None:
                # Dequantize for stream_generate compatibility (single-serve
                # path doesn't support quantized cache in generate_step)
                cache = _maybe_dequantize_cache(cache)

            # Enable prefill checkpoints now that cache/ctx are ready
            _checkpoint_state["ctx"] = ctx
            _checkpoint_state["cache_key"] = cache_key
            _checkpoint_state["cache"] = cache

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
                rqueue.put(
                    Response(
                        gen.text,
                        gen.token,
                        gen.logprobs[gen.token].item(),
                        gen.finish_reason,
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

            rqueue.put(None)

            # Save the KV cache again
            self._store_cache(
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

        ctx = response_queue.get()
        if isinstance(ctx, Exception):
            raise ctx

        return ctx, _inner()

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

        indent = "\t"  # Backslashes can't be inside of f-strings
        logging.debug(f"Incoming Request Body: {json.dumps(self.body, indent=indent)}")
        assert isinstance(
            self.body, dict
        ), f"Request should be dict, but got {type(self.body)}"

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

    def validate_model_parameters(self):
        """
        Validate the model parameters passed in the request for the correct types and values.
        """
        if not isinstance(self.stream, bool):
            raise ValueError("stream must be a boolean")

        if not isinstance(self.max_tokens, int) or self.max_tokens < 0:
            raise ValueError("max_tokens must be a non-negative integer")

        if not isinstance(self.temperature, (float, int)) or self.temperature < 0:
            raise ValueError("temperature must be a non-negative float")

        if not isinstance(self.top_p, (float, int)) or self.top_p < 0 or self.top_p > 1:
            raise ValueError("top_p must be a float between 0 and 1")

        if not isinstance(self.top_k, int) or self.top_k < 0:
            raise ValueError("top_k must be a non-negative integer")

        if not isinstance(self.min_p, (float, int)) or self.min_p < 0 or self.min_p > 1:
            raise ValueError("min_p must be a float between 0 and 1")

        if not isinstance(self.num_draft_tokens, int) or self.num_draft_tokens < 0:
            raise ValueError("num_draft_tokens must be a non-negative integer")

        if (
            not isinstance(self.repetition_penalty, (float, int))
            or self.repetition_penalty < 0
        ):
            raise ValueError("repetition_penalty must be a non-negative float")
        if (
            not isinstance(self.repetition_context_size, int)
            or self.repetition_context_size < 0
        ):
            raise ValueError("repetition_context_size must be a non-negative integer")
        if not isinstance(self.presence_penalty, (float, int)):
            raise ValueError("Presence penalty must be must be a float")
        if (
            not isinstance(self.presence_context_size, int)
            or self.presence_context_size < 0
        ):
            raise ValueError("presence_context_size must be a non-negative integer")
        if not isinstance(self.frequency_penalty, (float, int)):
            raise ValueError("Presence penalty must be must be a float")
        if (
            not isinstance(self.frequency_context_size, int)
            or self.frequency_context_size < 0
        ):
            raise ValueError("frequency_context_size must be a non-negative integer")

        if not isinstance(self.logprobs, bool):
            raise ValueError("logprobs must be a boolean")

        if self.top_logprobs != -1 and not (0 < self.top_logprobs <= 10):
            raise ValueError(
                f"top_logprobs must be between 1 and 10 but got {self.top_logprobs:,}"
            )

        if self.logit_bias is not None:
            if not isinstance(self.logit_bias, dict):
                raise ValueError("logit_bias must be a dict of int to float")

            try:
                self.logit_bias = {int(k): v for k, v in self.logit_bias.items()}
            except ValueError:
                raise ValueError("logit_bias must be a dict of int to float")
        if not (
            isinstance(self.xtc_probability, float)
            and 0.00 <= self.xtc_probability <= 1.00
        ):
            raise ValueError(f"xtc_probability must be a float between 0.00 and 1.00")
        if not (
            isinstance(self.xtc_threshold, float) and 0.00 <= self.xtc_threshold <= 0.50
        ):
            raise ValueError(f"xtc_threshold must be a float between 0.00 and 0.5")
        if not isinstance(self.requested_model, str):
            raise ValueError("model must be a string")
        if self.adapter is not None and not isinstance(self.adapter, str):
            raise ValueError("adapter must be a string")
        if self.seed is not None and not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")

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
            choice[key_name] = {
                "role": "assistant",
                "content": text,
                "reasoning": reasoning_text,
                "tool_calls": tool_calls,
            }
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
            stop_words (List[str]): A list of stop words passed to the
                stopping_criteria function
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

        # Create keepalive callback to send SSE comments during long prompt processing
        def keepalive_callback(processed_tokens, total_tokens):
            logging.info(
                f"Prompt processing progress: {processed_tokens}/{total_tokens}"
            )
            if self.stream:
                try:
                    # Send SSE comment for keepalive - invisible to clients but keeps connection alive
                    self.wfile.write(
                        f": keepalive {processed_tokens}/{total_tokens}\n\n".encode()
                    )
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    # Client disconnected, ignore
                    pass

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
            self.wfile.write(json.dumps({"error": f"{e}"}).encode())
            return

        # Prepare the headers
        if self.stream:
            self._set_stream_headers(200)
            self.end_headers()
            logging.debug("Starting stream:")
        else:
            self._set_completion_headers(200)
            logging.debug("Starting completion:")

        # Variables to save the tool calls in as they are being generated by
        # the model.
        in_tool_call = False
        made_tool_call = False
        tool_calls = []
        tool_text = ""
        tool_idx = 0

        def format_tool_call(tool_call):
            nonlocal tool_idx
            tool_call_id = tool_call.pop("id", None) or str(uuid.uuid4())
            tool_call["arguments"] = json.dumps(
                tool_call["arguments"], ensure_ascii=False
            )
            out = {
                "function": tool_call,
                "type": "function",
                "id": tool_call_id,
            }
            if self.stream:
                out["index"] = tool_idx
                tool_idx += 1
            return out

        def parse_tools(tool_calls):
            if not tool_calls:
                return []
            result = []
            for tool_text in tool_calls:
                parsed = ctx.tool_parser(tool_text, request.tools)
                if isinstance(parsed, list):
                    result.extend(format_tool_call(tc) for tc in parsed)
                else:
                    result.append(format_tool_call(parsed))
            return result

        # Start out in reasoning if the model is a reasoning model and the
        # prompt has an open think token but no closing think token
        in_reasoning = False
        if ctx.has_thinking:
            for i in range(len(ctx.prompt) - 1, -1, -1):
                if ctx.prompt[i] == ctx.think_end_id:
                    break
                elif ctx.prompt[i] == ctx.think_start_id:
                    in_reasoning = True
                    break
        reasoning_text = ""

        # Variables to save the generated tokens and the corresponding probs
        tokens = []
        token_logprobs = []
        top_tokens = []

        # Variables to save the generated text
        text = ""
        segment = ""

        # Well finally save the reason for stopping
        finish_reason = "length"
        # Process the generated tokens
        for gen in response:
            logging.debug(gen.text)

            # Gather the text in tool calling or text variables
            if in_reasoning:
                if gen.text == ctx.think_end:
                    in_reasoning = False
                else:
                    reasoning_text += gen.text
            elif ctx.has_tool_calling and gen.text == ctx.tool_call_start:
                made_tool_call = True
                in_tool_call = True
            elif in_tool_call:
                if gen.text == ctx.tool_call_end:
                    tool_calls.append(tool_text)
                    tool_text = ""
                    in_tool_call = False
                else:
                    tool_text += gen.text
            else:
                text += gen.text
                segment += gen.text

            # Save the token and its logprob
            tokens.append(gen.token)
            if args.logprobs:
                token_logprobs.append(gen.logprob)

            # If requested save the k top logprobs
            if args.top_logprobs > 0:
                top_tokens.append(gen.top_tokens)

            # Check if we should stop early
            stop_condition = stopping_criteria(
                tokens,
                ctx.eos_token_ids,
                ctx.stop_token_sequences,
                stop_words,
            )
            if stop_condition.stop_met:
                finish_reason = "tool_calls" if made_tool_call else "stop"
                ctx.stop()
                tokens = tokens[: len(tokens) - stop_condition.trim_length]
                text = text[: len(text) - stop_condition.trim_text_length]
                segment = ""
                break

            if self.stream and not in_tool_call:
                # If the end of tokens overlaps with a stop sequence, generate new
                # tokens until we know if the stop sequence is hit or not
                if any(
                    (
                        sequence_overlap(tokens, sequence)
                        for sequence in ctx.stop_token_sequences
                    )
                ):
                    continue
                elif segment or tool_calls or reasoning_text:
                    response = self.generate_response(
                        segment,
                        None,
                        tool_calls=parse_tools(tool_calls),
                        reasoning_text=reasoning_text,
                    )
                    self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())
                    self.wfile.flush()
                    reasoning_text = ""
                    segment = ""
                    tool_calls = []

            if gen.finish_reason is not None:
                finish_reason = gen.finish_reason

        # Flush any remaining tool text (e.g. when tool_call_end is empty)
        if in_tool_call and tool_text:
            tool_calls.append(tool_text)

        if self.stream:
            response = self.generate_response(
                segment,
                finish_reason,
                tool_calls=parse_tools(tool_calls),
                reasoning_text=reasoning_text,
            )
            self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())
            self.wfile.flush()
            if self.stream_options is not None and self.stream_options["include_usage"]:
                response = self.completion_usage_response(
                    len(ctx.prompt),
                    len(tokens),
                    ctx.prompt_cache_count,
                )
                self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())
                self.wfile.flush()
            self.wfile.write("data: [DONE]\n\n".encode())
            self.wfile.flush()
        else:
            response = self.generate_response(
                text,
                finish_reason,
                len(ctx.prompt),
                len(tokens),
                ctx.prompt_cache_count,
                token_logprobs=token_logprobs,
                top_tokens=top_tokens,
                tokens=tokens,
                reasoning_text=reasoning_text,
                tool_calls=parse_tools(tool_calls),
            )
            response_json = json.dumps(response).encode()
            indent = "\t"  # Backslashes can't be inside of f-strings
            logging.debug(f"Outgoing Response: {json.dumps(response, indent=indent)}")

            # Send an additional Content-Length header when it is known
            self.send_header("Content-Length", str(len(response_json)))
            self.end_headers()
            self.wfile.write(response_json)
            self.wfile.flush()

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
    cache_dir = getattr(model_provider.cli_args, "prompt_cache_dir", None)
    if cache_dir:
        from .disk_cache import DiskBackedPromptCache
        prompt_cache = DiskBackedPromptCache(
            max_size=model_provider.cli_args.prompt_cache_size,
            cache_dir=cache_dir,
        )
    else:
        prompt_cache = LRUPromptCache(model_provider.cli_args.prompt_cache_size)
    response_generator = ResponseGenerator(model_provider, prompt_cache)
    if group.rank() == 0:
        # Start HTTP server in a background thread.
        # Generation MUST run on the main thread because MLX streams
        # are thread-affine: async_eval requires the stream that was
        # active when the model was loaded (stream 0, main thread).
        http_thread = Thread(
            target=_run_http_server,
            args=(host, port, response_generator),
            daemon=True,
        )
        http_thread.start()
        try:
            response_generator._generate()
        except KeyboardInterrupt:
            logging.info("Shutting down...")
        finally:
            response_generator._stop = True
    else:
        try:
            response_generator._generate()
        except KeyboardInterrupt:
            pass


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
        "--no-batch",
        action="store_true",
        help="Disable batch processing, use single-serve mode for all requests",
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
        "--kv-cache-quantization",
        type=str,
        default=None,
        metavar="K_BITS,V_BITS",
        help="Quantize cached KV for memory savings, e.g. '8,4' for K@8-bit V@4-bit. "
        "Converts caches before storing in the prompt cache LRU. "
        "Saves ~20%% memory at 32K context with K8,V4.",
    )
    parser.add_argument(
        "--quantized-kv-start",
        type=int,
        default=0,
        help="Minimum number of tokens before quantizing a KV cache "
        "(default: 0, quantize all). Short caches below this threshold "
        "stay in fp16.",
    )
    parser.add_argument(
        "--prompt-cache-dir",
        type=str,
        default=None,
        help="Directory to persist prompt caches to disk. Survives server "
        "restarts — cached prompts are restored from disk on cache miss. "
        "Evicted caches are also saved here.",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Use pipelining instead of tensor parallelism",
    )
    args = parser.parse_args()

    # Parse --kv-cache-quantization into a (k_bits, v_bits) tuple
    if args.kv_cache_quantization is not None:
        parts = args.kv_cache_quantization.split(",")
        if len(parts) != 2:
            parser.error("--kv-cache-quantization must be K_BITS,V_BITS (e.g. '8,4')")
        try:
            k_bits, v_bits = int(parts[0]), int(parts[1])
        except ValueError:
            parser.error("--kv-cache-quantization values must be integers")
        valid_bits = {2, 4, 8}
        if k_bits not in valid_bits or v_bits not in valid_bits:
            parser.error(f"--kv-cache-quantization bits must be one of {valid_bits}")
        args.kv_quant_config = (k_bits, v_bits)
    else:
        args.kv_quant_config = None

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
