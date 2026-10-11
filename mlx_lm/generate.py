# Copyright © 2023 Apple Inc.

import argparse
import contextlib
import copy
import functools
import json
import math
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Callable, Generator, List, Optional, Sequence, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_reduce
from transformers import PreTrainedTokenizer

from .generate_utils import BatchCounters, BatchCountersSnapshot, BatchStats
from .models.cache import (
    KVCache,
    QuantizedKVCache,
    QuantizedRotatingKVCache,
    RotatingKVCache,
    TokenBuffer,
    can_trim_prompt_cache,
    load_prompt_cache,
    make_prompt_cache,
    trim_prompt_cache,
)
from .sample_utils import LogitsProcessor, Sampler, greedy_sampler, make_sampler
from .tokenizer_utils import TokenizerWrapper
from .utils import (
    does_model_support_input_embeddings,
    load,
    maybe_set_recommended_wired_limit,
)

DEFAULT_PROMPT = "hello"
DEFAULT_MAX_TOKENS = 100
DEFAULT_TEMP = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_MIN_P = 0.0
DEFAULT_TOP_K = 0
DEFAULT_XTC_PROBABILITY = 0.0
DEFAULT_XTC_THRESHOLD = 0.1
DEFAULT_MIN_TOKENS_TO_KEEP = 1
DEFAULT_SEED = None
DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
DEFAULT_QUANTIZED_KV_START = 5000
DEFAULT_PREFILL_STEP_SIZE = 2048


def str2bool(string):
    return string.lower() not in ["false", "f"]


def setup_arg_parser():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(description="LLM inference script")
    parser.add_argument(
        "--model",
        type=str,
        help=(
            "The path to the local model directory or Hugging Face repo. "
            f"If no model is specified, then {DEFAULT_MODEL} is used."
        ),
        default=None,
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Optional path for the trained adapter weights and config.",
    )
    parser.add_argument(
        "--extra-eos-token",
        type=str,
        default=(),
        nargs="+",
        help="Add tokens in the list of eos tokens that stop generation.",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="System prompt to be used for the chat template",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        default=DEFAULT_PROMPT,
        help="Message to be processed by the model ('-' reads from stdin)",
    )
    parser.add_argument(
        "--prefill-response",
        default=None,
        help="Prefill response to be used for the chat template",
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temp", type=float, default=DEFAULT_TEMP, help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p", type=float, default=DEFAULT_TOP_P, help="Sampling top-p"
    )
    parser.add_argument(
        "--min-p", type=float, default=DEFAULT_MIN_P, help="Sampling min-p"
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K, help="Sampling top-k"
    )
    parser.add_argument(
        "--xtc-probability",
        type=float,
        default=DEFAULT_XTC_PROBABILITY,
        help="Probability of XTC sampling to happen each next token",
    )
    parser.add_argument(
        "--xtc-threshold",
        type=float,
        default=DEFAULT_XTC_THRESHOLD,
        help="Threshold the probs of each next token candidate to be sampled by XTC",
    )
    parser.add_argument(
        "--min-tokens-to-keep",
        type=int,
        default=DEFAULT_MIN_TOKENS_TO_KEEP,
        help="Minimum tokens to keep for min-p sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="PRNG seed",
    )
    parser.add_argument(
        "--ignore-chat-template",
        action="store_true",
        help="Use the raw prompt without the tokenizer's chat template.",
    )
    parser.add_argument(
        "--use-default-chat-template",
        action="store_true",
        help="Use the default chat template",
    )
    parser.add_argument(
        "--chat-template-config",
        help="Additional config for `apply_chat_template`. Should be a dictionary of"
        " string keys to values represented as a JSON decodable string.",
        default=None,
    )
    parser.add_argument(
        "--verbose",
        type=str2bool,
        default=True,
        help="Log verbose output when 'True' or 'T' or only print the response when 'False' or 'F'",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        help="Set the maximum key-value cache size",
        default=None,
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=DEFAULT_PREFILL_STEP_SIZE,
        help="Number of prompt tokens to process at a time. Smaller values "
        f"lower peak memory during prefill (default: {DEFAULT_PREFILL_STEP_SIZE})",
    )
    parser.add_argument(
        "--prompt-cache-file",
        type=str,
        default=None,
        help="A file containing saved KV caches to avoid recomputing them",
    )
    parser.add_argument(
        "--quantize-activations",
        "-qa",
        action="store_true",
        help="Quantize activations using the same quantization config as the corresponding layer.",
    )
    parser.add_argument(
        "--kv-bits",
        type=int,
        help="Number of bits for KV cache quantization. Defaults to no quantization.",
        default=None,
    )
    parser.add_argument(
        "--kv-group-size",
        type=int,
        help="Group size for KV cache quantization.",
        default=64,
    )
    parser.add_argument(
        "--quantized-kv-start",
        help="When --kv-bits is set, start quantizing the KV cache "
        "from this step onwards.",
        type=int,
        default=DEFAULT_QUANTIZED_KV_START,
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
    return parser


# A stream on the default device just for generation
generation_stream = mx.new_thread_local_stream(mx.default_device())


@contextlib.contextmanager
def wired_limit(model: nn.Module, streams: Optional[List[mx.Stream]] = None):
    """
    A context manager to temporarily change the wired limit.

    Note, the wired limit should not be changed during an async eval.  If an
    async eval could be running pass in the streams to synchronize with prior
    to exiting the context manager.
    """
    old_limit = maybe_set_recommended_wired_limit()
    if old_limit is None:
        yield
        return

    model_bytes = tree_reduce(
        lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc, model, 0
    )
    max_rec_size = mx.device_info()["max_recommended_working_set_size"]
    if model_bytes > 0.9 * max_rec_size:
        model_mb = model_bytes // 2**20
        max_rec_mb = max_rec_size // 2**20
        print(
            f"[WARNING] Generating with a model that requires {model_mb} MB "
            f"which is close to the maximum recommended size of {max_rec_mb} "
            "MB. This can be slow. See the documentation for possible work-arounds: "
            "https://github.com/ml-explore/mlx-lm/tree/main#large-models"
        )
    try:
        yield
    finally:
        if streams is not None:
            for s in streams:
                mx.synchronize(s)
        else:
            mx.synchronize()
        mx.set_wired_limit(old_limit)


@dataclass
class GenerationResponse:
    """
    The output of :func:`stream_generate`.

    Args:
        text (str): The next segment of decoded text. This can be an empty string.
        token (int): The next token.
        from_draft (bool): Whether the token was generated by the draft model.
        logprobs (mx.array): A vector of log probabilities.
        prompt_tokens (int): The number of tokens in the prompt.
        prompt_tps (float): The prompt processing tokens-per-second.
        generation_tokens (int): The number of generated tokens.
        generation_tps (float): The tokens-per-second for generation.
        peak_memory (float): The peak memory used so far in GB.
        finish_reason (str): The reason the response is being sent: "length", "stop" or `None`
    """

    text: str
    token: int
    logprobs: mx.array
    from_draft: bool
    prompt_tokens: int
    prompt_tps: float
    generation_tokens: int
    generation_tps: float
    peak_memory: float
    finish_reason: Optional[str] = None


MIN_PREFILL_STEP = 16
DEFAULT_PREFILL_SCORE_BYTES = 4


@dataclass
class PrefillBudget:
    """Memory budget for the attention scores of one prefill chunk.

    Args:
        nbytes: bytes for the scores array of one layer.
        min_step: the step never goes below this (or the max step).
        score_bytes: bytes per score element assumed (4: fp32, safe).
    """

    nbytes: int
    min_step: int = MIN_PREFILL_STEP
    score_bytes: int = DEFAULT_PREFILL_SCORE_BYTES

# Head dims mx.fast.scaled_dot_product_attention runs fused (no scores array)
# for a prefill chunk (query length > 8), MLX 0.32. 192 and 256 are sent to
# the unfused path in most cases.
_FUSED_PREFILL_HEAD_DIMS = (64, 72, 80, 96, 128)


def adaptive_prefill_step(
    max_step: int,
    offset: int,
    n_heads: int,
    budget_bytes: Optional[int],
    score_bytes: int = DEFAULT_PREFILL_SCORE_BYTES,
    min_step: int = MIN_PREFILL_STEP,
) -> int:
    """The largest prefill chunk whose attention scores
    (``n_heads x step x (offset + step)`` of ``score_bytes``, one array of a
    layer) fit in ``budget_bytes``, clamped to
    ``[min(min_step, max_step), max_step]``. The scores grow with the cache
    offset: a fixed step runs out of memory on a long prompt. A
    ``budget_bytes`` of ``None`` or 0, or ``n_heads`` 0, keeps ``max_step``.
    """
    if not budget_bytes or n_heads <= 0:
        return max_step
    limit = budget_bytes / (n_heads * score_bytes)
    # Positive root of step^2 + offset * step - limit = 0.
    step = int((math.sqrt(offset * offset + 4 * limit) - offset) / 2)
    return max(min(min_step, max_step), min(max_step, step))


def _text_args(model):
    for m in (model, getattr(model, "language_model", None)):
        args = getattr(m, "args", None)
        for a in (args, getattr(args, "text_config", None)):
            a = a if isinstance(a, dict) else getattr(a, "__dict__", {})
            if a.get("num_attention_heads"):
                return a
    return None


def attention_scores_heads(model: nn.Module) -> Tuple[int, bool]:
    """``(n_heads, fused)`` for the full-attention layers of ``model``:
    ``fused`` is True when the fused SDPA kernel serves their prefill (no
    scores array unless the KV cache is quantized). A model with no
    ``num_attention_heads`` in its config (e.g. pure SSM) gives ``(0, True)``:
    no step limit. A known head count with an unknown head dim counts as
    unfused."""
    a = _text_args(model)
    if a is None:
        return 0, True
    n_heads = a["num_attention_heads"]
    if isinstance(n_heads, (list, tuple)):
        n_heads = max(n_heads)
    if a.get("qk_nope_head_dim") is not None and a.get("qk_rope_head_dim") is not None:
        q_dim = a["qk_nope_head_dim"] + a["qk_rope_head_dim"]
    else:
        q_dim = a.get("global_head_dim") or a.get("head_dim")
        if not q_dim and a.get("hidden_size"):
            q_dim = a["hidden_size"] // n_heads
    v_dim = a.get("v_head_dim") or q_dim
    fused = q_dim == v_dim and q_dim in _FUSED_PREFILL_HEAD_DIMS
    return n_heads, fused


def kv_cache_quantized(cache) -> bool:
    return any(hasattr(c, "bits") for c in cache if c is not None)


def make_prefill_step(
    model: nn.Module,
    max_step: int,
    budget: Union[None, int, PrefillBudget],
) -> Callable[[int, bool], int]:
    """``step(offset, quantized)``: the prefill step for a chunk at
    ``offset``. It is limited by ``budget`` (bytes, or a
    :class:`PrefillBudget`) only where the scores are materialized: unfused
    head dims, or a quantized KV cache (quantized SDPA)."""
    if budget is not None and not isinstance(budget, PrefillBudget):
        budget = PrefillBudget(budget)
    n_heads, fused = attention_scores_heads(model)
    if budget is None or not budget.nbytes or n_heads == 0:
        return lambda offset, quantized: max_step

    def step(offset, quantized):
        if fused and not quantized:
            return max_step
        return adaptive_prefill_step(
            max_step,
            offset,
            n_heads,
            budget.nbytes,
            score_bytes=budget.score_bytes,
            min_step=budget.min_step,
        )

    return step


def _cache_offset(cache) -> int:
    return next(
        (c.offset for c in cache if getattr(c, "offset", None) is not None), 0
    )


def maybe_quantize_kv_cache(prompt_cache, quantized_kv_start, kv_group_size, kv_bits):
    if kv_bits is None:
        return
    for e, c in enumerate(prompt_cache):
        if hasattr(c, "to_quantized") and c.offset >= quantized_kv_start:
            prompt_cache[e] = c.to_quantized(group_size=kv_group_size, bits=kv_bits)


def generate_step(
    prompt: mx.array,
    model: nn.Module,
    stream: mx.Stream | mx.ThreadLocalStream = generation_stream,
    *,
    max_tokens: int = 256,
    sampler: Optional[Sampler] = None,
    logits_processors: Optional[List[LogitsProcessor]] = None,
    max_kv_size: Optional[int] = None,
    prompt_cache: Optional[Any] = None,
    prefill_step_size: int = 2048,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = DEFAULT_QUANTIZED_KV_START,
    prompt_progress_callback: Optional[Callable[[int, int], None]] = None,
    input_embeddings: Optional[mx.array] = None,
    prefill_memory_budget: Union[None, int, PrefillBudget] = None,
    token_history: Optional[List[int]] = None,
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    """
    A generator producing token ids based on the given prompt from the model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        max_tokens (int): The maximum number of tokens. Use``-1`` for an infinite
          generator. Default: ``256``.
        sampler (Sampler, optional): A sampler for sampling a
          token from a vector of log probabilities. Default: ``None``.
        logits_processors (List[LogitsProcessor], optional):
          A list of functions that take tokens and logits and return the processed
          logits. Default: ``None``.
        max_kv_size (int, optional): Maximum size of the key-value cache. Old
          entries (except the first 4 tokens) will be overwritten.
        prompt_cache (List[Any], optional): A pre-computed prompt cache. Note, if
          provided, the cache will be updated in place.
        prefill_step_size (int): Step size for processing the prompt.
        kv_bits (int, optional): Number of bits to use for KV cache quantization.
          None implies no cache quantization. Default: ``None``.
        kv_group_size (int): Group size for KV cache quantization. Default: ``64``.
        quantized_kv_start (int): Step to begin using a quantized KV cache.
           when ``kv_bits`` is non-None. Default: ``5000``.
        prompt_progress_callback (Callable[[int, int], None]): A call-back which takes the
           prompt tokens processed so far and the total number of prompt tokens.
        input_embeddings (mx.array, optional): Input embeddings to use instead of or in
          conjunction with prompt tokens. Default: ``None``.
        prefill_memory_budget (int or PrefillBudget, optional): Bytes for the
          attention scores of one prefill chunk: the step shrinks as the
          cache grows (see :func:`adaptive_prefill_step`). Default: ``None``
          (fixed step).
        token_history (List[int], optional): Tokens before ``prompt`` that
          ``prompt_cache`` already holds. The logits processors see them.

    Yields:
        Tuple[mx.array, mx.array]: One token and a vector of log probabilities.
    """
    if input_embeddings is not None:
        if not does_model_support_input_embeddings(model):
            raise ValueError("Model does not support input embeddings.")
        elif len(prompt) > 0 and len(prompt) != len(input_embeddings):
            raise ValueError(
                f"When providing input_embeddings, their sequence length ({len(input_embeddings)}) "
                f"must match the sequence length of the prompt ({len(prompt)}), or the "
                "prompt must be empty."
            )
    elif len(prompt) == 0:
        raise ValueError(
            "Either input_embeddings or prompt (or both) must be provided."
        )

    tokens = mx.array(token_history, prompt.dtype) if token_history else None

    # Create the KV cache for generation
    if prompt_cache is None:
        prompt_cache = make_prompt_cache(
            model,
            max_kv_size=max_kv_size,
        )

    prompt_progress_callback = prompt_progress_callback or (lambda *_: None)

    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    sampler = sampler or greedy_sampler

    def _model_call(input_tokens: mx.array, input_embeddings: Optional[mx.array]):
        if input_embeddings is not None:
            return model(
                input_tokens, cache=prompt_cache, input_embeddings=input_embeddings
            )
        else:
            return model(input_tokens, cache=prompt_cache)

    def _step(input_tokens: mx.array, input_embeddings: Optional[mx.array] = None):
        nonlocal tokens

        with mx.stream(stream):
            logits = _model_call(
                input_tokens=input_tokens[None],
                input_embeddings=(
                    input_embeddings[None] if input_embeddings is not None else None
                ),
            )

            logits = logits[:, -1, :]

            if logits_processors and len(input_tokens) > 0:
                tokens = (
                    mx.concat([tokens, input_tokens])
                    if tokens is not None
                    else input_tokens
                )
                for processor in logits_processors:
                    logits = processor(tokens, logits)

            quantize_cache_fn(prompt_cache)

            logprobs = logits - mx.logsumexp(logits, keepdims=True)
            sampled = sampler(logprobs)
            return sampled, logprobs.squeeze(0)

    with mx.stream(stream):
        total_prompt_tokens = (
            len(input_embeddings) if input_embeddings is not None else len(prompt)
        )
        prompt_processed_tokens = 0
        prompt_progress_callback(prompt_processed_tokens, total_prompt_tokens)
        # A model can keep a chunk from splitting a span that must be
        # processed in one pass (Gemma 4: an image's tokens attend to each
        # other in both directions). Positions count what the cache holds.
        chunk_size = getattr(model, "prefill_chunk_size", None)
        cached = _cache_offset(prompt_cache)
        prefill_step = make_prefill_step(
            model, prefill_step_size, prefill_memory_budget
        )
        while total_prompt_tokens - prompt_processed_tokens > 1:
            remaining = (total_prompt_tokens - prompt_processed_tokens) - 1
            step = prefill_step(
                cached + prompt_processed_tokens, kv_cache_quantized(prompt_cache)
            )
            n_to_process = min(step, remaining)
            if chunk_size is not None:
                n_to_process = max(
                    1, min(chunk_size(cached + prompt_processed_tokens, n_to_process), remaining)
                )
            processed = prompt[:n_to_process]
            _model_call(
                input_tokens=processed[None],
                input_embeddings=(
                    input_embeddings[:n_to_process][None]
                    if input_embeddings is not None
                    else None
                ),
            )
            quantize_cache_fn(prompt_cache)
            mx.eval([c.state for c in prompt_cache])
            if logits_processors and len(processed) > 0:
                tokens = (
                    mx.concat([tokens, processed]) if tokens is not None else processed
                )
            prompt_processed_tokens += n_to_process
            prompt_progress_callback(prompt_processed_tokens, total_prompt_tokens)
            prompt = prompt[n_to_process:]
            input_embeddings = (
                input_embeddings[n_to_process:]
                if input_embeddings is not None
                else input_embeddings
            )
            mx.clear_cache()

        y, logprobs = _step(input_tokens=prompt, input_embeddings=input_embeddings)
        mx.async_eval(y, logprobs)

        n = 0
        while True:
            if n != max_tokens:
                next_y, next_logprobs = _step(y)
                mx.async_eval(next_y, next_logprobs)
            if n == 0:
                mx.eval(y)
                prompt_progress_callback(total_prompt_tokens, total_prompt_tokens)
            if n == max_tokens:
                break
            yield y.item(), logprobs
            if n % 256 == 0:
                mx.clear_cache()
            y, logprobs = next_y, next_logprobs
            n += 1


def speculative_generate_step(
    prompt: mx.array,
    model: nn.Module,
    draft_model: nn.Module,
    stream: mx.Stream | mx.ThreadLocalStream = generation_stream,
    *,
    num_draft_tokens: int = 2,
    max_tokens: int = 256,
    sampler: Optional[Sampler] = None,
    logits_processors: Optional[List[LogitsProcessor]] = None,
    prompt_cache: Optional[Any] = None,
    prefill_step_size: int = 512,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = DEFAULT_QUANTIZED_KV_START,
    prefill_memory_budget: Union[None, int, PrefillBudget] = None,
) -> Generator[Tuple[mx.array, mx.array, bool], None, None]:
    """
    A generator producing token ids based on the given prompt from the model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        draft_model (nn.Module): The draft model for speculative decoding.
        num_draft_tokens (int, optional): The number of draft tokens for
          speculative decoding. Default: ``2``.
        max_tokens (int): The maximum number of tokens. Use``-1`` for an infinite
          generator. Default: ``256``.
        sampler (Sampler, optional): A sampler for sampling a
          token from a vector of log probabilities. Default: ``None``.
        logits_processors (List[LogitsProcessor], optional):
          A list of functions that take tokens and logits and return the processed
          logits. Default: ``None``.
        prompt_cache (List[Any], optional): A pre-computed prompt cache. Note, if
          provided, the cache will be updated in place. The cache must be trimmable.
        prefill_step_size (int): Step size for processing the prompt.
        kv_bits (int, optional): Number of bits to use for KV cache quantization.
          None implies no cache quantization. Default: ``None``.
        kv_group_size (int): Group size for KV cache quantization. Default: ``64``.
        quantized_kv_start (int): Step to begin using a quantized KV cache.
           when ``kv_bits`` is non-None. Default: ``5000``.

    Yields:
        Tuple[mx.array, mx.array, bool]: One token, a vector of log probabilities,
          and a bool indicating if the token was generated by the draft model
    """

    y = prompt.astype(mx.uint32)
    prev_tokens = None

    # Create the KV cache for generation
    if prompt_cache is None:
        model_cache = make_prompt_cache(model)
        draft_cache = make_prompt_cache(draft_model)
    else:
        # Split by the model's own cache length (KV-shared layers have no
        # cache of their own: fewer entries than layers).
        n_main = len(make_prompt_cache(model))
        model_cache = prompt_cache[:n_main]
        draft_cache = prompt_cache[n_main:]

    if not can_trim_prompt_cache(model_cache):
        types = {type(c).__name__ for c in model_cache if not c.is_trimmable()}
        raise ValueError(
            f"Speculative decoding requires a trimmable prompt cache " f"(got {types})."
        )

    sampler = sampler or greedy_sampler

    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    def _process_and_sample(tokens, logits):
        if logits_processors:
            for processor in logits_processors:
                logits = processor(tokens, logits)

        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        y = sampler(logprobs)
        return y, logprobs

    def _step(model, cache, y, n_predict=1):
        logits = model(y[None], cache=cache)
        logits = logits[:, -n_predict:, :]

        quantize_cache_fn(cache)
        if logits_processors:
            nonlocal prev_tokens
            out_y, out_logprobs = [], []
            if n_predict > 1:
                y = y[: -(n_predict - 1)]
            for i in range(n_predict):
                prev_tokens = (
                    mx.concatenate([prev_tokens, y]) if prev_tokens is not None else y
                )
                y, logprobs = _process_and_sample(prev_tokens, logits[:, i, :])
                out_y.append(y)
                out_logprobs.append(logprobs)
            return mx.concatenate(out_y, axis=0), mx.concatenate(out_logprobs, axis=0)
        else:
            return _process_and_sample(None, logits.squeeze(0))

    def _prefill(model, cache, y):
        prefill_step = make_prefill_step(
            model, prefill_step_size, prefill_memory_budget
        )
        while y.size > 1:
            step = prefill_step(_cache_offset(cache), kv_cache_quantized(cache))
            n_to_process = min(step, y.size - 1)
            model(y[:n_to_process][None], cache=cache)
            quantize_cache_fn(cache)
            mx.eval([c.state for c in cache])
            y = y[n_to_process:]
            mx.clear_cache()
        return y

    def _rewind_cache(num_draft, num_accept):
        trim_prompt_cache(model_cache, num_draft - num_accept)
        trim_prompt_cache(draft_cache, max(num_draft - num_accept - 1, 0))

    def _draft_generate(y, num_draft):
        if num_draft == 0:
            return mx.array([], mx.uint32)
        ys = []
        for _ in range(num_draft):
            y, _ = _step(draft_model, draft_cache, y)
            mx.async_eval(y)
            ys.append(y)
        return mx.concatenate(ys)

    def _offset(cache):
        return next((c.offset for c in cache if hasattr(c, "offset")), None)

    # On exit each cache holds exactly the prompt and the yielded tokens, as
    # generate_step leaves it (a caller like the server stores it under that
    # key): the last token yielded -- more for the draft cache, and with no
    # drafting the prompt's last token too -- was never fed, so feed it.
    emitted = []
    last_prompt_token = prompt[-1:].tolist()
    targets = [
        (m, c, None if _offset(c) is None else _offset(c) + prompt.size)
        for m, c in ((model, model_cache), (draft_model, draft_cache))
    ]

    def _sync_caches():
        for m, cache, base in targets:
            if base is None:
                continue
            missing = base + len(emitted) - _offset(cache)
            history = last_prompt_token + emitted
            if missing < 0:
                trim_prompt_cache(cache, -missing)
            elif 0 < missing <= len(history):
                m(mx.array(history[-missing:], mx.uint32)[None], cache=cache)
                quantize_cache_fn(cache)
                mx.eval([c.state for c in cache])
        if prompt_cache is not None:
            # Quantizing replaced entries in the split lists: the caller's
            # own list must get them (it kept the stale unquantized ones).
            prompt_cache[:] = model_cache + draft_cache

    with mx.stream(stream):
        draft_y = _prefill(draft_model, draft_cache, y)
        y = _prefill(model, model_cache, y)

        ntoks = 0
        # Set these so the finally block doesn't raise
        num_draft = 0
        n = 0
        clean_exit = True
        try:
            while True:
                num_draft = min(max_tokens - ntoks, num_draft_tokens)
                draft_tokens = _draft_generate(draft_y, num_draft)
                if prev_tokens is not None:
                    prev_tokens = prev_tokens[
                        : prev_tokens.size - y.size - num_draft + 1
                    ]
                y = mx.concatenate([y, draft_tokens])
                tokens, logprobs = _step(model, model_cache, y, num_draft + 1)
                mx.eval(tokens, draft_tokens)
                draft_tokens = draft_tokens.tolist()
                tokens = tokens.tolist()
                n = 0
                while n < num_draft:
                    tn, dtn, lpn = tokens[n], draft_tokens[n], logprobs[n]
                    if tn != dtn:
                        break
                    n += 1
                    ntoks += 1
                    emitted.append(tn)
                    yield tn, lpn, True
                    if ntoks == max_tokens:
                        break
                if ntoks < max_tokens:
                    ntoks += 1
                    emitted.append(tokens[n])
                    yield tokens[n], logprobs[n], False

                if ntoks == max_tokens:
                    break

                y = mx.array([tokens[n]], mx.uint32)
                draft_y = y

                # If we accepted all the draft tokens, include the last
                # draft token in the next draft step since it hasn't been
                # processed yet by the draft model
                if n == num_draft:
                    draft_y = mx.concatenate(
                        [mx.array(draft_tokens[-1:], mx.uint32), draft_y]
                    )

                if prev_tokens is not None:
                    prev_tokens = prev_tokens[: -max(num_draft - n, 1)]
                _rewind_cache(num_draft, n)
        except GeneratorExit:
            raise
        except BaseException:
            clean_exit = False
            raise
        finally:
            _rewind_cache(num_draft, n)
            if clean_exit:
                _sync_caches()


def _model_supports_nemotron_h_mtp(model: nn.Module) -> bool:
    return (
        getattr(model, "mtp", None) is not None
        and hasattr(model, "mtp_step")
        and hasattr(model, "make_mtp_cache")
        and hasattr(model, "rollback_speculative_cache")
    )


def nemotron_h_mtp_generate_step(
    prompt: mx.array,
    model: nn.Module,
    *,
    max_tokens: int = 256,
    sampler: Optional[Sampler] = None,
    prompt_cache: Optional[Any] = None,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = DEFAULT_QUANTIZED_KV_START,
    prefill_step_size: int = 2048,
    prompt_progress_callback: Optional[Callable[[int, int], None]] = None,
    stream: mx.Stream | mx.ThreadLocalStream = generation_stream,
    prefill_memory_budget: Union[None, int, PrefillBudget] = None,
) -> Generator[Tuple[int, mx.array, bool], None, None]:
    """Self-speculative decoding using Nemotron-H's in-checkpoint MTP head.

    Each step verifies one MTP-drafted token per backbone forward pass: the
    backbone processes ``[last confirmed token, draft token]`` together, and
    the draft is accepted iff the backbone's own top-1 pick for the first
    position matches it. Acceptance yields a second "bonus" token for free
    from the same pass. Rejection substitutes the backbone's own pick and
    rolls the caches back via ``rollback_speculative_cache``.

    Single sequence (batch size 1) only. Greedy acceptance only -- the draft
    is accepted iff it equals the backbone's own top-1 pick.

    KV-bit quantization only ever applies to the backbone's own full-
    attention cache (``model_cache``): Mamba caches have no ``to_quantized``
    so ``maybe_quantize_kv_cache`` always skips them, and the MTP head's own
    tiny attention cache (``mtp_cache``, at most one block) is left at full
    precision -- quantizing it would save almost nothing and the head's own
    accept-rate math is unaffected either way. ``rollback_speculative_cache``
    needs no quantization-aware branch: ``QuantizedKVCache.trim()`` is a
    plain offset decrement, identical in effect to ``KVCache.trim()``.

    Yields:
        Tuple[int, mx.array, bool]: (token, log-probabilities, from_draft).
    """
    if not _model_supports_nemotron_h_mtp(model):
        raise ValueError("The model does not have a usable Nemotron-H MTP head.")
    if prompt.ndim != 1:
        raise ValueError(
            "nemotron_h_mtp_generate_step supports a single sequence only."
        )

    sampler = sampler or greedy_sampler
    model_cache = prompt_cache if prompt_cache is not None else model.make_cache()
    mtp_cache = model.make_mtp_cache()
    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    def _sample(logits: mx.array) -> Tuple[mx.array, mx.array]:
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        return sampler(logprobs), logprobs.squeeze(0)

    # Prefill in prefill_step_size chunks, like generate_step. Feeding the
    # whole prompt in one backbone call made peak memory grow with prompt
    # length without bound (MoE / Mamba activations for every prompt token at
    # once): on a 26GB Mac the 30B model hit Metal OOM at a 400-token prompt,
    # and a tester's coding agent reached 36+GB before OOM. Only the last
    # chunk's hidden state is needed (the MTP head starts from the last prompt
    # token), so chunking changes nothing else.
    if prompt_progress_callback is None:
        prompt_progress_callback = lambda *_: None
    total = len(prompt)
    prefill_step = make_prefill_step(model, prefill_step_size, prefill_memory_budget)
    cached = _cache_offset([c for c in model_cache if c is not None])

    def _step():
        return prefill_step(cached + done, kv_cache_quantized(model_cache))

    with mx.stream(stream):
        done = 0
        prompt_progress_callback(done, total)
        while total - done > (step := _step()):
            model.backbone(prompt[done : done + step][None], cache=model_cache)
            quantize_cache_fn(model_cache)
            mx.eval([c.state for c in model_cache if c is not None])
            done += step
            prompt_progress_callback(done, total)
            mx.clear_cache()
        hidden = model.backbone(prompt[done:][None], cache=model_cache)
        quantize_cache_fn(model_cache)
        tok, logprobs = _sample(model.lm_head(hidden[:, -1, :]))
        hidden_last = hidden[:, -1:, :]
        prompt_progress_callback(total, total)

    # On exit the backbone cache must hold exactly prompt + every token
    # yielded so far, like plain generate_step leaves it: mlx_lm.server
    # stores it keyed by those tokens and continues the next turn from it.
    # The backbone's own token (bonus / correction) is yielded before it has
    # been fed, so if the consumer stops there (max_tokens / EOS closes the
    # generator at that yield) it is fed on the way out; an accepted draft
    # is already in the cache.
    unfed = tok
    n = 1
    # max_tokens < 0 means no limit, as for generate_step.
    limit = max_tokens if max_tokens >= 0 else float("inf")
    try:
        yield tok.item(), logprobs, False

        while n < limit:
            unfed = None  # mid-iteration: tok is being fed by the verify pass
            with mx.stream(stream):
                draft_logits, _ = model.mtp_step(
                    hidden_last, tok.reshape(1, 1), mtp_cache
                )
                draft_tok, draft_logprobs = _sample(draft_logits[:, -1, :])

                block = mx.concatenate(
                    [tok.reshape(1, 1), draft_tok.reshape(1, 1)], axis=1
                )
                ssm_sink = []
                hidden2 = model.backbone(block, cache=model_cache, ssm_sink=ssm_sink)
                quantize_cache_fn(model_cache)
                logits2 = model.lm_head(hidden2)
                verify_tok, verify_logprobs = _sample(logits2[:, 0, :])
                accepted = bool((verify_tok == draft_tok).item())

                if accepted:
                    bonus_tok, bonus_logprobs = _sample(logits2[:, 1, :])
                    hidden_last = hidden2[:, 1:2, :]
                else:
                    model.rollback_speculative_cache(
                        model_cache, ssm_sink, keep=1, block_size=2
                    )
                    for c in mtp_cache:
                        if c is not None and c.is_trimmable():
                            c.trim(1)
                    hidden_last = hidden2[:, 0:1, :]

            if accepted:
                # The backbone's distribution at that position (the MTP
                # head's is only the drafter's guess), like every other
                # speculative path reports.
                yield draft_tok.item(), verify_logprobs, True
                n += 1
                if n >= limit:
                    break
                unfed = bonus_tok
                yield bonus_tok.item(), bonus_logprobs, False
                n += 1
                tok = bonus_tok
            else:
                unfed = verify_tok
                yield verify_tok.item(), verify_logprobs, False
                n += 1
                tok = verify_tok
    finally:
        if unfed is not None:
            with mx.stream(stream):
                model.backbone(unfed.reshape(1, 1), cache=model_cache)
                quantize_cache_fn(model_cache)
                mx.eval([c.state for c in model_cache if c is not None])

def _mtp_language_model(model: nn.Module) -> Optional[nn.Module]:
    """The language model whose MTP head mtp_generate_step drafts with
    (Qwen3.5 / 3.5-MoE, a VL checkpoint's or a text-only one), or None."""
    lm = getattr(model, "language_model", model)
    if getattr(lm, "mtp", None) is None:
        return None
    if not all(
        hasattr(lm, a)
        for a in ("backbone", "logits_from_backbone", "mtp_step", "make_mtp_cache")
    ):
        return None
    return lm


class DraftLength:
    """Picks the number of drafts for each MTP iteration, 0 (plain
    decoding) to `max_drafts`, by the expected tokens per second. Measured
    as generation goes: how often the i-th draft is accepted once the ones
    before it were (a chained draft is accepted less often than the first),
    and each length's iteration time. Every `probe_every` iterations a
    neighbour of the best length is tried, so the estimates stay current."""

    def __init__(self, max_drafts: int, probe_every: int = 16, decay: float = 0.95):
        self.max_drafts = max(0, max_drafts)
        self.probe_every = probe_every
        self.decay = decay
        # Per draft position: accepted and tried (decayed), from a 50% prior.
        self.accepted = [1.0] * self.max_drafts
        self.tried = [2.0] * self.max_drafts
        self.seconds = {}
        self.n = 0

    def expected_tokens(self, k: int) -> float:
        total, reach = 1.0, 1.0
        for i in range(k):
            reach *= self.accepted[i] / self.tried[i]
            total += reach
        return total

    def _time(self, k):
        if k in self.seconds:
            return self.seconds[k]
        if not self.seconds:
            return 1.0 + 0.25 * k
        # Each draft adds a head step, about a quarter of an iteration.
        k0, t0 = min(self.seconds.items(), key=lambda kv: abs(kv[0] - k))
        return t0 * (1.0 + 0.25 * k) / (1.0 + 0.25 * k0)

    def best(self) -> int:
        return max(
            range(self.max_drafts + 1),
            key=lambda k: self.expected_tokens(k) / self._time(k),
        )

    def choose(self) -> int:
        if self.max_drafts == 0:
            return 0
        self.n += 1
        k = self.best()
        if self.n % self.probe_every == 0:
            up = (self.n // self.probe_every) % 2 == 1 or k == 0
            k = min(k + 1, self.max_drafts) if up else k - 1
        return k

    def update(self, k: int, n_accepted: int, seconds: float):
        for i in range(k):
            if n_accepted < i:
                break
            self.tried[i] = self.tried[i] * self.decay + 1
            self.accepted[i] = self.accepted[i] * self.decay + (n_accepted > i)
        old = self.seconds.get(k)
        self.seconds[k] = seconds if old is None else 0.7 * old + 0.3 * seconds


def mtp_generate_step(
    prompt: mx.array,
    model: nn.Module,
    *,
    num_draft_tokens: int = 1,
    adaptive: bool = True,
    max_tokens: int = 256,
    sampler: Optional[Sampler] = None,
    prompt_cache: Optional[Any] = None,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = DEFAULT_QUANTIZED_KV_START,
    prefill_step_size: int = 2048,
    prompt_progress_callback: Optional[Callable[[int, int], None]] = None,
    stream: mx.Stream | mx.ThreadLocalStream = generation_stream,
    prefill_memory_budget: Union[None, int, PrefillBudget] = None,
) -> Generator[Tuple[int, mx.array, bool], None, None]:
    """Self-speculative decoding with the model's own MTP head (Qwen3.5).

    One iteration drafts ``num_draft_tokens`` tokens greedily with the head
    (chained on its own hidden states after the first; ``adaptive``: 0 to
    that many, as DraftLength expects to be fastest), runs the backbone
    once over ``[last token, drafts...]``, samples the backbone's own token
    at every position and accepts the longest prefix of drafts equal to
    those samples, then emits the backbone's sample right after it. Every
    emitted token is the backbone's own sample, so the output distribution
    is exactly plain decoding's (greedy: the same tokens), with any sampler.

    The rejected drafts are rolled back: attention caches trim, the
    GatedDeltaNet states are recomputed over the kept tokens from what the
    verify pass captured (rollback_speculative_cache).

    The head's KV cache is quantized like the backbone's (``kv_bits``).
    The head attends over its own cache of (hidden state, next token)
    pairs, filled during the prefill as well (one extra layer over the
    uncached prompt): it was trained on whole sequences, not on the last
    position alone.

    Single sequence only; no logits processors or input embeddings
    (stream_generate falls back to plain decoding for those).

    Yields:
        Tuple[int, mx.array, bool]: (token, log-probabilities, from_draft).
    """
    lm = _mtp_language_model(model)
    if lm is None:
        raise ValueError("The model has no usable MTP head.")
    if prompt.ndim != 1:
        raise ValueError("mtp_generate_step supports a single sequence only.")

    sampler = sampler or greedy_sampler
    # The caller's own list (the server stores it after the request).
    cache = prompt_cache if prompt_cache is not None else make_prompt_cache(model)
    mtp_cache = lm.make_mtp_cache()
    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )
    if prompt_progress_callback is None:
        prompt_progress_callback = lambda *_: None

    # One sampler call per row unless the rows are independent in one call
    # (XTC draws one coin per call).
    rows_together = sampler is greedy_sampler or getattr(
        sampler, "rows_independent", False
    )

    def _sample(logits):
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        if rows_together or logprobs.ndim == 1 or logprobs.shape[0] == 1:
            return sampler(logprobs), logprobs
        return (
            mx.concatenate([sampler(logprobs[i : i + 1]) for i in range(logprobs.shape[0])]),
            logprobs,
        )

    def _kv_room():
        # Drafts that keep the verify block short of the point where plain
        # decoding quantizes the KV cache: it quantizes there exactly.
        if kv_bits is None or kv_cache_quantized(cache):
            return None
        offset = next(c.offset for c in cache if c.is_trimmable())
        return max(0, quantized_kv_start - offset - 1)

    def _feed_head(hidden, tokens):
        # Only the head's cache is used (and evaluated): its logits aren't.
        # No drafts at all: the head isn't run.
        if num_draft_tokens > 0 and hidden is not None and hidden.shape[1] > 0:
            lm.mtp_step(hidden, tokens, mtp_cache)
            quantize_cache_fn(mtp_cache)

    def _append_pair(hs, ts, h, t):
        if hs is None:
            return h, t
        return mx.concatenate([hs, h], axis=1), mx.concatenate([ts, t], axis=1)

    total = len(prompt)
    base = _cache_offset(cache)
    prefill_step = make_prefill_step(model, prefill_step_size, prefill_memory_budget)
    with mx.stream(stream):
        done = 0
        prompt_progress_callback(done, total)
        # The last hidden state of the chunk before: its pair's token is the
        # first one of the next chunk.
        prev = None
        while total - done > 1:
            step = prefill_step(base + done, kv_cache_quantized(cache))
            n = min(step, total - done - 1)
            h = lm.backbone(prompt[done : done + n][None], cache=cache)
            quantize_cache_fn(cache)
            hs = h[:, :-1] if prev is None else mx.concatenate([prev, h[:, :-1]], axis=1)
            _feed_head(hs, prompt[done + (prev is None) : done + n][None])
            prev = h[:, -1:]
            mx.eval([c.state for c in cache] + [c.state for c in mtp_cache])
            done += n
            prompt_progress_callback(done, total)
            mx.clear_cache()
        h = lm.backbone(prompt[done:][None], cache=cache)
        quantize_cache_fn(cache)
        hs = h[:, :-1] if prev is None else mx.concatenate([prev, h[:, :-1]], axis=1)
        _feed_head(hs, prompt[done + (prev is None) :][None])
        tok, logprobs = _sample(lm.logits_from_backbone(h[:, -1, :]))
        hidden = h[:, -1:, :]
        mx.eval(tok, logprobs)
        prompt_progress_callback(total, total)

    # On exit the cache must hold exactly prompt + every token yielded so
    # far, like plain generate_step leaves it (the server stores it keyed by
    # those tokens). `fixup` says how to get there from the most recent
    # yield: feed the backbone's own token (not in the cache yet), or keep
    # only part of the accepted drafts (rolled back with that verify pass's
    # capture).
    fixup = ("feed", tok)
    # Pairs the head hasn't seen yet: the accepted drafts' true hidden
    # states (the drafting saw the head's own), fed with the next draft.
    pending_h, pending_t = None, None
    n_out = 1
    try:
        yield tok.item(), logprobs.squeeze(0), False

        limit = max_tokens if max_tokens >= 0 else float("inf")
        lengths = DraftLength(num_draft_tokens) if adaptive else None
        while n_out < limit:
            k = lengths.choose() if lengths else num_draft_tokens
            k = int(min(k, limit - n_out))
            room = _kv_room()
            if room is not None:
                k = min(k, room) if room > 0 else 0
            started = time.perf_counter()
            fixup = None  # mid-iteration: cache state not well defined
            with mx.stream(stream):
                drafts = []
                if k > 0:
                    hs, ts = hidden, tok.reshape(1, 1)
                    if pending_h is not None:
                        hs = mx.concatenate([pending_h, hs], axis=1)
                        ts = mx.concatenate([pending_t, ts], axis=1)
                    for _ in range(k):
                        logits, h = lm.mtp_step(hs, ts, mtp_cache)
                        quantize_cache_fn(mtp_cache)
                        d = mx.argmax(logits[:, -1, :], axis=-1).reshape(1, 1)
                        drafts.append(d)
                        hs, ts = h[:, -1:], d
                # Quantized here, on the accepted tokens only (like plain
                # decoding's cache before this forward), not after a verify
                # pass or a rollback the consumer may still cut short.
                quantize_cache_fn(cache)
                block = mx.concatenate([tok.reshape(1, 1)] + drafts, axis=1)
                sink = []
                hid = lm.backbone(block, cache=cache, ssm_sink=sink)
                toks, lps = _sample(lm.logits_from_backbone(hid[0]))
                draft_ids = (
                    mx.concatenate(drafts, axis=1)[0]
                    if drafts
                    else mx.array([], mx.uint32)
                )
                mx.eval(toks, lps, draft_ids)
            toks_l = toks.tolist()
            draft_l = draft_ids.tolist()
            n_acc = 0
            while n_acc < k and draft_l[n_acc] == toks_l[n_acc]:
                n_acc += 1
            if lengths:
                lengths.update(k, n_acc, time.perf_counter() - started)

            with mx.stream(stream):
                lm.rollback_speculative_cache(cache, sink, n_acc + 1, k + 1)
                # The chained drafts' pairs held the head's own hidden
                # states: replaced by the backbone's for the accepted ones.
                if k > 1:
                    for c in mtp_cache:
                        c.trim(k - 1)
                if k == 0:
                    # Plain step: the head missed this pair.
                    pending_h, pending_t = _append_pair(
                        pending_h, pending_t, hidden, tok.reshape(1, 1)
                    )
                    if pending_h.shape[1] >= 32:
                        _feed_head(pending_h, pending_t)
                        pending_h = pending_t = None
                elif n_acc > 0:
                    pending_h = hid[:, :n_acc]
                    pending_t = draft_ids[None, :n_acc]
                else:
                    pending_h = pending_t = None
            tok = toks[n_acc]
            hidden = hid[:, n_acc : n_acc + 1, :]

            for i in range(n_acc):
                # Stopping here keeps tok + the first i + 1 drafts.
                fixup = ("keep", sink, i + 2, n_acc + 1)
                yield toks_l[i], lps[i], True
                n_out += 1
                if n_out >= limit:
                    return
            fixup = ("feed", tok)
            yield toks_l[n_acc], lps[n_acc], False
            n_out += 1
    finally:
        if fixup is not None:
            with mx.stream(stream):
                if fixup[0] == "keep":
                    _, sink, keep, held = fixup
                    lm.rollback_speculative_cache(cache, sink, keep, held)
                else:
                    quantize_cache_fn(cache)
                    lm.backbone(fixup[1].reshape(1, 1), cache=cache)
                quantize_cache_fn(cache)
                mx.eval([c.state for c in cache])


def _is_gemma4_assistant(draft_model: Optional[nn.Module]) -> bool:
    return getattr(draft_model, "model_type", None) == "gemma4_assistant"


def _gemma4_text_model(model: nn.Module) -> nn.Module:
    # gemma4 (multimodal wrapper) or gemma4_text
    return getattr(model, "language_model", model)


def _kv_to_dense(c, x):
    if isinstance(x, (list, tuple)):  # quantized (packed, scales, biases)
        return mx.dequantize(*x, group_size=c.group_size, bits=c.bits)
    return x


def _slice_kv(x, start, stop):
    if isinstance(x, (list, tuple)):
        return [a[..., start:stop, :] for a in x]
    return x[..., start:stop, :]


def _gemma4_valid_kv(c, window: Optional[int]):
    """The accepted (post-rollback) keys/values held by one main-model cache,
    dense, optionally limited to the last `window` positions. Order does not
    matter to the drafter (keys already carry RoPE, and it attends with no
    mask), so a rotated buffer is fine as long as it has no stale slots."""
    if isinstance(c, (RotatingKVCache, QuantizedRotatingKVCache)):
        if c.offset >= c.max_size and c._idx != _kv_len(c.keys):
            # Rotated in-place buffer: every slot is valid, all max_size.
            keys, values = c.keys, c.values
        else:
            keys = _slice_kv(c.keys, 0, min(c._idx, c.offset))
            values = _slice_kv(c.values, 0, min(c._idx, c.offset))
        n = _kv_len(keys)
        if window is not None and n > window:
            keys = _slice_kv(keys, n - window, n)
            values = _slice_kv(values, n - window, n)
    else:
        keys = _slice_kv(c.keys, 0, c.offset)
        values = _slice_kv(c.values, 0, c.offset)
    return _kv_to_dense(c, keys), _kv_to_dense(c, values)


def _kv_len(x):
    return (x[0] if isinstance(x, (list, tuple)) else x).shape[2]


def _gemma4_trim_cache(caches, n: int):
    """Drop the last `n` positions from every main-model cache after a
    verify forward that rejected `n` draft tokens.

    KVCache / QuantizedKVCache trim is an offset decrement. RotatingKVCache
    refuses to trim once it has wrapped (is_trimmable() is False past
    max_size), which is why mlx-lm's generic speculative decoding cannot run
    Gemma 4's sliding layers past 1024 tokens. But the verify forward is
    always multi-token, so the rotating caches just went through
    `_update_concat` and hold a buffer in temporal order of length
    >= max_size - 1 + block: slicing its tail off is an exact rollback and
    still leaves every kept token its full window."""
    if n <= 0:
        return
    for c in caches:
        if isinstance(c, (RotatingKVCache, QuantizedRotatingKVCache)):
            keys = c._temporal_order(c.keys)
            values = c._temporal_order(c.values)
            keep = _kv_len(keys) - n
            c.keys = _slice_kv(keys, 0, keep)
            c.values = _slice_kv(values, 0, keep)
            c.offset -= n
            c._idx = keep
        else:
            c.trim(n)


def gemma4_mtp_generate_step(
    prompt: mx.array,
    model: nn.Module,
    draft_model: nn.Module,
    stream: mx.Stream | mx.ThreadLocalStream = generation_stream,
    *,
    num_draft_tokens: int = 3,
    max_tokens: int = 256,
    sampler: Optional[Sampler] = None,
    prompt_cache: Optional[Any] = None,
    prefill_step_size: int = 2048,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = DEFAULT_QUANTIZED_KV_START,
    prompt_progress_callback: Optional[Callable[[int, int], None]] = None,
    prefill_memory_budget: Union[None, int, PrefillBudget] = None,
) -> Generator[Tuple[int, mx.array, bool], None, None]:
    """Speculative decoding for Gemma 4 with its MTP drafter
    (``gemma4_assistant``, e.g. google/gemma-4-26B-A4B-it-assistant).

    Unlike a regular draft model the drafter has no KV cache and no token
    history of its own: each draft step reads the main model's final hidden
    state and the KV of the main model's last full- and last sliding-attention
    layers (see models/gemma4_assistant.py). One iteration:

    1. draft ``num_draft_tokens`` tokens greedily with the drafter, starting
       from (last token, main hidden state that predicted it);
    2. run the main model once over ``[last token, drafts...]``;
    3. sample the main model's own token at every position; accept the
       longest prefix of drafts equal to those samples, emit the main
       model's sample right after it (correction or bonus token);
    4. roll the main caches back over the rejected drafts.

    Every emitted token is the main model's own sample given the accepted
    prefix, so the output distribution is exactly the main model's (with the
    greedy sampler: the exact same tokens as plain decoding).

    Single sequence only; no logits processors (stream_generate falls back to
    plain decoding for those).

    Yields:
        Tuple[int, mx.array, bool]: (token, log-probabilities, from_draft).
    """
    if prompt.ndim != 1:
        raise ValueError("gemma4_mtp_generate_step supports a single sequence only.")
    lm = _gemma4_text_model(model)
    text = lm.model
    layer_types = lm.args.layer_types
    last_full = max(i for i, t in enumerate(layer_types) if t == "full_attention")
    last_swa = max(i for i, t in enumerate(layer_types) if t == "sliding_attention")
    window = draft_model.sliding_window

    sampler = sampler or greedy_sampler
    # Used (and quantized in place by quantize_cache_fn) as the caller's own
    # list, not a slice of it: the server keeps and re-stores that list.
    # The drafter contributes no entries (its make_cache() is empty).
    cache = prompt_cache if prompt_cache is not None else make_prompt_cache(model)
    if len(cache) != len(lm.layers):
        raise ValueError(
            f"Expected {len(lm.layers)} main-model cache entries, got {len(cache)}."
        )
    if prompt_progress_callback is None:
        prompt_progress_callback = lambda *_: None
    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    def _sample(logits):
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        return sampler(logprobs), logprobs

    def _embed(tokens):
        return text.embed_tokens(tokens) * text.embed_scale

    def _shared_kv():
        return {
            "full_attention": _gemma4_valid_kv(cache[last_full], None),
            "sliding_attention": _gemma4_valid_kv(cache[last_swa], window),
        }

    # Absolute position of the first token of `prompt`: callers such as the
    # server pass only the uncached tail of the prompt, with a cache that
    # already holds the prefix. The drafter's RoPE query position has to
    # count that prefix too.
    base = cache[last_full].offset

    with mx.stream(stream):
        total = len(prompt)
        done = 0
        prompt_progress_callback(done, total)
        prefill_step = make_prefill_step(
            model, prefill_step_size, prefill_memory_budget
        )
        while total - done > 1:
            step = prefill_step(base + done, kv_cache_quantized(cache))
            n = min(step, total - done - 1)
            text(prompt[done : done + n][None], cache=cache)
            quantize_cache_fn(cache)
            mx.eval([c.state for c in cache])
            done += n
            prompt_progress_callback(done, total)
            mx.clear_cache()
        hidden = text(prompt[done:][None], cache=cache)
        quantize_cache_fn(cache)
        tok, logprobs = _sample(lm.logits_from_hidden(hidden[:, -1, :]))
        hidden = hidden[:, -1:, :]
        mx.eval(tok, logprobs)
        prompt_progress_callback(total, total)

    # On exit the main cache must hold exactly prompt + every token yielded
    # so far, like plain generate_step leaves it: callers such as the
    # server store it keyed by those tokens and later trim / extend it for
    # the next request. stream_generate stops pulling at max_tokens / EOS
    # (the generator is closed at that yield), when this loop's cache is
    # either ahead (stopped partway through the accepted drafts, whose KV is
    # already in) or one behind (stopped at the main model's own token, not
    # yet fed). `fixup` records which, for the most recent yield.
    fixup = ("feed", tok)
    position = base + total  # absolute position of `tok` (not yet in the cache)
    n_out = 1
    try:
        yield tok.item(), logprobs.squeeze(0), False

        # max_tokens < 0 means no limit, as for generate_step.
        limit = max_tokens if max_tokens >= 0 else float("inf")
        while n_out < limit:
            k = int(min(num_draft_tokens, limit - n_out))
            fixup = None  # mid-iteration: cache state not well defined
            with mx.stream(stream):
                drafts = []
                if k > 0:
                    shared = _shared_kv()
                    d, h = tok.reshape(1, 1), hidden
                    for _ in range(k):
                        logits, h = draft_model(
                            mx.concatenate([_embed(d), h], axis=-1), shared, position
                        )
                        d = mx.argmax(logits[:, -1, :], axis=-1).reshape(1, 1)
                        drafts.append(d)
                block = mx.concatenate([tok.reshape(1, 1)] + drafts, axis=1)
                hid = text(block, cache=cache)
                quantize_cache_fn(cache)
                toks, lps = _sample(lm.logits_from_hidden(hid[0]))
                draft_ids = (
                    mx.concatenate(drafts, axis=1)[0]
                    if drafts
                    else mx.array([], mx.uint32)
                )
                mx.eval(toks, lps, draft_ids)
            toks_l = toks.tolist()
            draft_l = draft_ids.tolist()
            n_acc = 0
            while n_acc < k and draft_l[n_acc] == toks_l[n_acc]:
                n_acc += 1

            _gemma4_trim_cache(cache, k - n_acc)
            position += n_acc + 1
            tok = toks[n_acc]
            hidden = hid[:, n_acc : n_acc + 1, :]

            for i in range(n_acc):
                fixup = ("trim", n_acc - 1 - i)
                yield toks_l[i], lps[i], True
                n_out += 1
                if n_out >= limit:
                    return
            fixup = ("feed", tok)
            yield toks_l[n_acc], lps[n_acc], False
            n_out += 1
    finally:
        if fixup is not None:
            kind, arg = fixup
            if kind == "trim":
                _gemma4_trim_cache(cache, arg)
            else:
                with mx.stream(stream):
                    text(arg.reshape(1, 1), cache=cache)
                    quantize_cache_fn(cache)
                    mx.eval([c.state for c in cache])

def stream_generate(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: Union[str, mx.array, List[int]],
    max_tokens: int = 256,
    draft_model: Optional[nn.Module] = None,
    stream: mx.Stream | mx.ThreadLocalStream = generation_stream,
    **kwargs,
) -> Generator[GenerationResponse, None, None]:
    """
    A generator producing text based on the given prompt from the model.

    Args:
        model (nn.Module): The model to use for generation.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        prompt (Union[str, mx.array, List[int]]): The input prompt string or
          integer tokens.
        max_tokens (int): The maximum number of tokens to generate.
          Default: ``256``.
        draft_model (Optional[nn.Module]): An optional draft model. If provided
          then speculative decoding is used. The draft model must use the same
          tokenizer as the main model. Default: ``None``.
        kwargs: The remaining options get passed to :func:`generate_step`.
          See :func:`generate_step` for more details.

    Yields:
        GenerationResponse: An instance containing the generated text segment and
            associated metadata. See :class:`GenerationResponse` for details.
    """
    if max_tokens == 0:
        raise ValueError(
            "Maximum number of tokens must be non-zero (use -1 for no limit)."
        )

    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    if not isinstance(prompt, mx.array):
        if isinstance(prompt, str):
            # Try to infer if special tokens are needed
            add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
                tokenizer.bos_token
            )
            prompt = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        prompt = mx.array(prompt)

    detokenizer = tokenizer.detokenizer

    kwargs["max_tokens"] = max_tokens

    if draft_model is None:
        num_head_drafts = kwargs.pop("num_draft_tokens", None)
        # Auto-dispatch to the in-checkpoint MTP head instead of plain
        # decoding when the model has one: same output (bit-exact, see
        # tests/test_nemotron_h_mtp_generate.py), just faster. Only for the
        # exact subset of generate_step's surface nemotron_h_mtp_generate_step
        # actually supports (single sequence, greedy, no logits processors /
        # max_kv_size) -- anything outside that silently falls back to plain
        # generate_step rather than ignoring the request. KV-bit
        # quantization IS supported (see nemotron_h_mtp_generate_step's
        # docstring for why rollback needs no quantization-aware branch).
        # A Qwen3.5 MTP head: any sampler (the drafts are checked against
        # the backbone's own samples).
        use_head = (
            _mtp_language_model(model) is not None
            and num_head_drafts != 0
            and prompt.ndim == 1
            and not kwargs.get("logits_processors")
            and kwargs.get("max_kv_size") is None
            and kwargs.get("input_embeddings") is None
        )
        use_mtp = (
            not use_head
            and _model_supports_nemotron_h_mtp(model)
            and prompt.ndim == 1
            and kwargs.get("sampler") in (None, greedy_sampler)
            and not kwargs.get("logits_processors")
            and kwargs.get("max_kv_size") is None
            and kwargs.get("input_embeddings") is None
        )
        if use_head:
            token_generator = mtp_generate_step(
                prompt,
                model,
                num_draft_tokens=1 if num_head_drafts is None else num_head_drafts,
                max_tokens=kwargs["max_tokens"],
                sampler=kwargs.get("sampler"),
                prompt_cache=kwargs.get("prompt_cache"),
                kv_bits=kwargs.get("kv_bits"),
                kv_group_size=kwargs.get("kv_group_size", 64),
                quantized_kv_start=kwargs.get(
                    "quantized_kv_start", DEFAULT_QUANTIZED_KV_START
                ),
                prefill_step_size=kwargs.get("prefill_step_size", 2048),
                prompt_progress_callback=kwargs.get("prompt_progress_callback"),
                stream=stream,
                prefill_memory_budget=kwargs.get("prefill_memory_budget"),
            )
        elif use_mtp:
            token_generator = nemotron_h_mtp_generate_step(
                prompt,
                model,
                max_tokens=kwargs["max_tokens"],
                sampler=kwargs.get("sampler"),
                prompt_cache=kwargs.get("prompt_cache"),
                kv_bits=kwargs.get("kv_bits"),
                kv_group_size=kwargs.get("kv_group_size", 64),
                quantized_kv_start=kwargs.get(
                    "quantized_kv_start", DEFAULT_QUANTIZED_KV_START
                ),
                prefill_step_size=kwargs.get("prefill_step_size", 2048),
                prompt_progress_callback=kwargs.get("prompt_progress_callback"),
                stream=stream,
                prefill_memory_budget=kwargs.get("prefill_memory_budget"),
            )
        else:
            token_generator = generate_step(prompt, model, stream, **kwargs)
            # from_draft always false for non-speculative generation
            token_generator = (
                (token, logprobs, False) for token, logprobs in token_generator
            )
    elif _is_gemma4_assistant(draft_model):
        # Gemma 4 MTP drafter. Anything its step doesn't support (logits
        # processors, image input_embeddings, a bounded max_kv_size) falls
        # back to plain decoding without the drafter -- same output, just
        # not faster -- rather than silently ignoring the request option.
        use_mtp = (
            prompt.ndim == 1
            and not kwargs.get("logits_processors")
            and kwargs.get("max_kv_size") is None
            and kwargs.get("input_embeddings") is None
        )
        if use_mtp:
            token_generator = gemma4_mtp_generate_step(
                prompt,
                model,
                draft_model,
                stream,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k
                    in (
                        "num_draft_tokens",
                        "max_tokens",
                        "sampler",
                        "prompt_cache",
                        "prefill_step_size",
                        "kv_bits",
                        "kv_group_size",
                        "quantized_kv_start",
                        "prompt_progress_callback",
                        "prefill_memory_budget",
                    )
                },
            )
        else:
            kwargs.pop("num_draft_tokens", None)
            token_generator = (
                (token, logprobs, False)
                for token, logprobs in generate_step(prompt, model, stream, **kwargs)
            )
    elif kwargs.get("input_embeddings") is not None:
        # A classic draft model can't take embeddings (an image request):
        # plain decoding without it -- same output, not faster.
        kwargs.pop("num_draft_tokens", None)
        token_generator = (
            (token, logprobs, False)
            for token, logprobs in generate_step(prompt, model, stream, **kwargs)
        )
    else:
        kwargs.pop("max_kv_size", None)
        kwargs.pop("prompt_progress_callback", None)
        # The server always passes it (None for text): speculative_generate_step
        # has no such parameter, and every request failed with a TypeError.
        kwargs.pop("input_embeddings", None)
        # Its logits processors never see the prompt, cached or not.
        kwargs.pop("token_history", None)
        token_generator = speculative_generate_step(
            prompt, model, draft_model, stream, **kwargs
        )
    with wired_limit(model, [stream]):
        tic = time.perf_counter()
        for n, (token, logprobs, from_draft) in enumerate(token_generator):
            if n == 0:
                prompt_time = time.perf_counter() - tic
                prompt_tps = prompt.size / prompt_time
                tic = time.perf_counter()
            if token in tokenizer.eos_token_ids:
                break

            detokenizer.add_token(token)
            if (n + 1) == max_tokens:
                break

            yield GenerationResponse(
                text=detokenizer.last_segment,
                token=token,
                logprobs=logprobs,
                from_draft=from_draft,
                prompt_tokens=prompt.size,
                prompt_tps=prompt_tps,
                generation_tokens=n + 1,
                generation_tps=(n + 1) / (time.perf_counter() - tic),
                peak_memory=mx.get_peak_memory() / 1e9,
                finish_reason=None,
            )

        detokenizer.finalize()
        yield GenerationResponse(
            text=detokenizer.last_segment,
            token=token,
            logprobs=logprobs,
            from_draft=from_draft,
            prompt_tokens=prompt.size,
            prompt_tps=prompt_tps,
            generation_tokens=n + 1,
            generation_tps=(n + 1) / (time.perf_counter() - tic),
            peak_memory=mx.get_peak_memory() / 1e9,
            finish_reason="stop" if token in tokenizer.eos_token_ids else "length",
        )


def generate(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: Union[str, List[int]],
    verbose: bool = False,
    **kwargs,
) -> str:
    """
    Generate a complete response from the model.

    Args:
       model (nn.Module): The language model.
       tokenizer (PreTrainedTokenizer): The tokenizer.
       prompt (Union[str, List[int]]): The input prompt string or integer tokens.
       verbose (bool): If ``True``, print tokens and timing information.
           Default: ``False``.
       kwargs: The remaining options get passed to :func:`stream_generate`.
          See :func:`stream_generate` for more details.
    """
    if verbose:
        print("=" * 10)

    text = ""
    for response in stream_generate(model, tokenizer, prompt, **kwargs):
        if verbose:
            print(response.text, end="", flush=True)
        text += response.text

    if verbose:
        print()
        print("=" * 10)
        if len(text) == 0:
            print("No text generated for this prompt")
            return
        print(
            f"Prompt: {response.prompt_tokens} tokens, "
            f"{response.prompt_tps:.3f} tokens-per-sec"
        )
        print(
            f"Generation: {response.generation_tokens} tokens, "
            f"{response.generation_tps:.3f} tokens-per-sec"
        )
        print(f"Peak memory: {response.peak_memory:.3f} GB")
    return text


def _right_pad_prompts(prompts, max_length=None):
    if max_length is None:
        max_length = max(len(p) for p in prompts)
    return mx.array([p + [0] * (max_length - len(p)) for p in prompts])


def _batch_layers(cache, kv_bits, kv_group_size):
    """One sequence's cache made ready to join a batch: full-attention
    layers quantized when the batch quantizes its KV cache (``kv_bits``),
    full precision when it doesn't; a quantized sliding-window layer
    (QuantizedRotatingKVCache, from the unbatched path) back to full
    precision -- in a batch those stay unquantized, their size is the
    window's. Other layers (recurrent state, lists) as they are."""
    out = []
    for layer in cache:
        if type(layer) is KVCache:
            if kv_bits is not None:
                layer = layer.to_quantized(group_size=kv_group_size, bits=kv_bits)
        elif isinstance(layer, QuantizedKVCache):
            if kv_bits is None:
                layer = layer.to_unquantized()
            elif (layer.group_size, layer.bits) != (kv_group_size, kv_bits):
                layer = layer.to_unquantized().to_quantized(group_size=kv_group_size, bits=kv_bits)
        elif isinstance(layer, QuantizedRotatingKVCache):
            layer = layer.to_unquantized()
        out.append(layer)
    return out


def _merge_caches(caches):
    batch_cache = []

    if not caches:
        return batch_cache

    for i in range(len(caches[0])):
        if hasattr(caches[0][i], "merge"):
            batch_cache.append(caches[0][i].merge([c[i] for c in caches]))
        else:
            raise ValueError(
                f"{type(caches[0][i])} does not yet support batching with history"
            )
    return batch_cache


def _extend_cache(cache_a, cache_b):
    if not cache_a:
        return cache_b
    if not cache_b:
        return cache_a
    for ca, cb in zip(cache_a, cache_b):
        ca.extend(cb)
    return cache_a


def _normalize_samplers(samplers, *, n, fallback):
    """One concrete sampler per sequence."""
    if not samplers:
        return [fallback] * n
    return [fallback if s is None else s for s in samplers]


def _normalize_logits_processors(logits_processors, *, n):
    """One concrete tuple of processors per sequence."""
    if not logits_processors:
        return [() for _ in range(n)]
    return [tuple(lp) if lp else () for lp in logits_processors]


def _build_trie(sequences):
    """Build an Aho-Corasick trie from the provided sequences

    See https://en.wikipedia.org/wiki/Aho–Corasick_algorithm .
    """
    trie = {}
    for idx, seq in enumerate(sequences):
        node = trie
        try:
            for tok in seq:
                node = node.setdefault(tok, {})
            if node is trie:
                # An empty pattern would make the root match every token.
                continue
            node["__match__"] = (tuple(seq), idx)
        except TypeError:
            node = node.setdefault(seq, {})
            node["__match__"] = ((seq,), idx)

    # BFS to set failure links and propagate matches.
    queue = deque()
    for key, child in trie.items():
        if key == "__match__":
            continue
        child["__fail__"] = trie
        queue.append(child)
    while queue:
        parent = queue.popleft()
        for key, child in parent.items():
            if key in ("__fail__", "__match__"):
                continue
            queue.append(child)
            fail = parent["__fail__"]
            while key not in fail and fail is not trie:
                fail = fail["__fail__"]
            child["__fail__"] = fail[key] if key in fail else trie
            if "__match__" not in child and "__match__" in child["__fail__"]:
                child["__match__"] = child["__fail__"]["__match__"]
    return trie


class StopSequences:
    """An immutable Aho-Corasick automaton over stop sequences.

    Any matched sequence signals stop. A :class:`StopSequences.Matcher` holds
    one stream's position in it.
    """

    class Matcher:
        """A position in a :class:`StopSequences` automaton."""

        def __init__(self, root):
            self._root = root
            self._node = root

        def advance(self, token: int) -> bool:
            """Consume one token. Returns whether a stop sequence completed."""
            node = self._node
            # Fail back until the token continues a match, or we hit the root.
            while token not in node and node is not self._root:
                node = node["__fail__"]
            if token in node:
                node = node[token]
            self._node = node
            return node.get("__match__") is not None

    def __init__(self, stop_sequences: Optional[Sequence[Sequence[int]]] = None):
        self._root = _build_trie(stop_sequences) if stop_sequences else {}

    def matcher(self) -> "StopSequences.Matcher":
        """A fresh matcher at the start of this automaton."""
        return self.Matcher(self._root)


class TextStateMachine:
    """A state machine that matches decoded text to track state transitions
    (reasoning, tool calling) and strip the matched control sequences from the
    output.

    Transitions are provided as state -> [(text, new_state)]. Matching on text
    rather than token ids is robust to tokenization differences (e.g. a
    marker's trailing ``>`` being merged with the following byte).

    The runtime state carries a buffer holding text that might be part of a
    control sequence. Text is only emitted once it is known not to be part of
    any match.

    Example:

        sm = TextStateMachine(
            transitions={
                "normal": [("<think>", "reasoning"), ("<tool_call>", "tool")],
                "reasoning": [("</think>", "normal")],
                "tool": [("</tool_call>", "normal")],
            },
        )
        state = sm.make_state(initial="normal")
    """

    def __init__(self, transitions=None):
        self._states = {}
        for src, edges in (transitions or {}).items():
            strings, dst = zip(*edges) if edges else ([], [])
            self._states[src] = (_build_trie(strings), dst)

    def make_state(self, initial="normal"):
        """Create a fresh runtime state (state_name, trie_node, states, buffer)."""
        if initial not in self._states:
            self._states[initial] = (_build_trie([]), [])
        return (initial, self._states[initial][0], self._states, "")

    @staticmethod
    def step(state, text):
        """Consume a chunk of decoded text.

        Returns (new_state, emittable_text, current_state_name) where
        emittable_text is the text safe to show (control sequences stripped,
        possible partial matches held back in the buffer).
        """
        s, n, states, buf = state
        buf += text
        trie = states[s][0]
        emittable = ""
        # buf[:consumed] has been emitted or discarded; buf[consumed:] pending.
        consumed = 0

        for i in range(len(buf)):
            ch = buf[i]
            while ch not in n and n is not trie:
                n = n["__fail__"]
            if ch in n:
                n = n[ch]

            match = n.get("__match__")
            if match is not None:
                match_start = i + 1 - len(match[0])
                emittable += buf[consumed:match_start]
                consumed = i + 1
                s = states[s][1][match[1]]
                if s is None:
                    return (s, None, states, buf[consumed:]), emittable, s
                trie = states[s][0]
                n = trie
            elif n is trie:
                # At the root: no partial match in progress, everything is safe.
                emittable += buf[consumed : i + 1]
                consumed = i + 1

        return (s, n, states, buf[consumed:]), emittable, s

    @staticmethod
    def flush(state):
        """Emit the remaining buffer (use on finish_reason="length")."""
        s, _, states, buf = state
        trie = states[s][0] if s is not None else None
        return (s, trie, states, ""), buf, s

    @staticmethod
    def discard(state):
        """Drop the remaining buffer (use on finish_reason="stop")."""
        s, _, states, _ = state
        trie = states[s][0] if s is not None else None
        return (s, trie, states, ""), s


def make_stop_sequences(tokenizer, stop_words=None):
    """Build StopSequences from EOS tokens and stop words."""
    stop_sequences = [(t,) for t in tokenizer.eos_token_ids]
    for w in stop_words or []:
        stop_sequences.append(tuple(tokenizer.encode(w, add_special_tokens=False)))
    return StopSequences(stop_sequences)


def make_text_state_machine(tokenizer, stop_words=None):
    """Build a TextStateMachine with reasoning/tool transitions and stop words.

    Stop words are added as self-transitions in every state so they are
    stripped from the output without changing state.
    """
    transitions = {}

    if tokenizer.has_thinking:
        transitions.setdefault("normal", []).append(
            (tokenizer.think_start, "reasoning")
        )
        transitions["reasoning"] = [(tokenizer.think_end, "normal")]

    if tokenizer.has_tool_calling:
        transitions.setdefault("normal", []).append((tokenizer.tool_call_start, "tool"))
        if tokenizer.has_thinking:
            transitions["reasoning"].append((tokenizer.tool_call_start, "tool"))
        transitions["tool"] = (
            [(tokenizer.tool_call_end, "normal")] if tokenizer.tool_call_end else []
        )

    if tokenizer.structural_markers:
        for w in tokenizer.structural_markers:
            transitions.setdefault("normal", []).append((w, "normal"))

    if stop_words:
        for state_name in set(transitions) | {"normal"}:
            for w in stop_words:
                transitions.setdefault(state_name, []).append((w, state_name))

    return TextStateMachine(transitions or None)


class PromptProcessingBatch:
    """
    A batch processor for prompt tokens with support for incremental processing.

    This class handles batched prompt processing, managing KV caches and preparing
    tokens for generation. It supports extending, filtering, and splitting batches.
    """

    @dataclass
    class Response:
        uid: int
        progress: tuple
        end_of_segment: bool
        end_of_prompt: bool

    def __init__(
        self,
        model: nn.Module,
        uids: List[int],
        caches: List[List[Any]],
        tokens: Optional[List[List[int]]] = None,
        prefill_step_size: int = 2048,
        samplers: Optional[List[Sampler]] = None,
        fallback_sampler: Optional[Sampler] = None,
        logits_processors: Optional[List[List[LogitsProcessor]]] = None,
        stop_sequences: Optional[List[StopSequences]] = None,
        max_tokens: Optional[List[int]] = None,
    ):
        self.model = model
        self.uids = uids
        self.prompt_cache = _merge_caches(caches)
        self.tokens = tokens if tokens is not None else [[] for _ in uids]

        self.prefill_step_size = prefill_step_size
        self.samplers = samplers if samplers is not None else []
        self.fallback_sampler = fallback_sampler or greedy_sampler
        self.logits_processors = (
            logits_processors if logits_processors is not None else []
        )
        self.stop_sequences = (
            stop_sequences
            if stop_sequences is not None
            else [StopSequences()] * len(uids)
        )
        self.max_tokens = (
            max_tokens
            if max_tokens is not None
            else [DEFAULT_MAX_TOKENS] * len(self.uids)
        )

    def __len__(self):
        return len(self.uids)

    def extract_cache(self, idx: int) -> List[Any]:
        return [c.extract(idx) for c in self.prompt_cache]

    def extend(self, batch):
        self.uids.extend(batch.uids)
        self.prompt_cache = _extend_cache(self.prompt_cache, batch.prompt_cache)
        self.tokens.extend(batch.tokens)
        self.samplers.extend(batch.samplers)
        self.logits_processors.extend(batch.logits_processors)
        self.max_tokens.extend(batch.max_tokens)
        self.stop_sequences.extend(batch.stop_sequences)

    def _copy(self):
        new_batch = self.__class__.__new__(self.__class__)
        new_batch.model = self.model
        new_batch.uids = list(self.uids)
        new_batch.prompt_cache = copy.deepcopy(self.prompt_cache)
        new_batch.tokens = list(self.tokens)
        new_batch.prefill_step_size = self.prefill_step_size
        new_batch.samplers = list(self.samplers)
        new_batch.fallback_sampler = self.fallback_sampler
        new_batch.logits_processors = list(self.logits_processors)
        new_batch.stop_sequences = list(self.stop_sequences)
        new_batch.max_tokens = list(self.max_tokens)
        return new_batch

    def split(self, indices: List[int]):
        indices = sorted(indices)
        indices_left = sorted(set(range(len(self.uids))) - set(indices))
        new_batch = self._copy()
        self.filter(indices_left)
        new_batch.filter(indices)

        return new_batch

    def filter(self, keep: List[int]):
        self.uids = [self.uids[idx] for idx in keep]
        if not keep:
            self.prompt_cache.clear()
        else:
            for c in self.prompt_cache:
                c.filter(keep)
        self.tokens = [self.tokens[idx] for idx in keep]
        self.samplers = [self.samplers[idx] for idx in keep]
        self.logits_processors = [self.logits_processors[idx] for idx in keep]
        self.max_tokens = [self.max_tokens[idx] for idx in keep]
        self.stop_sequences = [self.stop_sequences[idx] for idx in keep]

    def prompt(self, tokens: List[List[int]]):
        """
        Process prompt tokens through the model.

        Args:
            tokens: List of token sequences to process.
        """
        if len(self.uids) != len(tokens):
            raise ValueError("The batch length doesn't match the number of inputs")

        if not tokens:
            return

        # Add the tokens to the self.tokens so they represent the tokens
        # contained in the KV Cache.
        for sti, ti in zip(self.tokens, tokens):
            sti += ti

        # Calculate if we need to pad
        lengths = [len(p) for p in tokens]
        max_length = max(lengths)
        padding = [max_length - l for l in lengths]
        max_padding = max(padding)

        # Prepare the caches and inputs. Right pad if needed otherwise just
        # cast to array.
        if max_padding > 0:
            tokens = _right_pad_prompts(tokens, max_length=max_length)
            for c in self.prompt_cache:
                c.prepare(lengths=lengths, right_padding=padding)
        else:
            tokens = mx.array(tokens)

        # Actual prompt processing loop
        while tokens.shape[1] > 0:
            n_to_process = min(self.prefill_step_size, tokens.shape[1])
            self.model(tokens[:, :n_to_process], cache=self.prompt_cache)
            mx.eval([c.state for c in self.prompt_cache])
            mx.clear_cache()
            tokens = tokens[:, n_to_process:]

        # Finalize the cache if there was any padding
        if max_padding > 0:
            for c in self.prompt_cache:
                c.finalize()
            mx.eval([c.state for c in self.prompt_cache])
            mx.clear_cache()

    def generate(self, tokens: List[List[int]]):
        """
        Transition from prompt processing to generation.

        Args:
            tokens: Final tokens for each sequence to start generation.

        Returns:
            A GenerationBatch ready for token generation.
        """
        if any(len(t) > 1 for t in tokens):
            self.prompt([t[:-1] for t in tokens])
        last_token = mx.array([t[-1] for t in tokens])

        generation = GenerationBatch(
            self.model,
            self.uids,
            last_token,
            self.prompt_cache,
            self.tokens,
            self.samplers,
            self.fallback_sampler,
            self.logits_processors,
            self.stop_sequences,
            self.max_tokens,
        )

        self.uids = []
        self.prompt_cache = []
        self.tokens = []
        self.samplers = []
        self.logits_processors = []
        self.max_tokens = []

        return generation

    @classmethod
    def empty(
        cls,
        model: nn.Module,
        fallback_sampler: Sampler,
        prefill_step_size: int = 2048,
    ):
        return cls(
            model=model,
            fallback_sampler=fallback_sampler,
            prefill_step_size=prefill_step_size,
            uids=[],
            caches=[],
            tokens=[],
            samplers=[],
            logits_processors=[],
            max_tokens=[],
            stop_sequences=[],
        )


class GenerationBatch:
    """
    A batched token generator that manages multiple sequences in parallel.

    This class handles the generation phase after prompt processing, managing
    KV caches, sampling, and stop sequence detection for multiple sequences.
    """

    @dataclass
    class Response:
        uid: int
        token: int
        logprobs: mx.array
        finish_reason: Optional[str]
        prompt_cache: Optional[List[Any]]
        all_tokens: Optional[List[int]]

    def __init__(
        self,
        model: nn.Module,
        uids: List[int],
        inputs: mx.array,
        prompt_cache: List[Any],
        tokens: List[List[int]],
        samplers: Optional[List[Sampler]],
        fallback_sampler: Sampler,
        logits_processors: Optional[List[List[LogitsProcessor]]],
        stop_sequences: List[StopSequences],
        max_tokens: List[int],
    ):
        self.model = model
        self.uids = uids
        self.prompt_cache = prompt_cache
        self.tokens = tokens

        self.samplers = samplers
        self.fallback_sampler = fallback_sampler
        self.logits_processors = logits_processors
        self.stop_sequences = stop_sequences
        self.max_tokens = max_tokens

        if self.samplers and len(self.samplers) != len(self.uids):
            raise ValueError("Insufficient number of samplers provided")
        if self.logits_processors and len(self.logits_processors) != len(self.uids):
            raise ValueError("Insufficient number of logits_processors provided")

        self._current_tokens = None
        self._current_logprobs = []
        self._next_tokens = inputs
        self._next_logprobs = []
        self._token_context = [TokenBuffer(t) for t in tokens]
        self._num_tokens = [0] * len(self.uids)
        self._matchers = [ss.matcher() for ss in stop_sequences]

        if self.uids:
            self._step()

    def __len__(self):
        return len(self.uids)

    def extend(self, batch):
        """Extend this batch with another generation batch."""
        self.uids.extend(batch.uids)
        self.prompt_cache = _extend_cache(self.prompt_cache, batch.prompt_cache)
        self.tokens.extend(batch.tokens)
        self.samplers.extend(batch.samplers)
        self.logits_processors.extend(batch.logits_processors)
        self.max_tokens.extend(batch.max_tokens)
        self.stop_sequences.extend(batch.stop_sequences)
        if self._current_tokens is None:
            self._current_tokens = batch._current_tokens
            self._current_logprobs = batch._current_logprobs
        elif batch._current_tokens is not None:
            self._current_tokens = mx.concatenate(
                [self._current_tokens, batch._current_tokens]
            )
            self._current_logprobs.extend(batch._current_logprobs)
        if self._next_tokens is None:
            self._next_tokens = batch._next_tokens
            self._next_logprobs = batch._next_logprobs
        elif batch._next_tokens is not None:
            self._next_tokens = mx.concatenate([self._next_tokens, batch._next_tokens])
            self._next_logprobs.extend(batch._next_logprobs)
        self._token_context.extend(batch._token_context)
        self._num_tokens.extend(batch._num_tokens)
        self._matchers.extend(batch._matchers)

    def _step(self) -> Tuple[List[int], List[mx.array]]:
        """
        Perform a single generation step.

        Returns:
            Tuple of token list and logprobs list.
        """
        self._current_tokens = self._next_tokens
        self._current_logprobs = self._next_logprobs
        inputs = self._current_tokens

        # Forward pass
        logits = self.model(inputs[:, None], cache=self.prompt_cache)
        logits = logits[:, -1, :]

        # Logits processors
        token_context = []
        if any(self.logits_processors):
            processed_logits = []
            for e in range(len(self.uids)):
                sample_logits = logits[e : e + 1]
                # Only sequences with processors need a context.
                processors = self.logits_processors[e] or ()
                if processors:
                    context = self._token_context[e].update_and_fetch(inputs[e : e + 1])
                    token_context.append(context)
                    for processor in processors:
                        sample_logits = processor(context, sample_logits)
                processed_logits.append(sample_logits)
            logits = mx.concatenate(processed_logits, axis=0)

        # Normalize the logits
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

        # One batched call when the batch shares a sampler.
        samplers = [
            self.fallback_sampler if s is None else s
            for s in (self.samplers or [None] * len(self.uids))
        ]
        if not samplers or all(s is samplers[0] for s in samplers):
            sampled = (samplers[0] if samplers else self.fallback_sampler)(logprobs)
        else:
            sampled = mx.concatenate(
                [s(logprobs[e : e + 1]) for e, s in enumerate(samplers)], axis=0
            )

        # Assign the next step to member variables and start computing it
        # asynchronously
        self._next_tokens = sampled
        self._next_logprobs = list(logprobs)
        mx.async_eval(self._next_tokens, self._next_logprobs, token_context)

        # Eval the current tokens and current logprobs. After that also add
        # them to self.tokens so that it always represents the tokens contained
        # in the KV Cache.
        mx.eval(inputs, self._current_logprobs)
        inputs = inputs.tolist()
        for sti, ti in zip(self.tokens, inputs):
            sti.append(ti)
        return inputs, self._current_logprobs

    def extract_cache(self, idx: int) -> List[Any]:
        return [c.extract(idx) for c in self.prompt_cache]

    def filter(self, keep: List[int]):
        """Filter the batch to keep only the specified indices."""
        self.uids = [self.uids[idx] for idx in keep]
        if not keep:
            self.prompt_cache.clear()
        else:
            for c in self.prompt_cache:
                c.filter(keep)
        self.tokens = [self.tokens[idx] for idx in keep]
        self.samplers = [self.samplers[idx] for idx in keep]
        self.logits_processors = [self.logits_processors[idx] for idx in keep]
        self.max_tokens = [self.max_tokens[idx] for idx in keep]
        self.stop_sequences = [self.stop_sequences[idx] for idx in keep]

        self._next_tokens = self._next_tokens[keep] if keep else None
        self._next_logprobs = [self._next_logprobs[idx] for idx in keep]
        self._token_context = [self._token_context[idx] for idx in keep]
        self._num_tokens = [self._num_tokens[idx] for idx in keep]
        self._matchers = [self._matchers[idx] for idx in keep]

    def next(self) -> List[Response]:
        """
        Generate the next batch of tokens.

        Returns:
            List of Response objects for each sequence in the batch.
        """
        if not self.uids:
            return []

        tokens, logprobs = self._step()

        keep = []
        responses = []
        for i in range(len(self.uids)):
            finish_reason = None

            self._num_tokens[i] += 1
            if self._num_tokens[i] >= self.max_tokens[i]:
                finish_reason = "length"

            if self._matchers[i].advance(tokens[i]):
                finish_reason = "stop"

            if finish_reason is not None:
                responses.append(
                    self.Response(
                        uid=self.uids[i],
                        token=tokens[i],
                        logprobs=logprobs[i],
                        finish_reason=finish_reason,
                        prompt_cache=self.extract_cache(i),
                        all_tokens=self.tokens[i],
                    )
                )
            else:
                keep.append(i)
                responses.append(
                    self.Response(
                        uid=self.uids[i],
                        token=tokens[i],
                        logprobs=logprobs[i],
                        finish_reason=None,
                        prompt_cache=None,
                        all_tokens=None,
                    )
                )

        if len(keep) < len(self.uids):
            self.filter(keep)

        return responses

    @classmethod
    def empty(
        cls,
        model: nn.Module,
        fallback_sampler: Sampler,
    ):
        return cls(
            model=model,
            fallback_sampler=fallback_sampler,
            uids=[],
            inputs=mx.array([], dtype=mx.uint32),
            prompt_cache=[],
            tokens=[],
            samplers=[],
            logits_processors=[],
            max_tokens=[],
            stop_sequences=[],
        )


class BatchGenerator:
    """
    A batch generator implements continuous batching.

    This class provides automatic management of prompt processing and generation
    batches, handling the transition between the two.

    It also allows for segmented prompt processing which guarantees that the
    generator will stop at these boundaries when processing an input.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        max_tokens: int = 128,
        stop_tokens: Optional[Sequence[Sequence[int]]] = None,
        sampler: Optional[Sampler] = None,
        logits_processors: Optional[List[LogitsProcessor]] = None,
        completion_batch_size: int = 32,
        prefill_batch_size: int = 8,
        prefill_step_size: int = 2048,
        max_kv_size: Optional[int] = None,
        stream=None,
        kv_bits: Optional[int] = None,
        kv_group_size: int = 64,
    ):
        self.model = model
        self.max_tokens = max_tokens
        # Full-attention layers' KV cache quantized in the batch, as the
        # unbatched path quantizes it (from the first token: a batch has
        # no quantized_kv_start).
        self.kv_bits = kv_bits
        self.kv_group_size = kv_group_size
        self.sampler = sampler or greedy_sampler
        self.logits_processors = logits_processors or []
        self.uid_count = 0
        self.prefill_step_size = prefill_step_size
        self.prefill_batch_size = prefill_batch_size
        self.completion_batch_size = max(completion_batch_size, prefill_batch_size)
        self.max_kv_size = max_kv_size

        self._stream = stream or generation_stream

        self._default_stop_sequences = StopSequences(
            stop_tokens if stop_tokens else None,
        )
        self._uid_count = 0
        self._prompt_batch = PromptProcessingBatch.empty(
            self.model,
            self.sampler,
            prefill_step_size=prefill_step_size,
        )
        self._generation_batch = GenerationBatch.empty(self.model, self.sampler)
        self._unprocessed_sequences = deque()
        self._currently_processing = []

        self._counters = BatchCounters()

        self._old_wired_limit = maybe_set_recommended_wired_limit()

    @property
    def stream(self):
        return self._stream

    def close(self):
        if self._old_wired_limit is not None:
            mx.synchronize(self._stream)
            mx.set_wired_limit(self._old_wired_limit)
            self._old_wired_limit = None

    def __del__(self):
        self.close()

    @contextlib.contextmanager
    def stats(self):
        """The work this generator does while the block runs.

        Readings are taken on entry and exit and subtracted, so nested windows
        each report their own interval.
        """
        stats = BatchStats()
        start = BatchCountersSnapshot(**asdict(self._counters))
        try:
            yield stats
        finally:
            end = BatchCountersSnapshot(**asdict(self._counters))
            stats += BatchCountersSnapshot.between(start, end)

    def insert(
        self,
        prompts: List[List[int]],
        max_tokens: Optional[List[int]] = None,
        caches: Optional[List[List[Any]]] = None,
        all_tokens: Optional[List[List[int]]] = None,
        samplers: Optional[List[Sampler]] = None,
        logits_processors: Optional[List[List[LogitsProcessor]]] = None,
        stop_sequences: Optional[List[StopSequences]] = None,
    ):
        return self.insert_segments(
            [[p] for p in prompts],
            max_tokens,
            caches,
            all_tokens,
            samplers,
            logits_processors,
            stop_sequences,
        )

    def insert_segments(
        self,
        segments: List[List[List[int]]],
        max_tokens: Optional[List[int]] = None,
        caches: Optional[List[List[Any]]] = None,
        all_tokens: Optional[List[List[int]]] = None,
        samplers: Optional[List[Sampler]] = None,
        logits_processors: Optional[List[List[LogitsProcessor]]] = None,
        stop_sequences: Optional[List[StopSequences]] = None,
    ):
        num_segments = len(segments)

        max_tokens = max_tokens or [self.max_tokens] * num_segments
        all_tokens = all_tokens or [[] for _ in segments]
        samplers = _normalize_samplers(samplers, n=num_segments, fallback=self.sampler)
        logits_processors = _normalize_logits_processors(
            logits_processors or [self.logits_processors] * num_segments,
            n=num_segments,
        )
        stop_sequences = stop_sequences or (
            [self._default_stop_sequences] * num_segments
        )

        caches = caches or [None] * num_segments

        # Validate before touching any state: ``zip`` pairs up to the shortest.
        opts = {
            "max_tokens": max_tokens,
            "caches": caches,
            "all_tokens": all_tokens,
            "samplers": samplers,
            "logits_processors": logits_processors,
            "stop_sequences": stop_sequences,
        }
        for k, v in opts.items():
            if len(v) != num_segments:
                raise ValueError(
                    f"Option {k} must have one entry per segment: expected "
                    f"{num_segments}, got {len(v)}."
                )

        uids = list(range(self._uid_count, self._uid_count + num_segments))
        pending = []
        for uid, seq, m, c, at, s, lp, sm in zip(
            uids,
            segments,
            max_tokens,
            caches,
            all_tokens,
            samplers,
            logits_processors,
            stop_sequences,
        ):
            seq = [segment for segment in seq if segment]
            if not seq:
                raise ValueError(f"Sequence {uid} has an empty prompt.")
            if m <= 0:
                raise ValueError(f"Sequence {uid}'s max_tokens must be > 0.")
            if len(seq[-1]) != 1:
                seq.append(seq[-1][-1:])
                seq[-2] = seq[-2][:-1]
            if c is None:
                c = make_prompt_cache(self.model, self.max_kv_size)
            c = _batch_layers(c, self.kv_bits, self.kv_group_size)
            pending.append((uid, seq, m, c, at, s, lp, sm))

        self._unprocessed_sequences.extend(pending)
        self._uid_count += num_segments

        return uids

    def _find_uids(self, uids):
        uids = set(uids)
        results = {}
        for i, uid_i in enumerate(self._generation_batch.uids):
            if uid_i in uids:
                results[uid_i] = (2, i)
        for i, uid_i in enumerate(self._prompt_batch.uids):
            if uid_i in uids:
                results[uid_i] = (1, i)
        for i, seq in enumerate(self._unprocessed_sequences):
            if seq[0] in uids:
                results[seq[0]] = (0, i)
        return results

    def extract_cache(self, uids):
        results = {}
        for uid, (stage, idx) in self._find_uids(uids).items():
            if stage == 0:
                results[uid] = self._unprocessed_sequences[idx][3:5]
            elif stage == 1:
                results[uid] = (
                    self._prompt_batch.extract_cache(idx),
                    self._prompt_batch.tokens[idx],
                )
            else:
                results[uid] = (
                    self._generation_batch.extract_cache(idx),
                    self._generation_batch.tokens[idx],
                )
        return results

    def remove(self, uids, return_prompt_caches=False):
        caches = {}
        if return_prompt_caches:
            caches = self.extract_cache(uids)

        keep = (
            set(range(len(self._unprocessed_sequences))),
            set(range(len(self._prompt_batch))),
            set(range(len(self._generation_batch))),
        )
        for stage, idx in self._find_uids(uids).values():
            keep[stage].remove(idx)

        if len(keep[0]) < len(self._unprocessed_sequences):
            self._unprocessed_sequences = deque(
                x for i, x in enumerate(self._unprocessed_sequences) if i in keep[0]
            )
        if len(keep[1]) < len(self._prompt_batch):
            self._prompt_batch.filter(sorted(keep[1]))
            self._currently_processing = [
                x for i, x in enumerate(self._currently_processing) if i in keep[1]
            ]
        if len(keep[2]) < len(self._generation_batch):
            self._generation_batch.filter(sorted(keep[2]))

        return caches

    @property
    def prompt_cache_nbytes(self):
        total = sum(c.nbytes for p in self._unprocessed_sequences for c in p[3])
        total += sum(c.nbytes for c in self._prompt_batch.prompt_cache)
        total += sum(c.nbytes for c in self._generation_batch.prompt_cache)
        return total

    def _make_batch(self, n: int):
        uids = []
        caches = []
        tokens = []
        samplers = []
        logits_processors = []
        max_tokens = []
        stop_sequences = []
        for _ in range(n):
            sequence = self._unprocessed_sequences.popleft()
            uids.append(sequence[0])
            caches.append(sequence[3])
            tokens.append(sequence[4])
            samplers.append(sequence[5])
            logits_processors.append(sequence[6])
            max_tokens.append(sequence[2])
            stop_sequences.append(sequence[7])
            self._currently_processing.append(
                [sequence[1], 0, sum(len(s) for s in sequence[1])]
            )

        return PromptProcessingBatch(
            model=self.model,
            uids=uids,
            caches=caches,
            tokens=tokens,
            prefill_step_size=self.prefill_step_size,
            samplers=samplers,
            fallback_sampler=self.sampler,
            logits_processors=logits_processors,
            stop_sequences=stop_sequences,
            max_tokens=max_tokens,
        )

    def _next(self):
        generation_responses = []
        prompt_responses = []

        # Generate tokens first
        if len(self._generation_batch) > 0:
            tic = time.perf_counter()
            generation_responses = self._generation_batch.next()
            self._counters.decode_time += time.perf_counter() - tic
            self._counters.generation_tokens += len(generation_responses)
            self._counters.generation_steps += 1
            if self._counters.generation_steps % 512 == 0:
                mx.clear_cache()

        # Exit early because we already have our hands full with decoding
        if len(self._generation_batch) >= self.completion_batch_size:
            return prompt_responses, generation_responses

        # Check if we have sequences and add them to the prompt batch
        n = min(
            self.prefill_batch_size - len(self._prompt_batch),
            self.completion_batch_size - len(self._generation_batch),
            len(self._unprocessed_sequences),
        )
        if n > 0:
            self._prompt_batch.extend(self._make_batch(n))

        # Split the prompt sequences to the ones moving to generation and the rest
        keep = []
        split = []
        for i, seq in enumerate(self._currently_processing):
            segments = seq[0]
            if len(segments) == 1 and len(segments[0]) == 1:
                split.append(i)
            else:
                keep.append(i)

        # Actually split off part of the prompt batch and start generation
        if split:
            last_inputs = [self._currently_processing[i][0][0] for i in split]
            progress = [(self._currently_processing[i][2],) * 2 for i in split]
            self._currently_processing = [self._currently_processing[i] for i in keep]
            gen_batch = self._prompt_batch.split(split).generate(last_inputs)
            for i, p in enumerate(progress):
                prompt_responses.append(
                    PromptProcessingBatch.Response(
                        gen_batch.uids[i],
                        p,
                        True,
                        True,
                    )
                )
            self._generation_batch.extend(gen_batch)

        # Extract the next prompts input
        prompts = []
        for i, seq in enumerate(self._currently_processing):
            response = PromptProcessingBatch.Response(
                self._prompt_batch.uids[i], 0, False, False
            )
            segments = seq[0]
            n = min(len(segments[0]), self.prefill_step_size)
            prompts.append(segments[0][:n])
            segments[0] = segments[0][n:]
            if len(segments[0]) == 0:
                segments.pop(0)
                response.end_of_segment = True
            seq[1] += len(prompts[-1])
            response.progress = (seq[1], seq[2])
            prompt_responses.append(response)

        # Process the prompts
        self._counters.prompt_tokens += sum(len(p) for p in prompts)
        tic = time.perf_counter()
        self._prompt_batch.prompt(prompts)
        self._counters.prompt_time += time.perf_counter() - tic

        return prompt_responses, generation_responses

    def next(self):
        """
        Get the next batch of responses.

        Returns:
            Tuple of prompt processing responses and generation responses.
        """
        with mx.stream(self._stream):
            return self._next()

    def next_generated(self):
        """
        Return only generated tokens ignoring batch generation responses.

        Returns:
            List of GenerationBatch.Response objects
        """
        with mx.stream(self._stream):
            while True:
                prompt_responses, generation_responses = self._next()
                if not generation_responses and prompt_responses:
                    continue
                return generation_responses


@dataclass
class BatchResponse:
    """
    A data object to hold a batch generation response.

    Args:
        texts: (List[str]): The generated text for each prompt.
        stats (BatchStats): Statistics about the generation.
        caches: Optional prompt caches for each sequence.
        token_ids (Optional[List[List[int]]]): The generated token IDs for each
            prompt. Only present when ``return_token_ids=True``.
        logprobs (Optional[List[List[float]]]): The per-token log-probabilities
            of the sampled tokens for each prompt. Only present when
            ``return_logprobs=True``.
    """

    texts: List[str]
    stats: BatchStats
    caches: Optional[List[List[Any]]]
    token_ids: Optional[List[List[int]]] = None
    logprobs: Optional[List[List[float]]] = None


def batch_generate(
    model,
    tokenizer,
    prompts: List[List[int]],
    prompt_caches: Optional[List[List[Any]]] = None,
    max_tokens: Union[int, List[int]] = 128,
    verbose: bool = False,
    return_prompt_caches: bool = False,
    return_token_ids: bool = False,
    return_logprobs: bool = False,
    **kwargs,
) -> BatchResponse:
    """
    Generate responses for the given batch of prompts.

    Args:
       model (nn.Module): The language model.
       tokenizer (PreTrainedTokenizer): The tokenizer.
       prompts (List[List[int]]): The input prompts.
       prompt_caches (List[List[Any]], optional): Pre-computed prompt-caches
          for each input prompt. Note, unlike ``generate_step``, the caches
          won't be updated in-place.
       verbose (bool): If ``True``, print tokens and timing information.
          Default: ``False``.
       max_tokens (Union[int, List[int]): Maximum number of output tokens. This
          can be per prompt if a list is provided.
       return_prompt_caches (bool): Return the prompt caches in the batch
          responses. Default: ``False``.
       return_token_ids (bool): Return the generated token IDs in the batch
          responses. Default: ``False``.
       return_logprobs (bool): Return the per-token log-probability of the
          sampled token for each generated token. Useful for reinforcement
          learning (e.g. RLOO, PPO) where behavior log-probabilities are needed
          for importance weighting. Default: ``False``.
       kwargs: The remaining options get passed to :obj:`BatchGenerator`.
          See :obj:`BatchGenerator` for more details.
    """

    gen = BatchGenerator(
        model,
        stop_tokens=[[t] for t in tokenizer.eos_token_ids],
        **kwargs,
    )
    num_samples = len(prompts)
    fin = 0
    if verbose:
        print(f"[batch_generate] Finished processing 0/{num_samples} ...", end="\r")

    if isinstance(max_tokens, int):
        max_tokens = [max_tokens] * len(prompts)

    uids = gen.insert(prompts, max_tokens, caches=prompt_caches)
    results = {uid: [] for uid in uids}
    logprob_results = {uid: [] for uid in uids} if return_logprobs else None
    prompt_caches = {}
    with gen.stats() as stats:
        while responses := gen.next_generated():
            for r in responses:
                if r.finish_reason is not None:
                    if return_prompt_caches:
                        prompt_caches[r.uid] = r.prompt_cache
                    if verbose:
                        fin += 1
                        print(
                            f"[batch_generate] Finished processing {fin}/{num_samples} ...",
                            end="\r",
                        )
                if r.finish_reason != "stop":
                    results[r.uid].append(r.token)
                    if return_logprobs:
                        logprob_results[r.uid].append(r.logprobs[r.token].item())
    gen.close()
    if verbose:
        print(f"[batch_generate] Finished processing {fin}/{num_samples}")

    # Return results in correct order
    texts = [tokenizer.decode(results[uid]) for uid in uids]
    caches = [prompt_caches[uid] for uid in uids] if return_prompt_caches else None
    token_ids = [results[uid] for uid in uids] if return_token_ids else None
    logprobs = [logprob_results[uid] for uid in uids] if return_logprobs else None
    if verbose:
        print(
            f"[batch_generate] Prompt: {stats.prompt_tokens} tokens, {stats.prompt_tps:.3f} tokens-per-sec"
        )
        print(
            f"[batch_generate] Generation: {stats.generation_tokens} tokens, "
            f"{stats.generation_tps:.3f} tokens-per-sec"
        )
        print(f"[batch_generate] Peak memory: {stats.peak_memory:.3f} GB")
    return BatchResponse(texts, stats, caches, token_ids, logprobs)


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    if args.seed is not None:
        mx.random.seed(args.seed)

    # Load the prompt cache and metadata if a cache file is provided
    using_cache = args.prompt_cache_file is not None
    if using_cache:
        prompt_cache, metadata = load_prompt_cache(
            args.prompt_cache_file,
            return_metadata=True,
        )
        if isinstance(prompt_cache[0], QuantizedKVCache):
            if args.kv_bits is not None and args.kv_bits != prompt_cache[0].bits:
                raise ValueError(
                    "--kv-bits does not match the kv cache loaded from --prompt-cache-file."
                )
            if args.kv_group_size != prompt_cache[0].group_size:
                raise ValueError(
                    "--kv-group-size does not match the kv cache loaded from --prompt-cache-file."
                )

    # Building tokenizer_config
    tokenizer_config = (
        {} if not using_cache else json.loads(metadata["tokenizer_config"])
    )
    tokenizer_config["trust_remote_code"] = args.trust_remote_code

    model_path = args.model
    if using_cache:
        if model_path is None:
            model_path = metadata["model"]
        elif model_path != metadata["model"]:
            raise ValueError(
                f"Providing a different model ({model_path}) than that "
                f"used to create the prompt cache ({metadata['model']}) "
                "is an error."
            )
    model_path = model_path or DEFAULT_MODEL

    model, tokenizer = load(
        model_path,
        adapter_path=args.adapter_path,
        tokenizer_config=tokenizer_config,
        model_config={"quantize_activations": args.quantize_activations},
        trust_remote_code=args.trust_remote_code,
    )
    for eos_token in args.extra_eos_token:
        tokenizer.add_eos_token(eos_token)

    template_kwargs = {}
    if args.chat_template_config is not None:
        template_kwargs = json.loads(args.chat_template_config)

    prompt = args.prompt.replace("\\n", "\n").replace("\\t", "\t")
    prompt = sys.stdin.read() if prompt == "-" else prompt
    if not args.ignore_chat_template and tokenizer.has_chat_template:
        if args.system_prompt is not None:
            messages = [{"role": "system", "content": args.system_prompt}]
        else:
            messages = []
        messages.append({"role": "user", "content": prompt})

        has_prefill = args.prefill_response is not None
        if has_prefill:
            messages.append({"role": "assistant", "content": args.prefill_response})
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            continue_final_message=has_prefill,
            add_generation_prompt=not has_prefill,
            **template_kwargs,
        )

        # Treat the prompt as a suffix assuming that the prefix is in the
        # stored kv cache.
        if using_cache:
            messages[-1]["content"] = "<query>"
            test_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                continue_final_message=has_prefill,
                add_generation_prompt=not has_prefill,
            )
            prompt = prompt[test_prompt.index("<query>") :]
        prompt = tokenizer.encode(prompt, add_special_tokens=False)
    else:
        prompt = tokenizer.encode(prompt)

    if args.draft_model is not None:
        draft_model, draft_tokenizer = load(args.draft_model)
        if draft_tokenizer.vocab_size != tokenizer.vocab_size:
            raise ValueError("Draft model tokenizer does not match model tokenizer.")
    else:
        draft_model = None
    sampler = make_sampler(
        args.temp,
        args.top_p,
        args.min_p,
        args.min_tokens_to_keep,
        top_k=args.top_k,
        xtc_probability=args.xtc_probability,
        xtc_threshold=args.xtc_threshold,
        xtc_special_tokens=tokenizer.encode("\n") + list(tokenizer.eos_token_ids),
    )
    response = generate(
        model,
        tokenizer,
        prompt,
        max_tokens=args.max_tokens,
        verbose=args.verbose,
        sampler=sampler,
        max_kv_size=args.max_kv_size,
        prefill_step_size=args.prefill_step_size,
        prompt_cache=prompt_cache if using_cache else None,
        kv_bits=args.kv_bits,
        kv_group_size=args.kv_group_size,
        quantized_kv_start=args.quantized_kv_start,
        draft_model=draft_model,
        num_draft_tokens=args.num_draft_tokens,
    )
    if not args.verbose:
        print(response)


if __name__ == "__main__":
    print(
        "Calling `python -m mlx_lm.generate...` directly is deprecated."
        " Use `mlx_lm.generate...` or `python -m mlx_lm generate ...` instead."
    )
    main()
