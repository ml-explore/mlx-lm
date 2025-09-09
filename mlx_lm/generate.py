# Copyright © 2023-2024 Apple Inc.

import argparse
import contextlib
import functools
import json
import sys
import time
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
)

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_reduce
from transformers import PreTrainedTokenizer

from .models import cache
from .models.cache import (
    BatchKVCache,
    QuantizedKVCache,
    load_prompt_cache,
)
from .sample_utils import make_sampler
from .tokenizer_utils import TokenizerWrapper
from .utils import does_model_support_input_embeddings, load

DEFAULT_PROMPT = "hello"
DEFAULT_MAX_TOKENS = 100
DEFAULT_TEMP = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_MIN_P = 0.0
DEFAULT_TOP_K = 0
DEFAULT_XTC_PROBABILITY = 0.0
DEFAULT_XTC_THRESHOLD = 0.0
DEFAULT_MIN_TOKENS_TO_KEEP = 1
DEFAULT_SEED = None
DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
DEFAULT_QUANTIZED_KV_START = 5000


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
        default=0.0,
        help="Thresold the probs of each next token candidate to be sampled by XTC",
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
        "--prompt-cache-file",
        type=str,
        default=None,
        help="A file containing saved KV caches to avoid recomputing them",
    )
    parser.add_argument(
        "--kv-bits",
        type=int,
        help="Number of bits for KV cache quantization. "
        "Defaults to no quantization.",
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
generation_stream = mx.new_stream(mx.default_device())


@contextlib.contextmanager
def wired_limit(model: nn.Module, streams: Optional[List[mx.Stream]] = None):
    """
    A context manager to temporarily change the wired limit.

    Note, the wired limit should not be changed during an async eval.  If an
    async eval could be running pass in the streams to synchronize with prior
    to exiting the context manager.
    """
    if not mx.metal.is_available():
        try:
            yield
        finally:
            pass
    else:
        model_bytes = tree_reduce(
            lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc, model, 0
        )
        max_rec_size = mx.metal.device_info()["max_recommended_working_set_size"]
        if model_bytes > 0.9 * max_rec_size:
            model_mb = model_bytes // 2**20
            max_rec_mb = max_rec_size // 2**20
            print(
                f"[WARNING] Generating with a model that requires {model_mb} MB "
                f"which is close to the maximum recommended size of {max_rec_mb} "
                "MB. This can be slow. See the documentation for possible work-arounds: "
                "https://github.com/ml-explore/mlx-lm/tree/main#large-models"
            )
        old_limit = mx.set_wired_limit(max_rec_size)
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


def maybe_quantize_kv_cache(prompt_cache, quantized_kv_start, kv_group_size, kv_bits):
    if (
        kv_bits is not None
        and not isinstance(prompt_cache[0], cache.QuantizedKVCache)
        and prompt_cache[0].offset > quantized_kv_start
    ):
        for i in range(len(prompt_cache)):
            if isinstance(prompt_cache[i], cache.KVCache):
                prompt_cache[i] = prompt_cache[i].to_quantized(
                    group_size=kv_group_size, bits=kv_bits
                )


def generate_step(
    prompt: mx.array,
    model: nn.Module,
    *,
    max_tokens: int = 256,
    sampler: Optional[Callable[mx.array, mx.array]] = None,
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
    max_kv_size: Optional[int] = None,
    prompt_cache: Optional[Any] = None,
    prefill_step_size: int = 2048,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    prompt_progress_callback: Optional[Callable[int, int]] = None,
    input_embeddings: Optional[mx.array] = None,
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    """
    A generator producing token ids based on the given prompt from the model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        max_tokens (int): The maximum number of tokens. Use``-1`` for an infinite
          generator. Default: ``256``.
        sampler (Callable[mx.array, mx.array], optional): A sampler for sampling a
          token from a vector of log probabilities. Default: ``None``.
        logits_processors (List[Callable[[mx.array, mx.array], mx.array]], optional):
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
           when ``kv_bits`` is non-None. Default: ``0``.
        prompt_progress_callback (Callable[int, int]): A call-back which takes the
           prompt tokens processed so far and the total number of prompt tokens.
        input_embeddings (mx.array, optional): Input embeddings to use instead of or in
          conjunction with prompt tokens. Default: ``None``.

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

    tokens = None

    # Create the KV cache for generation
    if prompt_cache is None:
        prompt_cache = cache.make_prompt_cache(
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

    sampler = sampler or (lambda x: mx.argmax(x, axis=-1))

    def _model_call(input_tokens: mx.array, input_embeddings: Optional[mx.array]):
        if input_embeddings is not None:
            return model(
                input_tokens, cache=prompt_cache, input_embeddings=input_embeddings
            )
        else:
            return model(input_tokens, cache=prompt_cache)

    def _step(input_tokens: mx.array, input_embeddings: Optional[mx.array] = None):
        nonlocal tokens

        with mx.stream(generation_stream):
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

    with mx.stream(generation_stream):
        total_prompt_tokens = (
            len(input_embeddings) if input_embeddings is not None else len(prompt)
        )
        prompt_processed_tokens = 0
        prompt_progress_callback(prompt_processed_tokens, total_prompt_tokens)
        while total_prompt_tokens - prompt_processed_tokens > 1:
            n_to_process = min(prefill_step_size, prompt.size - 1)
            _model_call(
                input_tokens=prompt[:n_to_process][None],
                input_embeddings=(
                    input_embeddings[:n_to_process][None]
                    if input_embeddings is not None
                    else None
                ),
            )
            quantize_cache_fn(prompt_cache)
            mx.eval([c.state for c in prompt_cache])
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
    *,
    num_draft_tokens=2,
    max_tokens: int = 256,
    sampler: Optional[Callable[mx.array, mx.array]] = None,
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
    prompt_cache: Optional[Any] = None,
    prefill_step_size: int = 512,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
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
        sampler (Callable[mx.array, mx.array], optional): A sampler for sampling a
          token from a vector of log probabilities. Default: ``None``.
        logits_processors (List[Callable[[mx.array, mx.array], mx.array]], optional):
          A list of functions that take tokens and logits and return the processed
          logits. Default: ``None``.
        prompt_cache (List[Any], optional): A pre-computed prompt cache. Note, if
          provided, the cache will be updated in place. The cache must be trimmable.
        prefill_step_size (int): Step size for processing the prompt.
        kv_bits (int, optional): Number of bits to use for KV cache quantization.
          None implies no cache quantization. Default: ``None``.
        kv_group_size (int): Group size for KV cache quantization. Default: ``64``.
        quantized_kv_start (int): Step to begin using a quantized KV cache.
           when ``kv_bits`` is non-None. Default: ``0``.

    Yields:
        Tuple[mx.array, mx.array, bool]: One token, a vector of log probabilities,
          and a bool indicating if the token was generated by the draft model
    """

    y = prompt.astype(mx.uint32)
    prev_tokens = None

    # Create the KV cache for generation
    if prompt_cache is None:
        model_cache = cache.make_prompt_cache(model)
        draft_cache = cache.make_prompt_cache(draft_model)
    else:
        model_cache = prompt_cache[: len(model.layers)]
        draft_cache = prompt_cache[len(model.layers) :]

    sampler = sampler or (lambda x: mx.argmax(x, axis=-1))

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
        with mx.stream(generation_stream):
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
                        mx.concat([prev_tokens, y]) if prev_tokens is not None else y
                    )
                    y, logprobs = _process_and_sample(prev_tokens, logits[:, i, :])
                    out_y.append(y)
                    out_logprobs.append(logprobs)
                return mx.concatenate(out_y, axis=0), mx.concatenate(
                    out_logprobs, axis=0
                )
            else:
                return _process_and_sample(None, logits.squeeze(0))

    def _prefill(model, cache, y):
        while y.size > prefill_step_size:
            model(y[:prefill_step_size][None], cache=cache)
            quantize_cache_fn(cache)
            mx.eval([c.state for c in cache])
            y = y[prefill_step_size:]
            mx.clear_cache()
        return y

    def _rewind_cache(num_draft, num_accept):
        cache.trim_prompt_cache(model_cache, num_draft - num_accept)
        cache.trim_prompt_cache(draft_cache, max(num_draft - num_accept - 1, 0))

    def _draft_generate(y, num_draft):
        if num_draft == 0:
            return mx.array([], mx.uint32)
        ys = []
        for _ in range(num_draft):
            y, _ = _step(draft_model, draft_cache, y)
            mx.async_eval(y)
            ys.append(y)
        return mx.concatenate(ys)

    with mx.stream(generation_stream):
        draft_y = _prefill(draft_model, draft_cache, y)
        y = _prefill(model, model_cache, y)

    ntoks = 0
    # Set these so the finally block doesn't raise
    num_draft = 0
    n = 0
    try:
        while True:
            num_draft = min(max_tokens - ntoks, num_draft_tokens)
            draft_tokens = _draft_generate(draft_y, num_draft)
            if prev_tokens is not None:
                prev_tokens = prev_tokens[: prev_tokens.size - y.size - num_draft + 1]
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
                yield tn, lpn, True
                if ntoks == max_tokens:
                    break
            if ntoks < max_tokens:
                ntoks += 1
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
    finally:
        _rewind_cache(num_draft, n)


def stream_generate(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: Union[str, mx.array, List[int]],
    draft_model: Optional[nn.Module] = None,
    **kwargs,
) -> Generator[GenerationResponse, None, None]:
    """
    A generator producing text based on the given prompt from the model.

    Args:
        model (nn.Module): The model to use for generation.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        prompt (Union[str, mx.array, List[int]]): The input prompt string or
          integer tokens.
        draft_model (Optional[nn.Module]): An optional draft model. If provided
          then speculative decoding is used. The draft model must use the same
          tokenizer as the main model. Default: ``None``.
        kwargs: The remaining options get passed to :func:`generate_step`.
          See :func:`generate_step` for more details.

    Yields:
        GenerationResponse: An instance containing the generated text segment and
            associated metadata. See :class:`GenerationResponse` for details.
    """
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

    if draft_model is None:
        kwargs.pop("num_draft_tokens", None)
        token_generator = generate_step(prompt, model, **kwargs)
        # from_draft always false for non-speculative generation
        token_generator = (
            (token, logprobs, False) for token, logprobs in token_generator
        )
    else:
        kwargs.pop("max_kv_size", None)
        kwargs.pop("prompt_progress_callback", None)
        token_generator = speculative_generate_step(
            prompt, model, draft_model, **kwargs
        )
    with wired_limit(model, [generation_stream]):
        tic = time.perf_counter()
        for n, (token, logprobs, from_draft) in enumerate(token_generator):
            if n == 0:
                prompt_time = time.perf_counter() - tic
                prompt_tps = prompt.size / prompt_time
                tic = time.perf_counter()
            if token in tokenizer.eos_token_ids:
                break

            detokenizer.add_token(token)

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


def _left_pad_prompts(prompts, max_length=None):
    if max_length is None:
        max_length = max(len(p) for p in prompts)
    return mx.array([[0] * (max_length - len(p)) + p for p in prompts])


@dataclass
class CompletionResponse:
    uid: int
    token: int
    logprobs: mx.array
    finish_reason: Optional[str]


@dataclass
class BatchStats:
    """
    An data object to hold generation stats.

    Args:
        prompt_tokens (int): The number of prompt tokens processed.
        prompt_tps (float): The prompt processing tokens-per-second.
        prompt_time (float): The time in seconds spent in prompt processing.
        generation_tokens (int): The number of generated tokens.
        generation_tps (float): The tokens-per-second for generation.
        generation_time (float): The time in seconds spent in generation .
        peak_memory (float): The peak memory used so far in GB.
    """

    prompt_tokens: int = 0
    prompt_tps: float = 0
    prompt_time: float = 0
    generation_tokens: int = 0
    generation_tps: float = 0
    generation_time: float = 0
    peak_memory: float = 0


class BatchGenerator:
    def __init__(self, model, max_tokens=128, stop_tokens=None, sampler=None):
        self.model = model
        self.all_prompts = set()
        self.unprocessed_prompts = []
        self.max_tokens = max_tokens
        self.stop_tokens = stop_tokens or set()
        self.sampler = sampler or (lambda x: mx.argmax(x, keepdims=True, axis=-1))
        self.uid_count = 0
        # TODO tune these values to reasonable defaults possibly unique per
        # machine
        self.prompt_tokens_per_batch = 2**14
        self.completion_batch_size = 32
        self._stats = BatchStats()

        self._reset_state()

    def insert(self, prompts):
        uids = []
        for p in prompts:
            self.unprocessed_prompts.append((self.uid_count, p))
            uids.append(self.uid_count)
            self.uid_count += 1
        self.all_prompts.update(uids)
        return uids

    def _forward(self, input_tokens):
        return self.model(input_tokens, cache=self.prompt_cache)

    def _process_prompts(self, prompts):
        lengths = [len(p) for p in prompts]
        self._stats.prompt_tokens += sum(lengths)
        max_length = max(lengths)
        left_padding = [max_length - l for l in lengths]
        prefill_step_size = self.prompt_tokens_per_batch // self.completion_batch_size
        batch_size = len(prompts)
        while (2 * batch_size * prefill_step_size) < self.prompt_tokens_per_batch:
            prefill_step_size *= 2

        input_tokens = _left_pad_prompts(prompts, max_length=max_length)
        # TODO the model may not use a regular KV Cache
        self.prompt_cache = [BatchKVCache(left_padding) for _ in self.model.layers]

        with mx.stream(generation_stream):
            while input_tokens.shape[1] > 1:
                n_to_process = min(prefill_step_size, input_tokens.shape[1] - 1)
                self._forward(input_tokens[:, :n_to_process])
                mx.eval([c.state for c in self.prompt_cache])
                input_tokens = input_tokens[:, n_to_process:]
                mx.clear_cache()

        y, logprobs = self._step(input_tokens)
        mx.async_eval(y, logprobs)
        self.next_y, self.next_logprobs = y, logprobs

    def _step(self, input_tokens: mx.array):
        with mx.stream(generation_stream):
            logits = self._forward(input_tokens)
            logits = logits[:, -1, :]
            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            sampled = self.sampler(logprobs).squeeze(1)
            return sampled, logprobs

    def _reset_state(self):
        self.active_prompts = []
        self.prompt_cache = None
        self.n_complete = 0
        self.next_y = None
        self.next_logprobs = None

    def stats(self):
        self._stats.prompt_tps = self._stats.prompt_tokens / self._stats.prompt_time
        self._stats.generation_tps = (
            self._stats.generation_tokens / self._stats.generation_time
        )
        self._stats.peak_memory = mx.get_peak_memory() / 1e9
        return self._stats

    def next(self):
        tic = time.perf_counter()
        if len(self.active_prompts) == 0:
            # Process prompts
            prompts = self.unprocessed_prompts[: self.completion_batch_size]
            if len(prompts) == 0:
                return []
            uids, prompts = zip(*prompts)
            self.active_prompts.extend(uids)
            self._process_prompts(prompts)
            self.unprocessed_prompts = self.unprocessed_prompts[
                self.completion_batch_size :
            ]

        y, logprobs = self.next_y, self.next_logprobs

        if (self.n_complete + 1) < self.max_tokens:
            self.next_y, self.next_logprobs = self._step(y[:, None])
            mx.async_eval(self.next_y, self.next_logprobs)

        self.n_complete += 1

        mx.eval(y)
        toc = time.perf_counter()
        if self.n_complete == 1:
            self._stats.prompt_time += toc - tic
        else:
            self._stats.generation_time += toc - tic
        y = y.tolist()
        keep_idx = []
        end_idx = []
        responses = []

        finish_reason = None
        if self.n_complete == self.max_tokens:
            finish_reason == "length"

        for e, (t, uid) in enumerate(zip(y, self.active_prompts)):
            if t in self.stop_tokens:
                end_idx.append(e)
                responses.append(CompletionResponse(uid, t, logprobs[e], "stop"))
            else:
                responses.append(CompletionResponse(uid, t, logprobs[e], finish_reason))
                keep_idx.append(e)

        # Remove any finished completions
        if len(end_idx):
            self.active_prompts = [self.active_prompts[k] for k in keep_idx]
            if len(keep_idx) > 0:
                keep_idx = mx.array(keep_idx)
                self.next_y = self.next_y[keep_idx]
                self.next_logprobs = self.next_logprobs[keep_idx]
                for c in self.prompt_cache:
                    c.filter(keep_idx)

        if self.n_complete == self.max_tokens:
            self._reset_state()

        # Periodically clear out allocator cache
        if self.n_complete % 256 == 0:
            mx.clear_cache()

        self._stats.generation_tokens += len(responses)
        return responses


def batch_generate(
    model, tokenizer, prompts: List[int], verbose: bool = False, **kwargs
):
    gen = BatchGenerator(model, stop_tokens=tokenizer.eos_token_ids, **kwargs)
    uids = gen.insert(prompts)
    results = {uid: [] for uid in uids}
    while responses := gen.next():
        for response in responses:
            results[response.uid].append(response.token)
    # Return results in correct order
    texts = [tokenizer.decode(results[uid]) for uid in uids]
    if verbose:
        stats = gen.stats()
        print(
            f"Prompt: {stats.prompt_tokens} tokens, {stats.prompt_tps:.3f} tokens-per-sec"
        )
        print(
            f"Generation: {stats.generation_tokens} tokens, "
            f"{stats.generation_tps:.3f} tokens-per-sec"
        )
        print(f"Peak memory: {stats.peak_memory:.3f} GB")
    return texts


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
    tokenizer_config["trust_remote_code"] = True if args.trust_remote_code else None

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
    )
    for eos_token in args.extra_eos_token:
        tokenizer.add_eos_token(eos_token)

    template_kwargs = {}
    if args.chat_template_config is not None:
        template_kwargs = json.loads(args.chat_template_config)

    if args.use_default_chat_template:
        if tokenizer.chat_template is None:
            tokenizer.chat_template = tokenizer.default_chat_template
    elif using_cache:
        tokenizer.chat_template = json.loads(metadata["chat_template"])

    prompt = args.prompt.replace("\\n", "\n").replace("\\t", "\t")
    prompt = sys.stdin.read() if prompt == "-" else prompt
    if not args.ignore_chat_template and tokenizer.chat_template is not None:
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
