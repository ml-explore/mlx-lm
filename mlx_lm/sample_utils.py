# Copyright © 2023-2024 Apple Inc.

import math
from collections import Counter
from functools import partial
from typing import Callable, Dict, List, Optional

import mlx.core as mx


def make_sampler(
    temp: float = 0.0,
    top_p: float = 0.0,
    min_p: float = 0.0,
    min_tokens_to_keep: int = 1,
    top_k: int = 0,
    xtc_probability: float = 0.0,
    xtc_threshold: float = 0.0,
    xtc_special_tokens: List[int] = [],
) -> Callable[[mx.array], mx.array]:
    """
    Make a sampler function for use with ``generate_step``.

    Args:
        temp (float): The temperature for sampling, if 0 the argmax is used.
          Default: ``0``.
        top_p (float, optional): Nulceus sampling, higher means model considers
          more less likely words.
        min_p (float, optional): The minimum value (scaled by the top token's
          probability) that a token probability must have to be considered.
        min_tokens_to_keep (int, optional): Minimum number of tokens that cannot
          be filtered by min_p sampling.
        top_k (int, optional): The top k tokens ranked by probability to constrain
          the sampling to.
        xtc_probability (float, optional): The probability of applying XTC
            sampling.
        xtc_threshold (float, optional): The threshold the probs need to reach
            for being sampled.
        xtc_special_tokens (list(int), optional): List of special tokens IDs to
            be excluded from XTC sampling.


    Returns:
        Callable[mx.array, mx.array]:
            A sampler which takes log-probabilities and returns tokens.
    """
    if temp == 0:
        return lambda x: mx.argmax(x, axis=-1)

    # Create sampler chain
    sampling_methods = []
    if top_p > 0 and top_p < 1.0:
        sampling_methods.append(lambda x: apply_top_p(x, top_p))
    if min_p != 0.0:
        sampling_methods.append(lambda x: apply_min_p(x, min_p, min_tokens_to_keep))
    if xtc_probability > 0.0:
        sampling_methods.append(
            lambda x: apply_xtc(x, xtc_probability, xtc_threshold, xtc_special_tokens)
        )
    if top_k > 0:
        sampling_methods.append(lambda x: apply_top_k(x, top_k))

    # Apply the sampling methods
    def sampler(logprobs):
        for method in sampling_methods:
            logprobs = method(logprobs)
        # Return the sampled token
        return categorical_sampling(logprobs, temp)

    return sampler


def make_logits_processors(
    logit_bias: Optional[Dict[int, float]] = None,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = 20,
    presence_penalty: Optional[float] = None,
    presence_context_size: Optional[int] = 20,
    frequency_penalty: Optional[float] = None,
    frequency_context_size: Optional[int] = 20,
):
    """
    Make logits processors for use with ``generate_step``.

    Args:
        repetition_penalty (float, optional): A (sign-aware) multiplicative
          penalty for repeating tokens.
        repetition_context_size (int, optional): The number of tokens to
          consider for repetition penalty. Default: ``20``.
        presence_penalty (float, optional): An additive penalty to reduce
          repeating tokens.
        presence_context_size (int, optional): The number of tokens to consider
          for the presence penalty. Default: ``20``.
        frequency_penalty (float, optional): An additive penalty to reduce
          repeating tokens. The tokens are penalized proportionally to their
          frequency.
        frequency_context_size (int, optional): The number of tokens to consider
          for the frequency penalty. Default: ``20``.
        logit_bias (dictionary, optional): Additive logit bias.

    Returns:
        List[Callable[[mx.array, mx.array], mx.array]]:
            A list of logits processors. Each processor in the list is a
            callable which takes an array of tokens and an array of logits
            and returns the updated logits.
    """
    logits_processors = []
    if logit_bias:
        indices = mx.array(list(logit_bias.keys()))
        values = mx.array(list(logit_bias.values()))

        def logit_bias_processor(_, logits):
            return logits.at[:, indices].add(values)

        logits_processors.append(logit_bias_processor)

    repetition_penalties = [
        (make_repetition_penalty, repetition_penalty, repetition_context_size),
        (make_presence_penalty, presence_penalty, presence_context_size),
        (make_frequency_penalty, frequency_penalty, frequency_context_size),
    ]

    for make_penalty, penalty, context_size in repetition_penalties:
        if penalty is not None and penalty != 0:
            logits_processors.append(make_penalty(penalty, context_size))

    return logits_processors


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def apply_top_k(
    logprobs: mx.array,
    top_k: int,
) -> mx.array:
    """
    Sample from only the top K tokens ranked by probability.

    Args:
        logprobs: A vector of log probabilities.
        top_k (int): Top k tokens to sample from.
    """
    vocab_size = logprobs.shape[-1]
    if not isinstance(top_k, int) or not (0 < top_k < vocab_size):
        raise ValueError(
            f"`top_k` has to be an integer in the (0, {vocab_size}) interval,"
            f" but is {top_k}."
        )
    mask_idx = mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[..., top_k:]
    masked_logprobs = mx.put_along_axis(
        logprobs, mask_idx, mx.array(-float("inf"), logprobs.dtype), axis=-1
    )
    return masked_logprobs


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def apply_min_p(
    logprobs: mx.array,
    min_p: float,
    min_tokens_to_keep: int = 1,
) -> mx.array:
    """
    Apply min-p sampling to the logprobs.

    Min-p keeps all tokens that are above a minimum probability, scaled by the
    probability of the most likely token. As a result, the filter is more
    aggressive given a very high-probability token.

    Args:
        logprobs: A vector of log probabilities.
        min_p (float): Minimum token probability. Typical values are in the
            0.01-0.2 range, comparably selective as setting `top_p` in the
            0.99-0.8 range.
        min_tokens_to_keep (int, optional): Minimum number of tokens that cannot
            be filtered. Default: ``1``.

    """
    if not (0 <= min_p <= 1.0):
        raise ValueError(
            f"`min_p` has to be a float in the [0, 1] interval, but is {min_p}"
        )
    if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
        raise ValueError(
            f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}"
        )

    # Mask tokens that have a probability less than the max(p) * min_p
    top_logprobs = mx.max(logprobs, axis=-1, keepdims=True)
    scaled_min_p = top_logprobs + math.log(min_p)
    tokens_to_remove = logprobs < scaled_min_p

    # Ensure at least min_tokens_to_keep survive the filter
    if min_tokens_to_keep > 1:
        top_indices = mx.argpartition(logprobs, kth=-min_tokens_to_keep, axis=-1)
        top_indices = top_indices[..., -min_tokens_to_keep:]
        tokens_to_remove = mx.put_along_axis(
            tokens_to_remove,
            top_indices,
            False,
            axis=-1,
        )

    return mx.where(tokens_to_remove, -float("inf"), logprobs)


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def apply_top_p(logprobs: mx.array, top_p: float) -> mx.array:
    """
    Apply top-p (nucleus) sampling to logits.

    Args:
        logprobs: A vector of log probabilities.
        top_p: The cumulative probability threshold for top-p filtering.
    Returns:
        token selected based on the top-p criterion.
    """
    # referenced implementation from https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L449-L460
    probs = mx.exp(logprobs)
    # sort in ascending order
    sorted_indices = mx.argsort(logprobs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)

    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

    # Rearrange cumulative probs back to original order
    inverse_indices = mx.put_along_axis(
        mx.zeros_like(sorted_indices),
        sorted_indices,
        mx.arange(sorted_indices.shape[-1], dtype=sorted_indices.dtype),
        axis=-1,
    )
    cumulative_probs = mx.take_along_axis(cumulative_probs, inverse_indices, axis=-1)

    # select tokens with cumulative probs below threshold
    return mx.where(
        cumulative_probs > 1 - top_p,
        logprobs,
        -float("inf"),
    )


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def apply_xtc(
    logits: mx.array,
    xtc_probability: float,
    xtc_threshold: float,
    xtc_special_tokens: List[int],
) -> mx.array:
    """
    Apply XTC sampling to the logits.

    Args:
        logits: The logits from the model's output.
        xtc_probability (float): Probability of XTC sampling to happen for each token
        xtc_threshold (float): The threshold the probs need to reach for being sampled.
        special_tokens_ids (list(int)): List of special tokens IDs to be excluded from XTC sampling.
    """
    if not (0 <= xtc_threshold <= 0.5):
        raise ValueError(
            f"`threshold` has to be a float in the [0, 0.5] interval, but is {xtc_threshold}"
        )
    if not (0 <= xtc_probability <= 1.0):
        raise ValueError(
            f"`probability` has to be a float in the [0, 1] interval, but is {xtc_probability}"
        )

    probs = mx.softmax(logits, -1)
    mask = probs > mx.where(probs > xtc_threshold, probs, mx.inf).min()
    if xtc_special_tokens:
        mask[..., xtc_special_tokens] = False

    return mx.where(
        mx.random.uniform(0, 1) > xtc_probability,
        logits,
        mx.where(mask, -mx.inf, logits),
    )


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def categorical_sampling(logits, temp):
    return mx.random.categorical(logits * (1 / temp))


def make_repetition_penalty(penalty: float, context_size: int = 20):
    """
    Make repetition penalty processor.

    Paper: https://arxiv.org/abs/1909.05858

    Args:
        penalty (float): The repetition penalty factor to be applied.
        context_size (int): The number of previous tokens to use.
            Default: ``20``.

    Returns:
        Callable[[mx.array, List[int]], mx.array]:
            The repetition penalty processor.
    """
    if penalty < 0 or not isinstance(penalty, (int, float)):
        raise ValueError(f"penalty must be a non-negative float, got {penalty}")

    def repetition_penalty_processor(tokens, logits):
        if len(tokens) > 0:
            tokens = tokens[-context_size:]
            selected_logits = logits[:, tokens]
            selected_logits = mx.where(
                selected_logits < 0,
                selected_logits * penalty,
                selected_logits / penalty,
            )
            logits[:, tokens] = selected_logits
        return logits

    return repetition_penalty_processor


def make_presence_penalty(penalty: float, context_size: int = 20):
    """
    Make a presence penalty processor.

    Corresponds to the OpenAI option with the same name. Namely, subtracts
    ``penalty`` from a logit if the token has occured at least once in the
    ``context_size`` previous tokens.

    Args:
        penalty (float): The presence penalty to be applied.
        context_size (int): The number of previous tokens to use.
            Default: ``20``.

    Returns:
        Callable[[mx.array, List[int]], mx.array]
    """

    def presence_penalty_processor(tokens, logits):
        if len(tokens) > 0:
            tokens = tokens[-context_size:]
            logits[:, tokens] -= penalty
        return logits

    return presence_penalty_processor


def make_frequency_penalty(penalty: float, context_size: int = 20):
    """
    Make a frequency penalty processor.

    Corresponds to the OpenAI option with the same name. Namely, subtracts
    ``penalty`` from a logit for every time that the token has occured in the
    ``context_size`` previous tokens.

    The difference with the presence penalty is that the more often a token
    occurs the more it will be penalized.

    Args:
        penalty (float): The frequency penalty to be applied.
        context_size (int): The number of previous tokens to use.
            Default: ``20``.

    Returns:
        Callable[[mx.array, List[int]], mx.array]
    """

    def frequency_penalty_processor(tokens, logits):
        if len(tokens) > 0:
            tokens = tokens[-context_size:]
            logits = logits.at[:, tokens].subtract(penalty)
        return logits

    return frequency_penalty_processor


def _is_token_cycle(ids, max_cycle, min_span, window=800):
    """True if the tail of ``ids`` is a period-``p`` cycle (1..max_cycle)
    repeated back-to-back for at least ``min_span`` tokens. Decode-free and
    not newline-aligned, so it catches loops in flowing prose that a
    line-based check misses. ``min_span`` scales the required repeat count
    with the cycle size (a 1-token cycle must repeat many times before it
    counts, a longer phrase only a few)."""
    t = ids[-window:]
    n = len(t)
    for p in range(1, min(max_cycle, n // 3) + 1):
        block = t[-p:]
        reps, i = 1, n - 2 * p
        while i >= 0 and t[i : i + p] == block:
            reps += 1
            i -= p
        if reps >= 3 and reps * p >= min_span:
            return True
    return False


def _is_line_repetition(text, min_line=20, max_repeats=3):
    """True if any substantive line (>= ``min_line`` chars) recurs at least
    ``max_repeats`` times — the signature of a trace that found its point and
    is now looping on it."""
    counts = Counter(
        ln.strip() for ln in (text or "").splitlines() if len(ln.strip()) >= min_line
    )
    return bool(counts) and counts.most_common(1)[0][1] >= max_repeats


def make_reasoning_budget(
    think_close: int,
    max_think_tokens: int,
    *,
    think_open: Optional[int] = None,
    tokenizer=None,
    check_every: int = 16,
    max_cycle: int = 80,
    min_cycle_span: int = 30,
):
    """
    Make a logits processor that bounds a runaway reasoning channel.

    Reasoning models can enter a "thinking" channel and never leave it —
    finding an answer, then looping or over-deliberating until the token cap,
    yielding a long trace and no usable answer. Soft sampling pressure
    (``repetition_penalty``) does not robustly bound this. This processor is a
    hard, adaptive cap: it stays out of the way while the model reasons and
    only intervenes — by forcing the ``think_close`` token, so generation
    leaves the reasoning channel and produces an answer from what it has —
    once the reasoning is provably running away.

    A trip fires on any of:

    * a hard budget of ``max_think_tokens`` spent inside the channel;
    * a repeated token cycle in the tail of the channel (decode-free);
    * a repeated substantive line (only if a ``tokenizer`` is given).

    Args:
        think_close (int): Token id that closes the reasoning channel; this is
            what gets forced on a trip (e.g. the id of ``</think>``).
        max_think_tokens (int): Hard ceiling on tokens spent inside the
            channel before the close is forced.
        think_open (int, optional): Token id that opens the channel. If
            ``None``, generation is assumed to start inside the channel (as
            with templates that open it for the model). Default: ``None``.
        tokenizer (optional): If given, enables the decode-based line-repetition
            detector. Only its ``decode`` method is used. Default: ``None``.
        check_every (int): How often (in channel tokens) to run the loop
            detectors. Default: ``16``.
        max_cycle (int): Longest token-cycle period considered. Default: ``80``.
        min_cycle_span (int): Minimum repeated span (in tokens) for a cycle to
            count. Default: ``30``.

    Returns:
        Callable[[mx.array, mx.array], mx.array]: The logits processor. It
        operates on a single sequence (as the other processors here do).
    """
    if max_think_tokens <= 0:
        raise ValueError(f"max_think_tokens must be positive, got {max_think_tokens}")

    state = {
        "n": 0,  # tokens consumed from the running `tokens` array so far
        "in_think": think_open is None,
        "ids": [],  # ids seen inside the current channel
        "since_check": 0,
    }

    def reasoning_budget_processor(tokens, logits):
        new = tokens[state["n"] :].tolist()
        state["n"] = len(tokens)
        for tid in new:
            if tid == think_close:
                state["in_think"] = False
                state["ids"] = []
                state["since_check"] = 0
            elif think_open is not None and tid == think_open:
                state["in_think"] = True
                state["ids"] = []
                state["since_check"] = 0
            elif state["in_think"]:
                state["ids"].append(tid)

        if not state["in_think"]:
            return logits

        ids = state["ids"]
        trip = len(ids) >= max_think_tokens
        if not trip:
            state["since_check"] += len(new)
            if state["since_check"] >= check_every:
                state["since_check"] = 0
                trip = _is_token_cycle(ids, max_cycle, min_cycle_span) or (
                    tokenizer is not None
                    and _is_line_repetition(tokenizer.decode(ids[-1500:]))
                )

        if not trip:
            return logits

        # Force the channel-close token: -inf everywhere else so the sampler
        # (greedy or stochastic) must emit `think_close` next.
        forced = mx.full(logits.shape, -float("inf"), dtype=logits.dtype)
        forced[:, think_close] = 0.0
        return forced

    return reasoning_budget_processor
