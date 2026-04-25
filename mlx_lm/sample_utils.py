# Copyright © 2023-2024 Apple Inc.

import math
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
            f"`top_k` has to be an integer in the (0, {vocab_size}] interval,"
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


def make_thinking_budget_processor(
    think_start_tokens: tuple,
    think_end_tokens: tuple,
    budget: int,
    early_stop_tokens: mx.array,
) -> Callable[[mx.array, mx.array], mx.array]:
    """
    Make a logits processor that enforces a thinking token budget.

    Tracks the thinking state by watching for ``think_start_tokens`` /
    ``think_end_tokens`` in the generated sequence.  When the number of
    thinking tokens exceeds ``budget``, the processor injects
    ``early_stop_tokens`` one token at a time by forcing each token's logit
    to ``0.0`` and setting all other logits to ``-inf``.  After the full
    injection sequence has been forced the processor returns to passthrough
    mode.

    Args:
        think_start_tokens (tuple): Token IDs that mark the start of a
            thinking block (e.g. the tokenization of ``<think>``).
        think_end_tokens (tuple): Token IDs that mark the end of a thinking
            block (e.g. the tokenization of ``</think>``).
        budget (int): Maximum number of thinking tokens allowed before
            early-stop injection begins.  Must be >= 0.
        early_stop_tokens (mx.array): 1-D array of token IDs to inject when
            the budget is exceeded.

    Returns:
        Callable[[mx.array, mx.array], mx.array]:
            A stateful logits processor with signature
            ``(tokens, logits) -> logits``.
    """
    if budget < 0:
        raise ValueError(f"budget must be >= 0, got {budget}")

    state = {
        "in_thinking": False,
        "thinking_count": 0,
        "forcing": False,
        "force_idx": 0,
        "prev_len": 0,
    }

    start = list(think_start_tokens)
    end = list(think_end_tokens)
    stop = early_stop_tokens.tolist()

    vocab_size = early_stop_tokens.shape[0] if early_stop_tokens.ndim == 1 else None

    def _force_logits(logits, forced_id):
        """Return logits with forced_id set to 0.0 and all others to -inf."""
        indices = mx.arange(logits.shape[-1])
        return mx.where(indices == forced_id, 0.0, float("-inf"))[None]

    def _ends_with(seq, pattern):
        """Return True if seq ends with pattern."""
        n = len(pattern)
        return len(seq) >= n and seq[-n:] == pattern

    def thinking_budget_processor(tokens, logits):
        token_list = tokens.tolist()
        prev_len = state["prev_len"]
        new_tokens = token_list[prev_len:]
        state["prev_len"] = len(token_list)

        # If currently forcing injection, continue regardless of new tokens.
        if state["forcing"]:
            forced_id = stop[state["force_idx"]]
            state["force_idx"] += 1
            if state["force_idx"] >= len(stop):
                # Injection complete; reset to not-thinking.
                state["forcing"] = False
                state["force_idx"] = 0
                state["in_thinking"] = False
                state["thinking_count"] = 0
            return _force_logits(logits, forced_id)

        # Process each new token to update thinking state.
        # Walk new_tokens by position to avoid index(tok) ambiguity.
        for pos, tok in enumerate(new_tokens, start=prev_len):
            prefix = token_list[: pos + 1]
            if not state["in_thinking"]:
                if _ends_with(prefix, start):
                    state["in_thinking"] = True
                    state["thinking_count"] = 0
            else:
                if _ends_with(prefix, end):
                    state["in_thinking"] = False
                    state["thinking_count"] = 0
                else:
                    state["thinking_count"] += 1
                    if state["thinking_count"] >= budget:
                        state["forcing"] = True
                        state["force_idx"] = 0
                        forced_id = stop[state["force_idx"]]
                        state["force_idx"] += 1
                        if state["force_idx"] >= len(stop):
                            state["forcing"] = False
                            state["force_idx"] = 0
                            state["in_thinking"] = False
                            state["thinking_count"] = 0
                        return _force_logits(logits, forced_id)

        return logits

    return thinking_budget_processor
