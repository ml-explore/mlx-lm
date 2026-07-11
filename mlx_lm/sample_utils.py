# Copyright © 2023-2026 Apple Inc.

import math
import time
from collections import Counter
from functools import partial
from numbers import Integral, Real
from typing import Callable, Dict, List, Optional, Union

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
    max_think_tokens: Union[int, Callable[[], int]],
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

    .. note::
        The processor tracks channel state from the running token stream and
        never rewinds it, so it does not support speculative decoding's
        draft-token rejection: tokens it has ingested are counted even if the
        generator later discards them. Use it with standard (non-speculative)
        generation.

    Args:
        think_close (int): Token id that closes the reasoning channel; this is
            what gets forced on a trip (e.g. the id of ``</think>``).
        max_think_tokens (int or callable): Hard ceiling on tokens spent
            inside the channel before the close is forced. May be a zero-arg
            callable returning the current ceiling, evaluated at every
            generation step until a trip fires — this makes the budget
            *dynamic*, e.g. one that tightens under live host cost (see
            ``make_cost_braked_budget``). The callable must return a positive
            integer; anything else raises a labeled ``ValueError``.
            Once a trip fires it latches until a ``think_close`` token is
            actually emitted: a budget that later grows — or an in-channel
            ``think_open`` seen before any close — cannot undo a forced
            close, and the callable is not invoked again while the latch
            holds.
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
    dynamic_budget = callable(max_think_tokens)
    if not dynamic_budget:
        if isinstance(max_think_tokens, bool) or not isinstance(
            max_think_tokens, Integral
        ):
            raise ValueError(
                "max_think_tokens must be a positive integer or a zero-arg "
                f"callable, got {max_think_tokens!r}"
            )
        if max_think_tokens <= 0:
            raise ValueError(
                f"max_think_tokens must be positive, got {max_think_tokens}"
            )

    state = {
        "n": 0,  # tokens consumed from the running `tokens` array so far
        "in_think": think_open is None,
        "ids": [],  # ids seen inside the current channel
        "since_check": 0,
        # Trip latch: with a *dynamic* budget the ceiling can grow between
        # steps (e.g. host cost dropped), which would otherwise "un-trip" a
        # budget trip before the forced close token is actually emitted. Once
        # tripped, keep forcing until a close token is actually seen — an
        # in-channel re-open does NOT release it (only a real close does).
        "tripped": False,
    }

    def reasoning_budget_processor(tokens, logits):
        new = tokens[state["n"] :].tolist()
        state["n"] = len(tokens)
        for tid in new:
            if tid == think_close:
                state["in_think"] = False
                state["ids"] = []
                state["since_check"] = 0
                state["tripped"] = False
            elif think_open is not None and tid == think_open:
                state["in_think"] = True
                state["ids"] = []
                state["since_check"] = 0
            elif state["in_think"]:
                state["ids"].append(tid)

        if not state["in_think"]:
            return logits

        ids = state["ids"]
        if state["tripped"]:
            # Latched: do not re-evaluate the budget (a dynamic callable is
            # not invoked again — a sensor failing after the trip cannot
            # crash the forced close).
            trip = True
        else:
            if dynamic_budget:
                raw_budget = max_think_tokens()
                if isinstance(raw_budget, bool) or not isinstance(raw_budget, Integral):
                    raise ValueError(
                        "make_reasoning_budget: the max_think_tokens callable "
                        f"must return a positive integer, got {raw_budget!r}"
                    )
                budget = int(raw_budget)
                if budget <= 0:
                    raise ValueError(
                        "make_reasoning_budget: the max_think_tokens callable "
                        f"must return a positive budget, got {budget}"
                    )
            else:
                budget = max_think_tokens
            trip = len(ids) >= budget
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

        state["tripped"] = True
        # Force the channel-close token: -inf everywhere else so the sampler
        # (greedy or stochastic) must emit `think_close` next.
        forced = mx.full(logits.shape, -float("inf"), dtype=logits.dtype)
        forced[:, think_close] = 0.0
        return forced

    return reasoning_budget_processor


# Base think-token budget by expected task size, and the difficulty scaling
# hard reasoning earns. Used by ``make_cost_braked_budget``.
REASONING_LEN_BASE = {"short": 512, "medium": 1280, "long": 3200}
REASONING_DIFF_MULT = {"easy": 1.0, "medium": 1.25, "hard": 1.5}


def make_cost_braked_budget(
    prompt_len: Optional[int] = None,
    difficulty: str = "medium",
    *,
    length_class: Optional[str] = None,
    floor: int = 256,
    ceil: int = 4096,
    cost_brake: float = 0.35,
    cost_fn: Optional[Callable[[], float]] = None,
    brake_horizon: float = 120.0,
    clock: Optional[Callable[[], float]] = None,
    short_prompt: int = 256,
    long_prompt: int = 2048,
) -> Callable[[], int]:
    """
    Make a dynamic, cost-braked think-token budget for ``make_reasoning_budget``.

    A static reasoning budget must be sized for the worst case; a dynamic one
    can be thin by default and spend where the task earns it. This helper
    builds the budget as::

        budget = clamp(base(length) * difficulty_mult * brake(cost), floor, ceil)

    * **base** comes from the task's expected size: an explicit
      ``length_class`` (``"short"``/``"medium"``/``"long"``), or one derived
      from ``prompt_len`` (below ``short_prompt`` tokens is short, at or above
      ``long_prompt`` is long, otherwise medium).
    * **difficulty** scales it — hard reasoning earns more room
      (easy 1.0x / medium 1.25x / hard 1.5x).
    * **brake** tightens the budget under live cost: with cost in ``[0, 1]``,
      ``brake = 1 - cost_brake * cost``, so at full cost the budget shrinks by
      ``cost_brake`` (default 35%). The cost signal is an injected zero-arg
      callable so any host metric can drive it (load, memory, GPU
      contention, ...); out-of-range values are clamped to ``[0, 1]`` and a
      NaN cost raises a labeled ``ValueError``. The default is a wall-clock
      brake: cost ramps 0 -> 1 over ``brake_horizon`` seconds from the first
      evaluation, so long generations progressively tighten.
    * **clamp** bounds it: a hard ``ceil`` is the anti-runaway guard (a loop
      cannot burn a worst-case flat cap), and ``floor`` guarantees even short
      tasks room to answer — the brake is discretionary, never below floor.

    The returned zero-arg callable is re-evaluated at every generation step by
    ``make_reasoning_budget``, so the brake is live, not sampled once.

    Build one budget per generation: the default wall-clock brake anchors on
    the first evaluation and never resets, so a callable reused across
    generations starts the later ones already braked.

    Args:
        prompt_len (int, optional): Prompt length in tokens, used to derive
            the length class when ``length_class`` is not given. ``None``
            means medium. Default: ``None``.
        difficulty (str): ``"easy"``, ``"medium"``, or ``"hard"``.
            Default: ``"medium"``.
        length_class (str, optional): Explicit ``"short"``/``"medium"``/
            ``"long"`` override; takes precedence over ``prompt_len``.
            Default: ``None``.
        floor (int): Minimum budget; the brake never cuts below it.
            Default: ``256``.
        ceil (int): Hard maximum budget. Default: ``4096``.
        cost_brake (float): Maximum fractional budget reduction at full cost,
            in ``[0, 1]``. Default: ``0.35``.
        cost_fn (callable, optional): Zero-arg callable returning the live
            cost in ``[0, 1]`` (out-of-range values are clamped; NaN raises).
            If ``None``, a wall-clock-elapsed brake is used. Default: ``None``.
        brake_horizon (float): Seconds over which the default wall-clock cost
            ramps from 0 to 1. Ignored when ``cost_fn`` is given.
            Default: ``120.0``.
        clock (callable, optional): Zero-arg monotonic time source for the
            default wall-clock brake (for testing). Ignored when ``cost_fn``
            is given. Default: ``time.monotonic``.
        short_prompt (int): ``prompt_len`` below this is short. Default: ``256``.
        long_prompt (int): ``prompt_len`` at or above this is long.
            Default: ``2048``.

    Returns:
        Callable[[], int]: A zero-arg callable returning the current budget;
        pass it as ``max_think_tokens`` to ``make_reasoning_budget``.
    """
    integer_args = {
        "floor": floor,
        "ceil": ceil,
        "short_prompt": short_prompt,
        "long_prompt": long_prompt,
    }
    if prompt_len is not None:
        integer_args["prompt_len"] = prompt_len
    for name, value in integer_args.items():
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be an integer, got {value!r}")

    if floor < 1:
        raise ValueError(f"floor must be positive, got {floor}")
    if ceil < floor:
        raise ValueError(f"ceil must be >= floor, got ceil={ceil}, floor={floor}")
    if isinstance(cost_brake, bool) or not isinstance(cost_brake, Real):
        raise ValueError(f"cost_brake must be a number in [0, 1], got {cost_brake!r}")
    if not math.isfinite(cost_brake) or not (0 <= cost_brake <= 1):
        raise ValueError(f"cost_brake must be in [0, 1], got {cost_brake}")
    if not isinstance(difficulty, str) or difficulty not in REASONING_DIFF_MULT:
        raise ValueError(
            f"difficulty must be one of {sorted(REASONING_DIFF_MULT)}, got {difficulty!r}"
        )
    if short_prompt < 0 or long_prompt < 0:
        raise ValueError(
            "short_prompt and long_prompt must be non-negative, got "
            f"short_prompt={short_prompt}, long_prompt={long_prompt}"
        )
    if short_prompt >= long_prompt:
        raise ValueError(
            f"short_prompt must be < long_prompt, got {short_prompt} >= {long_prompt}"
        )
    if prompt_len is not None and prompt_len < 0:
        raise ValueError(f"prompt_len must be non-negative, got {prompt_len}")
    if length_class is None:
        if prompt_len is None:
            length_class = "medium"
        elif prompt_len < short_prompt:
            length_class = "short"
        elif prompt_len >= long_prompt:
            length_class = "long"
        else:
            length_class = "medium"
    elif not isinstance(length_class, str) or length_class not in REASONING_LEN_BASE:
        raise ValueError(
            f"length_class must be one of {sorted(REASONING_LEN_BASE)}, got {length_class!r}"
        )

    base = REASONING_LEN_BASE[length_class] * REASONING_DIFF_MULT[difficulty]

    if cost_fn is None:
        if isinstance(brake_horizon, bool) or not isinstance(brake_horizon, Real):
            raise ValueError(
                f"brake_horizon must be a positive finite number, got {brake_horizon!r}"
            )
        if not math.isfinite(brake_horizon) or brake_horizon <= 0:
            raise ValueError(
                f"brake_horizon must be positive and finite, got {brake_horizon}"
            )
        if clock is not None and not callable(clock):
            raise ValueError(f"clock must be callable, got {clock!r}")
        read_clock = clock if clock is not None else time.monotonic
        start = None
        previous = None

        def cost_fn():
            nonlocal previous, start
            raw_now = read_clock()
            if isinstance(raw_now, bool) or not isinstance(raw_now, Real):
                raise ValueError(
                    "make_cost_braked_budget: clock must return a real number, "
                    f"got {raw_now!r}"
                )
            now = float(raw_now)
            if not math.isfinite(now):
                raise ValueError(
                    "make_cost_braked_budget: clock must return a finite number, "
                    f"got {raw_now!r}"
                )
            if start is None:
                start = now
            elif now < previous:
                raise ValueError(
                    "make_cost_braked_budget: clock moved backwards, "
                    f"from {previous} to {now}"
                )
            previous = now
            return (now - start) / brake_horizon

    elif not callable(cost_fn):
        raise ValueError(f"cost_fn must be callable, got {cost_fn!r}")

    def cost_braked_budget():
        raw_cost = cost_fn()
        if isinstance(raw_cost, bool) or not isinstance(raw_cost, Real):
            raise ValueError(
                "make_cost_braked_budget: cost_fn must return a real number, "
                f"got {raw_cost!r}"
            )
        cost = float(raw_cost)
        if math.isnan(cost):
            raise ValueError("make_cost_braked_budget: cost_fn returned NaN")
        cost = max(0.0, min(1.0, cost))
        brake = 1.0 - cost_brake * cost
        return int(max(floor, min(ceil, int(base * brake))))

    return cost_braked_budget
