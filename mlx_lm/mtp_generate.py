# Copyright © 2026 Apple Inc.

"""MTP speculative-decoding generator for Qwen3-Next / Qwen3.5-MoE.

Unlike ``speculative_generate_step`` in ``mlx_lm.generate`` -- which uses a
separate draft *model* -- this module uses an in-model :class:`MTPHead`
(loaded from the ``mtp.*`` weights shipped with the base checkpoint) as
the draft. The head is one decoder layer and shares ``embed_tokens`` and
``lm_head`` with the main model, so the memory and compute overhead is
small relative to a full draft model.

Algorithm (greedy, draft depth 1)::

    1. Prefill the prompt through the main model. Sample ``t_1`` from
       the last hidden state.

    2. Repeat until ``max_tokens`` generated:
       a. Draft.  ``MTPHead(last_hidden, embed(t_n)) -> d_1``.
       b. Mark caches to capture intermediate (post-t_n) state.
       c. Verify. Forward main on ``[t_n, d_1]`` (length-2 input), which
          advances main caches by 2 positions.
       d. Compute ``m_2 = argmax(lm_head(h_M))`` -- main's prediction of
          the token that should follow ``t_n``. If ``d_1 == m_2``: ACCEPT.
          Compute ``m_3 = argmax(lm_head(h_{M+1}))`` -- bonus token.
          Generated += ``[d_1, m_3]``; ``last_hidden`` = ``h_{M+1}``.
          Cache stays at offset M+2.
       e. If REJECT: rollback caches to the captured intermediate (post-t_n)
          state. Generated += ``[m_2]``; ``last_hidden`` = ``h_M``. Cache is
          now at offset M+1.

Performance ceiling at depth=1::

    speedup = (1 + a) / (2 - a)        with rollback-without-reforward
    where a = acceptance rate.
    a = 0.85 -> 1.32x;  a = 0.95 -> 1.85x.

Bit-exact under greedy (argmax) decoding.

Public entry point: :func:`mtp_generate_step`.
"""

from __future__ import annotations

from typing import Any, Generator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .models import cache as cache_mod
from .models.base import create_attention_mask, create_ssm_mask
from .models.gated_delta_spec import (
    install as install_gdn_spec,
    mark_for_capture,
    rollback_to_intermediate,
    unmark_capture,
)
from .models.mtp_head import MTPHead


def _full_forward_capture(model: nn.Module, input_ids: mx.array, prompt_cache: list):
    """Forward through the inner Qwen3-Next text model.

    Returns the post-final-norm hidden state at every position
    (shape ``(B, L, H)``). Equivalent to running the model up to (but not
    through) ``lm_head``, so the caller can decide whether to apply
    ``lm_head`` to one position or all of them.
    """
    inner = model.language_model.model
    hidden = inner.embed_tokens(input_ids)
    fa_mask = create_attention_mask(hidden, prompt_cache[inner.fa_idx])
    ssm_mask = create_ssm_mask(hidden, prompt_cache[inner.ssm_idx])
    for layer, c in zip(inner.layers, prompt_cache):
        m = ssm_mask if layer.is_linear else fa_mask
        hidden = layer(hidden, mask=m, cache=c)
    return inner.norm(hidden)


def _lm_head_of(model: nn.Module):
    text_model = model.language_model
    if hasattr(text_model, "lm_head"):
        return text_model.lm_head
    # Tied-embeddings path.
    return text_model.model.embed_tokens.as_linear


def _embed_of(model: nn.Module):
    return model.language_model.model.embed_tokens


def mtp_generate_step(
    prompt: mx.array,
    model: nn.Module,
    mtp_head: MTPHead,
    *,
    max_tokens: int = 256,
    prompt_cache: Optional[list] = None,
) -> Generator[Tuple[int, mx.array, bool], None, None]:
    """Generator yielding tokens from greedy MTP-spec decoding.

    Args:
        prompt: 1-D int prompt token array.
        model: the main Qwen3-Next / Qwen3.5-MoE model (must wrap a
            ``language_model.model`` with ``layers`` / ``embed_tokens`` /
            ``norm`` / ``fa_idx`` / ``ssm_idx``).
        mtp_head: a loaded :class:`MTPHead` matching ``model``'s
            architecture.
        max_tokens: maximum number of tokens to yield.
        prompt_cache: optional pre-built prompt cache. If ``None``, one
            is created with ``cache.make_prompt_cache(model)``.

    Yields:
        ``(token_id, logprobs_or_stub, from_draft)`` tuples, mirroring the
        shape used by :func:`mlx_lm.generate.speculative_generate_step`.
        ``logprobs_or_stub`` is either the actual softmax for fallback /
        prefill tokens, or a zero-vector stub for spec-decoded tokens
        (greedy decoding does not consume logprobs). ``from_draft`` is
        ``True`` for accepted MTP draft tokens.
    """
    # Lazy install of the GDN split-call patch. No-op if already installed.
    install_gdn_spec()

    lm_head = _lm_head_of(model)
    embed_tokens = _embed_of(model)

    if prompt_cache is None:
        prompt_cache = cache_mod.make_prompt_cache(model)

    prompt_arr = prompt.astype(mx.uint32)[None]
    post = _full_forward_capture(model, prompt_arr, prompt_cache)
    last_hidden = post[:, -1:, :]

    # Sample first token from the prompt's last hidden.
    first_logits = lm_head(last_hidden).squeeze(0).squeeze(0)
    t = int(mx.argmax(first_logits).item())
    n = 0
    if n < max_tokens:
        n += 1
        yield t, first_logits, False
    if n >= max_tokens:
        return

    # Stub logprobs for spec-yielded tokens (greedy mode doesn't use them
    # for sampling; computing a full vocab-wide softmax every cycle is
    # wasted work).
    vocab_size = first_logits.shape[-1]
    logprob_stub = mx.zeros((vocab_size,), dtype=mx.float32)

    while n < max_tokens:
        last_tok = t
        last_tok_arr = mx.array([[last_tok]])

        # MTP draft step. depth=1, cache=None: the head is one self-attn
        # layer and at L=1 there is nothing to attend to but itself, so a
        # K/V history within the chain is unnecessary.
        emb = embed_tokens(last_tok_arr)
        h_draft = mtp_head(last_hidden, emb, cache=None)
        d1_arr = mx.argmax(lm_head(h_draft), axis=-1)  # (1, 1)

        # Verify pass on [t_n, d_1]. Mark GDN caches to capture the
        # post-t_n intermediate so we can roll back on rejection without
        # a re-forward.
        mark_for_capture(prompt_cache)
        verify_input = mx.concatenate([last_tok_arr, d1_arr], axis=1)
        verify_post = _full_forward_capture(model, verify_input, prompt_cache)
        verify_logits = lm_head(verify_post)  # (1, 2, V)
        m_arr = mx.argmax(verify_logits, axis=-1)  # (1, 2)

        # One eval/sync per cycle.
        mx.eval(d1_arr, m_arr)
        unmark_capture(prompt_cache)

        d1 = int(d1_arr.item())
        m2 = int(m_arr[0, 0].item())
        m3 = int(m_arr[0, 1].item())

        if d1 == m2:
            # Accept. Cache is at offset+2 already; yield primary then bonus.
            last_hidden = verify_post[:, -1:, :]
            n += 1
            yield d1, logprob_stub, True
            if n >= max_tokens:
                return
            n += 1
            yield m3, logprob_stub, False
            t = m3
        else:
            # Reject. Roll cache back to post-t_n, advance to offset+1.
            ok = rollback_to_intermediate(prompt_cache)
            if not ok:
                raise RuntimeError(
                    "mtp_generate_step: GDN rollback found no captured state "
                    "(install_gdn_spec must run before the verify forward)."
                )
            last_hidden = verify_post[:, :1, :]
            n += 1
            yield m2, logprob_stub, False
            t = m2


def mtp_generate(
    model: nn.Module,
    mtp_head: MTPHead,
    tokenizer,
    prompt,
    *,
    max_tokens: int = 256,
) -> Tuple[str, dict]:
    """Convenience wrapper that decodes a prompt to a string under MTP-spec.

    Args:
        model: the main model.
        mtp_head: a loaded :class:`MTPHead`.
        tokenizer: a tokenizer (or :class:`TokenizerWrapper`) supporting
            ``encode`` and ``decode``.
        prompt: a string or a sequence of token ids.
        max_tokens: maximum number of tokens to generate.

    Returns:
        ``(text, stats)`` where ``stats`` contains ``n_tokens``,
        ``n_accepted_drafts``, ``n_cycles``, and ``acceptance_rate``.
    """
    if isinstance(prompt, str):
        ids = tokenizer.encode(prompt)
    else:
        ids = list(prompt)
    prompt_arr = mx.array(ids)

    out_tokens: List[int] = []
    n_accepted = 0
    n_cycles = 0
    n_prefill_yields = 0

    for tok, _, from_draft in mtp_generate_step(
        prompt_arr, model, mtp_head, max_tokens=max_tokens
    ):
        out_tokens.append(tok)
        if from_draft:
            n_accepted += 1
            n_cycles += 1
        else:
            # The first non-spec yield is from the prompt prefill; subsequent
            # non-spec yields are MTP rejects (one cycle each).
            if n_prefill_yields == 0:
                n_prefill_yields = 1
            else:
                n_cycles += 1

    text = tokenizer.decode(out_tokens)
    stats = {
        "n_tokens": len(out_tokens),
        "n_accepted_drafts": n_accepted,
        "n_cycles": n_cycles,
        "acceptance_rate": n_accepted / max(1, n_cycles),
    }
    return text, stats
