# Copyright © 2026 Apple Inc.

"""Commit-steering: a content-blind controller for reasoning run-on.

Reasoning models often *have* the answer well before they stop deliberating —
they re-derive, second-guess, and re-check long after the result is formed. This
"run-on" is largely a **suppressed stop token** (the exit is high-rank but held
down by a logit margin), not confusion. The behavior of committing/terminating
is a direction in the residual stream; nudging the hidden state along that
direction at the layer where the model decides to wrap up releases the answer
sooner.

The direction is learned **offline** by difference-of-means — the mean hidden
state over commitment spans minus the mean over reflection spans
(:func:`extract_commit_direction`). At inference the controller is
**content-blind**: it never reads the generated text, it only adds a fixed
vector at one decoder layer while the model is inside its think channel.

Two-mode controller (:class:`CommitSteerer`):

* an always-on low-dose **bias** while inside the think channel, and
* a late hard **hammer** if the model is still thinking past a token budget.

Integration is non-invasive. :class:`CommitSteerer` is a context manager that
taps one decoder layer, and exposes a :meth:`~CommitSteerer.logits_processor`
that plugs into the existing ``generate_step(..., logits_processors=[...])``
API. The processor never changes the sampled logits — it only updates which
steering vector the *next* forward injects, from the committed-token prefix.

Example::

    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.steer import CommitSteerer
    import numpy as np

    model, tok = load("mlx-community/Qwen3-8B-4bit")
    z = np.load("commit_vectors_qwen3-8b.npz")          # from extract_commit_direction
    steerer = CommitSteerer(model, tok, z["v_20"], z["rms_20"], layer=20)

    ids = mx.array(tok.encode(prompt))
    with steerer:
        for tok_id, _ in generate_step(
            ids, model, max_tokens=2048,
            logits_processors=[steerer.logits_processor],
        ):
            ...
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np

__all__ = [
    "CommitSteerer",
    "steer_layer",
    "build_vector",
    "extract_commit_direction",
]


# --------------------------------------------------------------------------- #
# Layer tap
# --------------------------------------------------------------------------- #
class _SteerTap:
    """Wraps one decoder layer: adds a (mutable) steering vector to its residual
    output and/or captures that output, delegating everything else — attention
    mask, cache, sinks, hybrid routing, ``is_linear``, ... — to the real layer.

    Because it runs the real layer's own forward, it works across architectures
    (dense, MoE, sliding-window/sink attention, latent-attention, GDN/SSM
    hybrids) without re-implementing any of them.

    ``holder[0]`` is the current steering vector (or ``None`` for a no-op).
    ``sink`` (optional) is a one-element list that receives the layer output for
    offline capture.
    """

    def __init__(self, inner, holder, sink=None):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_holder", holder)
        object.__setattr__(self, "_sink", sink)

    def __call__(self, *args, **kwargs):
        h = self._inner(*args, **kwargs)
        vec = self._holder[0]
        if vec is not None:
            h = h + vec
        if self._sink is not None:
            self._sink[0] = h
        return h

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_inner"), name)


@contextmanager
def steer_layer(model, layer_index: int, holder: List[Optional[mx.array]], sink=None):
    """Temporarily tap decoder layer ``layer_index``.

    While the context is open, ``holder[0]`` (if not ``None``) is added to the
    layer's residual output on every forward, and — if ``sink`` is given — the
    output is stashed into ``sink[0]``. The original layer is always restored on
    exit. Requires ``model.model.layers`` to be the decoder-layer list, which is
    the standard mlx-lm decoder layout.
    """
    layers = model.model.layers
    original = layers[layer_index]
    layers[layer_index] = _SteerTap(original, holder, sink)
    try:
        yield
    finally:
        layers[layer_index] = original


def build_vector(v_hat, rms: float, alpha: float) -> Optional[mx.array]:
    """Return ``alpha * rms * v_hat`` as a float16 array, or ``None`` if the
    direction is degenerate (non-finite or ~zero norm).

    Returning ``None`` makes a missing or invalid axis a safe no-op: a model with
    no extractable commit direction (e.g. a non-thinking model) simply runs
    unchanged instead of having ``NaN`` injected into its residual stream.
    """
    v = np.asarray(v_hat, dtype=np.float64)
    rms = float(rms)
    if (
        not np.isfinite(rms)
        or not np.all(np.isfinite(v))
        or float(np.linalg.norm(v)) < 1e-6
    ):
        return None
    return mx.array((alpha * rms * v).astype(np.float16))


def _single_token_id(tokenizer, text: str) -> int:
    """Token id for ``text`` if it encodes to exactly one token, else ``-1``
    (which never matches a real token). Non-thinking tokenizers that split
    ``</think>`` across tokens therefore yield ``-1`` and disable phase tracking.
    """
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    except Exception:
        return -1
    return ids[0] if len(ids) == 1 else -1


# --------------------------------------------------------------------------- #
# Runtime controller
# --------------------------------------------------------------------------- #
class CommitSteerer:
    """Two-mode commit-steering controller for a reasoning model.

    Args:
        model: an mlx-lm model exposing ``model.model.layers``.
        tokenizer: used only to resolve the reasoning-channel tag ids; the
          controller is content-blind otherwise.
        v_hat: unit commit direction at ``layer`` (length = hidden size).
        rms: typical residual norm at ``layer`` (scales the injected vector).
        layer (int): decoder layer to inject at. ~55-60% relative depth is a
          good default across models.
        alpha_bias (float): standing low-dose bias while inside the think
          channel. Default: ``0.2``.
        alpha_hammer (float): late hard dose if still thinking past
          ``hammer_budget``. Default: ``0.8``.
        hammer_budget (int): think-token count after which the hammer engages.
          Set to ``0``/``None`` for bias only. Default: ``900``.
        close_tag, open_tag (str): reasoning-channel tags. Defaults
          ``"</think>"`` / ``"<think>"``.

    If the commit direction is degenerate (:func:`build_vector` returns
    ``None``) the controller is disabled and behaves as a pure no-op.

    Use as a context manager and pass :meth:`logits_processor` to
    ``generate_step`` (see the module docstring for a full example).
    """

    def __init__(
        self,
        model,
        tokenizer,
        v_hat,
        rms,
        *,
        layer: int,
        alpha_bias: float = 0.2,
        alpha_hammer: float = 0.8,
        hammer_budget: Optional[int] = 900,
        close_tag: str = "</think>",
        open_tag: str = "<think>",
    ):
        self.model = model
        self.layer = layer
        self._bias = build_vector(v_hat, rms, alpha_bias)
        self._hammer = (
            build_vector(v_hat, rms, alpha_hammer) if hammer_budget else None
        )
        self.hammer_budget = int(hammer_budget or 0)
        self._close_id = _single_token_id(tokenizer, close_tag)
        self._open_id = _single_token_id(tokenizer, open_tag)
        # a degenerate axis (or no resolvable close tag) => pure no-op
        self.disabled = self._bias is None or self._close_id < 0
        # runtime state
        self._holder: List[Optional[mx.array]] = [None]
        self._in_think = True  # chat templates typically pre-open the channel
        self._think_count = 0
        self._seen = 0
        self._cm = None

    # -- context management: patch/unpatch the layer --------------------------
    def __enter__(self) -> "CommitSteerer":
        self.reset()
        if not self.disabled:
            self._cm = steer_layer(self.model, self.layer, self._holder)
            self._cm.__enter__()
        return self

    def __exit__(self, *exc) -> bool:
        if self._cm is not None:
            self._cm.__exit__(*exc)
            self._cm = None
        self._holder[0] = None
        return False

    def reset(self) -> None:
        """Reset the think-channel state for a fresh generation."""
        self._in_think = True
        self._think_count = 0
        self._seen = 0
        self._holder[0] = None

    # -- schedule -------------------------------------------------------------
    def _select(self) -> Optional[mx.array]:
        if self.disabled or not self._in_think:
            return None
        if (
            self.hammer_budget
            and self._think_count >= self.hammer_budget
            and self._hammer is not None
        ):
            return self._hammer
        return self._bias

    def logits_processor(self, tokens: mx.array, logits: mx.array) -> mx.array:
        """State-updater for the ``logits_processors`` list.

        Reads the committed-token prefix, updates the think-channel state, and
        sets the vector the *next* forward will inject. Returns ``logits``
        unchanged — steering happens mid-network, not on the sampled logits, so
        this composes with samplers and other logits processors.
        """
        if self.disabled:
            return logits
        n = int(tokens.shape[-1])
        if self._seen == 0:
            # first call: `tokens` is the prompt. Derive the think state from its
            # tail; do not count prompt tokens toward the hammer budget. Leave
            # the holder at None so prefill and the first decode step are
            # unsteered (steering begins one decode token in).
            for t in tokens[max(0, n - 64):].tolist():
                if t == self._open_id:
                    self._in_think = True
                elif t == self._close_id:
                    self._in_think = False
        else:
            for t in tokens[self._seen:].tolist():
                if t == self._open_id:
                    self._in_think = True
                elif t == self._close_id:
                    self._in_think = False
                elif self._in_think:
                    self._think_count += 1
            self._holder[0] = self._select()
        self._seen = n
        return logits


# --------------------------------------------------------------------------- #
# Offline extraction (content labels are used here only, never at runtime)
# --------------------------------------------------------------------------- #
# Lexical markers used *offline* to label reflection vs commitment spans. These
# are a convenience for English self-correcting reasoners; a position-based
# labeling (early-think vs the tokens just before </think>) generalizes to
# models that reason more linearly.
REFLECT_RE = re.compile(
    r"\b(Wait|But wait|Hmm|However|Actually|Alternatively|Hold on|"
    r"Let me (?:double-?check|re-?check|verify|check|reconsider))\b",
    re.IGNORECASE,
)
COMMIT_RE = re.compile(
    r"\b(Therefore|Thus|So the answer is|The answer is|the final answer|"
    r"final answer is|So the result|I'?m confident|answer:)\b",
    re.IGNORECASE,
)


def _hidden_dim(model) -> int:
    for attr in ("hidden_size", "model_dim", "d_model", "n_embd", "dim"):
        d = getattr(model.args, attr, None)
        if isinstance(d, int) and d > 0:
            return d
    return int(model.model.embed_tokens.weight.shape[-1])


@contextmanager
def _tap_layers(model, layers, holder, sinks):
    """Tap several layers at once (shared no-op ``holder``, per-layer ``sinks``),
    restoring all of them on exit."""
    layer_list = model.model.layers
    original = {L: layer_list[L] for L in layers}
    try:
        for L in layers:
            layer_list[L] = _SteerTap(layer_list[L], holder, sinks[L])
        yield
    finally:
        for L, layer in original.items():
            layer_list[L] = layer


def _capture_layers(model, ids: List[int], layers: List[int]) -> Dict[int, np.ndarray]:
    """One forward over ``ids`` (fresh cache), returning each layer's residual
    output as a (seq, hidden) numpy array."""
    from mlx_lm.models.cache import make_prompt_cache

    holder: List[Optional[mx.array]] = [None]
    sinks = {L: [None] for L in layers}
    with _tap_layers(model, layers, holder, sinks):
        model(mx.array(ids)[None], cache=make_prompt_cache(model))
    mx.eval(*[sinks[L][0] for L in layers])
    return {L: np.array(sinks[L][0][0].astype(mx.float32)) for L in layers}


def extract_commit_direction(
    model,
    tokenizer,
    traces: List[List[int]],
    layers: List[int],
    *,
    close_tag: str = "</think>",
    span_after: int = 6,
    commit_tail: int = 8,
) -> Dict[int, Tuple[np.ndarray, float]]:
    """Offline: extract the commit direction per layer by difference-of-means.

    Args:
        model, tokenizer: the model to extract from.
        traces: greedy reasoning traces, each a list of generated token ids.
        layers: decoder layers to extract at.
        close_tag: reasoning-channel close tag (its span end anchors commitment).
        span_after: tokens taken after each lexical marker as its span.
        commit_tail: in-think tokens just before ``</think>`` counted as commit.

    Returns ``{layer: (v_hat, rms)}`` where ``v_hat`` is the unit commit
    direction and ``rms`` the mean residual norm at that layer. Reflection and
    commitment spans are labeled by lexical markers here (offline only — the
    runtime controller in :class:`CommitSteerer` is content-blind).

    Cross-trace consistency of ``v_hat`` (mean pairwise cosine of the per-trace
    directions) is a good sanity check: a real axis is strongly positive
    (~+0.5 to +0.86 across the models we tested); near zero means no usable
    direction.
    """
    D = _hidden_dim(model)
    sums = {L: {"reflect": np.zeros(D), "commit": np.zeros(D)} for L in layers}
    counts = {"reflect": 0, "commit": 0}
    rms = {L: [] for L in layers}

    def _positions(text, starts, rx, limit):
        out: List[int] = []
        for m in rx.finditer(text[:limit] if limit else text):
            t0 = next((i for i, s in enumerate(starts) if s >= m.start()), None)
            if t0 is not None:
                out.extend(range(t0, min(t0 + span_after, len(starts))))
        return sorted(set(out))

    for ids in traces:
        # incremental detokenization gives clean text + per-token char offsets
        det = tokenizer.detokenizer
        det.reset()
        starts: List[int] = []
        for t in ids:
            starts.append(len(det.text))
            det.add_token(t)
        det.finalize()
        text = det.text

        close_char = text.find(close_tag)
        limit = close_char if close_char >= 0 else len(text)
        p_ref = _positions(text, starts, REFLECT_RE, limit)
        p_com = _positions(text, starts, COMMIT_RE, limit)
        if close_char >= 0:
            ct = next(
                (i for i, s in enumerate(starts) if s >= close_char), len(starts)
            )
            p_com = sorted(set(p_com) | set(range(max(0, ct - commit_tail), ct)))
        p_ref = [p for p in p_ref if p not in set(p_com)]
        if not p_ref or not p_com:
            continue

        H = _capture_layers(model, ids, layers)  # one forward, all layers
        for L in layers:
            hr = H[L][[p for p in p_ref if p < len(H[L])]]
            hc = H[L][[p for p in p_com if p < len(H[L])]]
            sums[L]["reflect"] += hr.sum(0)
            sums[L]["commit"] += hc.sum(0)
            rms[L].append(float(np.linalg.norm(H[L], axis=1).mean()))
        counts["reflect"] += len(p_ref)
        counts["commit"] += len(p_com)

    out: Dict[int, Tuple[np.ndarray, float]] = {}
    for L in layers:
        v = sums[L]["commit"] / max(counts["commit"], 1) - sums[L][
            "reflect"
        ] / max(counts["reflect"], 1)
        vhat = v / (np.linalg.norm(v) + 1e-8)
        out[L] = (vhat.astype(np.float32), float(np.mean(rms[L]) if rms[L] else 0.0))
    return out
