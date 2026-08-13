# Copyright © 2023-2024 Apple Inc.

import argparse
import contextlib
import copy
import functools
import json
import math
import sys
import threading
import time
import weakref
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    FrozenSet,
    Generator,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_reduce
from transformers import PreTrainedTokenizer

from .models import cache
from .models.cache import (
    ArraysCache,
    BatchKVCache,
    BatchRotatingKVCache,
    CacheList,
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
    TokenBuffer,
    load_prompt_cache,
)
from .sample_utils import apply_min_p, apply_top_k, apply_top_p, make_sampler
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


class GenerationForwardPhase(str, Enum):
    """A stable description of a model forward within generation.

    ``DRAFT`` and ``VERIFY`` are used by external speculative decoding.  The
    enum deliberately describes generation mechanics rather than a particular
    model architecture so callers can use the same callback with ordinary and
    speculative generation.
    """

    PREFILL = "prefill"
    DECODE = "decode"
    DRAFT = "draft"
    VERIFY = "verify"
    MTP_DRAFT = "mtp_draft"


LogicalPositionVector = Tuple[int, ...]
LogicalPositionMatrix = Tuple[LogicalPositionVector, ...]
GenerationLogicalPositions = Union[LogicalPositionVector, LogicalPositionMatrix]


def _logical_position_shape(
    logical_positions: GenerationLogicalPositions,
) -> Tuple[int, int]:
    """Validate immutable host positions and return their exact ``(B, S)``.

    A legacy vector is deliberately only valid for a single row.  Batched
    callers must supply one explicit immutable row per input batch entry, so a
    shared row can never be silently broadcast over another request.
    """

    if not isinstance(logical_positions, tuple) or not logical_positions:
        raise ValueError("generation_logical_positions_required")
    first = logical_positions[0]
    if isinstance(first, tuple):
        rows = logical_positions
        if not rows or any(not isinstance(row, tuple) or not row for row in rows):
            raise ValueError("generation_logical_position_matrix_ragged")
        sequence_length = len(rows[0])
        if any(len(row) != sequence_length for row in rows):
            raise ValueError("generation_logical_position_matrix_ragged")
        values = (position for row in rows for position in row)
        shape = (len(rows), sequence_length)
    else:
        values = iter(logical_positions)
        shape = (1, len(logical_positions))
    if any(
        isinstance(position, bool) or not isinstance(position, int) or position < 0
        for position in values
    ):
        raise ValueError("generation_logical_position_value_invalid")
    return shape


def _validate_logical_positions_for_input(
    logical_positions: GenerationLogicalPositions, input_shape: Tuple[int, ...]
) -> None:
    if len(input_shape) != 2:
        raise ValueError("generation_logical_position_input_rank_invalid")
    if input_shape[0] > 1 and not isinstance(logical_positions[0], tuple):
        raise ValueError("generation_logical_position_batch_requires_matrix")
    if input_shape[0] == 1 and isinstance(logical_positions[0], tuple):
        raise ValueError("generation_logical_position_single_row_requires_vector")
    position_shape = _logical_position_shape(logical_positions)
    if position_shape != input_shape:
        raise ValueError("generation_logical_position_input_shape_mismatch")


@dataclass(frozen=True)
class GenerationForward:
    """Immutable metadata for one Python model-call graph construction.

    The callback scope deliberately covers graph construction only.  MLX
    realization may remain asynchronously pipelined after the scope exits.
    ``input_tokens`` has the exact batched shape passed to the model.
    """

    model: nn.Module
    input_tokens: mx.array
    cache: Any
    phase: GenerationForwardPhase
    input_embeddings: Optional[mx.array] = None
    logical_positions: Optional[GenerationLogicalPositions] = None
    logical_position_ack: Optional["GenerationForwardPositionAck"] = None

    def __post_init__(self):
        if self.logical_positions is not None:
            _validate_logical_positions_for_input(
                self.logical_positions, tuple(self.input_tokens.shape)
            )
            if self.logical_position_ack is not None and (
                self.logical_position_ack._logical_positions != self.logical_positions
                or self.logical_position_ack._position_shape
                != tuple(self.input_tokens.shape)
            ):
                raise ValueError("generation_logical_position_ack_binding_mismatch")
        elif self.logical_position_ack is not None:
            raise ValueError("generation_logical_position_ack_requires_positions")


@dataclass(frozen=True)
class NativeMTPCapabilityFingerprint:
    """Stable value identity for a model's native-MTP capability snapshot."""

    type_module: str
    type_name: str
    supported: bool
    reason: str
    num_layers: Optional[int]

    @classmethod
    def from_model(cls, model: nn.Module) -> "NativeMTPCapabilityFingerprint":
        capability = getattr(model, "mtp_capability", None)
        if capability is None:
            raise RuntimeError("native_mtp_model_capability_missing")
        capability_type = type(capability)
        return cls(
            capability_type.__module__,
            capability_type.__qualname__,
            capability.supported,
            capability.reason,
            getattr(capability, "num_layers", None),
        )


_GENERATION_FORWARD_RECEIPT_ISSUER = object()
_GENERATION_FORWARD_RECEIPT_LOCK = threading.Lock()


class _GenerationForwardReceiptToken:
    __slots__ = ("__weakref__",)


_GENERATION_FORWARD_RECEIPTS = weakref.WeakKeyDictionary()


def _array_identity_evidence(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return type(value), tuple(_array_identity_evidence(item) for item in value)
    return id(value), tuple(value.shape), value.dtype


def _cache_state_identity_evidence(entries):
    evidence = []
    for entry in entries:
        if isinstance(entry, ArraysCache):
            state = tuple(_array_identity_evidence(value) for value in entry.cache)
            evidence.append(
                (
                    type(entry),
                    state,
                    _array_identity_evidence(entry.left_padding),
                    _array_identity_evidence(entry.lengths),
                )
            )
        else:
            evidence.append(
                (
                    type(entry),
                    entry.offset,
                    _array_identity_evidence(entry.keys),
                    _array_identity_evidence(entry.values),
                )
            )
    return tuple(evidence)


def _uint64_hash_constant(value):
    """Construct exact uint64 bits without MLX's unsigned Python-int cast."""

    value &= (1 << 64) - 1
    if value >= 1 << 63:
        value -= 1 << 64
    return mx.array(value, dtype=mx.int64).view(mx.uint64)


def _active_array_payload_hash(array):
    """Hash exact payload bits to two words; this is not cryptographic."""

    payload = array.view(mx.uint8)
    if payload.ndim == 0:
        payload = payload.reshape(1)
    words = payload.astype(mx.uint64)
    position = _uint64_hash_constant(0)
    position_constant = 0x517CC1B727220A95
    position_step = 0x6C8E9CF570932BD5
    mask = (1 << 64) - 1
    for axis, size in enumerate(payload.shape):
        coordinate_shape = [1] * payload.ndim
        coordinate_shape[axis] = size
        coordinates = mx.arange(size, dtype=mx.uint64).reshape(coordinate_shape)
        axis_constant = _uint64_hash_constant(
            (position_constant + axis * position_step) & mask
        )
        position = position + coordinates * axis_constant

    initial = mx.bitwise_xor(
        words + _uint64_hash_constant(0x243F6A8885A308D3),
        position + _uint64_hash_constant(0x13198A2E03707344),
    )
    mixed = initial * _uint64_hash_constant(0x9E3779B185EBCA87)
    mixed = mx.bitwise_xor(mixed, mx.right_shift(mixed, 29))
    mixed = mixed * _uint64_hash_constant(0xC2B2AE3D27D4EB4F)
    second = mx.bitwise_xor(
        mixed,
        position * _uint64_hash_constant(0x165667B19E3779F9)
        + _uint64_hash_constant(0x85EBCA77C2B2AE63),
    )
    lanes = mx.stack((mixed, second), axis=-1)
    return mx.sum(lanes, axis=tuple(range(payload.ndim)))


def _array_content_digest(value):
    """Return compact dtype-preserving hashes for a nested array payload."""

    flattened = []

    def append_arrays(item):
        if item is None:
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                append_arrays(child)
            return
        flattened.append(item)

    append_arrays(value)
    parts = [_active_array_payload_hash(array) for array in flattened]
    if not parts:
        return mx.zeros((0,), dtype=mx.uint64)
    return mx.concatenate(parts)


def _cache_content_digest(entries):
    active = []
    for entry in entries:
        if isinstance(entry, ArraysCache):
            active.extend(entry.cache)
            active.extend((entry.left_padding, entry.lengths))
            continue

        def active_prefix(value):
            if value is None:
                return None
            if isinstance(value, (list, tuple)):
                return type(value)(active_prefix(item) for item in value)
            return value[..., : entry.offset, :]

        active.extend((active_prefix(entry.keys), active_prefix(entry.values)))
    return _array_content_digest(active)


@dataclass(frozen=True)
class _GenerationForwardCanonicalRecord:
    receipt_id: int
    model_id: int
    cache: List[Any]
    cache_container_id: int
    cache_entry_ids: Tuple[int, ...]
    cache_state_evidence: tuple
    cache_content_digest: Optional[mx.array]
    capability: NativeMTPCapabilityFingerprint
    phase: GenerationForwardPhase
    logical_positions: Tuple[int, ...]
    token_ids: Tuple[int, ...]
    immediate_successor_token_ids: Tuple[int, ...]
    logits: Optional[mx.array]
    logits_evidence: Optional[tuple]
    logits_content_digest: Optional[mx.array]
    hidden_rows: mx.array
    hidden_evidence: tuple
    hidden_content_digest: mx.array


@dataclass
class _GenerationForwardAuthority:
    record: _GenerationForwardCanonicalRecord
    reservation: Optional[Any] = None


@dataclass(frozen=True, init=False)
class GenerationForwardPositionReceipt:
    """Unforgeable immutable evidence from one acknowledged target call."""

    model_id: int
    cache_container_id: int
    cache_entry_ids: Tuple[int, ...]
    capability: NativeMTPCapabilityFingerprint
    phase: GenerationForwardPhase
    logical_positions: Tuple[int, ...]
    token_ids: Tuple[int, ...]
    immediate_successor_token_ids: Tuple[int, ...]
    logits: Optional[mx.array]
    hidden_rows: mx.array
    _issuer_seal: Any
    _record_token: Any

    def __init__(
        self,
        *,
        _issuer,
        model_id,
        cache_container_id,
        cache_entry_ids,
        capability,
        phase,
        logical_positions,
        token_ids,
        immediate_successor_token_ids,
        logits,
        hidden_rows,
        record_token,
    ):
        if _issuer is not _GENERATION_FORWARD_RECEIPT_ISSUER:
            raise TypeError(
                "generation forward receipts are issued by an acknowledged call"
            )
        for name, value in (
            ("model_id", model_id),
            ("cache_container_id", cache_container_id),
            ("cache_entry_ids", cache_entry_ids),
            ("capability", capability),
            ("phase", phase),
            ("logical_positions", logical_positions),
            ("token_ids", token_ids),
            ("immediate_successor_token_ids", immediate_successor_token_ids),
            ("logits", logits),
            ("hidden_rows", hidden_rows),
            ("_issuer_seal", _issuer),
            ("_record_token", record_token),
        ):
            object.__setattr__(self, name, value)


class GenerationForwardPositionAck:
    """One-shot proof that a context consumer used exact logical positions.

    The generation wrapper activates the acknowledgement only for the model
    call itself.  A request-local position hook must acknowledge the immutable
    host positions while that call is in progress; acknowledging on context
    entry/exit, acknowledging twice, or acknowledging different positions is
    rejected.
    """

    def __init__(
        self,
        logical_positions: GenerationLogicalPositions,
        *,
        model: Optional[nn.Module] = None,
        cache: Optional[Any] = None,
        token_ids: Optional[Tuple[int, ...]] = None,
        immediate_successor_token_ids: Tuple[int, ...] = (),
        phase: Optional[GenerationForwardPhase] = None,
        _receipt_issuer=None,
    ):
        self._position_shape = _logical_position_shape(logical_positions)
        self._logical_positions = logical_positions
        self._model = model
        self._cache = cache
        self._token_ids = token_ids
        self._immediate_successor_token_ids = immediate_successor_token_ids
        self._phase = phase
        self._capability = (
            NativeMTPCapabilityFingerprint.from_model(model)
            if model is not None and getattr(model, "mtp_capability", None) is not None
            else None
        )
        self._receipt_issuer = _receipt_issuer
        self._active = False
        self._acknowledged = False
        self._finished = False
        self._receipt = None

    @property
    def receipt(self) -> Optional[GenerationForwardPositionReceipt]:
        return self._receipt

    def acknowledge(self, logical_positions: GenerationLogicalPositions) -> None:
        if not self._active or self._finished:
            raise RuntimeError("generation_logical_position_ack_outside_forward")
        if self._acknowledged:
            raise RuntimeError("generation_logical_position_ack_reused")
        if not isinstance(logical_positions, tuple) or (
            logical_positions != self._logical_positions
        ):
            raise RuntimeError("generation_logical_position_ack_mismatch")
        self._acknowledged = True

    def _assert_consumer_binding(self, *, model, cache, phase, input_shape) -> None:
        """Reject a context consumed by a different model-call graph."""

        if (
            self._model is not model
            or self._cache is not cache
            or self._phase is not phase
            or tuple(input_shape) != self._position_shape
        ):
            raise RuntimeError("generation_logical_position_consumer_mismatch")
        if self._capability is not None and (
            NativeMTPCapabilityFingerprint.from_model(model) != self._capability
        ):
            raise RuntimeError("generation_logical_position_capability_mismatch")

    def _activate(self) -> None:
        if self._active or self._finished:
            raise RuntimeError("generation_logical_position_ack_reused")
        self._active = True

    def _require_acknowledged(self) -> None:
        if not self._active or not self._acknowledged:
            raise RuntimeError("generation_logical_positions_not_acknowledged")

    def _finish(self) -> None:
        self._active = False
        self._finished = True

    def _issue_receipt(self, result) -> GenerationForwardPositionReceipt:
        if not self._finished or not self._acknowledged or self._receipt is not None:
            raise RuntimeError("generation_logical_position_receipt_not_ready")
        if self._receipt_issuer is not _GENERATION_FORWARD_RECEIPT_ISSUER:
            raise RuntimeError("generation_logical_position_receipt_untrusted")
        if (
            self._model is None
            or not isinstance(self._cache, list)
            or self._token_ids is None
            or self._phase is None
        ):
            raise RuntimeError("generation_logical_position_receipt_unbound")
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("generation_logical_position_receipt_missing_hidden")
        logits, hidden_rows = result
        retained_hidden_rows = hidden_rows + mx.zeros(
            hidden_rows.shape, dtype=hidden_rows.dtype
        )
        hidden_content_digest = _array_content_digest(retained_hidden_rows)
        retained_final_logits = None
        logits_content_digest = None
        cache_content_digest = None
        if len(self._immediate_successor_token_ids) == len(self._logical_positions) - 1:
            final_logits = logits[:, -1:, :]
            retained_final_logits = final_logits + mx.zeros(
                final_logits.shape, dtype=final_logits.dtype
            )
            logits_content_digest = _array_content_digest(retained_final_logits)
            cache_content_digest = _cache_content_digest(self._cache)
        canonical_values = [retained_hidden_rows, hidden_content_digest]
        if retained_final_logits is not None:
            canonical_values.append(retained_final_logits)
            canonical_values.append(logits_content_digest)
            canonical_values.append(cache_content_digest)
        mx.eval(canonical_values)
        record_token = _GenerationForwardReceiptToken()
        self._receipt = GenerationForwardPositionReceipt(
            _issuer=_GENERATION_FORWARD_RECEIPT_ISSUER,
            model_id=id(self._model),
            cache_container_id=id(self._cache),
            cache_entry_ids=tuple(id(entry) for entry in self._cache),
            capability=NativeMTPCapabilityFingerprint.from_model(self._model),
            phase=self._phase,
            logical_positions=self._logical_positions,
            token_ids=self._token_ids,
            immediate_successor_token_ids=self._immediate_successor_token_ids,
            logits=retained_final_logits,
            hidden_rows=retained_hidden_rows,
            record_token=record_token,
        )
        record = _GenerationForwardCanonicalRecord(
            receipt_id=id(self._receipt),
            model_id=id(self._model),
            cache=self._cache,
            cache_container_id=id(self._cache),
            cache_entry_ids=tuple(id(entry) for entry in self._cache),
            cache_state_evidence=_cache_state_identity_evidence(self._cache),
            cache_content_digest=cache_content_digest,
            capability=NativeMTPCapabilityFingerprint.from_model(self._model),
            phase=self._phase,
            logical_positions=self._logical_positions,
            token_ids=self._token_ids,
            immediate_successor_token_ids=self._immediate_successor_token_ids,
            logits=retained_final_logits,
            logits_evidence=_array_identity_evidence(retained_final_logits),
            logits_content_digest=logits_content_digest,
            hidden_rows=retained_hidden_rows,
            hidden_evidence=_array_identity_evidence(retained_hidden_rows),
            hidden_content_digest=hidden_content_digest,
        )
        with _GENERATION_FORWARD_RECEIPT_LOCK:
            _GENERATION_FORWARD_RECEIPTS[record_token] = _GenerationForwardAuthority(
                record
            )
        return self._receipt


GenerationForwardContext = Callable[[GenerationForward], ContextManager[None]]


def _native_mtp_position_context(
    model: nn.Module,
    external_context: Optional[GenerationForwardContext],
) -> GenerationForwardContext:
    """Choose the one authoritative position consumer for native MTP.

    Qwen publishes a model-owned context hook.  It is selected automatically
    so callers cannot accidentally request sparse logical positions without
    applying them inside the model.  Older third-party native-MTP models may
    still provide the existing external callback, but a Qwen hook and a
    different external callback would create ambiguous dual consumers and is
    rejected before any cache is constructed.
    """

    canonical_context = getattr(model, "generation_forward_context", None)
    if canonical_context is not None and not callable(canonical_context):
        raise TypeError("native_mtp_model_position_context_invalid")
    if canonical_context is not None:
        if external_context is not None and external_context is not canonical_context:
            raise ValueError("native_mtp_position_context_ambiguous")
        return canonical_context
    if external_context is None:
        raise ValueError("native_mtp_position_context_missing")
    return external_context


def attested_target_forward(
    model: nn.Module,
    token_ids: Tuple[int, ...],
    target_cache: List[Any],
    *,
    phase: GenerationForwardPhase,
    logical_positions: Tuple[int, ...],
    immediate_successor_token_ids: Tuple[int, ...],
    model_forward_context: GenerationForwardContext,
) -> Tuple[Tuple[mx.array, mx.array], GenerationForwardPositionReceipt]:
    """Run and attest one exact hidden-returning target forward.

    This is the only public receipt producer.  It owns the entire call rather
    than accepting caller-provided outputs, so a receipt can only be issued
    after the model itself returned hidden rows under an acknowledged position
    context and those lazy outputs realized successfully.
    """

    if not isinstance(target_cache, list):
        raise TypeError("attested target cache must be a caller-owned list")
    if not isinstance(token_ids, tuple) or not token_ids:
        raise ValueError("attested target forward requires non-empty host tokens")
    if any(
        isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
        for token_id in token_ids
    ):
        raise ValueError("attested target token IDs must be non-negative integers")
    if not isinstance(logical_positions, tuple) or len(logical_positions) != len(
        token_ids
    ):
        raise ValueError("attested target positions must match input tokens")
    if not isinstance(immediate_successor_token_ids, tuple) or len(
        immediate_successor_token_ids
    ) not in (len(logical_positions), len(logical_positions) - 1):
        raise ValueError("attested target successor count is invalid")
    input_tokens = mx.array(token_ids, dtype=mx.uint32)
    ack = GenerationForwardPositionAck(
        logical_positions,
        model=model,
        cache=target_cache,
        token_ids=token_ids,
        immediate_successor_token_ids=immediate_successor_token_ids,
        phase=phase,
        _receipt_issuer=_GENERATION_FORWARD_RECEIPT_ISSUER,
    )
    forward = GenerationForward(
        model=model,
        input_tokens=input_tokens[None],
        cache=target_cache,
        phase=phase,
        logical_positions=logical_positions,
        logical_position_ack=ack,
    )
    with model_forward_context(forward):
        ack._activate()
        try:
            result = model(input_tokens[None], cache=target_cache, return_hidden=True)
            ack._require_acknowledged()
            if not isinstance(result, tuple) or len(result) != 2:
                raise RuntimeError("attested target forward did not return hidden rows")
            mx.eval(result)
        finally:
            ack._finish()
    receipt = ack._issue_receipt(result)
    return result, receipt


@dataclass(frozen=True)
class NativeMTPSamplingConfig:
    """Replay-safe sampling contract for native MTP speculation."""

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    min_tokens_to_keep: int = 1
    seed: Optional[int] = None

    def __post_init__(self):
        for name in ("temperature", "top_p", "min_p"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"native MTP {name} must be a finite number")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int):
            raise ValueError("native MTP top_k must be an integer")
        if isinstance(self.min_tokens_to_keep, bool) or not isinstance(
            self.min_tokens_to_keep, int
        ):
            raise ValueError("native MTP min_tokens_to_keep must be an integer")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise ValueError("native MTP seed must be an integer or None")
        if self.temperature < 0:
            raise ValueError("native MTP temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("native MTP top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("native MTP top_k must be non-negative")
        if not 0 <= self.min_p <= 1:
            raise ValueError("native MTP min_p must be in [0, 1]")
        if self.min_tokens_to_keep < 1:
            raise ValueError("native MTP min_tokens_to_keep must be positive")
        if self.seed is not None and self.seed < 0:
            raise ValueError("native MTP seed must be non-negative")

    @property
    def stochastic(self) -> bool:
        return self.temperature > 0


class _NativeMTPSamplingState:
    """Replay-safe sampling and RNG state for one native-MTP row.

    The single-request generator deliberately keeps this object private: its
    public call signature and draw order are part of the existing B=1
    behaviour.  The batched native-MTP path keeps an equivalent UID-local
    retained key, rather than sharing a mutable global key across rows.
    """

    def __init__(
        self,
        config: NativeMTPSamplingConfig,
        *,
        sampler: Optional[Callable[[mx.array], mx.array]] = None,
        logits_processors: Optional[
            List[Callable[[mx.array, mx.array], mx.array]]
        ] = None,
        seed: Optional[int] = None,
    ):
        self.config = config
        self.sampler = sampler
        self.logits_processors = logits_processors or []
        self._rng_key = mx.random.key(
            seed if seed is not None else (time.time_ns() & ((1 << 63) - 1))
        )

    def next_rng_key(self):
        keys = mx.random.split(self._rng_key)
        self._rng_key = keys[0]
        return keys[1]

    def reported_logprobs(self, logits, tokens):
        if logits.ndim == 1:
            logits = logits[None]
        for processor in self.logits_processors:
            logits = processor(tokens, logits)
        return (logits - mx.logsumexp(logits, axis=-1, keepdims=True)).squeeze(0)

    def sampling_distribution(self, reported_logprobs):
        logprobs = reported_logprobs[None]
        if self.config.top_p < 1:
            logprobs = apply_top_p(logprobs, self.config.top_p)
        if self.config.min_p > 0:
            keep = self.config.min_tokens_to_keep
            vocab_size = logprobs.shape[-1]
            if keep > vocab_size:
                raise ValueError(
                    "native MTP min_tokens_to_keep cannot exceed vocabulary size"
                )
            if keep < vocab_size:
                # apply_min_p's multi-token branch passes a Python bool to
                # put_along_axis, which newer MLX releases reject. Preserve
                # the standard filter exactly by restoring requested top rows.
                unfiltered = logprobs
                logprobs = apply_min_p(logprobs, self.config.min_p)
                if keep > 1:
                    top_indices = mx.argpartition(unfiltered, kth=-keep, axis=-1)[
                        ..., -keep:
                    ]
                    top_values = mx.take_along_axis(unfiltered, top_indices, axis=-1)
                    logprobs = mx.put_along_axis(
                        logprobs, top_indices, top_values, axis=-1
                    )
        if self.config.top_k > 0:
            if self.config.top_k >= logprobs.shape[-1]:
                raise ValueError(
                    "native MTP top_k must be smaller than vocabulary size"
                )
            logprobs = apply_top_k(logprobs, self.config.top_k)
        if self.config.stochastic:
            logprobs = logprobs / self.config.temperature
        return (logprobs - mx.logsumexp(logprobs, axis=-1, keepdims=True)).squeeze(0)

    def sample(self, logprobs, *, rng_key=None):
        if self.config.stochastic:
            return mx.random.categorical(
                logprobs, key=self.next_rng_key() if rng_key is None else rng_key
            )
        if self.sampler is None:
            return mx.argmax(logprobs, axis=-1)
        return self.sampler(logprobs)

    def residual_sample(self, target_logprobs, draft_logprobs, *, rng_key=None):
        target_probs = mx.exp(target_logprobs)
        draft_probs = mx.exp(draft_logprobs)
        residual = mx.maximum(target_probs - draft_probs, 0)
        normalizer = mx.sum(residual)
        residual_logprobs = mx.where(
            normalizer > 0,
            mx.log(residual / normalizer),
            target_logprobs,
        )
        return self.sample(residual_logprobs, rng_key=rng_key), residual_logprobs


@dataclass(frozen=True)
class _NativeMTPCohortBinding:
    """Exact cache identities bound before a cohort forward may mutate them."""

    model_id: int
    backbone_container_id: int
    mtp_container_id: int
    uids: Tuple[int, ...]
    backbone_layout: Tuple[tuple, ...]
    mtp_layout: Tuple[tuple, ...]
    backbone_metadata: Tuple[tuple, ...]
    mtp_metadata: Tuple[tuple, ...]


@dataclass(frozen=True)
class _NativeMTPCohortCacheCheckpoint:
    """Recurrent-value / KV-offset transaction state for one cohort round."""

    backbone: Tuple[tuple, ...]
    mtp: Tuple[tuple, ...]


@dataclass(frozen=True)
class _NativeMTPCohortMutationDelta:
    """Caller-declared physical cache advance for one verified cohort forward."""

    backbone: Tuple[Tuple[int, ...], ...]
    mtp: Tuple[Tuple[int, ...], ...]


class _NativeMTPCohortCache:
    """Private move-only cache owner for a homogeneous native-MTP cohort.

    Rows enter as separately owned B=1 native-MTP requests, then become one
    merged cohort cache.  The caller must bind immediately before each model
    forward, so replacing a cache container or entry cannot silently redirect a
    transaction.  KV rollback records only offsets and cache references; only
    small recurrent ``ArraysCache`` values are copied at the transaction
    boundary.  Rotating and quantized layouts are deliberately refused.
    """

    _MAX_RECURRENT_STATE_ELEMENTS = 65536

    def __init__(
        self,
        model: nn.Module,
        rows: Sequence[cache.NativeMTPRequestCache],
        *,
        uids: Optional[Sequence[int]] = None,
    ):
        if not rows:
            raise ValueError("native_mtp_cohort_requires_rows")
        if any(not isinstance(row, cache.NativeMTPRequestCache) for row in rows):
            raise TypeError("native_mtp_cohort_requires_native_request_caches")
        if any(row.model is not model for row in rows):
            raise ValueError("native_mtp_cohort_model_mismatch")
        if any(row.closed for row in rows):
            raise RuntimeError("native_mtp_cohort_row_closed")
        if any(row.checkpoint_active for row in rows):
            raise RuntimeError("native_mtp_cohort_row_transaction_active")
        for row in rows:
            row.assert_aligned(
                backbone_tokens=row.state.backbone_tokens,
                mtp_tokens=row.state.mtp_tokens,
            )

        if uids is None:
            uids = tuple(range(len(rows)))
        if len(uids) != len(rows) or len(set(uids)) != len(uids):
            raise ValueError("native_mtp_cohort_uids_invalid")

        # Build every destination entry before consuming any source owner.  A
        # source request becomes closed only after the full merged cohort is
        # valid, which makes failed construction atomically non-consuming.
        backbone = self._merge_entries([row.backbone for row in rows])
        mtp = self._merge_entries([row.mtp for row in rows])
        self.model = model
        self._poisoned_reason = None
        self._checkpoint = None
        self._transaction_sealed = False
        self._size = len(rows)
        self.uids = tuple(uids)
        self.backbone = backbone
        self.mtp = mtp
        self._binding = self._make_binding()
        try:
            for row in rows:
                row.finish("generator_closed")
        except BaseException:
            for row in rows:
                row._checkpoint = None
                row._replay_required = None
                row._closed = True
            self.poison("native_mtp_cohort_source_consume_failed")
            raise

    @staticmethod
    def _reject_entry(entry):
        if isinstance(entry, (QuantizedKVCache, RotatingKVCache, BatchRotatingKVCache)):
            raise RuntimeError("native_mtp_cohort_cache_layout_unsupported")
        if not isinstance(entry, (KVCache, BatchKVCache, ArraysCache)):
            raise TypeError("native_mtp_cohort_cache_type_unsupported")

    @classmethod
    def _merge_entries(cls, rows):
        width = len(rows[0])
        if width == 0 or any(len(row) != width for row in rows):
            raise ValueError("native_mtp_cohort_cache_topology_mismatch")
        merged = []
        for index in range(width):
            entries = [row[index] for row in rows]
            for entry in entries:
                cls._reject_entry(entry)
            entry_type = type(entries[0])
            if any(type(entry) is not entry_type for entry in entries):
                raise TypeError("native_mtp_cohort_cache_type_mismatch")
            schema = cls._entry_schema(entries[0])
            if any(cls._entry_schema(entry) != schema for entry in entries[1:]):
                raise TypeError("native_mtp_cohort_cache_storage_schema_mismatch")
            merged.append(entry_type.merge(entries))
        return merged

    @property
    def size(self):
        return self._size

    @property
    def poisoned(self):
        return self._poisoned_reason is not None

    @staticmethod
    def _array_layout(value):
        if value is None:
            return None
        return id(value), tuple(value.shape), value.dtype

    @classmethod
    def _entry_schema(cls, entry):
        """Stable topology/storage schema; excludes mutable sequence capacity."""

        cls._reject_entry(entry)
        if isinstance(entry, ArraysCache):
            batch_size = entry.batch_size
            if any(
                value is not None
                and math.prod(value.shape[1:]) > cls._MAX_RECURRENT_STATE_ELEMENTS
                for value in entry.cache
            ):
                raise RuntimeError("native_mtp_cohort_recurrent_state_too_large")
            slots = []
            for value in entry.cache:
                if value is None:
                    slots.append(None)
                    continue
                if value.ndim < 1 or value.shape[0] != batch_size:
                    raise RuntimeError(
                        "native_mtp_cohort_recurrent_batch_size_mismatch"
                    )
                slots.append(
                    (value.ndim, value.shape[0], tuple(value.shape[1:]), value.dtype)
                )
            return (
                type(entry),
                batch_size,
                tuple(slots),
            )

        def kv_schema(value):
            if value is None:
                return None
            return (
                value.ndim,
                value.shape[0],
                value.shape[1],
                value.shape[3],
                value.dtype,
            )

        if entry.keys is None and isinstance(entry, BatchKVCache):
            if entry.offset.size != entry.left_padding.size:
                raise RuntimeError("native_mtp_cohort_batch_metadata_width_mismatch")
            batch_size = entry.offset.size
        else:
            batch_size = entry.keys.shape[0] if entry.keys is not None else 1

        return (
            type(entry),
            batch_size,
            kv_schema(entry.keys),
            kv_schema(entry.values),
        )

    @classmethod
    def _entry_layout(cls, entry):
        """Stable entry ownership plus schema; mutable backing storage is excluded."""

        return id(entry), cls._entry_schema(entry)

    @staticmethod
    def _metadata_spec(entry):
        if isinstance(entry, ArraysCache):
            return (entry.left_padding, entry.lengths)
        if not isinstance(entry, BatchKVCache):
            return (mx.array([entry.offset]), None)
        return (entry.offset, entry.left_padding)

    @classmethod
    def _host_metadata(cls, entries):
        """Read mutable cache metadata through exactly one aggregate decision."""

        arrays = []
        widths = []
        for entry in entries:
            entry_widths = []
            for value in cls._metadata_spec(entry):
                width = 0 if value is None else value.size
                entry_widths.append(width)
                if value is not None:
                    arrays.append(value.reshape(-1).astype(mx.int64))
            widths.append(tuple(entry_widths))
        flattened = mx.concatenate(arrays) if arrays else mx.array([], dtype=mx.int64)
        mx.eval(flattened)
        values = tuple(flattened.tolist())
        cursor = 0
        metadata = []
        for entry_widths in widths:
            entry_values = []
            for width in entry_widths:
                entry_values.append(values[cursor : cursor + width])
                cursor += width
            entry = entries[len(metadata)]
            if isinstance(entry, ArraysCache):
                metadata.append((None, *entry_values))
            else:
                metadata.append((entry._idx, *entry_values))
        return tuple(metadata)

    def _make_binding(self):
        entries = self.backbone + self.mtp
        metadata = self._host_metadata(entries)
        split = len(self.backbone)
        return _NativeMTPCohortBinding(
            model_id=id(self.model),
            backbone_container_id=id(self.backbone),
            mtp_container_id=id(self.mtp),
            uids=self.uids,
            backbone_layout=tuple(self._entry_layout(entry) for entry in self.backbone),
            mtp_layout=tuple(self._entry_layout(entry) for entry in self.mtp),
            backbone_metadata=metadata[:split],
            mtp_metadata=metadata[split:],
        )

    def _binding_topology_matches(self):
        """Check ownership identity before reading mutable cache internals.

        In particular, a same-type replacement can be deliberately incomplete
        (for example, a fresh ``KVCache`` in place of a merged
        ``BatchKVCache``).  Its identity mismatch is enough to reject it; do
        not walk its metadata just to discover that its representation is not
        a valid cohort entry.
        """

        binding = self._binding
        if (
            id(self.model) != binding.model_id
            or id(self.backbone) != binding.backbone_container_id
            or id(self.mtp) != binding.mtp_container_id
            or self.uids != binding.uids
        ):
            return False

        def entries_match(entries, layouts):
            return len(entries) == len(layouts) and all(
                id(entry) == entry_id and type(entry) is schema[0]
                for entry, (entry_id, schema) in zip(entries, layouts)
            )

        return entries_match(self.backbone, binding.backbone_layout) and entries_match(
            self.mtp, binding.mtp_layout
        )

    def bind_before_mutation(self):
        """Assert that the exact cache identities survive until the forward."""

        if self.poisoned:
            raise RuntimeError(self._poisoned_reason)
        if not self._binding_topology_matches():
            self.poison("native_mtp_cohort_cache_binding_changed")
            raise RuntimeError("native_mtp_cohort_cache_binding_changed")
        try:
            actual = self._make_binding()
        except BaseException as exc:
            self.poison("native_mtp_cohort_cache_binding_changed")
            raise RuntimeError("native_mtp_cohort_cache_binding_changed") from exc
        if actual != self._binding:
            self.poison("native_mtp_cohort_cache_binding_changed")
            raise RuntimeError("native_mtp_cohort_cache_binding_changed")
        return self.backbone, self.mtp

    @staticmethod
    def _validate_delta(name, before, after, expected):
        if len(before) != len(after) or len(before) != len(expected):
            raise ValueError(f"native_mtp_cohort_{name}_delta_topology_mismatch")
        for layer, (before_layer, after_layer, expected_layer) in enumerate(
            zip(before, after, expected)
        ):
            # KV metadata is (_idx, offsets, left_padding); Arrays metadata is
            # (None, left_padding, lengths).  Each ArraysCache vector is
            # independently optional: an absent vector is an advance no-op;
            # a present vector must retain its cohort width and exact delta.
            before_index, before_primary, *before_rest = before_layer
            after_index, after_primary, *after_rest = after_layer
            if before_index is None:
                if len(before_rest) != 1 or len(after_rest) != 1:
                    raise RuntimeError(
                        f"native_mtp_cohort_{name}_arrays_metadata_changed: layer={layer}"
                    )
                before_lengths, after_lengths = before_rest[0], after_rest[0]
                if not expected_layer:
                    if tuple(before_primary) != tuple(after_primary) or tuple(
                        before_lengths
                    ) != tuple(after_lengths):
                        raise RuntimeError(
                            f"native_mtp_cohort_{name}_arrays_metadata_changed: layer={layer}"
                        )
                    continue
                expected_delta = tuple(-value for value in expected_layer)
                for before_vector, after_vector in (
                    (before_primary, after_primary),
                    (before_lengths, after_lengths),
                ):
                    if not before_vector:
                        if after_vector:
                            raise RuntimeError(
                                f"native_mtp_cohort_{name}_arrays_metadata_changed: layer={layer}"
                            )
                        continue
                    if (
                        not after_vector
                        or len(before_vector) != len(expected_layer)
                        or len(after_vector) != len(expected_layer)
                    ):
                        raise RuntimeError(
                            f"native_mtp_cohort_{name}_arrays_metadata_changed: layer={layer}"
                        )
                    if (
                        tuple(
                            current - prior
                            for prior, current in zip(before_vector, after_vector)
                        )
                        != expected_delta
                    ):
                        raise RuntimeError(
                            f"native_mtp_cohort_{name}_array_delta_unsupported: layer={layer}"
                        )
                continue
            if tuple(after_rest) != tuple(before_rest):
                raise RuntimeError(
                    f"native_mtp_cohort_{name}_metadata_changed: layer={layer}"
                )
            actual_delta = tuple(
                current - prior for prior, current in zip(before_primary, after_primary)
            )
            if (
                len(before_primary) != len(after_primary)
                or tuple(expected_layer) != actual_delta
                or after_index - before_index
                != (actual_delta[0] if actual_delta else 0)
                or len(set(actual_delta)) > 1
            ):
                raise RuntimeError(
                    f"native_mtp_cohort_{name}_delta_mismatch: layer={layer}"
                )

    @classmethod
    def _schema_transition_allowed(cls, before, after):
        """Allow only one-way first-population transitions for one owner."""

        before_entry_id, before_schema = before
        after_entry_id, after_schema = after
        if (
            before_entry_id != after_entry_id
            or len(before_schema) < 2
            or len(after_schema) < 2
        ):
            return False

        before_type, before_batch = before_schema[:2]
        after_type, after_batch = after_schema[:2]
        if before_type is not after_type or before_batch != after_batch:
            return False

        def valid_dimension(value):
            return isinstance(value, int) and not isinstance(value, bool) and value > 0

        def valid_kv_schema(schema):
            return (
                isinstance(schema, tuple)
                and len(schema) == 5
                and schema[0] == 4
                and schema[1] == before_batch
                and valid_dimension(schema[2])
                and valid_dimension(schema[3])
                and schema[4] is not None
            )

        if before_type in (KVCache, BatchKVCache):
            if len(before_schema) != 4 or len(after_schema) != 4:
                return False
            before_keys, before_values = before_schema[2:]
            after_keys, after_values = after_schema[2:]
            return (
                before_keys is None
                and before_values is None
                and valid_kv_schema(after_keys)
                and valid_kv_schema(after_values)
            )

        if before_type is ArraysCache:
            if len(before_schema) != 3 or len(after_schema) != 3:
                return False
            before_slots, after_slots = before_schema[2], after_schema[2]
            if len(before_slots) != len(after_slots):
                return False

            def valid_array_schema(schema):
                return (
                    isinstance(schema, tuple)
                    and len(schema) == 4
                    and valid_dimension(schema[0])
                    and schema[1] == before_batch
                    and isinstance(schema[2], tuple)
                    and len(schema[2]) == schema[0] - 1
                    and all(
                        isinstance(dimension, int)
                        and not isinstance(dimension, bool)
                        and dimension >= 0
                        for dimension in schema[2]
                    )
                    and schema[3] is not None
                )

            return any(
                before_slot is None and after_slot is not None
                for before_slot, after_slot in zip(before_slots, after_slots)
            ) and all(
                before_slot == after_slot
                or (before_slot is None and valid_array_schema(after_slot))
                for before_slot, after_slot in zip(before_slots, after_slots)
            )

        return False

    def seal_after_mutation(self, expected: _NativeMTPCohortMutationDelta):
        """Record the exact post-forward layout before a later forward.

        Call immediately after the cohort's known model forward.  Subsequent
        ``bind_before_mutation`` calls reject any topology, row ordering,
        metadata, offset, or backing-array change made outside that boundary.
        """

        if self.poisoned:
            raise RuntimeError(self._poisoned_reason)
        if not isinstance(expected, _NativeMTPCohortMutationDelta):
            raise TypeError("native_mtp_cohort_expected_delta_required")
        before = self._binding
        try:
            after = self._make_binding()
        except BaseException as exc:
            self.poison("native_mtp_cohort_unexpected_layout_change")
            raise RuntimeError("native_mtp_cohort_unexpected_layout_change") from exc
        if (
            before.model_id != after.model_id
            or before.backbone_container_id != after.backbone_container_id
            or before.mtp_container_id != after.mtp_container_id
            or before.uids != after.uids
            or len(before.backbone_layout) != len(after.backbone_layout)
            or len(before.mtp_layout) != len(after.mtp_layout)
            or any(
                previous != current
                and not self._schema_transition_allowed(previous, current)
                for previous, current in zip(
                    before.backbone_layout, after.backbone_layout
                )
            )
            or any(
                previous != current
                and not self._schema_transition_allowed(previous, current)
                for previous, current in zip(before.mtp_layout, after.mtp_layout)
            )
        ):
            self.poison("native_mtp_cohort_unexpected_layout_change")
            raise RuntimeError("native_mtp_cohort_unexpected_layout_change")
        try:
            self._validate_delta(
                "backbone",
                before.backbone_metadata,
                after.backbone_metadata,
                expected.backbone,
            )
            self._validate_delta(
                "mtp", before.mtp_metadata, after.mtp_metadata, expected.mtp
            )
        except BaseException:
            self.poison("native_mtp_cohort_unexpected_metadata_change")
            raise
        self._binding = after
        self._transaction_sealed = True

    @staticmethod
    def _copy_recurrent(value):
        """Copy a bounded recurrent value without retaining its write target."""

        if value is None:
            return None
        return value + mx.zeros(value.shape, dtype=value.dtype)

    @staticmethod
    def _snapshot_entry(entry):
        if isinstance(entry, ArraysCache):
            return (
                entry,
                tuple(
                    _NativeMTPCohortCache._copy_recurrent(value)
                    for value in entry.cache
                ),
                _NativeMTPCohortCache._copy_recurrent(entry.left_padding),
                _NativeMTPCohortCache._copy_recurrent(entry.lengths),
            )
        if not isinstance(entry, BatchKVCache):
            raise TypeError("native_mtp_cohort_cache_type_unsupported")
        return (entry, entry._idx, entry.offset, entry.left_padding)

    @staticmethod
    def _restore_entry(snapshot):
        entry = snapshot[0]
        if isinstance(entry, ArraysCache):
            _, values, left_padding, lengths = snapshot
            entry.cache = list(values)
            entry.left_padding = left_padding
            entry.lengths = lengths
            return
        _, previous_idx, previous_offset, previous_left_padding = snapshot
        trim = entry._idx - previous_idx
        if trim < 0 or (trim and entry.trim(trim) != trim):
            raise RuntimeError("native_mtp_cohort_cache_rollback_inexact")
        entry.offset = previous_offset
        entry.left_padding = previous_left_padding

    def checkpoint(self):
        if self.poisoned:
            raise RuntimeError(self._poisoned_reason)
        if self._checkpoint is not None:
            raise RuntimeError("native_mtp_cohort_checkpoint_already_active")
        self.bind_before_mutation()
        self._checkpoint = _NativeMTPCohortCacheCheckpoint(
            backbone=tuple(self._snapshot_entry(entry) for entry in self.backbone),
            mtp=tuple(self._snapshot_entry(entry) for entry in self.mtp),
        )
        self._transaction_sealed = False
        return self._checkpoint

    def commit(self):
        if self._checkpoint is None:
            raise RuntimeError("native_mtp_cohort_checkpoint_missing")
        if not self._transaction_sealed:
            raise RuntimeError("native_mtp_cohort_mutation_not_sealed")
        self._checkpoint = None
        self._transaction_sealed = False

    def rollback(self):
        if self._checkpoint is None:
            raise RuntimeError("native_mtp_cohort_checkpoint_missing")
        checkpoint, self._checkpoint = self._checkpoint, None
        self._transaction_sealed = False
        try:
            for snapshot in checkpoint.backbone:
                self._restore_entry(snapshot)
            for snapshot in checkpoint.mtp:
                self._restore_entry(snapshot)
            self._binding = self._make_binding()
        except BaseException:
            self.poison("native_mtp_cohort_rollback_failed")
            raise

    def filter(self, keep: Sequence[int]):
        if self._checkpoint is not None:
            raise RuntimeError("native_mtp_cohort_filter_during_transaction")
        if len(set(keep)) != len(keep) or any(
            index < 0 or index >= self.size for index in keep
        ):
            raise ValueError("native_mtp_cohort_filter_indices_invalid")
        self.bind_before_mutation()
        try:
            partition = self._partition(keep)
        except BaseException:
            self.poison("native_mtp_cohort_filter_failed")
            raise
        self.backbone, self.mtp, self.uids = partition
        self._size = len(keep)
        self._binding = self._make_binding()

    def split(self, keep: Sequence[int]):
        """Move selected rows into a new owner; neither owner shares a cache."""

        if self._checkpoint is not None:
            raise RuntimeError("native_mtp_cohort_split_during_transaction")
        keep = tuple(keep)
        if len(set(keep)) != len(keep) or any(
            index < 0 or index >= self.size for index in keep
        ):
            raise ValueError("native_mtp_cohort_split_indices_invalid")
        self.bind_before_mutation()
        try:
            selected = self._from_partition(keep)
            remaining = tuple(
                index for index in range(self.size) if index not in set(keep)
            )
            remaining_partition = self._partition(remaining)
        except BaseException:
            self.poison("native_mtp_cohort_split_failed")
            raise
        self.backbone, self.mtp, self.uids = remaining_partition
        self._size = len(remaining)
        self._binding = self._make_binding()
        return selected

    @staticmethod
    def _gather_batch_rows(value, indices):
        """Select row indices on-device while preserving every trailing axis."""

        if value is None:
            return None
        return mx.take(value, indices, axis=0)

    @staticmethod
    def _partition_entry(entry, indices):
        """Device-side whole-cohort partition; never extract host rows."""

        indices = mx.array(indices, dtype=mx.int32)
        clone = copy.copy(entry)
        if isinstance(entry, ArraysCache):
            clone.cache = [
                _NativeMTPCohortCache._gather_batch_rows(value, indices)
                for value in entry.cache
            ]
            clone.left_padding = _NativeMTPCohortCache._gather_batch_rows(
                entry.left_padding, indices
            )
            clone.lengths = _NativeMTPCohortCache._gather_batch_rows(
                entry.lengths, indices
            )
            return clone
        clone.keys = _NativeMTPCohortCache._gather_batch_rows(entry.keys, indices)
        clone.values = _NativeMTPCohortCache._gather_batch_rows(entry.values, indices)
        clone.offset = _NativeMTPCohortCache._gather_batch_rows(entry.offset, indices)
        clone.left_padding = _NativeMTPCohortCache._gather_batch_rows(
            entry.left_padding, indices
        )
        clone._right_padding = None
        return clone

    def _partition(self, indices):
        if not indices:
            return [], [], ()
        return (
            [self._partition_entry(entry, indices) for entry in self.backbone],
            [self._partition_entry(entry, indices) for entry in self.mtp],
            tuple(self.uids[index] for index in indices),
        )

    @classmethod
    def _from_merged(cls, model, backbone, mtp, uids):
        owner = object.__new__(cls)
        owner.model = model
        owner._poisoned_reason = None
        owner._checkpoint = None
        owner._transaction_sealed = False
        owner._size = len(uids)
        owner.uids = tuple(uids)
        owner.backbone = backbone
        owner.mtp = mtp
        owner._binding = owner._make_binding()
        return owner

    def _from_partition(self, indices):
        return self._from_merged(self.model, *self._partition(indices))

    def join(self, other):
        if not isinstance(other, type(self)) or other.model is not self.model:
            raise ValueError("native_mtp_cohort_join_model_mismatch")
        if self._checkpoint is not None or other._checkpoint is not None:
            raise RuntimeError("native_mtp_cohort_join_during_transaction")
        self.bind_before_mutation()
        other.bind_before_mutation()
        if self.size == 0:
            self.backbone, self.mtp, self.uids = other.backbone, other.mtp, other.uids
            self._size = other.size
            self._binding = self._make_binding()
            other.poison("native_mtp_cohort_moved")
            return
        if other.size == 0:
            other.poison("native_mtp_cohort_moved")
            return
        try:
            staged_backbone = [copy.copy(entry) for entry in self.backbone]
            staged_mtp = [copy.copy(entry) for entry in self.mtp]
            for own, incoming in zip(
                staged_backbone + staged_mtp, other.backbone + other.mtp
            ):
                own.extend(incoming)
            staged_uids = self.uids + other.uids
            if len(set(staged_uids)) != len(staged_uids):
                raise ValueError("native_mtp_cohort_join_uid_collision")
        except BaseException:
            self.poison("native_mtp_cohort_join_failed")
            other.poison("native_mtp_cohort_join_failed")
            raise
        self.backbone, self.mtp, self.uids = staged_backbone, staged_mtp, staged_uids
        self._size += other.size
        self._binding = self._make_binding()
        other.poison("native_mtp_cohort_moved")

    def poison(self, reason):
        self._checkpoint = None
        self._transaction_sealed = False
        self._poisoned_reason = reason


class _NativeMTPPhase(str, Enum):
    """Public, linear phases of a native-MTP batch epoch.

    This is intentionally a lifecycle API, not a second batch generator.  The
    ordinary ``BatchGenerator`` owns its existing scheduling contract.  Native
    MTP has a stricter transaction: a caller can advance only through the
    phase handles returned by :class:`NativeMTPBatchGenerator`.
    """

    INITIAL = "initial"
    READY = "ready"
    DECISION = "decision"
    ACCEPTED = "accepted"
    BONUS = "bonus"
    REJECTED = "rejected"
    CLOSED = "closed"


@dataclass(frozen=True)
class NativeMTPRowSpec:
    """Immutable admission contract for one native-MTP row.

    ``seed`` is deliberately row-owned.  Batch split, reorder, and deferred
    ready-join code must preserve this UID-to-seed binding rather than derive
    random state from a cohort index.
    """

    uid: int
    prompt: Tuple[int, ...]
    max_tokens: int
    seed: Optional[int] = None
    eos_token_ids: FrozenSet[int] = frozenset()
    sampling_config: NativeMTPSamplingConfig = NativeMTPSamplingConfig()

    def __post_init__(self):
        if isinstance(self.uid, bool) or not isinstance(self.uid, int):
            raise TypeError("native_mtp_row_uid_invalid")
        if not isinstance(self.prompt, tuple) or not self.prompt:
            raise ValueError("native_mtp_row_prompt_required")
        if any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in self.prompt
        ):
            raise ValueError("native_mtp_row_prompt_token_invalid")
        if (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or self.max_tokens < 0
        ):
            raise ValueError("native_mtp_row_max_tokens_invalid")
        if self.seed is not None and (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("native_mtp_row_seed_invalid")
        if not isinstance(self.eos_token_ids, frozenset) or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in self.eos_token_ids
        ):
            raise ValueError("native_mtp_row_eos_invalid")
        if not isinstance(self.sampling_config, NativeMTPSamplingConfig):
            raise TypeError("native_mtp_row_sampling_config_invalid")


@dataclass(frozen=True)
class NativeMTPEmission:
    """One immutable token result at one public MTP emission boundary."""

    uid: int
    token: int
    logprobs: mx.array
    from_draft: bool
    finish_reason: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.uid, bool) or not isinstance(self.uid, int):
            raise TypeError("native_mtp_emission_uid_invalid")
        if (
            isinstance(self.token, bool)
            or not isinstance(self.token, int)
            or self.token < 0
        ):
            raise ValueError("native_mtp_emission_token_invalid")
        if not isinstance(self.logprobs, mx.array) or self.logprobs.ndim != 1:
            raise TypeError("native_mtp_emission_logprobs_invalid")
        if not isinstance(self.from_draft, bool):
            raise TypeError("native_mtp_emission_draft_flag_invalid")
        if self.finish_reason not in (None, "eos", "length", "cancelled"):
            raise ValueError("native_mtp_emission_finish_reason_invalid")


@dataclass(frozen=True)
class _NativeMTPTelemetryEvent:
    """Append-only public audit event; payload is never mutated in place."""

    phase: _NativeMTPPhase
    uids: Tuple[int, ...]
    event: str


@dataclass(frozen=True)
class NativeMTPAdmission:
    """Validated native-MTP cohort admission.

    Admission is intentionally fail-closed.  Prefix reuse, media, external
    drafters, sparse receipts, opaque processors, rotating caches and
    quantized caches have different rollback/position contracts and must be
    handled by later, separately qualified integrations.
    """

    model: nn.Module
    rows: Tuple[NativeMTPRowSpec, ...]
    cohort: _NativeMTPCohortCache
    # Sparse admission is deliberately represented by the already-claimed
    # canonical records, never by caller-provided duplicate values.  Keeping
    # this private also prevents a later caller from reusing a bootstrap after
    # its receipt authority was consumed.
    _sparse_claims: Optional[Tuple["_NativeMTPSparseClaim", ...]] = None
    _position_context: Optional[GenerationForwardContext] = None

    @classmethod
    def create(
        cls,
        model: nn.Module,
        rows: Sequence[NativeMTPRowSpec],
        request_caches: Sequence[cache.NativeMTPRequestCache],
        *,
        prefix_cache=None,
        media=None,
        external_draft=None,
        sparse_bootstrap=None,
        logits_processors=None,
        kv_bits=None,
        max_kv_size=None,
    ) -> "NativeMTPAdmission":
        rows = tuple(rows)
        request_caches = tuple(request_caches)
        if not rows or len(rows) != len(request_caches):
            raise ValueError("native_mtp_batch_rows_and_caches_required")
        if any(not isinstance(row, NativeMTPRowSpec) for row in rows):
            raise TypeError("native_mtp_batch_row_spec_required")
        if len({row.uid for row in rows}) != len(rows):
            raise ValueError("native_mtp_batch_uid_collision")
        if prefix_cache is not None:
            raise ValueError("native_mtp_batch_prefix_reuse_unsupported")
        if media is not None:
            raise ValueError("native_mtp_batch_media_unsupported")
        if external_draft is not None:
            raise ValueError("native_mtp_batch_external_draft_unsupported")
        if sparse_bootstrap is not None:
            raise ValueError("native_mtp_batch_sparse_bootstrap_unsupported")
        if logits_processors:
            raise ValueError("native_mtp_batch_logits_processors_unsupported")
        if kv_bits is not None:
            raise ValueError("native_mtp_batch_quantized_cache_unsupported")
        if max_kv_size is not None:
            raise ValueError("native_mtp_batch_rotating_cache_unsupported")
        capability = getattr(model, "mtp_capability", None)
        if capability is None or not capability.supported:
            raise RuntimeError(
                "native_mtp_model_capability_missing"
                if capability is None
                else capability.reason
            )
        return cls(
            model,
            rows,
            _NativeMTPCohortCache(
                model, request_caches, uids=tuple(row.uid for row in rows)
            ),
        )

    @classmethod
    def create_from_sparse_bootstraps(
        cls,
        model: nn.Module,
        rows: Sequence[NativeMTPRowSpec],
        bootstraps: Sequence["NativeMTPSparseBootstrap"],
    ) -> "NativeMTPAdmission":
        """Atomically adopt B=1 sparse target receipts into one cohort.

        Every bootstrap is claimed before any MTP cache is made observable.
        Its target cache remains the attested cache; only the retained hidden
        rows and their original immediate successors initialize the fresh MTP
        cache.  Destination cohort construction happens last, after every
        source request has reached the exact ``target=N, mtp=N-1`` state.
        """

        rows = tuple(rows)
        bootstraps = tuple(bootstraps)
        if not rows or len(rows) != len(bootstraps):
            raise ValueError("native_mtp_sparse_batch_rows_and_bootstraps_required")
        if any(not isinstance(row, NativeMTPRowSpec) for row in rows):
            raise TypeError("native_mtp_batch_row_spec_required")
        if any(not isinstance(item, NativeMTPSparseBootstrap) for item in bootstraps):
            raise TypeError("native_mtp_sparse_bootstrap_invalid")
        if len({row.uid for row in rows}) != len(rows):
            raise ValueError("native_mtp_batch_uid_collision")
        # Reject an incoherent caller contract before reserving or consuming a
        # single receipt.  This is admission validation, not a partial batch
        # failure, so the caller may correct the row metadata and retry its
        # still-live sparse evidence.
        for row, bootstrap in zip(rows, bootstraps):
            if row.prompt != bootstrap.selected_token_ids:
                raise ValueError("native_mtp_sparse_row_prompt_mismatch")
        capability = getattr(model, "mtp_capability", None)
        if capability is None or not capability.supported:
            raise RuntimeError(
                "native_mtp_model_capability_missing"
                if capability is None
                else capability.reason
            )

        # Establish the only valid model-owned consumer before consuming
        # receipts.  This fails without touching receipt authority.
        position_context = _native_mtp_position_context(model, None)
        claims = []
        requests = []
        cohort = None
        try:
            for bootstrap in bootstraps:
                claim = bootstrap.claim(model)
                claims.append(claim)
                request = cache.NativeMTPRequestCache.adopt_sparse_target(
                    model,
                    target_cache=claim.target_cache,
                    target_tokens=len(claim.selected_token_ids),
                    next_logical_position=claim.next_logical_position,
                )
                requests.append(request)
                cls._initialize_sparse_mtp_request(
                    model, request, claim, position_context
                )
            cohort = _NativeMTPCohortCache(
                model, requests, uids=tuple(row.uid for row in rows)
            )
            return cls(
                model,
                rows,
                cohort,
                _sparse_claims=tuple(claims),
                _position_context=position_context,
            )
        except BaseException:
            # ``claim`` consumes receipt authority even if validation fails;
            # close every not-yet-claimed bootstrap as well so a failed batch
            # can never be retried as an accidental partial cohort.
            for bootstrap in bootstraps:
                bootstrap.close()
            for request in requests:
                if not request.closed:
                    request.finish("cancelled")
            if cohort is not None:
                cohort.poison("native_mtp_sparse_batch_admission_failed")
            raise

    @staticmethod
    def _initialize_sparse_mtp_request(model, request, claim, position_context):
        """Populate one fresh MTP cache from attested hidden/successor pairs."""

        try:
            for record in claim.records:
                if not record.immediate_successor_token_ids:
                    continue
                positions = record.logical_positions[
                    : len(record.immediate_successor_token_ids)
                ]
                tokens = mx.array(
                    record.immediate_successor_token_ids, dtype=mx.uint32
                )[None]
                ack = GenerationForwardPositionAck(
                    positions,
                    model=model,
                    cache=request.mtp,
                    phase=GenerationForwardPhase.MTP_DRAFT,
                )
                forward = GenerationForward(
                    model=model,
                    input_tokens=tokens,
                    cache=request.mtp,
                    phase=GenerationForwardPhase.MTP_DRAFT,
                    logical_positions=positions,
                    logical_position_ack=ack,
                )
                with position_context(forward):
                    ack._activate()
                    try:
                        model.mtp_forward(
                            record.hidden_rows[:, : len(positions), :],
                            tokens,
                            request.mtp,
                        )
                        ack._require_acknowledged()
                    finally:
                        ack._finish()
            request.seal_verified(
                backbone_tokens=len(claim.selected_token_ids),
                mtp_tokens=len(claim.immediate_successor_token_ids),
            )
            request.commit(
                backbone_tokens=len(claim.selected_token_ids),
                mtp_tokens=len(claim.immediate_successor_token_ids),
            )
        except BaseException:
            if not request.closed:
                request.finish("cancelled")
            raise


class _NativeMTPEpoch:
    """Move-only base for public phase handles."""

    def __init__(self, generator, phase, active_uids):
        self._generator = generator
        self._phase = phase
        self._active_uids = tuple(active_uids)
        self._moved = False

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def active_uids(self) -> Tuple[int, ...]:
        return self._active_uids

    def _consume(self):
        if self._moved or self._generator._epoch is not self:
            raise RuntimeError("native_mtp_epoch_moved")
        self._moved = True

    def cancel(self) -> None:
        self._consume()
        self._generator._close("cancelled")


class NativeMTPInitialEpoch(_NativeMTPEpoch):
    def resume(self) -> "NativeMTPReadyEpoch":
        return self._generator._resume_initial(self)


class NativeMTPReadyEpoch(_NativeMTPEpoch):
    def decide(self) -> "NativeMTPDecisionEpoch":
        return self._generator._decide(self)


class NativeMTPDecisionEpoch(_NativeMTPEpoch):
    def accept(
        self,
    ) -> Tuple[Tuple[NativeMTPEmission, ...], "NativeMTPAcceptedEpoch"]:
        return self._generator._accept(self)

    def reject(
        self,
    ) -> Tuple[Tuple[NativeMTPEmission, ...], "NativeMTPRejectedEpoch"]:
        return self._generator._reject(self)

    def resolve(self):
        return self._generator._resolve_mixed(self)


class NativeMTPAcceptedEpoch(_NativeMTPEpoch):
    def bonus(self) -> "NativeMTPBonusEpoch":
        return self._generator._bonus(self)


class NativeMTPBonusEpoch(_NativeMTPEpoch):
    def catch_up(self) -> "NativeMTPReadyEpoch":
        return self._generator._catch_up(self)


class NativeMTPRejectedEpoch(_NativeMTPEpoch):
    def redraft(self) -> "NativeMTPReadyEpoch":
        return self._generator._redraft(self)


class NativeMTPMixedContinuation:
    """Opaque branch owner after one mixed decision emission boundary."""

    def __init__(self, generator, accepted_uids, rejected_uids):
        self._generator = generator
        self._accepted_uids = accepted_uids
        self._rejected_uids = rejected_uids
        self._moved = False

    def resume_after_resolution(self):
        if self._moved:
            raise RuntimeError("native_mtp_epoch_moved")
        self._moved = True
        try:
            return self._generator._mixed_after_resolution(self)
        except BaseException:
            self._generator._poison_all("native_mtp_batch_mixed_resolution_failed")
            raise

    def cancel(self) -> None:
        if self._moved:
            raise RuntimeError("native_mtp_epoch_moved")
        self._moved = True
        self._generator._poison_all("native_mtp_batch_cancelled")


class NativeMTPMixedBonusContinuation:
    """Opaque accepted-bonus owner awaiting catch-up and Ready join."""

    def __init__(self, generator):
        self._generator = generator
        self._moved = False

    def resume_after_bonus(self):
        if self._moved:
            raise RuntimeError("native_mtp_epoch_moved")
        self._moved = True
        try:
            return self._generator._mixed_after_bonus(self)
        except BaseException:
            self._generator._poison_all("native_mtp_batch_mixed_bonus_failed")
            raise

    def cancel(self) -> None:
        if self._moved:
            raise RuntimeError("native_mtp_epoch_moved")
        self._moved = True
        self._generator._poison_all("native_mtp_batch_cancelled")


class NativeMTPBatchGenerator:
    """Typed public lifecycle for a committed native-MTP cohort cache.

    The numeric model calls remain private implementation detail.  This class
    owns the externally observable epoch ordering, one-token emission rule,
    terminal filtering, and immutable telemetry needed to make those calls
    safe to batch.  It deliberately does not wire into ``BatchGenerator``.
    """

    def __init__(self, admission: NativeMTPAdmission):
        if not isinstance(admission, NativeMTPAdmission):
            raise TypeError("native_mtp_batch_admission_required")
        self._admission = admission
        self._cohort = admission.cohort
        self._rows = {row.uid: row for row in admission.rows}
        self._generated = {row.uid: 0 for row in admission.rows}
        self._active = tuple(row.uid for row in admission.rows)
        self._epoch = None
        self._closed = False
        self._events = ()
        self._sampling = {
            row.uid: _NativeMTPSamplingState(
                row.sampling_config,
                seed=row.seed if row.seed is not None else row.sampling_config.seed,
            )
            for row in admission.rows
        }
        self._history = {
            row.uid: mx.array(row.prompt, dtype=mx.uint32) for row in admission.rows
        }
        self._head = {}
        self._hidden = {}
        self._draft = {}
        self._draft_logprobs = {}
        self._rng_key = {
            row.uid: mx.random.key(
                row.seed if row.seed is not None else (row.sampling_config.seed or 0)
            )
            for row in admission.rows
        }
        self._last_failed_owners = ()
        self._sparse_claims = admission._sparse_claims
        self._position_context = admission._position_context
        self._logical_cursor = (
            {
                row.uid: claim.next_logical_position
                for row, claim in zip(admission.rows, self._sparse_claims)
            }
            if self._sparse_claims is not None
            else None
        )
        self._initial_mtp_positions = (
            {
                row.uid: claim.selected_logical_positions[-1]
                for row, claim in zip(admission.rows, self._sparse_claims)
            }
            if self._sparse_claims is not None
            else None
        )
        if self._sparse_claims is not None:
            # Canonical B=1 sparse generation deliberately begins processor
            # history at the final retained target anchor, not the caller's
            # original selected-token sequence.  The latter would alter
            # stateful/replay-safe processors and make a batched sparse row
            # diverge before its first emitted head.
            self._history = {
                row.uid: mx.array([claim.selected_token_ids[-1]], dtype=mx.uint32)
                for row, claim in zip(admission.rows, self._sparse_claims)
            }

    def start_sparse(self):
        """Start a sparse admission from its retained final target evidence."""

        if self._sparse_claims is None:
            raise RuntimeError("native_mtp_sparse_batch_admission_required")
        if self._epoch is not None or self._closed:
            raise RuntimeError("native_mtp_batch_initial_already_started")
        return self._sparse_initial()

    @property
    def telemetry(self) -> tuple:
        return self._events

    @property
    def closed(self) -> bool:
        return self._closed

    def _initial(self, emissions: Iterable[NativeMTPEmission]) -> NativeMTPInitialEpoch:
        active = self._record_emissions(
            _NativeMTPPhase.INITIAL, emissions, expected=self._active
        )
        return self._replace(
            NativeMTPInitialEpoch(self, _NativeMTPPhase.INITIAL, active),
            "initial_emission",
        )

    def prefill(self, *, prefill_step_size: int = 512):
        """Run isolated merged, variable-length shifted prefill and emit heads.

        Rows are temporarily partitioned only when their remaining prompt work
        differs.  Every target/MTP call still receives a merged cache owner;
        the resulting temporary owner is joined back without host row-cache
        extraction.  This is the required safe path for unequal prompt sizes.
        """

        if self._epoch is not None or self._closed:
            raise RuntimeError("native_mtp_batch_initial_already_started")
        if self._sparse_claims is not None:
            raise RuntimeError("native_mtp_sparse_batch_requires_start_sparse")
        if (
            isinstance(prefill_step_size, bool)
            or not isinstance(prefill_step_size, int)
            or prefill_step_size < 1
        ):
            raise ValueError("native_mtp_batch_prefill_step_size_invalid")
        positions = {uid: 0 for uid in self._active}
        while True:
            pending = tuple(
                uid
                for uid in self._active
                if positions[uid] < len(self._rows[uid].prompt) - 1
            )
            if not pending:
                break
            # A chunk cohort contains only equal remaining work.  Unequal rows
            # are partitioned on-device, processed as a real B×N call, then
            # rejoined before the next size class.
            by_count = {}
            for uid in pending:
                count = min(
                    prefill_step_size, len(self._rows[uid].prompt) - 1 - positions[uid]
                )
                by_count.setdefault(count, []).append(uid)
            for count, uids in by_count.items():
                uids = tuple(uids)
                remaining = self._move_to_front(uids)
                selected = self._cohort
                try:
                    tokens = tuple(
                        self._rows[uid].prompt[positions[uid] : positions[uid] + count]
                        for uid in uids
                    )
                    successors = tuple(
                        self._rows[uid].prompt[
                            positions[uid] + 1 : positions[uid] + count + 1
                        ]
                        for uid in uids
                    )
                    _, hidden = self._target_forward(
                        tokens, phase=GenerationForwardPhase.PREFILL
                    )
                    self._mtp_forward(hidden, successors)
                    for uid in uids:
                        positions[uid] += count
                except BaseException:
                    self._poison_all(
                        "native_mtp_batch_prefill_failed", selected, remaining
                    )
                    raise
                else:
                    try:
                        self._cohort = selected
                        self._restore_from_front(remaining)
                    except BaseException:
                        self._poison_all(
                            "native_mtp_batch_prefill_join_failed", selected, remaining
                        )
                        raise
        final_uids = tuple(self._active)
        remaining = self._move_to_front(final_uids)
        selected = self._cohort
        try:
            tokens = tuple(self._rows[uid].prompt[-1] for uid in final_uids)
            logits, hidden = self._target_forward(
                tokens, phase=GenerationForwardPhase.PREFILL
            )
            emissions = []
            for index, uid in enumerate(final_uids):
                state = self._sampling[uid]
                reported = state.reported_logprobs(
                    logits[index, -1, :], self._history[uid]
                )
                sampled = state.sample(
                    state.sampling_distribution(reported),
                    rng_key=(
                        self._next_rng_key(uid) if state.config.stochastic else None
                    ),
                ).reshape(-1)
                mx.eval(sampled, reported)
                token = sampled.item()
                self._head[uid] = sampled
                self._hidden[uid] = hidden[index : index + 1, -1:, :]
                emissions.append(self._make_emission(uid, token, reported, False))
        except BaseException:
            self._poison_all("native_mtp_batch_prefill_failed", selected, remaining)
            raise
        else:
            try:
                self._cohort = selected
                self._restore_from_front(remaining)
            except BaseException:
                self._poison_all(
                    "native_mtp_batch_prefill_join_failed", selected, remaining
                )
                raise
        return tuple(emissions), self._initial(emissions)

    def _sparse_initial(self):
        """Turn retained final sparse logits into the Initial epoch only."""

        emissions = []
        try:
            for row, claim in zip(self._admission.rows, self._sparse_claims):
                uid = row.uid
                state = self._sampling[uid]
                reported = state.reported_logprobs(
                    claim.final_target_logits[0, -1, :], self._history[uid]
                )
                sampled = state.sample(
                    state.sampling_distribution(reported),
                    rng_key=(
                        self._next_rng_key(uid) if state.config.stochastic else None
                    ),
                ).reshape(-1)
                mx.eval(sampled, reported)
                self._head[uid] = sampled
                self._hidden[uid] = claim.final_target_hidden
                emissions.append(
                    self._make_emission(uid, sampled.item(), reported, False)
                )
        except BaseException:
            self._poison_all("native_mtp_sparse_initial_failed")
            raise
        return tuple(emissions), self._initial(emissions)

    def _move_to_front(self, uids):
        """Return a selected move-only owner while keeping the selected UIDs first."""

        uids = tuple(uids)
        if not uids:
            return None
        indices = tuple(self._cohort.uids.index(uid) for uid in uids)
        remaining = self._cohort
        self._cohort = remaining.split(indices)
        return remaining

    def _restore_from_front(self, remaining):
        if remaining is None:
            return
        self._cohort.join(remaining)

    def _poison_all(self, reason, *extra_owners):
        """Close every owned cohort after an irreversible lifecycle failure."""

        owners = {id(self._cohort): self._cohort}
        for name in ("_mixed_accepted_owner", "_mixed_rejected_owner"):
            owner = getattr(self, name, None)
            if owner is not None:
                owners[id(owner)] = owner
        for owner in extra_owners:
            if owner is not None:
                owners[id(owner)] = owner
        for owner in owners.values():
            owner.poison(reason)
        self._last_failed_owners = tuple(owners.values())
        self._closed = True
        self._active = ()
        self._epoch = None

    @staticmethod
    def _delta(entries, advance):
        return tuple(
            (
                (advance,) * entry.batch_size
                if isinstance(entry, ArraysCache)
                else (advance,) * entry.offset.size
            )
            for entry in entries
        )

    def _seal_delta(self, *, backbone, mtp):
        self._cohort.seal_after_mutation(
            _NativeMTPCohortMutationDelta(
                backbone=self._delta(self._cohort.backbone, backbone),
                mtp=self._delta(self._cohort.mtp, mtp),
            )
        )

    def _target_forward(self, tokens, *, phase, logical_positions=None):
        try:
            self._cohort.bind_before_mutation()
            inputs = mx.array(tokens, dtype=mx.uint32)
            if inputs.ndim == 1:
                inputs = inputs[:, None]
            result = self._positioned_target_call(inputs, phase, logical_positions)
            if not isinstance(result, tuple) or len(result) != 2:
                raise RuntimeError("native_mtp_batch_target_hidden_missing")
            self._seal_delta(backbone=inputs.shape[1], mtp=0)
            return result
        except BaseException:
            self._poison_all("native_mtp_batch_target_failed")
            raise

    def _mtp_forward(self, hidden, tokens, *, logical_positions=None):
        try:
            self._cohort.bind_before_mutation()
            inputs = mx.array(tokens, dtype=mx.uint32)
            if inputs.ndim == 1:
                inputs = inputs[:, None]
            result = self._positioned_mtp_call(hidden, inputs, logical_positions)
            self._seal_delta(backbone=0, mtp=inputs.shape[1])
            return result
        except BaseException:
            self._poison_all("native_mtp_batch_mtp_failed")
            raise

    def _positioned_target_call(self, inputs, phase, logical_positions):
        if logical_positions is None:
            return self._admission.model(
                inputs, cache=self._cohort.backbone, return_hidden=True
            )
        if self._position_context is None:
            raise RuntimeError("native_mtp_sparse_position_context_missing")
        ack = GenerationForwardPositionAck(
            logical_positions,
            model=self._admission.model,
            cache=self._cohort.backbone,
            phase=phase,
        )
        forward = GenerationForward(
            model=self._admission.model,
            input_tokens=inputs,
            cache=self._cohort.backbone,
            phase=phase,
            logical_positions=logical_positions,
            logical_position_ack=ack,
        )
        with self._position_context(forward):
            ack._activate()
            try:
                result = self._admission.model(
                    inputs, cache=self._cohort.backbone, return_hidden=True
                )
                ack._require_acknowledged()
                return result
            finally:
                ack._finish()

    def _positioned_mtp_call(self, hidden, inputs, logical_positions):
        if logical_positions is None:
            return self._admission.model.mtp_forward(hidden, inputs, self._cohort.mtp)
        if self._position_context is None:
            raise RuntimeError("native_mtp_sparse_position_context_missing")
        ack = GenerationForwardPositionAck(
            logical_positions,
            model=self._admission.model,
            cache=self._cohort.mtp,
            phase=GenerationForwardPhase.MTP_DRAFT,
        )
        forward = GenerationForward(
            model=self._admission.model,
            input_tokens=inputs,
            cache=self._cohort.mtp,
            phase=GenerationForwardPhase.MTP_DRAFT,
            logical_positions=logical_positions,
            logical_position_ack=ack,
        )
        with self._position_context(forward):
            ack._activate()
            try:
                result = self._admission.model.mtp_forward(
                    hidden, inputs, self._cohort.mtp
                )
                ack._require_acknowledged()
                return result
            finally:
                ack._finish()

    @staticmethod
    def _position_argument(rows):
        """Keep B=1 on the canonical vector contract, B>1 on matrices."""

        return rows[0] if len(rows) == 1 else tuple(rows)

    def _position_rows(self, uids, width, *, start_delta):
        if self._logical_cursor is None:
            return None
        return tuple(
            tuple(
                self._logical_cursor[uid] + start_delta + index
                for index in range(width)
            )
            for uid in uids
        )

    def _positions(self, uids, width, *, start_delta):
        """Return exact B=1 vectors or explicit B>1 position matrices."""

        rows = self._position_rows(uids, width, start_delta=start_delta)
        return None if rows is None else self._position_argument(rows)

    def _make_emission(self, uid, token, logprobs, from_draft):
        row = self._rows[uid]
        total = self._generated[uid] + 1
        reason = (
            "eos"
            if token in row.eos_token_ids
            else ("length" if total >= row.max_tokens else None)
        )
        return NativeMTPEmission(uid, token, logprobs, from_draft, reason)

    def _next_rng_key(self, uid):
        """Consume one B=1-compatible stochastic draw for one stable UID."""

        retained, draw = mx.random.split(self._rng_key[uid])
        self._rng_key[uid] = retained
        return draw

    def _replace(self, epoch, event):
        self._epoch = epoch
        self._events = self._events + (
            _NativeMTPTelemetryEvent(epoch.phase, epoch.active_uids, event),
        )
        return epoch

    def _record_emissions(self, phase, emissions, *, expected):
        emissions = tuple(emissions)
        if len(emissions) != len(expected):
            raise ValueError("native_mtp_batch_emission_boundary_incomplete")
        if any(not isinstance(item, NativeMTPEmission) for item in emissions):
            raise TypeError("native_mtp_batch_emission_required")
        if {item.uid for item in emissions} != set(expected):
            raise ValueError("native_mtp_batch_emission_uid_mismatch")
        next_active = []
        for emission in emissions:
            row = self._rows[emission.uid]
            self._generated[emission.uid] += 1
            if self._logical_cursor is not None:
                self._logical_cursor[emission.uid] += 1
            terminal = (
                emission.finish_reason is not None
                or self._generated[emission.uid] >= row.max_tokens
            )
            if not terminal:
                next_active.append(emission.uid)
        self._events = self._events + (
            _NativeMTPTelemetryEvent(phase, tuple(expected), "emission"),
        )
        self._active = tuple(next_active)
        return self._active

    @staticmethod
    def _filter_owner_uids(owner, uids):
        if owner is None:
            return None
        if not uids:
            owner.poison("native_mtp_batch_terminal_rows_removed")
            return None
        indices = tuple(owner.uids.index(uid) for uid in uids)
        owner.filter(indices)
        return owner

    def _prune_mixed_terminal_owners(self, emissions, accepted_uids, rejected_uids):
        """Drop terminal branch rows before a later branch forward or join."""

        emission_by_uid = {emission.uid: emission for emission in emissions}
        self._mixed_accepted_uids = tuple(
            uid for uid in accepted_uids if emission_by_uid[uid].finish_reason is None
        )
        self._mixed_rejected_uids = tuple(
            uid for uid in rejected_uids if emission_by_uid[uid].finish_reason is None
        )
        self._mixed_accepted_owner = self._filter_owner_uids(
            self._mixed_accepted_owner, self._mixed_accepted_uids
        )
        self._mixed_rejected_owner = self._filter_owner_uids(
            self._mixed_rejected_owner, self._mixed_rejected_uids
        )

    def _resume_initial(self, epoch):
        epoch._consume()
        drafts = tuple(epoch.active_uids)
        if drafts:
            logits = self._mtp_forward(
                mx.concatenate([self._hidden[uid] for uid in drafts], axis=0),
                tuple(self._head[uid].item() for uid in drafts),
                logical_positions=(
                    self._position_argument(
                        tuple((self._initial_mtp_positions[uid],) for uid in drafts)
                    )
                    if self._initial_mtp_positions is not None
                    else None
                ),
            )[:, -1, :]
            for index, uid in enumerate(drafts):
                state = self._sampling[uid]
                self._history[uid] = mx.concatenate(
                    [self._history[uid], self._head[uid]]
                )
                reported = state.reported_logprobs(logits[index], self._history[uid])
                draft_logprobs = state.sampling_distribution(reported)
                draft = state.sample(
                    draft_logprobs,
                    rng_key=(
                        self._next_rng_key(uid) if state.config.stochastic else None
                    ),
                ).reshape(-1)
                mx.eval(draft, reported)
                self._draft[uid] = draft
                self._draft_logprobs[uid] = draft_logprobs
        return self._replace(
            NativeMTPReadyEpoch(self, _NativeMTPPhase.READY, drafts),
            "initial_head_plus_one_draft",
        )

    def _decide(self, epoch):
        epoch._consume()
        try:
            self._cohort.checkpoint()
            verify_rows = self._position_rows(epoch.active_uids, 2, start_delta=-1)
            self._verify_positions = (
                self._position_argument(verify_rows)
                if verify_rows is not None
                else None
            )
            self._verify_position_by_uid = (
                dict(zip(epoch.active_uids, verify_rows))
                if verify_rows is not None
                else None
            )
            inputs = mx.stack(
                [
                    mx.concatenate([self._head[uid], self._draft[uid]])
                    for uid in epoch.active_uids
                ],
                axis=0,
            )
            verify_logits, verify_hidden = self._target_forward(
                inputs,
                phase=GenerationForwardPhase.VERIFY,
                logical_positions=self._verify_positions,
            )
            (
                self._verify_hidden,
                self._verify_logits,
                self._verify_logprobs,
                self._replacement,
            ) = ({}, {}, {}, {})
            accepted_values = []
            for index, uid in enumerate(epoch.active_uids):
                state = self._sampling[uid]
                reported = state.reported_logprobs(
                    verify_logits[index, 0, :], self._history[uid]
                )
                sampled = state.sampling_distribution(reported)
                if state.config.stochastic:
                    draft_id = self._draft[uid].item()
                    probability = mx.minimum(
                        mx.exp(sampled[draft_id] - self._draft_logprobs[uid][draft_id]),
                        1.0,
                    )
                    accepted_value = (
                        mx.random.uniform(key=self._next_rng_key(uid)) < probability
                    )
                    mx.eval(accepted_value)
                    is_accepted = bool(accepted_value.item())
                    replacement = self._draft[uid]
                    if not is_accepted:
                        replacement, _ = state.residual_sample(
                            sampled,
                            self._draft_logprobs[uid],
                            rng_key=self._next_rng_key(uid),
                        )
                        replacement = replacement.reshape(-1)
                else:
                    replacement = state.sample(sampled).reshape(-1)
                    is_accepted = replacement.item() == self._draft[uid].item()
                mx.eval(replacement, reported)
                self._verify_hidden[uid] = verify_hidden[index : index + 1]
                self._verify_logits[uid] = verify_logits[index : index + 1, 1:2, :]
                self._verify_logprobs[uid] = reported
                self._replacement[uid] = replacement
                if is_accepted:
                    accepted_values.append(uid)
            accepted = tuple(accepted_values)
        except BaseException:
            self._poison_all("native_mtp_batch_decision_failed")
            raise
        rejected = tuple(uid for uid in epoch.active_uids if uid not in set(accepted))
        result = NativeMTPDecisionEpoch(
            self, _NativeMTPPhase.DECISION, epoch.active_uids
        )
        result.accepted_uids = accepted
        result.rejected_uids = rejected
        return self._replace(result, "target_plus_two_checkpoint_decision")

    def _accept(self, epoch):
        epoch._consume()
        try:
            if epoch.rejected_uids:
                raise RuntimeError(
                    "native_mtp_batch_mixed_decision_requires_branch_resolution"
                )
            self._cohort.commit()
            emissions = tuple(
                self._make_emission(
                    uid, self._draft[uid].item(), self._verify_logprobs[uid], True
                )
                for uid in epoch.accepted_uids
            )
        except BaseException:
            self._poison_all("native_mtp_batch_accept_failed")
            raise
        active = self._record_emissions(
            _NativeMTPPhase.ACCEPTED, emissions, expected=epoch.accepted_uids
        )
        result = NativeMTPAcceptedEpoch(self, _NativeMTPPhase.ACCEPTED, active)
        result._deferred_rejected = epoch.rejected_uids
        return emissions, self._replace(result, "accepted_draft_emission")

    def _resolve_mixed(self, epoch):
        """Split only after rollback; branch owners never share recurrent state."""

        epoch._consume()
        accepted_owner = rejected_owner = None
        original_owner = self._cohort
        try:
            self._cohort.rollback()
            accepted_indices = tuple(
                self._cohort.uids.index(uid) for uid in epoch.accepted_uids
            )
            if accepted_indices:
                accepted_owner = self._cohort.split(accepted_indices)
                rejected_owner = self._cohort
            else:
                accepted_owner, rejected_owner = None, self._cohort
            self._mixed_accepted_owner = accepted_owner
            self._mixed_rejected_owner = rejected_owner
            self._mixed_accepted_uids = epoch.accepted_uids
            self._mixed_rejected_uids = epoch.rejected_uids
            emissions = []
            # Rerun the verified target call on the accepted owner, then commit its
            # all-or-nothing recurrent state before its draft token is observable.
            if accepted_owner is not None:
                self._cohort = accepted_owner
                self._cohort.checkpoint()
                inputs = mx.stack(
                    [
                        mx.concatenate([self._head[uid], self._draft[uid]])
                        for uid in epoch.accepted_uids
                    ],
                    axis=0,
                )
                accepted_positions = (
                    self._position_argument(
                        tuple(
                            self._verify_position_by_uid[uid]
                            for uid in epoch.accepted_uids
                        )
                    )
                    if self._verify_position_by_uid is not None
                    else None
                )
                verify_logits, verify_hidden = self._target_forward(
                    inputs,
                    phase=GenerationForwardPhase.VERIFY,
                    logical_positions=accepted_positions,
                )
                self._cohort.commit()
                for index, uid in enumerate(epoch.accepted_uids):
                    self._verify_hidden[uid] = verify_hidden[index : index + 1]
                    self._verify_logits[uid] = verify_logits[index : index + 1, 1:2, :]
                    emissions.append(
                        self._make_emission(
                            uid,
                            self._draft[uid].item(),
                            self._verify_logprobs[uid],
                            True,
                        )
                    )
            for uid in epoch.rejected_uids:
                emissions.append(
                    self._make_emission(
                        uid,
                        self._replacement[uid].item(),
                        self._verify_logprobs[uid],
                        False,
                    )
                )
            # This is the required exactly-one-token/UID resolution boundary.
            self._record_emissions(
                _NativeMTPPhase.DECISION, emissions, expected=epoch.active_uids
            )
            self._prune_mixed_terminal_owners(
                emissions, epoch.accepted_uids, epoch.rejected_uids
            )
            self._events = self._events + (
                _NativeMTPTelemetryEvent(
                    _NativeMTPPhase.DECISION,
                    epoch.active_uids,
                    "mixed_resolution_emission",
                ),
            )
            return tuple(emissions), NativeMTPMixedContinuation(
                self, self._mixed_accepted_uids, self._mixed_rejected_uids
            )
        except BaseException:
            self._poison_all(
                "native_mtp_batch_mixed_resolve_failed",
                original_owner,
                accepted_owner,
                rejected_owner,
            )
            raise

    def _mixed_after_resolution(self, continuation):
        """Replay rejected head, redraft it, and emit accepted bonus only."""

        rejected_ready = ()
        if self._mixed_rejected_uids:
            self._cohort = self._mixed_rejected_owner
            self._replay_position_by_uid = (
                {
                    uid: self._logical_cursor[uid] - 2
                    for uid in self._mixed_rejected_uids
                }
                if self._logical_cursor is not None
                else None
            )
            _, replay_hidden = self._target_forward(
                tuple(self._head[uid].item() for uid in self._mixed_rejected_uids),
                phase=GenerationForwardPhase.DECODE,
                logical_positions=(
                    self._position_argument(
                        tuple(
                            (self._replay_position_by_uid[uid],)
                            for uid in self._mixed_rejected_uids
                        )
                    )
                    if self._replay_position_by_uid is not None
                    else None
                ),
            )
            self._replay_hidden = {}
            for index, uid in enumerate(self._mixed_rejected_uids):
                self._replay_hidden[uid] = replay_hidden[index : index + 1, -1:, :]
            logits = self._mtp_forward(
                mx.concatenate(
                    [self._replay_hidden[uid] for uid in self._mixed_rejected_uids],
                    axis=0,
                ),
                tuple(
                    self._replacement[uid].item() for uid in self._mixed_rejected_uids
                ),
                logical_positions=(
                    self._position_argument(
                        tuple(
                            (self._replay_position_by_uid[uid],)
                            for uid in self._mixed_rejected_uids
                        )
                    )
                    if self._replay_position_by_uid is not None
                    else None
                ),
            )[:, -1, :]
            rejected_ready = self._mixed_rejected_uids
            for index, uid in enumerate(rejected_ready):
                state = self._sampling[uid]
                self._head[uid] = self._replacement[uid]
                self._history[uid] = mx.concatenate(
                    [self._history[uid], self._head[uid]]
                )
                reported = state.reported_logprobs(logits[index], self._history[uid])
                draft_logprobs = state.sampling_distribution(reported)
                self._draft[uid] = state.sample(
                    draft_logprobs,
                    rng_key=(
                        self._next_rng_key(uid) if state.config.stochastic else None
                    ),
                ).reshape(-1)
                self._draft_logprobs[uid] = draft_logprobs
        emissions = []
        if self._mixed_accepted_owner is not None:
            self._cohort = self._mixed_accepted_owner
            for uid in self._mixed_accepted_uids:
                state = self._sampling[uid]
                history = mx.concatenate([self._history[uid], self._draft[uid]])
                reported = state.reported_logprobs(
                    self._verify_logits[uid].squeeze((0, 1)), history
                )
                token = state.sample(
                    state.sampling_distribution(reported),
                    rng_key=(
                        self._next_rng_key(uid) if state.config.stochastic else None
                    ),
                ).reshape(-1)
                mx.eval(token, reported)
                self._history[uid] = mx.concatenate([history, token])
                self._head[uid] = token
                emission = self._make_emission(uid, token.item(), reported, False)
                emissions.append(emission)
        # This is a real public emission boundary.  Record it exactly once
        # before terminal pruning so generated counts, finish semantics and
        # sparse logical cursors remain B=1-equivalent.
        accepted_bonus = self._record_emissions(
            _NativeMTPPhase.BONUS,
            emissions,
            expected=self._mixed_accepted_uids,
        )
        # Terminal accepted rows must not be caught up or rejoined with the
        # rejected Ready owner.
        self._mixed_accepted_uids = accepted_bonus
        self._mixed_accepted_owner = self._filter_owner_uids(
            self._mixed_accepted_owner, accepted_bonus
        )
        self._mixed_rejected_ready = rejected_ready
        self._mixed_accepted_ready = accepted_bonus
        return tuple(emissions), NativeMTPMixedBonusContinuation(self)

    def _mixed_after_bonus(self, continuation):
        """Catch up accepted MTP state, then join compatible Ready owners."""

        if self._mixed_accepted_owner is not None and self._mixed_accepted_ready:
            self._cohort = self._mixed_accepted_owner
            uids = self._mixed_accepted_ready
            hidden = mx.concatenate([self._verify_hidden[uid] for uid in uids], axis=0)
            tokens = mx.array(
                [(self._draft[uid].item(), self._head[uid].item()) for uid in uids],
                dtype=mx.uint32,
            )
            self._cohort.bind_before_mutation()
            accepted_positions = (
                self._position_argument(
                    tuple(self._verify_position_by_uid[uid] for uid in uids)
                )
                if self._verify_position_by_uid is not None
                else None
            )
            logits = self._positioned_mtp_call(hidden, tokens, accepted_positions)[
                :, -1, :
            ]
            self._seal_delta(backbone=0, mtp=2)
            for index, uid in enumerate(uids):
                state = self._sampling[uid]
                reported = state.reported_logprobs(logits[index], self._history[uid])
                draft_logprobs = state.sampling_distribution(reported)
                self._draft[uid] = state.sample(
                    draft_logprobs,
                    rng_key=(
                        self._next_rng_key(uid) if state.config.stochastic else None
                    ),
                ).reshape(-1)
                self._draft_logprobs[uid] = draft_logprobs
        owners = [
            owner
            for owner in (self._mixed_accepted_owner, self._mixed_rejected_owner)
            if owner is not None
        ]
        if not owners:
            self._close("cancelled")
            raise RuntimeError("native_mtp_batch_mixed_owner_missing")
        ready_owner = owners[0]
        for owner in owners[1:]:
            ready_owner.join(owner)
        self._cohort = ready_owner
        ready_uids = tuple(self._mixed_accepted_ready + self._mixed_rejected_ready)
        if tuple(ready_owner.uids) != ready_uids:
            self._poison_all("native_mtp_batch_mixed_ready_uid_mismatch", *owners)
            raise RuntimeError("native_mtp_batch_mixed_ready_uid_mismatch")
        self._active = ready_uids
        return self._replace(
            NativeMTPReadyEpoch(self, _NativeMTPPhase.READY, ready_uids),
            "mixed_ready_join",
        )

    def _bonus(self, epoch):
        epoch._consume()
        emissions = []
        for uid in epoch.active_uids:
            state = self._sampling[uid]
            history = mx.concatenate([self._history[uid], self._draft[uid]])
            reported = state.reported_logprobs(
                self._verify_logits[uid].squeeze((0, 1)), history
            )
            token = state.sample(
                state.sampling_distribution(reported),
                rng_key=self._next_rng_key(uid) if state.config.stochastic else None,
            ).reshape(-1)
            mx.eval(token, reported)
            self._history[uid] = mx.concatenate([history, token])
            self._head[uid] = token
            emissions.append(self._make_emission(uid, token.item(), reported, False))
        active = self._record_emissions(
            _NativeMTPPhase.BONUS, emissions, expected=epoch.active_uids
        )
        result = NativeMTPBonusEpoch(self, _NativeMTPPhase.BONUS, active)
        result._deferred_rejected = epoch._deferred_rejected
        return self._replace(result, "bonus_emission")

    def _catch_up(self, epoch):
        epoch._consume()
        drafts = tuple(epoch.active_uids)
        if drafts:
            hidden = mx.concatenate(
                [self._verify_hidden[uid] for uid in drafts], axis=0
            )
            tokens = mx.array(
                [(self._draft[uid].item(), self._head[uid].item()) for uid in drafts],
                dtype=mx.uint32,
            )
            self._cohort.bind_before_mutation()
            logits = self._positioned_mtp_call(
                hidden,
                tokens,
                (
                    self._position_argument(
                        tuple(self._verify_position_by_uid[uid] for uid in drafts)
                    )
                    if getattr(self, "_verify_position_by_uid", None) is not None
                    else None
                ),
            )[:, -1, :]
            self._seal_delta(backbone=0, mtp=2)
            for index, uid in enumerate(drafts):
                state = self._sampling[uid]
                reported = state.reported_logprobs(logits[index], self._history[uid])
                draft_logprobs = state.sampling_distribution(reported)
                draft = state.sample(
                    draft_logprobs,
                    rng_key=(
                        self._next_rng_key(uid) if state.config.stochastic else None
                    ),
                ).reshape(-1)
                mx.eval(draft, reported)
                self._draft[uid] = draft
                self._draft_logprobs[uid] = draft_logprobs
        return self._replace(
            NativeMTPReadyEpoch(self, _NativeMTPPhase.READY, drafts),
            "head_catch_up_plus_two",
        )

    def _reject(self, epoch):
        epoch._consume()
        try:
            if epoch.accepted_uids:
                raise RuntimeError(
                    "native_mtp_batch_mixed_decision_requires_branch_resolution"
                )
            self._cohort.rollback()
            self._replay_position_by_uid = (
                {uid: self._logical_cursor[uid] - 1 for uid in epoch.rejected_uids}
                if self._logical_cursor is not None
                else None
            )
            _, replay_hidden = self._target_forward(
                tuple(self._head[uid].item() for uid in epoch.rejected_uids),
                phase=GenerationForwardPhase.DECODE,
                logical_positions=(
                    self._position_argument(
                        tuple(
                            (self._replay_position_by_uid[uid],)
                            for uid in epoch.rejected_uids
                        )
                    )
                    if self._replay_position_by_uid is not None
                    else None
                ),
            )
            self._replay_hidden = {}
            emissions = []
            for index, uid in enumerate(epoch.rejected_uids):
                self._replay_hidden[uid] = replay_hidden[index : index + 1, -1:, :]
                emissions.append(
                    self._make_emission(
                        uid,
                        self._replacement[uid].item(),
                        self._verify_logprobs[uid],
                        False,
                    )
                )
        except BaseException:
            self._poison_all("native_mtp_batch_reject_failed")
            raise
        active = self._record_emissions(
            _NativeMTPPhase.REJECTED, emissions, expected=epoch.rejected_uids
        )
        return emissions, self._replace(
            NativeMTPRejectedEpoch(self, _NativeMTPPhase.REJECTED, active),
            "rollback_replay_replacement_emission",
        )

    def _redraft(self, epoch):
        epoch._consume()
        drafts = tuple(epoch.active_uids)
        if drafts:
            logits = self._mtp_forward(
                mx.concatenate([self._replay_hidden[uid] for uid in drafts], axis=0),
                tuple(self._replacement[uid].item() for uid in drafts),
                logical_positions=(
                    self._position_argument(
                        tuple((self._replay_position_by_uid[uid],) for uid in drafts)
                    )
                    if getattr(self, "_replay_position_by_uid", None) is not None
                    else None
                ),
            )[:, -1, :]
            for index, uid in enumerate(drafts):
                state = self._sampling[uid]
                self._head[uid] = self._replacement[uid]
                self._history[uid] = mx.concatenate(
                    [self._history[uid], self._head[uid]]
                )
                reported = state.reported_logprobs(logits[index], self._history[uid])
                draft_logprobs = state.sampling_distribution(reported)
                draft = state.sample(
                    draft_logprobs,
                    rng_key=(
                        self._next_rng_key(uid) if state.config.stochastic else None
                    ),
                ).reshape(-1)
                mx.eval(draft, reported)
                self._draft[uid] = draft
                self._draft_logprobs[uid] = draft_logprobs
        return self._replace(
            NativeMTPReadyEpoch(self, _NativeMTPPhase.READY, drafts),
            "head_redraft_plus_one",
        )

    def _close(self, reason):
        if self._closed:
            return
        self._cohort.poison("native_mtp_batch_" + reason)
        self._closed = True
        self._epoch = None
        self._active = ()
        self._events = self._events + (
            _NativeMTPTelemetryEvent(_NativeMTPPhase.CLOSED, (), reason),
        )


def _validate_receipt_against_record(receipt, record):
    if (
        not isinstance(receipt, GenerationForwardPositionReceipt)
        or getattr(receipt, "_issuer_seal", None)
        is not _GENERATION_FORWARD_RECEIPT_ISSUER
        or id(receipt) != record.receipt_id
        or receipt.model_id != record.model_id
        or receipt.cache_container_id != record.cache_container_id
        or receipt.cache_entry_ids != record.cache_entry_ids
        or receipt.capability != record.capability
        or receipt.phase is not record.phase
        or receipt.logical_positions != record.logical_positions
        or receipt.token_ids != record.token_ids
        or receipt.immediate_successor_token_ids != record.immediate_successor_token_ids
        or receipt.logits is not record.logits
        or receipt.hidden_rows is not record.hidden_rows
        or _array_identity_evidence(receipt.logits) != record.logits_evidence
        or _array_identity_evidence(receipt.hidden_rows) != record.hidden_evidence
    ):
        raise RuntimeError("native_mtp_sparse_receipt_canonical_mismatch")


def _sparse_attestation_host_decision(decision):
    mx.eval(decision)
    return bool(decision.item())


def _verify_sparse_canonical_content(records):
    """Realize all mutable evidence checks with one aggregate host decision."""

    checks = [
        mx.array_equal(
            _array_content_digest(record.hidden_rows), record.hidden_content_digest
        )
        for record in records
    ]
    final = records[-1]
    checks.extend(
        (
            mx.array_equal(
                _array_content_digest(final.logits), final.logits_content_digest
            ),
            mx.array_equal(
                _cache_content_digest(final.cache), final.cache_content_digest
            ),
        )
    )
    decision = mx.all(mx.stack(checks))
    if not _sparse_attestation_host_decision(decision):
        raise RuntimeError("native_mtp_sparse_evidence_content_mismatch")


@dataclass(frozen=True)
class _NativeMTPSparseClaim:
    records: Tuple[_GenerationForwardCanonicalRecord, ...]
    selected_logical_positions: Tuple[int, ...]
    selected_token_ids: Tuple[int, ...]
    immediate_successor_token_ids: Tuple[int, ...]
    target_cache: List[Any]
    next_logical_position: int

    @property
    def final_target_hidden(self):
        return self.records[-1].hidden_rows[:, -1:, :]

    @property
    def final_target_logits(self):
        return self.records[-1].logits


def abandon_native_mtp_sparse_receipts(
    receipts: Sequence[GenerationForwardPositionReceipt],
) -> None:
    """Consume issued, unreserved sparse receipts without touching model state.

    This is the narrow owner for a partial ordered receipt set when a later
    attested target chunk fails before a complete bootstrap can be formed.
    It is deliberately best-effort for malformed caller containers: only a
    live canonical receipt can remove its own direct weak-registry authority.
    A claim reservation always wins over abandonment and remains responsible
    for its existing finally-path cleanup.
    """

    if not isinstance(receipts, (list, tuple)):
        return
    with _GENERATION_FORWARD_RECEIPT_LOCK:
        for receipt in receipts:
            record_token = getattr(receipt, "_record_token", None)
            if not isinstance(record_token, _GenerationForwardReceiptToken):
                continue
            authority = _GENERATION_FORWARD_RECEIPTS.get(record_token)
            if (
                authority is not None
                and authority.reservation is None
                and authority.record.receipt_id == id(receipt)
            ):
                del _GENERATION_FORWARD_RECEIPTS[record_token]


@dataclass(frozen=True)
class NativeMTPSparseBootstrap:
    """Attested sparse target state from which native MTP may start.

    The receipts retain the exact target outputs.  The explicit host tuples
    make the caller contract inspectable, but never act as a second authority:
    validation requires them to equal the ordered receipt contents exactly.
    """

    receipts: Tuple[GenerationForwardPositionReceipt, ...]
    selected_logical_positions: Tuple[int, ...]
    selected_token_ids: Tuple[int, ...]
    immediate_successor_token_ids: Tuple[int, ...]
    target_cache: List[Any]
    next_logical_position: int

    @property
    def final_target_hidden(self) -> mx.array:
        return self.receipts[-1].hidden_rows[:, -1:, :]

    @property
    def final_target_logits(self) -> mx.array:
        return self.receipts[-1].logits

    def close(self) -> None:
        """Abandon this bootstrap's still-unclaimed receipt authority.

        Closing is idempotent and does not sample, construct an MTP request,
        or mutate the attested target cache. A claim that already reserved the
        same receipts remains its sole owner and consumes them in its own
        finally path.
        """

        abandon_native_mtp_sparse_receipts(self.receipts)

    def _abandonable_authority_locked(self):
        """Return this exact unreserved authority set using host metadata only."""

        if not isinstance(self.receipts, tuple) or not self.receipts:
            return None
        if not isinstance(self.target_cache, list) or not self.target_cache:
            return None
        if not all(
            isinstance(values, tuple)
            for values in (
                self.selected_logical_positions,
                self.selected_token_ids,
                self.immediate_successor_token_ids,
            )
        ):
            return None
        if isinstance(self.next_logical_position, bool) or not isinstance(
            self.next_logical_position, int
        ):
            return None

        records = []
        record_tokens = []
        for receipt in self.receipts:
            record_token = getattr(receipt, "_record_token", None)
            if not isinstance(record_token, _GenerationForwardReceiptToken):
                return None
            authority = _GENERATION_FORWARD_RECEIPTS.get(record_token)
            if authority is None or authority.reservation is not None:
                return None
            try:
                _validate_receipt_against_record(receipt, authority.record)
            except Exception:
                return None
            records.append(authority.record)
            record_tokens.append(record_token)
        if len(record_tokens) != len(set(record_tokens)):
            return None

        final = records[-1]
        try:
            current_entry_ids = tuple(id(entry) for entry in self.target_cache)
            current_cache_state = _cache_state_identity_evidence(self.target_cache)
        except Exception:
            return None
        if (
            id(self.target_cache) != final.cache_container_id
            or current_entry_ids != final.cache_entry_ids
            or current_cache_state != final.cache_state_evidence
        ):
            return None

        positions = tuple(
            position for record in records for position in record.logical_positions
        )
        token_ids = tuple(
            token_id for record in records for token_id in record.token_ids
        )
        successors = tuple(
            token_id
            for record in records
            for token_id in record.immediate_successor_token_ids
        )
        if (
            positions != self.selected_logical_positions
            or token_ids != self.selected_token_ids
            or successors != self.immediate_successor_token_ids
            or not positions
            or self.next_logical_position != positions[-1] + 1
        ):
            return None
        previous_position = -1
        for position in positions:
            if (
                isinstance(position, bool)
                or not isinstance(position, int)
                or position <= previous_position
            ):
                return None
            previous_position = position
        if any(
            isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
            for token_id in (*token_ids, *successors)
        ):
            return None

        for index, record in enumerate(records):
            is_final = index == len(records) - 1
            expected_successors = len(record.token_ids) - int(is_final)
            if (
                record.cache is not self.target_cache
                or record.phase is not GenerationForwardPhase.PREFILL
                or len(record.logical_positions) != len(record.token_ids)
                or len(record.immediate_successor_token_ids) != expected_successors
                or (record.logits is None) is is_final
                or (record.logits_content_digest is None) is is_final
                or (record.cache_content_digest is None) is is_final
            ):
                return None
            if is_final and (
                record.logits.ndim != 3
                or record.logits.shape[0] != 1
                or record.logits.shape[1] != 1
            ):
                return None
        return tuple(records), tuple(record_tokens)

    def try_abandon_unclaimed(self) -> bool:
        """Atomically consume valid unclaimed authority for safe dense replay.

        Canonical payload hashes are verified off-lock. The final locked pass
        rechecks every exact receipt and reservation before deleting all keys;
        a concurrent claim therefore either owns the complete set or loses to
        this abandonment, never a partial mixture.
        """

        try:
            with _GENERATION_FORWARD_RECEIPT_LOCK:
                snapshot = self._abandonable_authority_locked()
        except Exception:
            return False
        if snapshot is None:
            return False
        records, _ = snapshot
        try:
            _verify_sparse_canonical_content(records)
        except Exception:
            return False

        try:
            with _GENERATION_FORWARD_RECEIPT_LOCK:
                current = self._abandonable_authority_locked()
                if current is None:
                    return False
                current_records, record_tokens = current
                if any(
                    current_record is not snapshot_record
                    for current_record, snapshot_record in zip(current_records, records)
                ):
                    return False
                for record_token in record_tokens:
                    del _GENERATION_FORWARD_RECEIPTS[record_token]
        except Exception:
            return False
        return True

    def _canonical_records_locked(self, reservation=None):
        records = []
        tokens = []
        authorities = []
        for receipt in self.receipts:
            record_token = getattr(receipt, "_record_token", None)
            if not isinstance(record_token, _GenerationForwardReceiptToken):
                raise RuntimeError("native_mtp_sparse_receipt_canonical_mismatch")
            authority = _GENERATION_FORWARD_RECEIPTS.get(record_token)
            if authority is None or authority.reservation is not None:
                raise RuntimeError("native_mtp_sparse_bootstrap_already_claimed")
            record = authority.record
            _validate_receipt_against_record(receipt, record)
            records.append(record)
            tokens.append(record_token)
            authorities.append(authority)
        if len(tokens) != len(set(tokens)):
            raise RuntimeError("native_mtp_sparse_receipt_reused")
        if reservation is not None:
            for authority in authorities:
                authority.reservation = reservation
        return tuple(records), tuple(tokens)

    def _validate_records(self, model: nn.Module, canonical_records) -> None:
        """Validate provenance, values, topology-facing shapes, and cursor."""

        if not isinstance(self.receipts, tuple) or not self.receipts:
            raise ValueError("native_mtp_sparse_receipts_required")
        if not isinstance(self.target_cache, list) or not self.target_cache:
            raise TypeError("native_mtp_sparse_target_cache_must_be_list")
        for name, values in (
            ("positions", self.selected_logical_positions),
            ("tokens", self.selected_token_ids),
            ("successors", self.immediate_successor_token_ids),
        ):
            if not isinstance(values, tuple):
                raise TypeError(f"native MTP sparse {name} must be a host tuple")

        positions = self.selected_logical_positions
        token_ids = self.selected_token_ids
        successors = self.immediate_successor_token_ids
        if not positions or len(token_ids) != len(positions):
            raise ValueError("native_mtp_sparse_selected_length_mismatch")
        if len(successors) != len(positions) - 1:
            raise ValueError("native_mtp_sparse_successor_length_mismatch")

        previous = -1
        for position in positions:
            if (
                isinstance(position, bool)
                or not isinstance(position, int)
                or position <= previous
            ):
                raise ValueError("native_mtp_sparse_positions_not_strict")
            previous = position
        if (
            isinstance(self.next_logical_position, bool)
            or not isinstance(self.next_logical_position, int)
            or self.next_logical_position != positions[-1] + 1
        ):
            raise ValueError("native_mtp_sparse_cursor_mismatch")

        vocab_size = getattr(model, "vocab_size", None)
        if vocab_size is None:
            vocab_size = getattr(getattr(model, "args", None), "vocab_size", None)
        for token_id in (*token_ids, *successors):
            if (
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                or (vocab_size is not None and token_id >= vocab_size)
            ):
                raise ValueError("native_mtp_sparse_token_id_invalid")

        capability = NativeMTPCapabilityFingerprint.from_model(model)
        expected_cache_id = id(self.target_cache)
        expected_entry_ids = tuple(id(entry) for entry in self.target_cache)
        model_args = getattr(model, "args", None)
        if getattr(model_args, "hidden_size", None) is None:
            model_args = getattr(getattr(model, "language_model", None), "args", None)
        expected_hidden_width = getattr(model_args, "hidden_size", None)
        if canonical_records[-1].logits is None:
            raise RuntimeError("native_mtp_sparse_final_logits_evidence_missing")
        if canonical_records[-1].logits_content_digest is None:
            raise RuntimeError("native_mtp_sparse_final_logits_digest_missing")
        if canonical_records[-1].cache_content_digest is None:
            raise RuntimeError("native_mtp_sparse_final_cache_evidence_missing")
        if any(
            record.logits is not None
            or record.logits_content_digest is not None
            or record.cache_content_digest is not None
            for record in canonical_records[:-1]
        ):
            raise RuntimeError("native_mtp_sparse_final_logits_evidence_order_mismatch")
        receipt_positions = []
        receipt_tokens = []
        receipt_successors = []
        hidden_width = None
        floating_dtypes = (mx.float16, mx.float32, mx.bfloat16)
        for index, (receipt, record) in enumerate(
            zip(self.receipts, canonical_records)
        ):
            if not isinstance(receipt, GenerationForwardPositionReceipt):
                raise TypeError("native_mtp_sparse_receipt_invalid")
            if (
                getattr(receipt, "_issuer_seal", None)
                is not _GENERATION_FORWARD_RECEIPT_ISSUER
            ):
                raise TypeError("native_mtp_sparse_receipt_not_issued")
            if receipt.phase is not GenerationForwardPhase.PREFILL:
                raise ValueError("native_mtp_sparse_receipt_phase_mismatch")
            if (
                receipt.model_id != id(model)
                or receipt.cache_container_id != expected_cache_id
                or receipt.cache_entry_ids != expected_entry_ids
                or receipt.capability != capability
            ):
                raise RuntimeError("native_mtp_sparse_receipt_provenance_mismatch")
            if record.cache is not self.target_cache:
                raise RuntimeError("native_mtp_sparse_target_cache_identity_mismatch")
            if not receipt.logical_positions or len(receipt.token_ids) != len(
                receipt.logical_positions
            ):
                raise ValueError("native_mtp_sparse_receipt_length_mismatch")
            expected_successors = len(receipt.token_ids)
            if index == len(self.receipts) - 1:
                expected_successors -= 1
            if len(receipt.immediate_successor_token_ids) != expected_successors:
                raise ValueError("native_mtp_sparse_receipt_successor_mismatch")

            hidden = receipt.hidden_rows
            logits = receipt.logits
            chunk_size = len(receipt.token_ids)
            if (
                not isinstance(hidden, mx.array)
                or hidden.ndim != 3
                or hidden.shape[0] != 1
                or hidden.shape[1] != chunk_size
                or hidden.dtype not in floating_dtypes
            ):
                raise ValueError("native_mtp_sparse_hidden_shape_or_dtype_mismatch")
            if hidden_width is None:
                hidden_width = hidden.shape[2]
            elif hidden.shape[2] != hidden_width:
                raise ValueError("native_mtp_sparse_hidden_width_mismatch")
            if (
                expected_hidden_width is not None
                and hidden.shape[2] != expected_hidden_width
            ):
                raise ValueError("native_mtp_sparse_hidden_model_width_mismatch")
            is_final = index == len(self.receipts) - 1
            if is_final:
                if (
                    not isinstance(logits, mx.array)
                    or logits.ndim != 3
                    or logits.shape[0] != 1
                    or logits.shape[1] != 1
                    or logits.dtype not in floating_dtypes
                    or (vocab_size is not None and logits.shape[2] != vocab_size)
                ):
                    raise ValueError(
                        "native_mtp_sparse_final_logits_shape_or_dtype_mismatch"
                    )
            elif logits is not None:
                raise ValueError("native_mtp_sparse_nonfinal_logits_retained")

            receipt_positions.extend(receipt.logical_positions)
            receipt_tokens.extend(receipt.token_ids)
            receipt_successors.extend(receipt.immediate_successor_token_ids)

        if tuple(receipt_positions) != positions:
            raise ValueError("native_mtp_sparse_receipt_positions_mismatch")
        if tuple(receipt_tokens) != token_ids:
            raise ValueError("native_mtp_sparse_receipt_tokens_mismatch")
        if tuple(receipt_successors) != successors:
            raise ValueError("native_mtp_sparse_receipt_successors_mismatch")
        if (
            _cache_state_identity_evidence(self.target_cache)
            != canonical_records[-1].cache_state_evidence
        ):
            raise RuntimeError("native_mtp_sparse_target_cache_state_mismatch")

    def validate(self, model: nn.Module) -> None:
        """Validate currently issued authority without consuming it."""

        if not isinstance(self.receipts, tuple) or not self.receipts:
            raise ValueError("native_mtp_sparse_receipts_required")
        with _GENERATION_FORWARD_RECEIPT_LOCK:
            records, _ = self._canonical_records_locked()
        self._validate_records(model, records)
        _verify_sparse_canonical_content(records)

    def claim(self, model: nn.Module) -> _NativeMTPSparseClaim:
        """Reserve, verify off-lock, and permanently consume receipt authority."""

        if not isinstance(self.receipts, tuple) or not self.receipts:
            raise ValueError("native_mtp_sparse_receipts_required")
        reservation = object()
        with _GENERATION_FORWARD_RECEIPT_LOCK:
            records, record_tokens = self._canonical_records_locked(reservation)
        try:
            self._validate_records(model, records)
            _verify_sparse_canonical_content(records)
        finally:
            with _GENERATION_FORWARD_RECEIPT_LOCK:
                for record_token in record_tokens:
                    authority = _GENERATION_FORWARD_RECEIPTS.get(record_token)
                    if authority is not None and authority.reservation is reservation:
                        del _GENERATION_FORWARD_RECEIPTS[record_token]

        positions = tuple(
            position for record in records for position in record.logical_positions
        )
        token_ids = tuple(
            token_id for record in records for token_id in record.token_ids
        )
        successors = tuple(
            token_id
            for record in records
            for token_id in record.immediate_successor_token_ids
        )
        final = records[-1]
        return _NativeMTPSparseClaim(
            records=records,
            selected_logical_positions=positions,
            selected_token_ids=token_ids,
            immediate_successor_token_ids=successors,
            target_cache=final.cache,
            next_logical_position=positions[-1] + 1,
        )


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
        default=0.1,
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
    if not mx.metal.is_available():
        try:
            yield
        finally:
            pass
    else:
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
    mtp_drafts: int = 0
    mtp_accepted: int = 0
    mtp_bypass_reason: Optional[str] = None


def maybe_quantize_kv_cache(prompt_cache, quantized_kv_start, kv_group_size, kv_bits):
    if kv_bits is None:
        return
    for e, c in enumerate(prompt_cache):
        if hasattr(c, "to_quantized") and c.offset >= quantized_kv_start:
            prompt_cache[e] = c.to_quantized(group_size=kv_group_size, bits=kv_bits)


def _prompt_cache_has_tokens(prompt_cache: Any) -> bool:
    """Return whether a supplied prompt cache already represents a prefix.

    ``offset`` is the public occupancy contract of the standard KV cache
    families.  A few cache implementations expose only ``empty()``; use that
    as a conservative fallback without asking callers to identify cache types.
    """
    for entry in prompt_cache:
        offset = getattr(entry, "offset", None)
        if isinstance(offset, int) and offset > 0:
            return True
        is_empty = getattr(entry, "empty", None)
        if callable(is_empty) and not is_empty():
            return True
    return False


def generate_step(
    prompt: mx.array,
    model: nn.Module,
    *,
    max_tokens: int = 256,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
    max_kv_size: Optional[int] = None,
    prompt_cache: Optional[Any] = None,
    prefill_step_size: int = 2048,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    prompt_progress_callback: Optional[Callable[[int, int], None]] = None,
    input_embeddings: Optional[mx.array] = None,
    model_forward_context: Optional[GenerationForwardContext] = None,
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
        prompt_progress_callback (Callable[[int, int], None]): A call-back which takes the
           prompt tokens processed so far and the total number of prompt tokens.
        input_embeddings (mx.array, optional): Input embeddings to use instead of or in
          conjunction with prompt tokens. Default: ``None``.
        model_forward_context (Callable[[GenerationForward], ContextManager], optional):
          A request-local context factory invoked around each Python model call.
          The scope covers MLX graph construction, not deferred realization.

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

    # The final prompt token is a semantic prefill for a fresh cache, but a
    # decode when the caller supplies an already-populated continuation cache.
    has_cached_prefix = (
        model_forward_context is not None
        and prompt_cache is not None
        and _prompt_cache_has_tokens(prompt_cache)
    )

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

    def _model_call(
        input_tokens: mx.array,
        input_embeddings: Optional[mx.array],
        phase: GenerationForwardPhase,
    ):
        # Preserve the ordinary generation path when no caller needs forward
        # metadata: no context-manager or metadata allocation occurs here.
        if model_forward_context is None:
            if input_embeddings is not None:
                return model(
                    input_tokens, cache=prompt_cache, input_embeddings=input_embeddings
                )
            return model(input_tokens, cache=prompt_cache)

        forward = GenerationForward(
            model=model,
            input_tokens=input_tokens,
            cache=prompt_cache,
            phase=phase,
            input_embeddings=input_embeddings,
        )
        with model_forward_context(forward):
            if input_embeddings is not None:
                return model(
                    input_tokens, cache=prompt_cache, input_embeddings=input_embeddings
                )
            return model(input_tokens, cache=prompt_cache)

    def _step(
        input_tokens: mx.array,
        input_embeddings: Optional[mx.array] = None,
        phase: GenerationForwardPhase = GenerationForwardPhase.DECODE,
    ):
        nonlocal tokens

        with mx.stream(generation_stream):
            logits = _model_call(
                input_tokens=input_tokens[None],
                input_embeddings=(
                    input_embeddings[None] if input_embeddings is not None else None
                ),
                phase=phase,
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
            remaining = (total_prompt_tokens - prompt_processed_tokens) - 1
            n_to_process = min(prefill_step_size, remaining)
            _model_call(
                input_tokens=prompt[:n_to_process][None],
                input_embeddings=(
                    input_embeddings[:n_to_process][None]
                    if input_embeddings is not None
                    else None
                ),
                phase=GenerationForwardPhase.PREFILL,
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

        y, logprobs = _step(
            input_tokens=prompt,
            input_embeddings=input_embeddings,
            phase=(
                GenerationForwardPhase.DECODE
                if has_cached_prefix
                else GenerationForwardPhase.PREFILL
            ),
        )

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
    num_draft_tokens: int = 2,
    max_tokens: int = 256,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
    prompt_cache: Optional[Any] = None,
    prefill_step_size: int = 512,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    model_forward_context: Optional[GenerationForwardContext] = None,
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
        sampler (Callable[[mx.array], mx.array], optional): A sampler for sampling a
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
        model_forward_context (Callable[[GenerationForward], ContextManager], optional):
          A request-local context factory invoked around each Python model call.
          The scope covers MLX graph construction, not deferred realization.

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

    if not cache.can_trim_prompt_cache(model_cache):
        types = {type(c).__name__ for c in model_cache if not c.is_trimmable()}
        raise ValueError(
            f"Speculative decoding requires a trimmable prompt cache " f"(got {types})."
        )

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

    def _model_call(model, cache, input_tokens, phase):
        # Keep the no-callback path direct: this is the hot path for ordinary
        # speculative decoding and does not allocate a context or metadata.
        if model_forward_context is None:
            return model(input_tokens, cache=cache)
        forward = GenerationForward(
            model=model,
            input_tokens=input_tokens,
            cache=cache,
            phase=phase,
        )
        with model_forward_context(forward):
            return model(input_tokens, cache=cache)

    def _step(model, cache, y, n_predict=1, phase=GenerationForwardPhase.VERIFY):
        with mx.stream(generation_stream):
            logits = _model_call(model, cache, y[None], phase)
            logits = logits[:, -n_predict:, :]

            quantize_cache_fn(cache)
            if logits_processors:
                nonlocal prev_tokens
                out_y, out_logprobs = [], []
                if n_predict > 1:
                    y = y[: -(n_predict - 1)]
                for i in range(n_predict):
                    prev_tokens = (
                        mx.concatenate([prev_tokens, y])
                        if prev_tokens is not None
                        else y
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
        while y.size > 1:
            n_to_process = min(prefill_step_size, y.size - 1)
            _model_call(
                model,
                cache,
                y[:n_to_process][None],
                GenerationForwardPhase.PREFILL,
            )
            quantize_cache_fn(cache)
            mx.eval([c.state for c in cache])
            y = y[n_to_process:]
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
            y, _ = _step(
                draft_model,
                draft_cache,
                y,
                phase=GenerationForwardPhase.DRAFT,
            )
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
    first_target_round = True
    try:
        while True:
            num_draft = min(max_tokens - ntoks, num_draft_tokens)
            draft_tokens = _draft_generate(draft_y, num_draft)
            if prev_tokens is not None:
                prev_tokens = prev_tokens[: prev_tokens.size - y.size - num_draft + 1]
            y = mx.concatenate([y, draft_tokens])
            tokens, logprobs = _step(
                model,
                model_cache,
                y,
                num_draft + 1,
                phase=(
                    GenerationForwardPhase.VERIFY
                    if num_draft
                    else (
                        GenerationForwardPhase.PREFILL
                        if first_target_round
                        else GenerationForwardPhase.DECODE
                    )
                ),
            )
            first_target_round = False
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


def mtp_generate_step(
    prompt: Optional[mx.array],
    model: nn.Module,
    *,
    max_tokens: int = 256,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
    prompt_cache: Optional[Any] = None,
    prefill_step_size: int = 512,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    model_forward_context: Optional[GenerationForwardContext] = None,
    sampling_config: Optional[NativeMTPSamplingConfig] = None,
    eos_token_ids: Optional[Sequence[int]] = None,
    telemetry: Optional[Dict[str, Any]] = None,
    prompt_logical_positions: Optional[Sequence[int]] = None,
    sparse_bootstrap: Optional[NativeMTPSparseBootstrap] = None,
) -> Generator[Tuple[int, mx.array, bool], None, None]:
    """Stream native-Qwen MTP with request-local transactional cache state.

    This implementation intentionally supports a single sequence and no
    prefix cache.  A full verification commits atomically.  A rejected draft
    rolls the full transaction back and replays the already-emitted token
    before continuing, which is required for the Qwen recurrent layers.

    Stochastic sampling is described by :class:`NativeMTPSamplingConfig` and
    uses an explicit per-request MLX key.  Opaque filters and stateful logits
    processors fail before cache construction because they cannot be replayed
    exactly after a rejected recurrent verification.  Optional sparse prompt
    coordinates are immutable host metadata only: Qwen selects its canonical
    in-model context automatically and every forward must acknowledge exact
    consumption before its request cache can be retained.
    """

    if prompt_cache is not None:
        raise ValueError("native_mtp_prefix_reuse_unsupported")
    capability = getattr(model, "mtp_capability", None)
    if capability is None or not capability.supported:
        reason = (
            "native_mtp_model_capability_missing"
            if capability is None
            else capability.reason
        )
        if telemetry is not None:
            telemetry["mtp_bypass_reason"] = reason
        raise RuntimeError(reason)
    if sparse_bootstrap is None:
        if prompt is None or prompt.ndim != 1 or prompt.size == 0:
            raise ValueError("native MTP requires one non-empty unbatched prompt")
    else:
        if prompt is not None:
            raise ValueError("native_mtp_sparse_bootstrap_owns_selected_tokens")
        if prompt_logical_positions is not None:
            raise ValueError("native_mtp_sparse_bootstrap_owns_logical_positions")
    if max_tokens < 0:
        raise ValueError("native MTP requires a finite non-negative max_tokens")
    if (
        isinstance(prefill_step_size, bool)
        or not isinstance(prefill_step_size, int)
        or prefill_step_size < 1
    ):
        raise ValueError("native MTP prefill_step_size must be a positive integer")
    sampling_config = sampling_config or NativeMTPSamplingConfig()
    model_vocab_size = getattr(model, "vocab_size", None)
    if model_vocab_size is None:
        model_vocab_size = getattr(getattr(model, "args", None), "vocab_size", None)
    if (
        sampling_config.top_k > 0
        and model_vocab_size is not None
        and sampling_config.top_k >= model_vocab_size
    ):
        raise ValueError("native MTP top_k must be smaller than vocabulary size")
    if sampler is not None and (
        sampling_config.stochastic
        or not getattr(sampler, "native_mtp_deterministic", False)
    ):
        raise ValueError("native_mtp_opaque_sampler_unsupported")
    if logits_processors and any(
        not getattr(processor, "native_mtp_replay_safe", False)
        for processor in logits_processors
    ):
        raise ValueError("native_mtp_non_replay_safe_logits_processor")

    logical_prompt = None
    if prompt_logical_positions is not None:
        if isinstance(prompt_logical_positions, mx.array) or not isinstance(
            prompt_logical_positions, Sequence
        ):
            raise TypeError(
                "native MTP prompt logical positions must be a host sequence"
            )
        logical_prompt = tuple(prompt_logical_positions)
        if len(logical_prompt) != prompt.size:
            raise ValueError("native MTP prompt logical positions must match prompt")
        previous_position = -1
        for position in logical_prompt:
            if (
                isinstance(position, bool)
                or not isinstance(position, int)
                or position < 0
            ):
                raise ValueError(
                    "native MTP prompt logical positions must be non-negative integers"
                )
            if position <= previous_position:
                raise ValueError(
                    "native MTP prompt logical positions must be strictly increasing"
                )
            previous_position = position

    logical_positions_active = (
        logical_prompt is not None or sparse_bootstrap is not None
    )
    position_forward_context = (
        _native_mtp_position_context(model, model_forward_context)
        if logical_positions_active
        else None
    )

    request = None
    sparse_claim = None

    def _target_call(
        input_tokens,
        phase,
        *,
        return_hidden=False,
        logical_positions=None,
        immediate_successor_token_ids=(),
    ):
        batched = input_tokens[None]
        forward_context = (
            position_forward_context
            if logical_positions is not None
            else model_forward_context
        )
        if forward_context is None:
            return model(batched, cache=request.backbone, return_hidden=return_hidden)
        position_ack = (
            GenerationForwardPositionAck(
                logical_positions,
                model=model,
                cache=request.backbone,
                phase=phase,
            )
            if logical_positions is not None
            else None
        )
        forward = GenerationForward(
            model=model,
            input_tokens=batched,
            cache=request.backbone,
            phase=phase,
            logical_positions=logical_positions,
            logical_position_ack=position_ack,
        )
        with forward_context(forward):
            if position_ack is None:
                return model(
                    batched, cache=request.backbone, return_hidden=return_hidden
                )
            position_ack._activate()
            try:
                result = model(
                    batched, cache=request.backbone, return_hidden=return_hidden
                )
                position_ack._require_acknowledged()
            finally:
                position_ack._finish()
        return result

    def _mtp_call(hidden, next_tokens, *, logical_positions=None):
        next_ids = next_tokens.reshape(1, -1)
        forward_context = (
            position_forward_context
            if logical_positions is not None
            else model_forward_context
        )
        if forward_context is None:
            return model.mtp_forward(hidden, next_ids, request.mtp)
        position_ack = (
            GenerationForwardPositionAck(
                logical_positions,
                model=model,
                cache=request.mtp,
                phase=GenerationForwardPhase.MTP_DRAFT,
            )
            if logical_positions is not None
            else None
        )
        forward = GenerationForward(
            model=model,
            input_tokens=next_ids,
            cache=request.mtp,
            phase=GenerationForwardPhase.MTP_DRAFT,
            logical_positions=logical_positions,
            logical_position_ack=position_ack,
        )
        with forward_context(forward):
            if position_ack is None:
                return model.mtp_forward(hidden, next_ids, request.mtp)
            position_ack._activate()
            try:
                result = model.mtp_forward(hidden, next_ids, request.mtp)
                position_ack._require_acknowledged()
                return result
            finally:
                position_ack._finish()

    def _quantize():
        request.quantize(
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            start=quantized_kv_start,
        )

    def _assert_target_mtp_alignment():
        state = request.state
        if state.backbone_tokens != state.mtp_tokens:
            raise RuntimeError("native_mtp_target_head_alignment_mismatch")
        request.assert_aligned(
            backbone_tokens=state.backbone_tokens,
            mtp_tokens=state.mtp_tokens,
        )

    def _draft(hidden, next_tokens, token_history, logical_positions=None):
        with mx.stream(generation_stream):
            logits = _mtp_call(
                hidden, next_tokens, logical_positions=logical_positions
            )[:, -1, :]
            _quantize()
            request.retain(
                backbone_tokens=request.state.backbone_tokens,
                mtp_tokens=request.state.mtp_tokens + next_tokens.size,
            )
            _assert_target_mtp_alignment()
            draft_reported_logprobs = sampling_state.reported_logprobs(
                logits.squeeze(0), token_history
            )
            draft_logprobs = sampling_state.sampling_distribution(
                draft_reported_logprobs
            )
            draft_token = sampling_state.sample(draft_logprobs)
        mx.eval(draft_token, draft_logprobs)
        if telemetry is not None:
            telemetry["mtp_drafts"] += 1
        return draft_token.reshape(-1), draft_logprobs

    finish_reason = "generator_closed"
    generated = 0
    try:
        if sparse_bootstrap is None:
            request = model.make_mtp_request_cache()
        else:
            sparse_claim = sparse_bootstrap.claim(model)
            request = cache.NativeMTPRequestCache.adopt_sparse_target(
                model,
                target_cache=sparse_claim.target_cache,
                target_tokens=len(sparse_claim.selected_token_ids),
                next_logical_position=sparse_claim.next_logical_position,
            )
        if telemetry is not None:
            telemetry.update(
                mtp_drafts=0,
                mtp_accepted=0,
                mtp_bypass_reason=None,
            )
        eos_token_ids = frozenset(eos_token_ids or ())
        sampling_state = _NativeMTPSamplingState(
            sampling_config,
            sampler=sampler,
            logits_processors=logits_processors,
            seed=sampling_config.seed,
        )
        if max_tokens == 0:
            finish_reason = "length"
            return

        if sparse_bootstrap is None:
            # Shifted hidden/token pairs prefill the MTP head to the same
            # physical and logical position as the target before decode.
            remaining = prompt.astype(mx.uint32)
            prompt_position_index = 0 if logical_prompt is not None else None
            with mx.stream(generation_stream):
                while remaining.size > 1:
                    count = min(prefill_step_size, remaining.size - 1)
                    forward_positions = (
                        logical_prompt[
                            prompt_position_index : prompt_position_index + count
                        ]
                        if logical_prompt is not None
                        else None
                    )
                    _, hidden = _target_call(
                        remaining[:count],
                        GenerationForwardPhase.PREFILL,
                        return_hidden=True,
                        logical_positions=forward_positions,
                        immediate_successor_token_ids=remaining[1 : count + 1].tolist(),
                    )
                    _mtp_call(
                        hidden,
                        remaining[1 : count + 1],
                        logical_positions=forward_positions,
                    )
                    _quantize()
                    mx.eval(
                        [entry.state for entry in request.backbone],
                        [entry.state for entry in request.mtp],
                    )
                    if forward_positions is None:
                        request.retain(
                            backbone_tokens=request.state.backbone_tokens + count,
                            mtp_tokens=request.state.mtp_tokens + count,
                        )
                    else:
                        request.retain_logical_position(
                            backbone_tokens=request.state.backbone_tokens + count,
                            mtp_tokens=request.state.mtp_tokens + count,
                            next_logical_position=forward_positions[-1] + 1,
                        )
                    _assert_target_mtp_alignment()
                    remaining = remaining[count:]
                    if logical_prompt is not None:
                        prompt_position_index += count
                    mx.clear_cache()

                final_prompt_positions = (
                    (logical_prompt[prompt_position_index],)
                    if logical_prompt is not None
                    else None
                )
                initial_logits, initial_hidden = _target_call(
                    remaining,
                    GenerationForwardPhase.PREFILL,
                    return_hidden=True,
                    logical_positions=final_prompt_positions,
                )
                _quantize()
                if final_prompt_positions is None:
                    request.retain(
                        backbone_tokens=request.state.backbone_tokens + 1,
                        mtp_tokens=request.state.mtp_tokens,
                    )
                else:
                    request.retain_logical_position(
                        backbone_tokens=request.state.backbone_tokens + 1,
                        mtp_tokens=request.state.mtp_tokens,
                        next_logical_position=final_prompt_positions[0] + 1,
                    )
                # Match generate_step: processors see only the final step.
                history = remaining.astype(mx.uint32)
                current_logprobs = sampling_state.reported_logprobs(
                    initial_logits[:, -1, :].squeeze(0), history
                )
                current_sampling_logprobs = sampling_state.sampling_distribution(
                    current_logprobs
                )
                current_token = sampling_state.sample(
                    current_sampling_logprobs
                ).reshape(-1)
            current_hidden = initial_hidden[:, -1:, :]
        else:
            selected_count = len(sparse_claim.selected_token_ids)
            final_prompt_positions = (sparse_claim.selected_logical_positions[-1],)
            with mx.stream(generation_stream):
                for record in sparse_claim.records:
                    pair_count = len(record.immediate_successor_token_ids)
                    pair_start = 0
                    while pair_start < pair_count:
                        pair_end = min(pair_start + prefill_step_size, pair_count)
                        _mtp_call(
                            record.hidden_rows[:, pair_start:pair_end, :],
                            mx.array(
                                record.immediate_successor_token_ids[
                                    pair_start:pair_end
                                ],
                                dtype=mx.uint32,
                            ),
                            logical_positions=record.logical_positions[
                                pair_start:pair_end
                            ],
                        )
                        _quantize()
                        mx.eval([entry.state for entry in request.mtp])
                        pair_start = pair_end
                        mx.clear_cache()
                mx.eval(
                    tuple(record.hidden_rows for record in sparse_claim.records),
                    sparse_claim.final_target_logits,
                    [entry.state for entry in request.backbone],
                    [entry.state for entry in request.mtp],
                )
                history = mx.array(
                    [sparse_claim.selected_token_ids[-1]], dtype=mx.uint32
                )
                current_logprobs = sampling_state.reported_logprobs(
                    sparse_claim.final_target_logits[:, -1, :].squeeze(0),
                    history,
                )
                current_sampling_logprobs = sampling_state.sampling_distribution(
                    current_logprobs
                )
                current_token = sampling_state.sample(
                    current_sampling_logprobs
                ).reshape(-1)
            current_hidden = sparse_claim.final_target_hidden
        mx.eval(current_token, current_logprobs)
        if sparse_bootstrap is not None:
            request.seal_verified(
                backbone_tokens=selected_count,
                mtp_tokens=selected_count - 1,
            )
            request.commit(
                backbone_tokens=selected_count,
                mtp_tokens=selected_count - 1,
            )
        history = mx.concatenate([history, current_token])

        generated = 1
        if logical_positions_active:
            request.advance_logical_position()
        current_id = current_token.item()
        terminal = current_id in eos_token_ids or generated >= max_tokens
        if terminal:
            finish_reason = "eos" if current_id in eos_token_ids else "length"
            request.finish(finish_reason)
        yield current_id, current_logprobs, False
        if terminal:
            return
        draft_token, draft_logprobs = _draft(
            current_hidden,
            current_token,
            history,
            logical_positions=final_prompt_positions,
        )

        while True:

            request.checkpoint()
            verify_inputs = mx.concatenate([current_token, draft_token])
            verify_positions = (
                (
                    request.state.next_logical_position - 1,
                    request.state.next_logical_position,
                )
                if logical_positions_active
                else None
            )
            with mx.stream(generation_stream):
                verify_logits, verify_hidden = _target_call(
                    verify_inputs,
                    GenerationForwardPhase.VERIFY,
                    return_hidden=True,
                    logical_positions=verify_positions,
                )
                _quantize()
                request.seal_verified(
                    backbone_tokens=request.state.backbone_tokens + 2,
                    mtp_tokens=request.state.mtp_tokens,
                )
                verify_logprobs = sampling_state.reported_logprobs(
                    verify_logits[:, 0, :].squeeze(0), history
                )
                verify_sampling_logprobs = sampling_state.sampling_distribution(
                    verify_logprobs
                )

            if sampling_config.stochastic:
                draft_id = draft_token.item()
                acceptance = mx.minimum(
                    mx.exp(
                        verify_sampling_logprobs[draft_id] - draft_logprobs[draft_id]
                    ),
                    1.0,
                )
                accepted = (
                    mx.random.uniform(key=sampling_state.next_rng_key()) < acceptance
                )
                mx.eval(accepted)
                accepted = bool(accepted.item())
                replacement_token = replacement_logprobs = None
                if not accepted:
                    replacement_token, replacement_logprobs = (
                        sampling_state.residual_sample(
                            verify_sampling_logprobs, draft_logprobs
                        )
                    )
                    replacement_token = replacement_token.reshape(-1)
                    mx.eval(replacement_token, replacement_logprobs)
            else:
                replacement_token = sampling_state.sample(
                    verify_sampling_logprobs
                ).reshape(-1)
                mx.eval(replacement_token)
                accepted = replacement_token.item() == draft_token.item()
                replacement_logprobs = verify_sampling_logprobs

            if accepted:
                request.commit(
                    backbone_tokens=request.state.backbone_tokens + 2,
                    mtp_tokens=request.state.mtp_tokens,
                )
                if telemetry is not None:
                    telemetry["mtp_accepted"] += 1
                generated += 1
                if logical_positions_active:
                    request.advance_logical_position()
                draft_id = draft_token.item()
                draft_terminal = draft_id in eos_token_ids or generated >= max_tokens
                if draft_terminal:
                    finish_reason = "eos" if draft_id in eos_token_ids else "length"
                    request.finish(finish_reason)
                yield draft_id, verify_logprobs, True
                if draft_terminal:
                    return

                # Bonus processing begins only after the accepted draft has
                # crossed its yield/resume boundary.
                bonus_history = mx.concatenate([history, draft_token])
                bonus_logprobs = sampling_state.reported_logprobs(
                    verify_logits[:, 1, :].squeeze(0), bonus_history
                )
                bonus_sampling_logprobs = sampling_state.sampling_distribution(
                    bonus_logprobs
                )
                bonus_token = sampling_state.sample(bonus_sampling_logprobs).reshape(-1)
                mx.eval(bonus_token, bonus_logprobs)
                next_history = mx.concatenate([bonus_history, bonus_token])
                generated += 1
                if logical_positions_active:
                    request.advance_logical_position()
                bonus_id = bonus_token.item()
                bonus_terminal = bonus_id in eos_token_ids or generated >= max_tokens
                if bonus_terminal:
                    finish_reason = "eos" if bonus_id in eos_token_ids else "length"
                    request.finish(finish_reason)
                yield bonus_id, bonus_logprobs, False
                if bonus_terminal:
                    return

                # Catch the MTP state up and create the next draft only after
                # the bonus token has crossed its yield/resume boundary.
                mtp_hidden = mx.concatenate(
                    [verify_hidden[:, 0:1, :], verify_hidden[:, 1:2, :]], axis=1
                )
                mtp_tokens = mx.concatenate([draft_token, bonus_token])
                next_draft, next_draft_logprobs = _draft(
                    mtp_hidden,
                    mtp_tokens,
                    next_history,
                    logical_positions=verify_positions,
                )
                history = next_history
                current_token = bonus_token
                current_logprobs = bonus_logprobs
                current_hidden = verify_hidden[:, 1:2, :]
                draft_token = next_draft
                draft_logprobs = next_draft_logprobs
                continue

            # The prior output token was physically included by verify but
            # recurrent state cannot retain a prefix of that forward.  Restore
            # then replay exactly that token before drafting from the residual.
            request.reject_partial(accepted_backbone_tokens=1, accepted_mtp_tokens=0)
            replay_positions = (
                (request.state.next_logical_position - 1,)
                if logical_positions_active
                else None
            )
            with mx.stream(generation_stream):
                _, replay_hidden = _target_call(
                    current_token,
                    GenerationForwardPhase.DECODE,
                    return_hidden=True,
                    logical_positions=replay_positions,
                )
                _quantize()
                request.replay_retained(
                    backbone_tokens=request.state.backbone_tokens + 1,
                    mtp_tokens=request.state.mtp_tokens,
                )
            next_history = mx.concatenate([history, replacement_token])
            generated += 1
            if logical_positions_active:
                request.advance_logical_position()
            replacement_id = replacement_token.item()
            replacement_terminal = (
                replacement_id in eos_token_ids or generated >= max_tokens
            )
            if replacement_terminal:
                finish_reason = "eos" if replacement_id in eos_token_ids else "length"
                request.finish(finish_reason)
            yield replacement_id, verify_logprobs, False
            if replacement_terminal:
                return

            next_draft, next_draft_logprobs = _draft(
                replay_hidden[:, -1:, :],
                replacement_token,
                next_history,
                logical_positions=replay_positions,
            )
            history = next_history
            current_token = replacement_token
            current_logprobs = verify_logprobs
            current_hidden = replay_hidden[:, -1:, :]
            draft_token = next_draft
            draft_logprobs = next_draft_logprobs
    except GeneratorExit:
        finish_reason = "generator_closed"
        raise
    except BaseException:
        finish_reason = "cancelled"
        raise
    finally:
        if request is not None and not request.closed:
            request.finish(finish_reason)


def stream_generate(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: Optional[Union[str, mx.array, List[int]]],
    max_tokens: int = 256,
    draft_model: Optional[nn.Module] = None,
    mtp: bool = False,
    mtp_sampling_config: Optional[NativeMTPSamplingConfig] = None,
    model_forward_context: Optional[GenerationForwardContext] = None,
    sparse_bootstrap: Optional[NativeMTPSparseBootstrap] = None,
    **kwargs,
) -> Generator[GenerationResponse, None, None]:
    """
    A generator producing text based on the given prompt from the model.

    Args:
        model (nn.Module): The model to use for generation.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        prompt (Union[str, mx.array, List[int]], optional): The input prompt
          string or integer tokens. ``None`` is only valid with native MTP and
          an attested ``sparse_bootstrap`` that owns the selected prompt state.
        max_tokens (int): The maximum number of tokens to generate.
          Default: ``256``.
        draft_model (Optional[nn.Module]): An optional draft model. If provided
          then speculative decoding is used. The draft model must use the same
          tokenizer as the main model. Default: ``None``.
        mtp (bool): Request native model-owned MTP.  This is text-only,
          B=1, and does not support prefix cache reuse.
        mtp_sampling_config (NativeMTPSamplingConfig, optional): Immutable,
          request-local native MTP sampling and filtering policy.
        model_forward_context (Callable[[GenerationForward], ContextManager], optional):
          A request-local context factory forwarded to the selected generation
          implementation. The scope covers Python model-call graph construction.
        sparse_bootstrap (NativeMTPSparseBootstrap, optional): Attested sparse
          target state for native MTP. It owns the prompt tokens and logical
          positions, and cannot be combined with ``prompt`` or an external
          draft model.
        kwargs: The remaining options get passed to :func:`generate_step`.
          See :func:`generate_step` for more details.

    Yields:
        GenerationResponse: An instance containing the generated text segment and
            associated metadata. See :class:`GenerationResponse` for details.
    """
    sparse_prompt_tokens = None
    if sparse_bootstrap is not None:
        if not isinstance(sparse_bootstrap, NativeMTPSparseBootstrap):
            raise TypeError("native_mtp_sparse_bootstrap_invalid")
        if not mtp:
            raise ValueError("native_mtp_sparse_bootstrap_requires_mtp")
        if draft_model is not None:
            raise ValueError("native_mtp_sparse_bootstrap_external_draft_unsupported")
        if prompt is not None:
            raise ValueError("native_mtp_sparse_bootstrap_owns_selected_tokens")
        if kwargs.get("prompt_logical_positions") is not None:
            raise ValueError("native_mtp_sparse_bootstrap_owns_logical_positions")
        if kwargs.get("prompt_cache") is not None:
            raise ValueError("native_mtp_prefix_reuse_unsupported")
        capability = getattr(model, "mtp_capability", None)
        if capability is None or not capability.supported:
            reason = (
                "native_mtp_model_capability_missing"
                if capability is None
                else capability.reason
            )
            raise RuntimeError(reason)
        # Sparse prefill completed before this public stream starts. The next
        # logical position is immutable host metadata and preserves original
        # prompt-token usage without forcing device realization.
        sparse_prompt_tokens = sparse_bootstrap.next_logical_position
    elif prompt is None:
        raise ValueError("stream_generate requires a prompt")

    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    if sparse_bootstrap is None and not isinstance(prompt, mx.array):
        if isinstance(prompt, str):
            # Try to infer if special tokens are needed
            add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
                tokenizer.bos_token
            )
            prompt = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        prompt = mx.array(prompt)

    prompt_tokens = (
        prompt.size if sparse_prompt_tokens is None else sparse_prompt_tokens
    )

    detokenizer = tokenizer.detokenizer

    kwargs["max_tokens"] = max_tokens
    if model_forward_context is not None:
        kwargs["model_forward_context"] = model_forward_context

    mtp_telemetry = {
        "mtp_drafts": 0,
        "mtp_accepted": 0,
        "mtp_bypass_reason": None,
    }
    if draft_model is None and mtp:
        capability = getattr(model, "mtp_capability", None)
        if capability is not None and capability.supported:
            kwargs.pop("max_kv_size", None)
            kwargs.pop("prompt_progress_callback", None)
            kwargs.pop("num_draft_tokens", None)
            native_mtp_kwargs = {
                "sampling_config": mtp_sampling_config,
                "eos_token_ids": tokenizer.eos_token_ids,
                "telemetry": mtp_telemetry,
            }
            if sparse_bootstrap is not None:
                native_mtp_kwargs["sparse_bootstrap"] = sparse_bootstrap
            token_generator = mtp_generate_step(
                prompt,
                model,
                **native_mtp_kwargs,
                **kwargs,
            )
        else:
            mtp_telemetry["mtp_bypass_reason"] = (
                "native_mtp_model_capability_missing"
                if capability is None
                else capability.reason
            )
            kwargs.pop("num_draft_tokens", None)
            token_generator = generate_step(prompt, model, **kwargs)
            token_generator = (
                (token, logprobs, False) for token, logprobs in token_generator
            )
    elif draft_model is None:
        kwargs.pop("num_draft_tokens", None)
        token_generator = generate_step(prompt, model, **kwargs)
        # from_draft always false for non-speculative generation
        token_generator = (
            (token, logprobs, False) for token, logprobs in token_generator
        )
    else:
        if mtp:
            mtp_telemetry["mtp_bypass_reason"] = "native_mtp_external_draft_unsupported"
        kwargs.pop("max_kv_size", None)
        kwargs.pop("prompt_progress_callback", None)
        token_generator = speculative_generate_step(
            prompt, model, draft_model, **kwargs
        )
    with wired_limit(model, [generation_stream]):
        tic = time.perf_counter()
        token = 0
        logprobs = mx.array([])
        from_draft = False
        prompt_tps = 0.0
        n = -1
        try:
            for n, (token, logprobs, from_draft) in enumerate(token_generator):
                if n == 0:
                    prompt_time = time.perf_counter() - tic
                    prompt_tps = (
                        0.0
                        if sparse_bootstrap is not None
                        else prompt_tokens / prompt_time
                    )
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
                    prompt_tokens=prompt_tokens,
                    prompt_tps=prompt_tps,
                    generation_tokens=n + 1,
                    generation_tps=(n + 1) / (time.perf_counter() - tic),
                    peak_memory=mx.get_peak_memory() / 1e9,
                    finish_reason=None,
                    mtp_drafts=mtp_telemetry["mtp_drafts"],
                    mtp_accepted=mtp_telemetry["mtp_accepted"],
                    mtp_bypass_reason=mtp_telemetry["mtp_bypass_reason"],
                )
        finally:
            close = getattr(token_generator, "close", None)
            if close is not None:
                close()

        detokenizer.finalize()
        yield GenerationResponse(
            text=detokenizer.last_segment,
            token=token,
            logprobs=logprobs,
            from_draft=from_draft,
            prompt_tokens=prompt_tokens,
            prompt_tps=prompt_tps,
            generation_tokens=n + 1,
            generation_tps=(n + 1) / (time.perf_counter() - tic),
            peak_memory=mx.get_peak_memory() / 1e9,
            finish_reason="stop" if token in tokenizer.eos_token_ids else "length",
            mtp_drafts=mtp_telemetry["mtp_drafts"],
            mtp_accepted=mtp_telemetry["mtp_accepted"],
            mtp_bypass_reason=mtp_telemetry["mtp_bypass_reason"],
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


def _right_pad_prompts(prompts, max_length=None):
    if max_length is None:
        max_length = max(len(p) for p in prompts)
    return mx.array([p + [0] * (max_length - len(p)) for p in prompts])


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


def _make_cache(model, left_padding, max_kv_size):
    """
    Convert a list of regular caches into their corresponding
    batch-aware caches.
    """

    def to_batch_cache(c):
        if type(c) is KVCache:
            return BatchKVCache(left_padding)
        elif isinstance(c, ArraysCache):
            c.left_padding = mx.array(left_padding)
            return c
        elif isinstance(c, RotatingKVCache):
            if c.keep > 0:
                raise ValueError("RotatingKVCache with keep tokens is not supported.")
            return BatchRotatingKVCache(c.max_size, left_padding)
        elif isinstance(c, CacheList):
            return CacheList(*(to_batch_cache(sub_c) for sub_c in c.caches))
        else:
            raise ValueError(f"{type(c)} does not yet support batching")

    if hasattr(model, "make_cache"):
        cache = model.make_cache()
        return [to_batch_cache(c) for c in cache]
    else:
        if max_kv_size is not None:
            return [
                BatchRotatingKVCache(max_kv_size, left_padding) for _ in model.layers
            ]
        return [BatchKVCache(left_padding) for _ in model.layers]


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


def _step_trie(node, trie, x):
    """One step in the Aho-Corasick trie."""
    while x not in node and node is not trie:
        node = node["__fail__"]
    if x in node:
        node = node[x]
    return node


class StopSequenceMatcher:
    """Detect stop sequences in a stream of tokens using an Aho-Corasick trie.

    Any matched sequence signals stop. Used by the batch generator for EOS and
    stop word detection.
    """

    def __init__(self, stop_sequences=None):
        self._trie = _build_trie(stop_sequences) if stop_sequences else {}

    def __deepcopy__(self, memo):
        new = object.__new__(StopSequenceMatcher)
        new._trie = self._trie
        return new

    def make_state(self):
        return self._trie

    @staticmethod
    def match(state, trie, x):
        """Advance by one token. Returns (new_state, matched)."""
        node = _step_trie(state, trie, x)
        return node, node.get("__match__") is not None


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
        s, n, states, buf = state
        trie = states[s][0] if s is not None else None
        return (s, trie, states, ""), buf, s

    @staticmethod
    def discard(state):
        """Drop the remaining buffer (use on finish_reason="stop")."""
        s, n, states, buf = state
        trie = states[s][0] if s is not None else None
        return (s, trie, states, ""), s


def make_stop_matcher(tokenizer, stop_words=None):
    """Build a StopSequenceMatcher from EOS tokens and stop words."""
    stop_sequences = [(t,) for t in tokenizer.eos_token_ids]
    for w in stop_words or []:
        stop_sequences.append(tuple(tokenizer.encode(w, add_special_tokens=False)))
    return StopSequenceMatcher(stop_sequences)


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
        samplers: Optional[List[Callable[[mx.array], mx.array]]] = None,
        fallback_sampler: Optional[Callable[[mx.array], mx.array]] = None,
        logits_processors: Optional[
            List[List[Callable[[mx.array, mx.array], mx.array]]]
        ] = None,
        stop_matchers: Optional[List[StopSequenceMatcher]] = None,
        max_tokens: Optional[List[int]] = None,
    ):
        self.model = model
        self.uids = uids
        self.prompt_cache = _merge_caches(caches)
        self.tokens = tokens if tokens is not None else [[] for _ in uids]

        self.prefill_step_size = prefill_step_size
        self.samplers = samplers if samplers is not None else []
        self.fallback_sampler = fallback_sampler or (lambda x: mx.argmax(x, axis=-1))
        self.logits_processors = (
            logits_processors if logits_processors is not None else []
        )
        self.stop_matchers = (
            stop_matchers
            if stop_matchers is not None
            else [StopSequenceMatcher()] * len(uids)
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
        if not any(self.samplers):
            self.samplers = [None] * len(self.uids)
        if not any(self.logits_processors):
            self.logits_processors = [None] * len(self.uids)
        samplers = batch.samplers if any(batch.samplers) else [None] * len(batch.uids)
        logits_processors = (
            batch.logits_processors
            if any(batch.logits_processors)
            else [None] * len(batch.uids)
        )

        self.uids.extend(batch.uids)
        self.prompt_cache = _extend_cache(self.prompt_cache, batch.prompt_cache)
        self.tokens.extend(batch.tokens)
        self.samplers.extend(samplers)
        self.logits_processors.extend(logits_processors)
        self.max_tokens.extend(batch.max_tokens)
        self.stop_matchers.extend(batch.stop_matchers)

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
        new_batch.stop_matchers = list(self.stop_matchers)
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
        if any(self.samplers):
            self.samplers = [self.samplers[idx] for idx in keep]
        else:
            self.samplers = [None] * len(keep)
        if any(self.logits_processors):
            self.logits_processors = [self.logits_processors[idx] for idx in keep]
        else:
            self.logits_processors = [[]] * len(keep)
        self.max_tokens = [self.max_tokens[idx] for idx in keep]
        self.stop_matchers = [self.stop_matchers[idx] for idx in keep]

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
            self.stop_matchers,
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
        fallback_sampler: Callable[[mx.array], mx.array],
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
            stop_matchers=[],
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
        samplers: Optional[List[Callable[[mx.array], mx.array]]],
        fallback_sampler: Callable[[mx.array], mx.array],
        logits_processors: Optional[
            List[List[Callable[[mx.array, mx.array], mx.array]]]
        ],
        stop_matchers: List[StopSequenceMatcher],
        max_tokens: List[int],
    ):
        self.model = model
        self.uids = uids
        self.prompt_cache = prompt_cache
        self.tokens = tokens

        self.samplers = samplers
        self.fallback_sampler = fallback_sampler
        self.logits_processors = logits_processors
        self.stop_matchers = stop_matchers
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
        self._matcher_states = [m.make_state() for m in stop_matchers]

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
        self.stop_matchers.extend(batch.stop_matchers)
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
        self._matcher_states.extend(batch._matcher_states)

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
            # Update the token context that will be used by the logits processors
            token_context = [
                tc.update_and_fetch(inputs[i : i + 1])
                for i, tc in enumerate(self._token_context)
            ]
            processed_logits = []
            for e in range(len(self.uids)):
                sample_logits = logits[e : e + 1]
                for processor in self.logits_processors[e]:
                    sample_logits = processor(token_context[e], sample_logits)
                processed_logits.append(sample_logits)
            logits = mx.concatenate(processed_logits, axis=0)

        # Normalize the logits
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

        # Sample
        if any(self.samplers):
            all_samples = []
            for e in range(len(self.uids)):
                sample_sampler = self.samplers[e] or self.fallback_sampler
                sampled = sample_sampler(logprobs[e : e + 1])
                all_samples.append(sampled)
            sampled = mx.concatenate(all_samples, axis=0)
        else:
            sampled = self.fallback_sampler(logprobs)

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
        if any(self.samplers):
            self.samplers = [self.samplers[idx] for idx in keep]
        if any(self.logits_processors):
            self.logits_processors = [self.logits_processors[idx] for idx in keep]
        self.max_tokens = [self.max_tokens[idx] for idx in keep]
        self.stop_matchers = [self.stop_matchers[idx] for idx in keep]

        self._next_tokens = self._next_tokens[keep] if keep else None
        self._next_logprobs = [self._next_logprobs[idx] for idx in keep]
        self._token_context = [self._token_context[idx] for idx in keep]
        self._num_tokens = [self._num_tokens[idx] for idx in keep]
        self._matcher_states = [self._matcher_states[idx] for idx in keep]

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

            self._matcher_states[i], matched = StopSequenceMatcher.match(
                self._matcher_states[i],
                self.stop_matchers[i]._trie,
                tokens[i],
            )
            if matched:
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
        fallback_sampler: Callable[[mx.array], mx.array],
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
            stop_matchers=[],
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
        sampler: Optional[Callable[[mx.array], mx.array]] = None,
        logits_processors: Optional[
            List[Callable[[mx.array, mx.array], mx.array]]
        ] = None,
        completion_batch_size: int = 32,
        prefill_batch_size: int = 8,
        prefill_step_size: int = 2048,
        max_kv_size: Optional[int] = None,
        stream=None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.sampler = sampler or (lambda x: mx.argmax(x, axis=-1))
        self.logits_processors = logits_processors or []
        self.uid_count = 0
        self.prefill_step_size = prefill_step_size
        self.prefill_batch_size = prefill_batch_size
        self.completion_batch_size = max(completion_batch_size, prefill_batch_size)
        self.max_kv_size = max_kv_size

        self._stream = stream or generation_stream

        self._default_stop_matcher = StopSequenceMatcher(
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

        self._prompt_tokens_counter = 0
        self._prompt_time_counter = 0
        self._gen_tokens_counter = 0
        self._steps_counter = 0

        if mx.metal.is_available():
            self._old_wired_limit = mx.set_wired_limit(
                mx.device_info()["max_recommended_working_set_size"]
            )
        else:
            self._old_wired_limit = None

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
    def stats(self, stats=None):
        stats = stats or BatchStats()
        self._prompt_tokens_counter = 0
        self._prompt_time_counter = 0
        self._gen_tokens_counter = 0
        tic = time.perf_counter()
        try:
            yield stats
        finally:
            toc = time.perf_counter()
            total_time = toc - tic
            gen_time = total_time - self._prompt_time_counter
            stats.prompt_tokens += self._prompt_tokens_counter
            stats.prompt_time += self._prompt_time_counter
            stats.prompt_tps = stats.prompt_tokens / stats.prompt_time
            stats.generation_tokens += self._gen_tokens_counter
            stats.generation_time += gen_time
            stats.generation_tps = stats.generation_tokens / stats.generation_time
            stats.peak_memory = max(stats.peak_memory, mx.get_peak_memory() / 1e9)

    def insert(
        self,
        prompts: List[List[int]],
        max_tokens: Optional[List[int]] = None,
        caches: Optional[List[List[Any]]] = None,
        all_tokens: Optional[List[List[int]]] = None,
        samplers: Optional[List[Callable[[mx.array], mx.array]]] = None,
        logits_processors: Optional[
            List[List[Callable[[mx.array, mx.array], mx.array]]]
        ] = None,
        stop_matchers: Optional[List[StopSequenceMatcher]] = None,
    ):
        return self.insert_segments(
            [[p] for p in prompts],
            max_tokens,
            caches,
            all_tokens,
            samplers,
            logits_processors,
            stop_matchers,
        )

    def insert_segments(
        self,
        segments: List[List[List[int]]],
        max_tokens: Optional[List[int]] = None,
        caches: Optional[List[List[Any]]] = None,
        all_tokens: Optional[List[List[int]]] = None,
        samplers: Optional[List[Callable[[mx.array], mx.array]]] = None,
        logits_processors: Optional[
            List[List[Callable[[mx.array, mx.array], mx.array]]]
        ] = None,
        stop_matchers: Optional[List[StopSequenceMatcher]] = None,
    ):
        uids = []

        max_tokens = max_tokens or [self.max_tokens] * len(segments)
        all_tokens = all_tokens or [[] for _ in segments]
        samplers = samplers or [None] * len(segments)
        logits_processors = logits_processors or (
            [self.logits_processors] * len(segments)
        )
        stop_matchers = stop_matchers or ([self._default_stop_matcher] * len(segments))

        caches = caches or [None] * len(segments)
        for i in range(len(segments)):
            if caches[i] is None:
                caches[i] = self._make_new_cache()

        for seq, m, c, at, s, lp, sm in zip(
            segments,
            max_tokens,
            caches,
            all_tokens,
            samplers,
            logits_processors,
            stop_matchers,
        ):
            seq = list(seq)
            if len(seq[-1]) != 1:
                seq.append(seq[-1][-1:])
                seq[-2] = seq[-2][:-1]
            self._unprocessed_sequences.append(
                (self._uid_count, seq, m, c, at, s, lp, sm)
            )
            uids.append(self._uid_count)
            self._uid_count += 1

        return uids

    def _make_new_cache(self):
        if self.max_kv_size is None:
            return cache.make_prompt_cache(self.model)

        return [
            (
                RotatingKVCache(max_size=self.max_kv_size)
                if isinstance(ci, KVCache)
                else ci
            )
            for ci in cache.make_prompt_cache(self.model)
        ]

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
        stop_matchers = []
        for _ in range(n):
            sequence = self._unprocessed_sequences.popleft()
            uids.append(sequence[0])
            caches.append(sequence[3])
            tokens.append(sequence[4])
            samplers.append(sequence[5])
            logits_processors.append(sequence[6])
            max_tokens.append(sequence[2])
            stop_matchers.append(sequence[7])
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
            stop_matchers=stop_matchers,
            max_tokens=max_tokens,
        )

    def _next(self):
        generation_responses = []
        prompt_responses = []

        # Generate tokens first
        if len(self._generation_batch) > 0:
            generation_responses = self._generation_batch.next()
            self._gen_tokens_counter += len(generation_responses)
            self._steps_counter += 1
            if self._steps_counter % 512 == 0:
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
        self._prompt_tokens_counter += sum(len(p) for p in prompts)
        tic = time.perf_counter()
        self._prompt_batch.prompt(prompts)
        toc = time.perf_counter()
        self._prompt_time_counter += toc - tic

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
