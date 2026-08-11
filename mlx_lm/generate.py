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
    Generator,
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
    logical_positions: Optional[Tuple[int, ...]] = None
    logical_position_ack: Optional["GenerationForwardPositionAck"] = None


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
    logits: mx.array
    logits_evidence: tuple
    canonical_final_logits: Optional[mx.array]
    hidden_rows: mx.array
    hidden_evidence: tuple
    canonical_hidden_rows: mx.array


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
    logits: mx.array
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
        logical_positions: Tuple[int, ...],
        *,
        model: Optional[nn.Module] = None,
        cache: Optional[Any] = None,
        token_ids: Optional[Tuple[int, ...]] = None,
        immediate_successor_token_ids: Tuple[int, ...] = (),
        phase: Optional[GenerationForwardPhase] = None,
        _receipt_issuer=None,
    ):
        self._logical_positions = logical_positions
        self._model = model
        self._cache = cache
        self._token_ids = token_ids
        self._immediate_successor_token_ids = immediate_successor_token_ids
        self._phase = phase
        self._receipt_issuer = _receipt_issuer
        self._active = False
        self._acknowledged = False
        self._finished = False
        self._receipt = None

    @property
    def receipt(self) -> Optional[GenerationForwardPositionReceipt]:
        return self._receipt

    def acknowledge(self, logical_positions: Tuple[int, ...]) -> None:
        if not self._active or self._finished:
            raise RuntimeError("generation_logical_position_ack_outside_forward")
        if self._acknowledged:
            raise RuntimeError("generation_logical_position_ack_reused")
        if not isinstance(logical_positions, tuple) or (
            logical_positions != self._logical_positions
        ):
            raise RuntimeError("generation_logical_position_ack_mismatch")
        self._acknowledged = True

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
        canonical_hidden_rows = hidden_rows + mx.zeros(
            hidden_rows.shape, dtype=hidden_rows.dtype
        )
        canonical_final_logits = None
        cache_content_digest = None
        if len(self._immediate_successor_token_ids) == len(self._logical_positions) - 1:
            final_logits = logits[:, -1:, :]
            canonical_final_logits = final_logits + mx.zeros(
                final_logits.shape, dtype=final_logits.dtype
            )
            cache_content_digest = _cache_content_digest(self._cache)
        canonical_values = [canonical_hidden_rows]
        if canonical_final_logits is not None:
            canonical_values.append(canonical_final_logits)
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
            logits=logits,
            hidden_rows=hidden_rows,
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
            logits=logits,
            logits_evidence=_array_identity_evidence(logits),
            canonical_final_logits=canonical_final_logits,
            hidden_rows=hidden_rows,
            hidden_evidence=_array_identity_evidence(hidden_rows),
            canonical_hidden_rows=canonical_hidden_rows,
        )
        with _GENERATION_FORWARD_RECEIPT_LOCK:
            _GENERATION_FORWARD_RECEIPTS[record_token] = _GenerationForwardAuthority(
                record
            )
        return self._receipt


GenerationForwardContext = Callable[[GenerationForward], ContextManager[None]]


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
        mx.array_equal(record.hidden_rows, record.canonical_hidden_rows)
        for record in records
    ]
    final = records[-1]
    checks.extend(
        (
            mx.array_equal(final.logits[:, -1:, :], final.canonical_final_logits),
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
        return self.records[-1].canonical_hidden_rows[:, -1:, :]

    @property
    def final_target_logits(self):
        return self.records[-1].canonical_final_logits


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
        if canonical_records[-1].canonical_final_logits is None:
            raise RuntimeError("native_mtp_sparse_final_logits_evidence_missing")
        if canonical_records[-1].cache_content_digest is None:
            raise RuntimeError("native_mtp_sparse_final_cache_evidence_missing")
        if any(
            record.canonical_final_logits is not None
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
            if (
                not isinstance(logits, mx.array)
                or logits.ndim != 3
                or logits.shape[0] != 1
                or logits.shape[1] != chunk_size
                or logits.dtype not in floating_dtypes
                or (vocab_size is not None and logits.shape[2] != vocab_size)
            ):
                raise ValueError("native_mtp_sparse_logits_shape_or_dtype_mismatch")

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
    coordinates are immutable host metadata only: every forward must be
    acknowledged by the request-local context consumer that applies them.
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
        if model_forward_context is None:
            raise ValueError("native_mtp_sparse_bootstrap_requires_position_context")
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
        if model_forward_context is None:
            raise ValueError("native_mtp_logical_positions_require_forward_context")
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

    request = None
    sparse_claim = None

    def _next_rng_key():
        nonlocal rng_key
        keys = mx.random.split(rng_key)
        rng_key = keys[0]
        return keys[1]

    def _sample(logprobs):
        if sampling_config.stochastic:
            return mx.random.categorical(logprobs, key=_next_rng_key())
        if sampler is None:
            return mx.argmax(logprobs, axis=-1)
        return sampler(logprobs)

    def _reported_logprobs(logits, tokens):
        if logits.ndim == 1:
            logits = logits[None]
        if logits_processors:
            for processor in logits_processors:
                logits = processor(tokens, logits)
        return (logits - mx.logsumexp(logits, axis=-1, keepdims=True)).squeeze(0)

    def _sampling_distribution(reported_logprobs):
        logprobs = reported_logprobs[None]
        if sampling_config.top_p < 1:
            logprobs = apply_top_p(logprobs, sampling_config.top_p)
        if sampling_config.min_p > 0:
            keep = sampling_config.min_tokens_to_keep
            vocab_size = logprobs.shape[-1]
            if keep > vocab_size:
                raise ValueError(
                    "native MTP min_tokens_to_keep cannot exceed vocabulary size"
                )
            if keep < vocab_size:
                # apply_min_p's multi-token branch passes a Python bool to
                # put_along_axis, which newer MLX releases reject.  Preserve
                # the standard filter exactly by restoring the requested top
                # tokens with array-valued replacements.
                unfiltered = logprobs
                logprobs = apply_min_p(logprobs, sampling_config.min_p)
                if keep > 1:
                    top_indices = mx.argpartition(unfiltered, kth=-keep, axis=-1)[
                        ..., -keep:
                    ]
                    top_values = mx.take_along_axis(unfiltered, top_indices, axis=-1)
                    logprobs = mx.put_along_axis(
                        logprobs, top_indices, top_values, axis=-1
                    )
        if sampling_config.top_k > 0:
            if sampling_config.top_k >= logprobs.shape[-1]:
                raise ValueError(
                    "native MTP top_k must be smaller than vocabulary size"
                )
            logprobs = apply_top_k(logprobs, sampling_config.top_k)
        if sampling_config.stochastic:
            logprobs = logprobs / sampling_config.temperature
        return (logprobs - mx.logsumexp(logprobs, axis=-1, keepdims=True)).squeeze(0)

    def _target_call(
        input_tokens,
        phase,
        *,
        return_hidden=False,
        logical_positions=None,
        immediate_successor_token_ids=(),
    ):
        batched = input_tokens[None]
        if model_forward_context is None:
            return model(batched, cache=request.backbone, return_hidden=return_hidden)
        position_ack = (
            GenerationForwardPositionAck(logical_positions)
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
        with model_forward_context(forward):
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
        if model_forward_context is None:
            return model.mtp_forward(hidden, next_ids, request.mtp)
        position_ack = (
            GenerationForwardPositionAck(logical_positions)
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
        with model_forward_context(forward):
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
            draft_reported_logprobs = _reported_logprobs(
                logits.squeeze(0), token_history
            )
            draft_logprobs = _sampling_distribution(draft_reported_logprobs)
            draft_token = _sample(draft_logprobs)
        mx.eval(draft_token, draft_logprobs)
        if telemetry is not None:
            telemetry["mtp_drafts"] += 1
        return draft_token.reshape(-1), draft_logprobs

    def _residual_sample(target_logprobs, draft_logprobs):
        target_probs = mx.exp(target_logprobs)
        draft_probs = mx.exp(draft_logprobs)
        residual = mx.maximum(target_probs - draft_probs, 0)
        normalizer = mx.sum(residual)
        residual_logprobs = mx.where(
            normalizer > 0,
            mx.log(residual / normalizer),
            target_logprobs,
        )
        return _sample(residual_logprobs), residual_logprobs

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
        rng_key = mx.random.key(
            sampling_config.seed
            if sampling_config.seed is not None
            else (time.time_ns() & ((1 << 63) - 1))
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
                current_logprobs = _reported_logprobs(
                    initial_logits[:, -1, :].squeeze(0), history
                )
                current_sampling_logprobs = _sampling_distribution(current_logprobs)
                current_token = _sample(current_sampling_logprobs).reshape(-1)
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
                            record.canonical_hidden_rows[:, pair_start:pair_end, :],
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
                    tuple(
                        record.canonical_hidden_rows for record in sparse_claim.records
                    ),
                    sparse_claim.final_target_logits,
                    [entry.state for entry in request.backbone],
                    [entry.state for entry in request.mtp],
                )
                history = mx.array(
                    [sparse_claim.selected_token_ids[-1]], dtype=mx.uint32
                )
                current_logprobs = _reported_logprobs(
                    sparse_claim.final_target_logits[:, -1, :].squeeze(0),
                    history,
                )
                current_sampling_logprobs = _sampling_distribution(current_logprobs)
                current_token = _sample(current_sampling_logprobs).reshape(-1)
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
                verify_logprobs = _reported_logprobs(
                    verify_logits[:, 0, :].squeeze(0), history
                )
                verify_sampling_logprobs = _sampling_distribution(verify_logprobs)

            if sampling_config.stochastic:
                draft_id = draft_token.item()
                acceptance = mx.minimum(
                    mx.exp(
                        verify_sampling_logprobs[draft_id] - draft_logprobs[draft_id]
                    ),
                    1.0,
                )
                accepted = mx.random.uniform(key=_next_rng_key()) < acceptance
                mx.eval(accepted)
                accepted = bool(accepted.item())
                replacement_token = replacement_logprobs = None
                if not accepted:
                    replacement_token, replacement_logprobs = _residual_sample(
                        verify_sampling_logprobs, draft_logprobs
                    )
                    replacement_token = replacement_token.reshape(-1)
                    mx.eval(replacement_token, replacement_logprobs)
            else:
                replacement_token = _sample(verify_sampling_logprobs).reshape(-1)
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
                bonus_logprobs = _reported_logprobs(
                    verify_logits[:, 1, :].squeeze(0), bonus_history
                )
                bonus_sampling_logprobs = _sampling_distribution(bonus_logprobs)
                bonus_token = _sample(bonus_sampling_logprobs).reshape(-1)
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
        if model_forward_context is None:
            raise ValueError("native_mtp_sparse_bootstrap_requires_position_context")
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

    prompt_tokens = prompt.size if sparse_prompt_tokens is None else sparse_prompt_tokens

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
