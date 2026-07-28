# Copyright © 2023-2024 Apple Inc.

import copy
import operator
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map, tree_reduce, tree_unflatten

from .base import create_causal_mask


def make_prompt_cache(
    model: nn.Module,
    max_kv_size: Optional[int] = None,
) -> List[Any]:
    """
    Construct the model's cache for use in generation.

    This function will defer the cache construction to the model if it has a
    ``make_cache`` method, otherwise it will make a default KV cache.

    Args:
        model (nn.Module): The language model.
        max_kv_size (Optional[int]): If provided and the model does not have a
            ``make_cache`` method, a ``RotatingKVCache`` is used with a maximum
            size of ``max_kv_size``
    """
    if hasattr(model, "make_cache"):
        return model.make_cache()

    num_layers = len(model.layers)
    if max_kv_size is not None:
        return [
            RotatingKVCache(max_size=max_kv_size, keep=4) for _ in range(num_layers)
        ]
    else:
        return [KVCache() for _ in range(num_layers)]


def save_prompt_cache(file_name: str, cache: List[Any], metadata: Dict[str, str] = {}):
    """
    Save a pre-computed prompt cache to a file.

    Args:
        file_name (str): The ``.safetensors`` file name.
        cache (List[Any]): The model state.
        metadata (Dict[str, str]): Optional metadata to save along with model
            state.
    """
    cache_data = [c.state for c in cache]
    cache_info = [c.meta_state for c in cache]
    cache_data = dict(tree_flatten(cache_data))
    cache_classes = [type(c).__name__ for c in cache]
    cache_metadata = [cache_info, metadata, cache_classes]
    cache_metadata = dict(tree_flatten(cache_metadata))
    mx.save_safetensors(file_name, cache_data, cache_metadata)


def load_prompt_cache(file_name, return_metadata=False):
    """
    Load a prompt cache from a file.

    Args:
        file_name (str): The ``.safetensors`` file name.
        return_metadata (bool): Whether or not to return metadata.
            Default: ``False``.

    Returns:
        List[Any] or Tuple[List[Any], Dict[str, str]]: The prompt cache and
            the metadata if requested.
    """
    arrays, cache_metadata = mx.load(file_name, return_metadata=True)
    arrays = tree_unflatten(list(arrays.items()))
    cache_metadata = tree_unflatten(list(cache_metadata.items()))
    info, metadata, classes = cache_metadata
    cache = [
        globals()[c].from_state(state, meta_state)
        for c, state, meta_state in zip(classes, arrays, info)
    ]
    if return_metadata:
        return cache, metadata
    return cache


def can_trim_prompt_cache(cache: List[Any]) -> bool:
    """
    Check if model's cache can be trimmed.
    """
    return all(c.is_trimmable() for c in cache)


def trim_prompt_cache(cache: List[Any], num_tokens: int) -> List[Any]:
    """
    Trim the model's cache by the given number of tokens.

    This function will trim the cache if possible (in-place) and return the
    number of tokens that were trimmed.

    Args:
        cache (List[Any]): The model's cache.
        num_tokens (int): The number of tokens to trim.

    Returns:
        (int): The number of tokens that were trimmed.
    """
    if not can_trim_prompt_cache(cache) or len(cache) == 0:
        return 0
    return [c.trim(num_tokens) for c in cache][0]


def create_attention_mask(
    N: int, offset: int, return_array: bool, window_size: Optional[int]
):
    if window_size is not None:
        return create_causal_mask(N, offset, window_size=window_size)
    elif N == 1:
        return None
    elif return_array:
        return create_causal_mask(N, offset, window_size=window_size)
    else:
        return "causal"


class _BaseCache:
    @property
    def state(self):
        return []

    @state.setter
    def state(self, v):
        if v is not None and v:
            raise ValueError("This cache has no state but a state was set.")

    @property
    def meta_state(self):
        return ""

    @meta_state.setter
    def meta_state(self, v):
        if v is not None and v:
            raise ValueError("This cache has no meta_state but a meta_state was set.")

    def is_trimmable(self):
        return False

    def size(self):
        """
        Return the size (i.e. sequence length) of the cache.

        Not every cache is required to implement this, in which case the size
        will always be 0 (though the cache may not be empty).
        """
        return 0

    @property
    def nbytes(self):
        """Return the size of this cache in bytes"""
        raise NotImplementedError("Cache sub-class must implement nbytes")

    def empty(self):
        """
        Return if the cache is empty or not.
        """
        raise NotImplementedError("Cache sub-class must implement this.")

    @classmethod
    def from_state(cls, state, meta_state):
        # Create an instance of cls without calling __init__
        obj = cls.__new__(cls)
        obj.state = state
        obj.meta_state = meta_state
        return obj


class ConcatenateKVCache(_BaseCache):
    """ConcatenateKVCache the simplest KV cache implementation.

    Can be used as a mock KV cache or when large blocks are being processed at
    a time in which case KVCache isn't necessarily faster. Consider using the
    KVCache with a larger step size before using this cache.
    """

    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(self, keys, values):
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=-2)
            self.values = mx.concatenate([self.values, values], axis=-2)
        self.offset = self.keys.shape[-2]

        return self.keys, self.values

    @property
    def state(self):
        return self.keys, self.values

    @state.setter
    def state(self, v):
        self.keys, self.values = v
        self.offset = self.keys.shape[-2]

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes


class QuantizedKVCache(_BaseCache):
    step = 256

    def __init__(self, group_size: int = 64, bits: int = 8):
        self.keys = None
        self.values = None
        self.offset = 0
        self.group_size = group_size
        self.bits = bits

    def update_and_fetch(self, keys, values):
        B, n_kv_heads, num_steps, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        prev = self.offset

        if self.keys is None or (prev + num_steps) > self.keys[0].shape[-2]:
            el_per_int = 8 * mx.uint32.size // self.bits
            new_steps = (self.step + num_steps - 1) // self.step * self.step
            shape = (B, n_kv_heads, new_steps)

            def init_quant(dim):
                return (
                    mx.zeros((*shape, dim // el_per_int), dtype=mx.uint32),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                )

            def expand_quant(x):
                new_x = mx.zeros((*shape, x.shape[-1]), dtype=x.dtype)
                return mx.concatenate([x, new_x], axis=-2)

            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys, self.values = tree_map(
                        lambda x: x[..., :prev, :], (self.keys, self.values)
                    )

                self.keys, self.values = tree_map(
                    expand_quant, (self.keys, self.values)
                )
            else:
                self.keys, self.values = init_quant(k_head_dim), init_quant(v_head_dim)

        self.offset += num_steps

        keys = mx.quantize(keys, group_size=self.group_size, bits=self.bits)
        values = mx.quantize(values, group_size=self.group_size, bits=self.bits)
        for i in range(len(self.keys)):
            self.keys[i][..., prev : self.offset, :] = keys[i]
            self.values[i][..., prev : self.offset, :] = values[i]

        return tree_map(lambda x: x[..., : self.offset, :], (self.keys, self.values))

    @property
    def state(self):
        if self.offset == self.keys[0].shape[2]:
            return self.keys, self.values
        else:
            return tree_map(
                lambda x: x[..., : self.offset, :], (self.keys, self.values)
            )

    @state.setter
    def state(self, v):
        self.keys, self.values = v

    @property
    def meta_state(self):
        return tuple(map(str, (self.offset, self.group_size, self.bits)))

    @meta_state.setter
    def meta_state(self, v):
        self.offset, self.group_size, self.bits = map(int, v)

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        return tree_reduce(lambda a, x: a + x.nbytes, (self.keys, self.values), 0)


class KVCache(_BaseCache):
    step = 256

    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(self, keys, values):
        prev = self.offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    def size(self):
        return self.offset

    @property
    def state(self):
        if self.offset == self.keys.shape[2]:
            return self.keys, self.values
        else:
            return (
                self.keys[..., : self.offset, :],
                self.values[..., : self.offset, :],
            )

    @state.setter
    def state(self, v):
        self.keys, self.values = v
        self.offset = self.keys.shape[2]

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def to_quantized(self, group_size: int = 64, bits: int = 4) -> QuantizedKVCache:
        quant_cache = QuantizedKVCache(group_size=group_size, bits=bits)
        quant_cache.offset = self.offset
        if self.keys is not None:
            quant_cache.keys = mx.quantize(self.keys, group_size=group_size, bits=bits)
            quant_cache.values = mx.quantize(
                self.values, group_size=group_size, bits=bits
            )
        return quant_cache

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    @classmethod
    def merge(_, caches):
        return BatchKVCache.merge(caches)

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes


class RotatingKVCache(_BaseCache):
    step = 256

    def __init__(self, max_size, keep=0):
        self.keep = keep
        self.keys = None
        self.values = None
        self.offset = 0
        self.max_size = max_size
        self._idx = 0

    def _trim(self, trim_size, v, append=None):
        to_cat = []
        if trim_size > 0:
            to_cat = [v[..., : self.keep, :], v[..., trim_size + self.keep :, :]]
        else:
            to_cat = [v]
        if append is not None:
            to_cat.append(append)
        return mx.concatenate(to_cat, axis=2)

    def _temporal_order(self, v):
        """
        Rearrange the cache into temporal order, slicing off the end if unused.
        """
        if self._idx == v.shape[2]:
            return v
        elif self._idx < self.offset:
            return mx.concatenate(
                [
                    v[..., : self.keep, :],
                    v[..., self._idx :, :],
                    v[..., self.keep : self._idx, :],
                ],
                axis=2,
            )
        else:
            return v[..., : self._idx, :]

    def _update_concat(self, keys, values):
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            # Put the keys/values in temporal order to
            # preserve context
            self.keys = self._temporal_order(self.keys)
            self.values = self._temporal_order(self.values)
            self._idx = self.keys.shape[2]

            # The largest size is self.max_size + S - 1 to ensure
            # every token gets at least self.max_size context
            trim_size = self._idx - self.max_size + 1
            self.keys = self._trim(trim_size, self.keys, keys)
            self.values = self._trim(trim_size, self.values, values)
        self.offset += keys.shape[2]
        self._idx = self.keys.shape[2]
        return self.keys, self.values

    def _update_in_place(self, keys, values):
        # May not have hit the max size yet, so potentially
        # keep growing the cache
        B, n_kv_heads, S, k_head_dim = keys.shape
        prev = self.offset
        if self.keys is None or (
            prev >= self.keys.shape[2] and self.keys.shape[2] < self.max_size
        ):
            v_head_dim = values.shape[3]
            new_size = min(self.step, self.max_size - prev)
            k_shape = (B, n_kv_heads, new_size, k_head_dim)
            v_shape = (B, n_kv_heads, new_size, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v
            self._idx = prev

        # Trim if needed
        trim_size = self.keys.shape[2] - self.max_size
        if trim_size > 0:
            self.keys = self._trim(trim_size, self.keys)
            self.values = self._trim(trim_size, self.values)
            self._idx = self.max_size

        # Rotate
        if self._idx == self.max_size:
            self._idx = self.keep

        # Assign
        self.keys[..., self._idx : self._idx + S, :] = keys
        self.values[..., self._idx : self._idx + S, :] = values
        self.offset += S
        self._idx += S

        # If the buffer is not full, slice off the end
        if self.offset < self.max_size:
            return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
        return self.keys, self.values

    def update_and_fetch(self, keys, values):
        if keys.shape[2] == 1:
            return self._update_in_place(keys, values)
        return self._update_concat(keys, values)

    def size(self):
        return min(self.offset, self.max_size)

    @property
    def state(self):
        if self.offset < self.keys.shape[2]:
            return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
        else:
            return self.keys, self.values

    @state.setter
    def state(self, v):
        self.keys, self.values = v

    @property
    def meta_state(self):
        return tuple(map(str, (self.keep, self.max_size, self.offset, self._idx)))

    @meta_state.setter
    def meta_state(self, v):
        self.keep, self.max_size, self.offset, self._idx = map(
            int,
            v,
        )

    def is_trimmable(self):
        return self.offset < self.max_size

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        self._idx -= n
        return n

    def to_quantized(self, group_size: int = 64, bits: int = 4) -> QuantizedKVCache:
        raise NotImplementedError("RotatingKVCache Quantization NYI")

    def make_mask(
        self, N: int, window_size: Optional[int] = None, return_array: bool = False
    ):
        if N > 1:
            window_size = window_size or self.max_size
            offset = min(self.max_size - 1, self.offset)
            if offset + N > window_size or return_array:
                return create_causal_mask(N, offset, window_size=window_size)
            else:
                return "causal"
        else:
            if window_size is None:
                return None
            # May need a mask for when window_size < max_size
            if self.offset >= window_size and self.max_size > window_size:
                idx = self._idx
                if idx >= self.max_size:
                    idx = 0
                if self.offset < self.max_size:
                    mask_size = self.offset + 1
                else:
                    mask_size = self.max_size
                mask = mx.arange(mask_size) >= (mask_size - window_size)
                mask = mx.roll(mask, shift=idx + 1)
                return mask

    @classmethod
    def merge(_, caches):
        return BatchRotatingKVCache.merge(caches)

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes


def _try_schedule(*arrays):
    """Schedule an async evaluation unless it is disallowed: inside a
    graph transformation (``mx.compile`` traces captured state by
    temporarily swapping tracers into the captured containers, e.g.
    ``inputs=vars(cache)``) ``async_eval`` on a tracer raises, and a
    metadata access must instead behave like stock's plain attribute
    read -- return the tracer, schedule nothing, and leave any pending
    fold for the next eager access. Returns False in that case."""
    try:
        mx.async_eval(*arrays)
        return True
    except ValueError as e:
        if "graph transformation" not in str(e):
            raise
        return False


class ArraysCache(_BaseCache):
    # ``left_padding`` and ``lengths`` are logically decremented by
    # ``advance()`` on every layer call during decode. Doing that decrement
    # with mx.array arithmetic (``self.left_padding -= N``) builds one
    # unevaluated graph node per layer per token. The model only ever
    # rebuilds a mask from ONE of these caches (``cache[ssm_idx]``), so for
    # every other layer the chain is dead: it is never evaluated, it grows
    # without bound, and every link pins the small constant array (and its
    # live Metal buffer object) that was subtracted. The process then dies
    # with ``[metal::malloc] Resource limit (499000) exceeded`` -- a count
    # limit, not bytes -- after a few tens of thousands of decode tokens
    # (see ml-explore/mlx-lm#1185, #1332; ml-explore/mlx#3564, #3539).
    #
    # Fix: track the cumulative decrement as a plain python int per field
    # (``_lp_advance``/``_len_advance``; nonzero only while the matching
    # array exists) and fold it into the stored array when that field is
    # read (property access or ``make_mask``), replaced, or finalized.
    # ``advance()`` itself creates no graph nodes and never evaluates
    # anything; folding one field never touches the other. Folds are
    # applied in place (``-=``, preserving the array object) on the
    # metadata side and scheduled with ``mx.async_eval``, so no public
    # operation sequence accumulates unevaluated graph on metadata
    # nothing consumes.
    #
    # Deliberate deviations from the eager decrement (nothing in this
    # repo relies on them): a decrement becomes visible to aliases of a
    # metadata array at the next fold of that field, not at ``advance()``
    # time -- alias reads/writes before that fold interleave accordingly,
    # and storing the same array object in both fields applies each
    # field's decrement at its own fold; ``copy.copy`` folds first, so a
    # shallow copy cannot re-apply pending decrements to the shared
    # arrays. Deferred totals fold as the dtype-congruent scalar (MLX
    # converts python int scalars through int64, so uint64 uses the
    # signed representative modulo 2**64), which lands integer metadata
    # exactly where stock's per-step subtractions did -- including
    # totals that overflow the dtype. Offsets outside the scalar range
    # MLX accepts for the dtype raise a uniform ValueError at
    # ``advance()``, where stock's eager subtraction rejected them (as
    # ValueError, or an opaque std::bad_cast at the int64 gate).
    # Floating totals beyond the int64 gate fold in gate-sized chunks,
    # each scheduled as it is applied; per-step float *rounding* is not
    # reproduced. ``operator.index`` normalizes integral offsets
    # (python bool and numpy integer scalars become python ints, value
    # preserved) -- stock instead converted exotic offset types through
    # array promotion, whose dtype side effects (a bool offset keeping
    # bool metadata bool, numpy scalar offsets promoting the metadata
    # dtype) are not reproduced. bool metadata arithmetic is reproduced
    # at int32 precision (stock's promotion): the dtype stays bool
    # through zero-net advance histories, and after the first advance()
    # on the field -- even a zero or cancelling one, and through a bool
    # array shared by both fields -- later offsets validate against
    # int32 exactly as stock's already-promoted array would. A read that encounters a *tracer*
    # -- metadata swapped into a trace by
    # ``mx.compile(..., inputs=vars(cache))`` -- is pure: no fold, no
    # scheduling, like stock's attribute access, and the pending
    # decrement folds at the next eager access; a closure-captured
    # cache holds concrete arrays, so reads inside a trace fold eagerly
    # with stock-identical values. Compiling a function that *calls*
    # ``advance()`` is not supported: the decrement is host-side
    # bookkeeping and runs at trace time only (stock incidentally
    # recorded it as graph arithmetic -- the same arithmetic that
    # leaks), so call advance() outside compiled functions. Likewise,
    # mutating metadata (assignment, ``finalize``, ``filter``,
    # ``extend``) while a pending decrement cannot fold -- the field is
    # captured as a tracer -- raises rather than silently discarding
    # the decrement (``copy.copy`` and ``copy.deepcopy`` guard the same
    # way -- a copy taken mid-trace would escape holding the transient
    # tracer); mutate outside compiled functions. In-place item
    # assignment through a captured tracer read cannot be intercepted
    # (``__setitem__`` on the returned array bypasses the property
    # machinery): the write lands before the deferred decrement -- the
    # alias-write window above -- where stock's eager decrement landed
    # first. Reads are serialized per cache: a reentrant lock guards the
    # advance bookkeeping, folds, scheduling, and copies, because
    # stock's pure attribute reads were trivially thread-safe while
    # unserialized concurrent folds of one array deadlock on their
    # thread-local streams. Cross-thread *mutation* (filter/extend/
    # finalize racing anything) remains unsupported, as in stock.

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance._left_padding = None
        instance._lengths = None
        instance._lp_advance = 0
        instance._len_advance = 0
        # bool-metadata promotion trackers: stock's eager subtraction
        # promoted a bool field to int32 on the FIRST advance() -- even a
        # zero or cancelling one -- so validation must remember that
        # independently of the pending total (see _logical_dtype)
        instance._lp_promoted = False
        instance._len_promoted = False
        # Serializes the deferred-decrement machinery (advance
        # bookkeeping, folds, scheduling, copies). Stock's pure
        # attribute reads were trivially thread-safe; deferred folding
        # makes reads mutate, and two unserialized folds of the same
        # array from different threads deadlock on their thread-local
        # streams. Reentrant: mutators call _sync/_fold under it.
        instance._fold_lock = threading.RLock()
        return instance

    def __init__(self, size, left_padding: Optional[List[int]] = None):
        self.cache = [None] * size
        if left_padding:
            self.left_padding = mx.array(left_padding)

    # Integer metadata dtypes as (bits, signed): used to mirror stock's
    # eager per-call scalar range check in advance() and to wrap the
    # accumulated total to the dtype's modular range at fold time.
    _INT_DTYPES = {
        "int8": (8, True),
        "int16": (16, True),
        "int32": (32, True),
        "int64": (64, True),
        "uint8": (8, False),
        "uint16": (16, False),
        "uint32": (32, False),
        "uint64": (64, False),
    }

    @classmethod
    def _int_spec(cls, dtype):
        return cls._INT_DTYPES.get(str(dtype).split(".")[-1])

    @classmethod
    def _wrap_advance(cls, total, dtype):
        """Wrap an accumulated advance() total to the congruent scalar
        MLX converts exactly like stock's per-step decrements did.
        Sequential wrapping subtractions equal one subtraction of the
        sum modulo 2**bits, so the wrapped fold lands on stock's value
        even when the raw total overflows the dtype. Python int scalars
        convert through int64 (mlx python/src/utils.cpp), so uint64
        takes the *signed* int64 representative -- congruent modulo
        2**64 and accepted by the conversion. Non-integer dtypes pass
        through (the fold subtracts them in gate-sized chunks). bool
        metadata wraps at int32: stock's subtraction promoted it, and
        int32 wrapping matches the promoted arithmetic (including the
        two's-complement truncation MLX's C++ cast applies to
        out-of-range scalars)."""
        if dtype == mx.bool_:
            spec = (32, True)
        else:
            spec = cls._int_spec(dtype)
        if spec is None:
            return total
        bits, signed = spec
        total %= 1 << bits
        if (signed or bits == 64) and total >= 1 << (bits - 1):
            total -= 1 << bits
        return total

    @classmethod
    def _check_advance(cls, N, dtype):
        """Mirror stock's eager scalar conversion: python ints convert
        through int64, so every dtype rejects offsets outside int64's
        range, and integer dtypes narrower than 64 bits are additionally
        range-checked (uint64 is not: negatives wrap modulo 2**64).
        Runs per armed field, in stock's field order, because that is
        where stock's ``-= N`` raised (ValueError, or std::bad_cast at
        the int64 gate; here it is uniformly ValueError)."""
        lo, hi = -(1 << 63), (1 << 63) - 1
        spec = cls._int_spec(dtype)
        if spec is not None and spec[0] < 64:
            bits, signed = spec
            lo = -(1 << (bits - 1)) if signed else 0
            hi = ((1 << (bits - 1)) if signed else (1 << bits)) - 1
        if not lo <= N <= hi:
            raise ValueError(
                f"ArraysCache.advance offset {N} is out of range "
                f"for {dtype} metadata"
            )

    def _fold(self, arr_attr, adv_attr):
        """Fold the pending decrement into the backing array -- in place,
        so aliases observe it -- and schedule the evaluation. A no-op on
        tracers (see _try_schedule): the fold stays pending, the counter
        untouched. Integer totals fold as one pre-wrapped exact scalar;
        other dtypes subtract in int64-gate-sized chunks (an accumulated
        floating total can exceed the scalar range even when every
        offset was valid), each chunk scheduled as it is applied so a
        many-chunk fold cannot itself build an unbounded chain. The
        counter commits chunk-by-chunk, BEFORE each subtraction: a
        decrement is never applied twice, and a failure mid-fold leaves
        the unapplied remainder pending except for at most the single
        in-flight chunk (an asynchronous interrupt or failed subtraction
        between the counter commit and the subtraction drops it)."""
        with self._fold_lock:
            total = getattr(self, adv_attr)
            if not total:
                return
            arr = getattr(self, arr_attr)
            if not _try_schedule(arr):
                return
            total = self._wrap_advance(total, arr.dtype)
            setattr(self, adv_attr, total)
            lo, hi = -(1 << 63), (1 << 63) - 1
            while True:
                chunk = min(max(total, lo), hi)
                total -= chunk
                setattr(self, adv_attr, total)
                arr -= chunk
                setattr(self, arr_attr, arr)
                if not total:
                    break
                mx.async_eval(arr)
            mx.async_eval(arr)

    def _fold_lp(self):
        self._fold("_left_padding", "_lp_advance")

    def _fold_len(self):
        self._fold("_lengths", "_len_advance")

    @property
    def left_padding(self):
        if self._left_padding is None:
            return None
        with self._fold_lock:
            if self._lp_advance:
                self._fold_lp()
            else:
                # Schedule on every access: external in-place writes
                # through the returned array are otherwise never
                # evaluated for metadata nothing consumes (a property
                # access is not an evaluation).
                _try_schedule(self._left_padding)
            return self._left_padding

    @left_padding.setter
    def left_padding(self, v):
        # Fold the outgoing array first so earlier aliases still observe
        # the decrement; otherwise replacement would discard it forever.
        with self._fold_lock:
            if self._left_padding is not None:
                self._fold_lp()
                if self._lp_advance:
                    # The fold declined: the outgoing array is a tracer
                    # (captured by a graph transformation). Proceeding
                    # would silently discard the decrement.
                    raise RuntimeError(
                        "ArraysCache metadata cannot be replaced inside "
                        "a graph transformation while an advance() "
                        "decrement is pending"
                    )
            same = v is self._left_padding
            if v is not None:
                # Bound chains from repeatedly assigning lazy expressions
                _try_schedule(v)
            self._left_padding = v
            self._lp_advance = 0
            if not same:
                # Only a genuinely new array resets the bool-promotion
                # record; re-assigning the backing array is not a reset
                self._lp_promoted = False

    @property
    def lengths(self):
        if self._lengths is None:
            return None
        with self._fold_lock:
            if self._len_advance:
                self._fold_len()
            else:
                _try_schedule(self._lengths)
            return self._lengths

    @lengths.setter
    def lengths(self, v):
        with self._fold_lock:
            if self._lengths is not None:
                self._fold_len()
                if self._len_advance:
                    raise RuntimeError(
                        "ArraysCache metadata cannot be replaced inside "
                        "a graph transformation while an advance() "
                        "decrement is pending"
                    )
            same = v is self._lengths
            if v is not None:
                _try_schedule(v)
            self._lengths = v
            self._len_advance = 0
            if not same:
                self._len_promoted = False

    def _sync(self):
        """Fold both pending integer decrements into the stored arrays."""
        self._fold_lp()
        self._fold_len()

    def _require_folded(self, op):
        """Guard for mutators: a pending counter after _sync() means the
        backing array is a tracer (captured by a graph transformation),
        and rebinding or clearing the field now would silently discard
        the decrement. Reads stay supported under capture; mutations
        must run eagerly. Never fires on eager paths, where folds
        always land."""
        if self._lp_advance or self._len_advance:
            raise RuntimeError(
                f"ArraysCache.{op}: metadata is captured by a graph "
                "transformation with an advance() decrement pending; "
                "apply cache mutations outside compiled functions"
            )

    def __copy__(self):
        # A shallow copy shares the backing arrays (as stock's did) --
        # and the fold lock, so folds on the shared arrays stay
        # serialized across both objects. Fold first: otherwise the
        # copied pending counters would apply this cache's decrement
        # once per object to the shared array. If the fold cannot land
        # (captured tracer), a copy would escape the trace holding the
        # transient tracer -- raise like the other guarded mutations.
        with self._fold_lock:
            self._sync()
            self._require_folded("__copy__")
            new = self.__class__.__new__(self.__class__)
            new.__dict__.update(self.__dict__)
            return new

    def __deepcopy__(self, memo):
        # Same guard as __copy__: python's default deepcopy would copy
        # __dict__ directly, letting a deep copy taken mid-trace escape
        # with the transient tracer and the pending counter. The copy
        # gets its own fold lock (its arrays are independent).
        with self._fold_lock:
            self._sync()
            self._require_folded("__deepcopy__")
            new = self.__class__.__new__(self.__class__)
            memo[id(self)] = new
            for k, v in self.__dict__.items():
                if k != "_fold_lock":
                    new.__dict__[k] = copy.deepcopy(v, memo)
            return new

    def _materialize(self):
        """Schedule evaluation of the metadata arrays. filter/extend run
        once per batch change, but only one layer's metadata is ever
        consumed downstream, so batch churn on a long-lived server would
        otherwise grow an unevaluated chain for every other layer (the
        same dead-graph pattern as advance(), at per-request rate).
        async_eval, not eval: a synchronous eval here would block behind
        whatever the generation loop already queued on the stream."""
        arrs = [a for a in (self._left_padding, self._lengths) if a is not None]
        if arrs:
            _try_schedule(arrs)

    @property
    def batch_size(self):
        for c in self.cache:
            if c is not None:
                return c.shape[0]
        if self._left_padding is not None:
            return self._left_padding.size
        elif self._lengths is not None:
            return self._lengths.size
        else:
            return 1

    def __setitem__(self, idx, value):
        self.cache[idx] = value

    def __getitem__(self, idx):
        return self.cache[idx]

    @property
    def state(self):
        return self.cache

    @state.setter
    def state(self, v):
        self.cache = v

    # dtypes the metadata codec accepts, with the value parser for each.
    # Deliberately numeric-only: metadata holds padding/length counts.
    _META_DTYPES = {
        "int8": int,
        "int16": int,
        "int32": int,
        "int64": int,
        "uint8": int,
        "uint16": int,
        "uint32": int,
        "uint64": int,
        "float16": float,
        "float32": float,
        "float64": float,
        "bfloat16": float,
    }

    @property
    def meta_state(self):
        # Metadata is not part of ``state`` (its fields are optional and
        # ``state`` slots may not be None), so serialize it as strings:
        # "" for an absent field, "<dtype>:<comma-joined values>"
        # otherwise (an empty array stays distinct from None and dtype
        # survives the round trip).
        if self._left_padding is None and self._lengths is None:
            return ""
        self._sync()

        def encode(a):
            if a is None:
                return ""
            if a.ndim != 1:
                # The encoding is 1-D only -- the shape make_mask/merge/
                # filter arithmetic assumes. Fail loudly instead of
                # emitting an entry the decoder cannot parse.
                raise ValueError(
                    f"ArraysCache metadata must be 1-D to serialize, "
                    f"got shape {a.shape}"
                )
            dtype = str(a.dtype).split(".")[-1]
            if dtype not in self._META_DTYPES:
                raise TypeError(
                    f"ArraysCache metadata with dtype {a.dtype} cannot be serialized"
                )
            return dtype + ":" + ",".join(str(x) for x in a.tolist())

        return (encode(self._left_padding), encode(self._lengths))

    @meta_state.setter
    def meta_state(self, v):
        # Files written before metadata serialization carry an empty entry
        if not v:
            return

        def decode(s):
            if not s:
                return None
            dtype_name, sep, vals = s.partition(":")
            cast = self._META_DTYPES.get(dtype_name)
            if not sep or cast is None:
                raise ValueError(f"Malformed ArraysCache metadata entry: {s!r}")
            return mx.array(
                [cast(x) for x in vals.split(",")] if vals else [],
                dtype=getattr(mx, dtype_name),
            )

        lp, ln = v
        self.left_padding = decode(lp)
        self.lengths = decode(ln)

    def filter(self, batch_indices):
        """
        In-place filter to keep just the given indices in the cache.
        """
        self.cache = [c[batch_indices] if c is not None else None for c in self.cache]
        self._sync()
        self._require_folded("filter")
        if self._left_padding is not None:
            self._left_padding = self._left_padding[batch_indices]
        if self._lengths is not None:
            self._lengths = self._lengths[batch_indices]
        self._materialize()

    def extend(self, other):
        """
        In-place extend this cache with the other cache.
        """

        a_batch = self.batch_size
        b_batch = other.batch_size

        def cat(a, b):
            shape = dtype = None
            if a is not None:
                shape = a.shape
                dtype = a.dtype
            if b is not None:
                shape = b.shape
                dtype = b.dtype

            if shape is None:
                return None

            if a is None:
                a = mx.zeros((a_batch,) + shape[1:], dtype=dtype)
            if b is None:
                b = mx.zeros((b_batch,) + shape[1:], dtype=dtype)

            return mx.concatenate([a, b])

        self._sync()
        other._sync()
        self._require_folded("extend")
        other._require_folded("extend")
        self.cache = [cat(c, o) for c, o in zip(self.cache, other.cache)]
        self._left_padding = cat(self._left_padding, other._left_padding)
        self._lengths = cat(self._lengths, other._lengths)
        self._lp_promoted = self._lp_promoted or other._lp_promoted
        self._len_promoted = self._len_promoted or other._len_promoted
        self._materialize()

    def extract(self, idx):
        cache = ArraysCache(len(self.cache))
        cache.cache = [c[idx : idx + 1] for c in self.cache]
        return cache

    def prepare(self, lengths=None, **kwargs):
        self.lengths = mx.array(lengths)

    def finalize(self):
        # Fold before clearing so aliases held by callers still observe
        # the decrements that happened while the fields were live
        self._sync()
        self._require_folded("finalize")
        self._lengths = None
        self._left_padding = None
        self._lp_promoted = False
        self._len_promoted = False

    def advance(self, N):
        # Integer bookkeeping only: building this with mx.array arithmetic
        # leaks one live buffer object per layer per token (see class note).
        # mx.array offsets are rejected loudly rather than coerced --
        # coercion would force an eval here, truncate floats, and accept
        # non-scalars that silently corrupt the deferred arithmetic.
        # operator.index accepts any integral python scalar and rejects
        # floats. Type validation runs regardless of whether any metadata
        # is set, so that contract does not depend on cache state; the
        # dtype range check below is per armed field, in stock's field
        # order, because that is exactly where stock's eager subtraction
        # would have rejected the offset.
        if isinstance(N, mx.array):
            raise TypeError("ArraysCache.advance requires a python int, not mx.array")
        N = operator.index(N)
        with self._fold_lock:
            if self._lengths is not None:
                promoted = self._len_promoted or (
                    self._lengths is self._left_padding and self._lp_promoted
                )
                self._check_advance(N, self._logical_dtype(self._lengths, promoted))
                self._len_advance += N
                if self._lengths.dtype == mx.bool_:
                    self._len_promoted = True
            if self._left_padding is not None:
                promoted = self._lp_promoted or (
                    self._left_padding is self._lengths and self._len_promoted
                )
                self._check_advance(
                    N, self._logical_dtype(self._left_padding, promoted)
                )
                self._lp_advance += N
                if self._left_padding.dtype == mx.bool_:
                    self._lp_promoted = True

    @staticmethod
    def _logical_dtype(arr, promoted):
        # Stock's eager subtraction promoted bool metadata to int32 on
        # the FIRST advance() -- even a zero or cancelling one, and in
        # place, so a bool array shared between both fields promoted
        # both (hence the identity checks at the call sites). With the
        # fold deferred, the stored dtype stays bool, so later offsets
        # must validate against the promoted dtype exactly as stock's
        # stored array would have.
        if promoted and arr.dtype == mx.bool_:
            return mx.int32
        return arr.dtype

    def make_mask(self, N: int):
        if self._left_padding is None and self._lengths is None:
            return None
        pos = mx.arange(N)
        if self._left_padding is not None:
            # Fold (and schedule) only the field the mask uses; the other
            # field's counter is untouched. Scheduling matters even here:
            # a caller that discards the mask would otherwise leave the
            # fold unevaluated, one node per call.
            self._fold_lp()
            return pos >= self._left_padding[:, None]
        self._fold_len()
        return pos < self._lengths[:, None]

    @classmethod
    def merge(cls, caches):
        n_state = len(caches[0].cache)
        B = len(caches)
        cache = cls(n_state)

        # All caches are empty so return early
        if all(c.empty() for c in caches):
            cache.left_padding = mx.array([0] * B)
            return cache

        for e in range(n_state):
            c_init = next(iter(c[e] for c in caches if c[e] is not None))
            shape = list(c_init.shape)
            shape[0] = B
            cache[e] = mx.zeros(shape, c_init.dtype)
            for i in range(B):
                if caches[i][e] is None:
                    continue
                cache[e][i : i + 1] = caches[i][e]
        return cache

    def empty(self):
        return self.cache[0] is None

    @property
    def nbytes(self):
        return sum(c.nbytes for c in self.cache if c is not None)


class ChunkedKVCache(_BaseCache):
    step = 256

    def __init__(self, chunk_size):
        self.keys = None
        self.values = None
        self.offset = 0
        self.chunk_size = chunk_size
        self.start_position = 0

    def maybe_trim_front(self):
        # Maintain the cache below the chunk size
        if self.keys is not None and self.keys.shape[2] >= self.chunk_size:
            self.start_position += self.keys.shape[2] - self.chunk_size
            self.keys = self.keys[..., -self.chunk_size :, :]
            self.values = self.values[..., -self.chunk_size :, :]

    def update_and_fetch(self, keys, values):
        prev = self.offset - self.start_position
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += keys.shape[2]
        end = self.offset - self.start_position
        self.keys[..., prev:end, :] = keys
        self.values[..., prev:end, :] = values
        return self.keys[..., :end, :], self.values[..., :end, :]

    @property
    def state(self):
        if self.offset == self.keys.shape[2]:
            return self.keys, self.values
        else:
            return (
                self.keys[..., : self.offset, :],
                self.values[..., : self.offset, :],
            )

    @state.setter
    def state(self, v):
        self.keys, self.values = v
        self.offset = self.keys.shape[2]

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset - self.start_position, n)
        self.offset -= n
        return n

    @property
    def meta_state(self):
        return tuple(map(str, (self.chunk_size, self.start_position)))

    @meta_state.setter
    def meta_state(self, v):
        self.chunk_size, self.start_position = map(int, v)

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes


class CacheList(_BaseCache):
    def __init__(self, *caches):
        self.caches = caches

    def __getitem__(self, idx):
        return self.caches[idx]

    def is_trimmable(self):
        return all(c.is_trimmable() for c in self.caches)

    def trim(self, n):
        for c in self.caches:
            m = c.trim(n)
        return m

    @property
    def state(self):
        return [c.state for c in self.caches]

    @state.setter
    def state(self, v):
        for c, s in zip(self.caches, v):
            c.state = s

    @property
    def meta_state(self):
        return (
            [type(c).__name__ for c in self.caches],
            [c.meta_state for c in self.caches],
        )

    @meta_state.setter
    def meta_state(self, v):
        for c, m in zip(self.caches, v[1]):
            c.meta_state = m

    def filter(self, batch_indices):
        """
        In-place filter to keep just the given indices in the cache.
        """
        for c in self.caches:
            c.filter(batch_indices)

    def extend(self, other):
        """
        In-place extend this cache with the other cache.
        """
        for c, o in zip(self.caches, other.caches):
            c.extend(o)

    @classmethod
    def merge(cls, caches):
        cache = cls()
        cache.caches = tuple(
            caches[0].caches[i].merge([c.caches[i] for c in caches])
            for i in range(len(caches[0].caches))
        )
        return cache

    def extract(self, idx):
        return CacheList(*(c.extract(idx) for c in self.caches))

    def prepare(self, **kwargs):
        for c in self.caches:
            c.prepare(**kwargs)

    def finalize(self):
        for c in self.caches:
            c.finalize()

    def size(self):
        return max(c.size() for c in self.caches)

    def empty(self):
        return self.caches[0].empty()

    @property
    def nbytes(self):
        return sum(c.nbytes for c in self.caches)

    @classmethod
    def from_state(cls, state, meta_state):
        obj = cls.__new__(cls)
        obj.caches = [
            globals()[c].from_state(s, m) for s, c, m in zip(state, *meta_state)
        ]
        return obj


def dynamic_roll(x, shifts, axis):
    n = x.shape[axis]
    expand_shifts = (...,) + (None,) * (x.ndim - axis)
    expand_indices = expand_shifts[:-1]
    idx = (mx.arange(n)[expand_indices] - shifts[expand_shifts]) % n
    rolled = mx.take_along_axis(x, idx, axis=axis)
    return rolled


class BatchKVCache(_BaseCache):
    step = 256

    def __init__(self, left_padding: List[int]):
        """
        The BatchKV cache expects inputs to be left-padded.

        E.g. the following prompts:

            [1, 3, 5]
            [7]
            [2, 6, 8, 9]

        Should be padded like so:

            [0, 1, 3, 5]
            [0, 0, 0, 7]
            [2, 6, 8, 9]

        And ``left_padding`` specifies the amount of padding for each.
        In this case, ``left_padding = [1, 3, 0]``.
        """
        self.keys = None
        self.values = None
        self.left_padding = mx.array(left_padding)
        self.offset = mx.array([-l for l in left_padding])
        self._idx = 0

        self._right_padding = None

    def update_and_fetch(self, keys, values):
        prev = self._idx
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += keys.shape[2]
        self._idx += keys.shape[2]
        self.keys[..., prev : self._idx, :] = keys
        self.values[..., prev : self._idx, :] = values
        return self.keys[..., : self._idx, :], self.values[..., : self._idx, :]

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        if left_padding is not None:
            if self.keys is not None:
                raise ValueError(
                    "Left padding can only be added to an empty BatchKVCache"
                )
            left_padding = mx.array(left_padding)
            self.left_padding += left_padding
            self.offset -= left_padding

        if right_padding is not None and max(right_padding) > 0:
            self._right_padding = mx.array(right_padding)

    def finalize(self):
        if self._right_padding is not None:
            padding = self._right_padding
            self.keys = dynamic_roll(self.keys, padding[:, None], axis=2)
            self.values = dynamic_roll(self.values, padding[:, None], axis=2)
            self.offset -= padding
            self.left_padding += padding
            self._right_padding = None

    @property
    def state(self):
        k, v = self.keys, self.values
        if self._idx < k.shape[2]:
            k = k[..., : self._idx, :]
            v = v[..., : self._idx, :]
        return k, v, self.offset, self.left_padding

    @state.setter
    def state(self, v):
        self.keys, self.values, self.offset, self.left_padding = v
        self._idx = self.keys.shape[2]

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self._idx, n)
        self._idx -= n
        self.offset -= n
        return n

    def make_mask(self, N: int, return_array: bool = False, **kwargs):
        return create_causal_mask(
            N, offset=self._idx, left_padding=self.left_padding, **kwargs
        )

    def filter(self, batch_indices):
        """
        In-place filter to keep just the given indices in the cache.
        """
        if self.keys is not None:
            self.keys = self.keys[batch_indices]
            self.values = self.values[batch_indices]
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]

        # Shift left to reduce padding
        min_left_pad = self.left_padding.min().item()
        if min_left_pad > 0:
            if self.keys is not None:
                self.keys = self.keys[..., min_left_pad:, :]
                self.values = self.values[..., min_left_pad:, :]
            self._idx -= min_left_pad
            self.left_padding -= min_left_pad

    def extend(self, other):
        """
        In-place extend this cache with the other cache.
        """
        if self.keys is None and other.keys is None:
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            self.offset = mx.concatenate([self.offset, other.offset])
            return

        max_idx = max(self._idx, other._idx)
        L1 = L2 = 0
        if self.keys is not None:
            B, H, L1, D = self.keys.shape
            M = self.values.shape[3]
        if other.keys is not None:
            B, H, L2, D = other.keys.shape
            M = other.values.shape[3]
        max_size = max(L1, L2)

        # Pad the keys and values so they are right-justified
        # with the index and the same size
        def pad(c):
            k, v = c.keys, c.values
            if k is None:
                Bc = c.offset.shape[0]
                k = mx.array([]).reshape(Bc, H, 0, D)
                v = mx.array([]).reshape(Bc, H, 0, M)
            left = max_idx - c._idx
            right = max_size - k.shape[2] - left
            if right < 0:
                k = k[..., :right, :]
                v = v[..., :right, :]
                right = 0
            if left != 0 or right != 0:
                pad = [(0, 0), (0, 0), (left, right), (0, 0)]
                k = mx.pad(k, pad)
                v = mx.pad(v, pad)
            left_padding = c.left_padding + left
            return k, v, c.offset, left_padding

        self.keys, self.values, self.offset, self.left_padding = map(
            mx.concatenate, zip(*(pad(self), pad(other)))
        )
        self._idx = max_idx

    def extract(self, idx):
        cache = KVCache()
        padding = self.left_padding[idx].item()
        cache.keys = mx.contiguous(self.keys[idx : idx + 1, :, padding : self._idx])
        cache.values = mx.contiguous(self.values[idx : idx + 1, :, padding : self._idx])
        cache.offset = cache.keys.shape[2]
        return cache

    @classmethod
    def merge(cls, caches):
        lengths = [c.size() for c in caches]
        max_length = max(lengths)

        # No cache has content so make an empty one
        if max_length == 0:
            return BatchKVCache([0] * len(caches))

        padding = [max_length - l for l in lengths]
        B = len(caches)
        H = max(c.keys.shape[1] for c in caches if c.keys is not None)
        Dk = max(c.keys.shape[3] for c in caches if c.keys is not None)
        Dv = max(c.values.shape[3] for c in caches if c.values is not None)
        dt = next(iter(c.keys.dtype for c in caches if c.keys is not None))

        keys = mx.zeros((B, H, max_length, Dk), dtype=dt)
        values = mx.zeros((B, H, max_length, Dv), dtype=dt)
        for i, (p, c) in enumerate(zip(padding, caches)):
            if c.keys is None:
                continue
            keys[i : i + 1, :, p : p + c.offset] = c.keys[..., : c.offset, :]
            values[i : i + 1, :, p : p + c.offset] = c.values[..., : c.offset, :]

        cache = cls(padding)
        cache.keys = keys
        cache.values = values
        cache.offset += keys.shape[2]
        cache._idx = keys.shape[2]

        return cache

    def size(self):
        return self._idx

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes


class BatchRotatingKVCache(_BaseCache):
    step = 256

    def __init__(self, max_size, left_padding: List[int]):
        self.keys = None
        self.values = None

        self.left_padding = mx.array(left_padding)
        self.offset = mx.array([-l for l in left_padding])

        self.max_size = max_size
        self._idx = 0
        self._offset = 0
        self.rotated = False

        # Lengths for right_padded inputs to make sure that padding tokens do
        # not evict valid tokens.
        self._lengths = None

    def _trim(self, trim_size, v, append=None):
        if trim_size > 0:
            v = v[..., trim_size:, :]
        if append is not None:
            return mx.concatenate([v, append], axis=2)
        return v

    def _temporal_order(self):
        """
        Rearrange the cache into temporal order.
        """
        if self.rotated:
            self.keys = mx.roll(self.keys, -self._idx, axis=2)
            self.values = mx.roll(self.values, -self._idx, axis=2)
            self._idx = self.keys.shape[2]
            self.rotated = False

    def _update_concat(self, keys, values):
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            # Put the keys/values in temporal order to
            # preserve context
            self._temporal_order()

            # Slice off the end if needed
            if self.keys.shape[2] > self._idx:
                self.keys = self.keys[..., : self._idx, :]
                self.values = self.values[..., : self._idx, :]

            # Roll right sequences that are padded to make sure that we don't
            # trim valid cache entries
            if self._lengths is not None:
                roll = mx.maximum(0, self.offset - self._lengths)
                self.keys = dynamic_roll(self.keys, roll[:, None], axis=2)
                self.values = dynamic_roll(self.values, roll[:, None], axis=2)
                self.left_padding += roll
                self.offset -= roll

            # The largest size is self.max_size + S - 1 to ensure
            # every token gets at least self.max_size context
            trim_size = self._idx - self.max_size + 1
            if trim_size > 0:
                self.left_padding -= trim_size
            self.keys = self._trim(trim_size, self.keys, keys)
            self.values = self._trim(trim_size, self.values, values)
        self.offset += keys.shape[2]
        self._offset += keys.shape[2]
        self._idx = self.keys.shape[2]

        # Make sure left_padding and offset are evaluated
        self.keys = mx.depends(self.keys, (self.left_padding, self.offset))

        return self.keys, self.values

    def _update_in_place(self, keys, values):
        if self._lengths is not None:
            raise RuntimeError(
                "finalize() should be called before deocoding with BatchRotatingKVCache"
            )

        # May not have hit the max size yet, so potentially
        # keep growing the cache
        B, n_kv_heads, S, k_head_dim = keys.shape
        prev = self._offset
        if self.keys is None or (
            prev >= self.keys.shape[2] and self.keys.shape[2] < self.max_size
        ):
            v_head_dim = values.shape[3]
            new_size = min(self.step, self.max_size - prev)
            k_shape = (B, n_kv_heads, new_size, k_head_dim)
            v_shape = (B, n_kv_heads, new_size, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v
            self._idx = prev

        # Trim if needed
        trim_size = self.keys.shape[2] - self.max_size
        if trim_size > 0:
            self.keys = self._trim(trim_size, self.keys)
            self.values = self._trim(trim_size, self.values)
            self._idx = self.max_size
            self.left_padding -= trim_size

        # Rotate
        if self._idx == self.max_size:
            self.rotated = True
            self._idx = 0
        if self.rotated:
            self.left_padding -= S

        # Assign
        self.keys[..., self._idx : self._idx + S, :] = keys
        self.values[..., self._idx : self._idx + S, :] = values
        self._offset += S
        self.offset += S
        self._idx += S

        # Make sure left_padding and offset are evaluated
        self.keys = mx.depends(self.keys, (self.left_padding, self.offset))

        # If the buffer is not full, slice off the end
        if self._offset < self.max_size:
            return (
                self.keys[..., : self._offset, :],
                self.values[..., : self._offset, :],
            )
        return self.keys, self.values

    def update_and_fetch(self, keys, values):
        if keys.shape[2] == 1:
            return self._update_in_place(keys, values)
        return self._update_concat(keys, values)

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        if left_padding is not None:
            if self.keys is not None:
                raise ValueError(
                    "Left padding can only be added to an empty BatchRotatingKVCache"
                )
            left_padding = mx.array(left_padding)
            self.left_padding += left_padding
            self.offset -= left_padding

        if right_padding is not None and max(right_padding) > 0:
            self._lengths = mx.array(lengths) + self.offset

    def finalize(self):
        if self._lengths is not None:
            roll = mx.maximum(0, self.offset - self._lengths)
            self.keys = dynamic_roll(self.keys, roll[:, None], axis=2)
            self.values = dynamic_roll(self.values, roll[:, None], axis=2)
            self.left_padding += roll
            self.offset -= roll
            self._lengths = None

    @property
    def state(self):
        k, v = self.keys, self.values
        if self._offset < k.shape[2]:
            k, v = k[..., : self._offset, :], v[..., : self._offset, :]
        return k, v, self.offset, self.left_padding

    @state.setter
    def state(self, v):
        self.keys, self.values, self.offset, self.left_padding = v

    @property
    def meta_state(self):
        return tuple(map(str, (self.max_size, self._offset, self._idx, self.rotated)))

    @meta_state.setter
    def meta_state(self, v):
        self.max_size, self._offset, self._idx = map(
            int,
            v[:3],
        )
        self.rotated = bool(v[3])

    def is_trimmable(self):
        return self._offset < self.max_size

    def trim(self, n):
        n = min(self._offset, n)
        self._offset -= n
        self._idx -= n
        self.offset -= n
        return n

    def to_quantized(self, group_size: int = 64, bits: int = 4) -> QuantizedKVCache:
        raise NotImplementedError("BatchRotatingKVCache Quantization NYI")

    def make_mask(
        self, N: int, window_size: Optional[int] = None, return_array: bool = False
    ):
        left_padding = self.left_padding
        window_size = window_size or self.max_size
        offset = min(self.max_size - 1, self._offset)
        rinds = mx.arange(offset + N)
        linds = mx.arange(offset, offset + N) if offset else rinds
        linds = linds[:, None]
        rinds = rinds[None]
        mask = linds >= rinds
        mask &= linds < rinds + window_size
        if (trim_size := self._idx - self.max_size + int(N > 1)) > 0:
            left_padding = left_padding - trim_size

        rotated = N == 1 and (self.rotated or self._idx >= self.max_size)
        if rotated:
            left_padding = left_padding - 1

        mask = mask & (rinds >= mx.expand_dims(left_padding, (1, 2, 3)))

        if rotated:
            idx = self._idx
            if idx >= self.max_size:
                idx = 0
            mask = mx.roll(mask, shift=idx + 1, axis=-1)

        return mask

    def filter(self, batch_indices):
        """
        In-place filter to keep just the given indices in the cache.
        """
        if self.keys is not None:
            self.keys = self.keys[batch_indices]
            self.values = self.values[batch_indices]
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]

    def extend(self, other):
        """
        In-place extend this cache with the other cache.
        """
        if self.keys is None and other.keys is None:
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            self.offset = mx.concatenate([self.offset, other.offset])
            return

        if (self.rotated != other.rotated) or self._idx != other._idx:
            self._temporal_order()
            other._temporal_order()

        max_idx = max(self._idx, other._idx)
        L1 = L2 = 0
        if self.keys is not None:
            B, H, L1, D = self.keys.shape
            M = self.values.shape[3]
        if other.keys is not None:
            B, H, L2, D = other.keys.shape
            M = other.values.shape[3]
        max_size = max(L1, L2)

        def pad(c):
            left = max_idx - c._idx
            k, v = c.keys, c.values
            if k is None:
                Bc = c.offset.shape[0]
                k = mx.array([]).reshape(Bc, H, 0, D)
                v = mx.array([]).reshape(Bc, H, 0, M)
            right = max_size - k.shape[2] - left
            if right < 0:
                k = k[..., :right, :]
                v = v[..., :right, :]
                right = 0
            if left != 0 or right != 0:
                pad = [(0, 0), (0, 0), (left, right), (0, 0)]
                k = mx.pad(k, pad)
                v = mx.pad(v, pad)
            left_padding = c.left_padding + left
            return k, v, c.offset, left_padding

        self.keys, self.values, self.offset, self.left_padding = map(
            mx.concatenate, zip(*(pad(self), pad(other)))
        )
        self._idx = max_idx
        self._offset = max(self._offset, other._offset)

    def extract(self, idx):
        mx.eval(self.left_padding, self.offset)
        cache = RotatingKVCache(self.max_size)
        padding = max(0, self.left_padding.tolist()[idx])
        offset = self.offset.tolist()[idx]
        cache.keys = self.keys[idx : idx + 1]
        cache.values = self.values[idx : idx + 1]
        cache._idx = self._idx
        if self.rotated:
            cache.keys = mx.roll(cache.keys, -self._idx, axis=2)
            cache.values = mx.roll(cache.values, -self._idx, axis=2)
            cache._idx = self.max_size
        cache.keys = mx.contiguous(cache.keys[:, :, padding : cache._idx])
        cache.values = mx.contiguous(cache.values[:, :, padding : cache._idx])
        cache.offset = offset
        cache._idx = cache.keys.shape[2]
        return cache

    @classmethod
    def merge(cls, caches):
        if not all(c.max_size == caches[0].max_size for c in caches):
            raise ValueError(
                "BatchRotatingKVCache can only merge caches with the same maximum size"
            )

        offsets = [c.offset for c in caches]
        lengths = [c.size() for c in caches]
        max_length = max(lengths)

        # No cache has content so make an empty one
        if max_length == 0:
            return cls(caches[0].max_size, [0] * len(caches))

        padding = [max_length - l for l in lengths]
        B = len(caches)
        H = max(c.keys.shape[1] for c in caches if c.keys is not None)
        Dk = max(c.keys.shape[3] for c in caches if c.keys is not None)
        Dv = max(c.values.shape[3] for c in caches if c.values is not None)
        dt = next(iter(c.keys.dtype for c in caches if c.keys is not None))

        keys = mx.zeros((B, H, max_length, Dk), dtype=dt)
        values = mx.zeros((B, H, max_length, Dv), dtype=dt)
        for i, (p, l, c) in enumerate(zip(padding, lengths, caches)):
            if c.keys is None:
                continue
            keys[i : i + 1, :, p : p + l] = c._temporal_order(c.keys)[..., -l:, :]
            values[i : i + 1, :, p : p + l] = c._temporal_order(c.values)[..., -l:, :]

        cache = cls(caches[0].max_size, padding)
        cache.keys = keys
        cache.values = values
        cache.offset = mx.array(offsets)
        cache._idx = keys.shape[2]
        cache._offset = keys.shape[2]

        return cache

    def size(self):
        return min(self._offset, self.max_size)

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes


class TokenBuffer:
    """A simple token buffer that can be efficiently appended to in a similar
    fashion to the KVCache.

    Perhaps these could share some logic in the future.
    """

    step = 256

    def __init__(self, tokens=[]):
        self._buffer = mx.array(tokens, dtype=mx.int32)
        self._size = len(tokens)

    def update_and_fetch(self, tokens):
        start = self._size
        end = start + len(tokens)

        new_size = ((end + self.step - 1) // self.step) * self.step
        if new_size > self._buffer.size:
            self._buffer = mx.concatenate(
                [self._buffer, mx.zeros(new_size - self._buffer.size, dtype=mx.int32)]
            )
        self._buffer[start:end] = tokens
        self._size = end

        return self._buffer[:end]

    @property
    def state(self):
        return self._buffer

    @property
    def tokens(self):
        return self._buffer[: self._size]


@dataclass
class PromptTrieResult:
    model: Any
    exact: Optional[List[int]]  # Exact match found
    shorter: Optional[List[int]]  # Longest prefix with a value
    longer: Optional[List[int]]  # Shortest value that extends beyond tokens
    common_prefix: int  # Length of common prefix with any path


class PromptTrie:
    def __init__(self):
        self._trie = {}

    def add(self, model: Any, tokens: List[int], value: Any):
        if model not in self._trie:
            self._trie[model] = {}

        current = self._trie[model]
        for tok in tokens:
            if tok not in current:
                current[tok] = {}
            current = current[tok]
        prev = current.get("__value__", None)
        current["__value__"] = value
        return prev

    def get(self, model: Any, tokens: List[int]):
        current = self._trie[model]
        for tok in tokens:
            current = current[tok]
        return current["__value__"]

    def pop(self, model: Any, tokens: List[int]):
        path = [self._trie[model]]
        for tok in tokens:
            path.append(path[-1][tok])
        value = path[-1].pop("__value__")
        for i in range(len(tokens), 0, -1):
            node = path[i]
            parent = path[i - 1]
            tok = tokens[i - 1]
            if len(node) > 0:
                break
            del parent[tok]
        return value

    def pop_prefixes(self, model: Any, tokens: List[int]):
        values = []
        current = self._trie[model]
        for i, tok in enumerate(tokens):
            if "__value__" in current:
                values.append((i, current.pop("__value__")))
            current = current[tok]
        return values

    def search(self, model: Any, tokens: List[int]) -> PromptTrieResult:
        if model not in self._trie:
            return PromptTrieResult(model, None, None, None, 0)

        current = self._trie[model]

        if not tokens and "__value__" in current:
            return PromptTrieResult(model, [], None, None, 0)

        # Walk the tokens as far as we can
        last_index = -1
        index = 0
        while index < len(tokens) and tokens[index] in current:
            current = current[tokens[index]]
            if "__value__" in current:
                last_index = index
            index += 1

        # Got an exact match
        if last_index == len(tokens) - 1 >= 0:
            return PromptTrieResult(model, tokens, None, None, 0)

        # Check if we found a prefix at any point
        shorter = None
        if last_index > 0:
            shorter = tokens[: last_index + 1]

        # Check for sequences that are longer
        longer = None
        common_prefix = index
        if index > 0:
            best = None
            stack = [(current, [])]
            while stack:
                current, extra = stack.pop()
                if "__value__" in current:
                    if best is None or len(extra) < len(best):
                        best = extra
                elif best is None or len(extra) < len(best):
                    for tok in current:
                        stack.append((current[tok], extra + [tok]))
            longer = tokens[:index] + best
        return PromptTrieResult(model, None, shorter, longer, common_prefix)


class LRUPromptCache:
    @dataclass
    class CacheEntry:
        prompt_cache: List[Any]
        nbytes: int
        cache_type: str

    class CacheOrder:
        def __init__(self, ordering: List[str] = ["assistant", "user", "system"]):
            self._ordering = ordering
            self._lrus = {k: deque() for k in ordering}

        def __len__(self):
            return sum(len(lru) for lru in self._lrus.values())

        def push(self, model: Any, tokens: List[Any], cache_type: str = "assistant"):
            self._lrus[cache_type].append((model, tokens))

        def remove(self, model: Any, tokens: List[Any]):
            for cache_type in self._ordering:
                try:
                    self._lrus[cache_type].remove((model, tokens))
                    break
                except ValueError:
                    pass

        def pop(self):
            i = 0
            while i + 1 < len(self._ordering):
                lru_a = self._lrus[self._ordering[i]]
                lru_b = self._lrus[self._ordering[i + 1]]
                if lru_a and len(lru_a) >= len(lru_b):
                    return lru_a.popleft()
                i += 1
            return lru_b.popleft()

    def __init__(self, max_size: int = 10, max_bytes: int = 1 << 63):
        self.max_size = max_size
        self.max_bytes = max_bytes
        self._trie = PromptTrie()
        self._lru = LRUPromptCache.CacheOrder()
        self._n_bytes = 0
        self._n_bytes_by_type = {k: 0 for k in self._lru._ordering}

    def __len__(self):
        return len(self._lru)

    @property
    def nbytes(self):
        return self._n_bytes

    def fetch_nearest_cache(self, model: Any, tokens: List[int]):
        result = self._trie.search(model, tokens)
        if result.exact is not None:
            cache_entry = self._trie.get(result.model, result.exact)
            return copy.deepcopy(cache_entry.prompt_cache), []

        short_length = len(result.shorter) if result.shorter is not None else 0
        if result.longer is not None and result.common_prefix > short_length:
            cache_entry = self._trie.get(result.model, result.longer)
            if can_trim_prompt_cache(cache_entry.prompt_cache):
                cache = copy.deepcopy(cache_entry.prompt_cache)
                prefix = min(len(tokens) - 1, result.common_prefix)
                num_to_trim = len(result.longer) - prefix
                trim_prompt_cache(cache, num_to_trim)
                return cache, tokens[prefix:]

        if short_length > 0:
            cache_entry = self._trie.get(result.model, result.shorter)
            return copy.deepcopy(cache_entry.prompt_cache), tokens[short_length:]

        return None, tokens

    def insert_cache(
        self,
        model: Any,
        tokens: List[int],
        prompt_cache: List[Any],
        *,
        cache_type: str = "assistant",
    ):
        # Make the cache entry
        entry = LRUPromptCache.CacheEntry(
            prompt_cache, sum(c.nbytes for c in prompt_cache), cache_type
        )

        # Insert into the trie and update the byte counter and lru position
        self._n_bytes += entry.nbytes
        self._n_bytes_by_type[cache_type] += entry.nbytes
        prev = self._trie.add(model, tokens, entry)
        if prev is not None:
            self._n_bytes -= prev.nbytes
            self._n_bytes_by_type[prev.cache_type] -= prev.nbytes
            self._lru.remove(model, tokens)
        self._lru.push(model, tokens, cache_type)

        # If it is a trimmable cache remove all prefixes cause they just take
        # space
        if can_trim_prompt_cache(prompt_cache):
            for prefix_len, entry in self._trie.pop_prefixes(model, tokens):
                self._n_bytes -= entry.nbytes
                self._n_bytes_by_type[entry.cache_type] -= entry.nbytes
                self._lru.remove(model, tokens[:prefix_len])

        # Ensure we match the constraints
        if len(self._lru) > self.max_size:
            model, tokens = self._lru.pop()
            entry = self._trie.pop(model, tokens)
            self._n_bytes -= entry.nbytes
            self._n_bytes_by_type[entry.cache_type] -= entry.nbytes
        while self._n_bytes > self.max_bytes:
            model, tokens = self._lru.pop()
            entry = self._trie.pop(model, tokens)
            self._n_bytes -= entry.nbytes
            self._n_bytes_by_type[entry.cache_type] -= entry.nbytes

    def trim_to(
        self, *, n_sequences: Optional[int] = None, n_bytes: Optional[int] = None
    ):
        n_sequences = max(0, n_sequences) if n_sequences is not None else 1 << 63
        n_bytes = max(0, n_bytes) if n_bytes is not None else 1 << 63

        while len(self._lru) > n_sequences:
            model, tokens = self._lru.pop()
            entry = self._trie.pop(model, tokens)
            self._n_bytes -= entry.nbytes
            self._n_bytes_by_type[entry.cache_type] -= entry.nbytes
        while self._n_bytes > n_bytes:
            model, tokens = self._lru.pop()
            entry = self._trie.pop(model, tokens)
            self._n_bytes -= entry.nbytes
            self._n_bytes_by_type[entry.cache_type] -= entry.nbytes

    def stats_by_type(self):
        result = {}
        for cache_type in self._lru._ordering:
            result[cache_type] = {
                "n_sequences": len(self._lru._lrus[cache_type]),
                "n_bytes": self._n_bytes_by_type[cache_type],
            }
        return result
