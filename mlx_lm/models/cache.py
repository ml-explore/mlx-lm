# Copyright © 2023-2024 Apple Inc.

import copy
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


class ArraysCache(_BaseCache):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.left_padding = None
        instance.lengths = None
        return instance

    def __init__(self, size, left_padding: Optional[List[int]] = None):
        self.cache = [None] * size
        if left_padding:
            self.left_padding = mx.array(left_padding)

    @property
    def batch_size(self):
        for c in self.cache:
            if c is not None:
                return c.shape[0]
        if self.left_padding is not None:
            return self.left_padding.size
        elif self.lengths is not None:
            return self.lengths.size
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

    def filter(self, batch_indices):
        """
        In-place filter to keep just the given indices in the cache.
        """
        self.cache = [c[batch_indices] if c is not None else None for c in self.cache]
        if self.left_padding is not None:
            self.left_padding = self.left_padding[batch_indices]
        if self.lengths is not None:
            self.lengths = self.lengths[batch_indices]

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

        self.cache = [cat(c, o) for c, o in zip(self.cache, other.cache)]
        self.left_padding = cat(self.left_padding, other.left_padding)
        self.lengths = cat(self.lengths, other.lengths)

    def extract(self, idx):
        cache = ArraysCache(len(self.cache))
        cache.cache = [c[idx : idx + 1] for c in self.cache]
        return cache

    def prepare(self, lengths=None, **kwargs):
        self.lengths = mx.array(lengths)

    def finalize(self):
        self.lengths = None
        self.left_padding = None

    def advance(self, N):
        if self.lengths is not None:
            self.lengths -= N
        if self.left_padding is not None:
            self.left_padding -= N

    def make_mask(self, N: int):
        if self.left_padding is not None:
            pos = mx.arange(N)
            return pos >= self.left_padding[:, None]
        elif self.lengths is not None:
            pos = mx.arange(N)
            return pos < self.lengths[:, None]
        else:
            return None

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


# ---------------------------------------------------------------------------
# SnapKV-D: post-prefill KV eviction with position-preserving decode
#
# Long-context decode reads the whole KV cache every step. After the prompt is
# prefilled, most middle prompt rows contribute little to future attention, so
# keeping attention sinks + a recent window + the top observation-window-scored
# middle rows within a budget (SnapKV, arXiv:2404.14469) and evicting the rest
# cuts the per-token KV read proportionally. The retained rows are a sparse
# subset of the prompt, so RoPE position and physical row count must diverge:
# PositionPreservingKVCache tracks the true sequence position in ``offset`` for
# future rotations while storing only the retained rows, and records each row's
# true position so a prefix trim (prompt-cache reuse) stays exact.
# ---------------------------------------------------------------------------


def snapkv_keep_indices(
    seq_len: int,
    budget: int,
    scores: Sequence[float],
    *,
    sink_tokens: int = 4,
    recent_tokens: Optional[int] = None,
    min_tokens: int = 128,
) -> Tuple[int, ...]:
    """Return the sorted prompt positions retained by the SnapKV-D policy.

    Keeps ``sink_tokens`` leading rows, a recent window, and the highest-scoring
    remaining rows up to ``budget``. Returns all positions unchanged when the
    prompt is at or below ``min_tokens`` or the budget covers it.
    """
    if seq_len < 0:
        raise ValueError("seq_len must be non-negative")
    if budget <= 0:
        raise ValueError("budget must be positive")
    if len(scores) < seq_len:
        raise ValueError(f"scores length {len(scores)} < seq_len {seq_len}")
    if seq_len == 0 or seq_len <= min_tokens or budget >= seq_len:
        return tuple(range(seq_len))

    budget = min(budget, seq_len)
    sink_count = min(max(0, sink_tokens), max(1, budget // 8), budget)
    retained = set(range(sink_count))

    remaining = budget - len(retained)
    if remaining > 0:
        recent_count = (
            recent_tokens if recent_tokens is not None else max(1, budget // 8)
        )
        recent_count = min(recent_count, remaining)
        retained.update(range(seq_len - recent_count, seq_len))

    remaining = budget - len(retained)
    if remaining > 0:
        ranked = sorted(
            (i for i in range(seq_len) if i not in retained),
            key=lambda i: (float(scores[i]), i),
            reverse=True,
        )
        retained.update(ranked[:remaining])

    return tuple(sorted(retained))


class PositionPreservingKVCache(_BaseCache):
    """KV cache with a true-position ``offset`` and compact retained storage.

    Intended for post-prefill eviction followed by single-token decode. The
    physical K/V rows may be a sparse subset of the logical prompt; the
    ``positions`` metadata records each row's true sequence position so a prefix
    trim (prompt-cache reuse) can shorten the logical ``offset`` without
    pretending sparse rows are contiguous. Speculative rollback trims a
    generated suffix and is tracked separately.
    """

    step = 256

    def __init__(
        self,
        keys=None,
        values=None,
        *,
        offset: int = 0,
        protected_stored: Optional[int] = None,
        positions: Optional[Sequence[int]] = None,
    ):
        self.keys = keys
        self.values = values
        self.offset = int(offset)
        self._stored = 0 if keys is None else int(keys.shape[2])
        self._protected_stored = (
            self._stored if protected_stored is None else int(protected_stored)
        )
        if self._protected_stored < 0 or self._protected_stored > self._stored:
            raise ValueError("protected_stored must be between 0 and stored rows")
        self._positions = self._coerce_positions(positions)
        self._speculating = False
        self._speculative_appends = deque()

    def _coerce_positions(self, positions):
        if positions is None:
            if self._stored == 0:
                return ()
            if self.offset == self._stored:
                return tuple(range(self._stored))
            return None
        out = tuple(int(p) for p in positions)
        if len(out) != self._stored:
            raise ValueError("positions length must match stored rows")
        if any(p < 0 for p in out):
            raise ValueError("positions must be non-negative")
        if tuple(sorted(out)) != out:
            raise ValueError("positions must be sorted")
        return out

    @staticmethod
    def _encode_positions(positions):
        if positions is None:
            return "-"
        return ",".join(map(str, positions))

    @staticmethod
    def _decode_positions(raw):
        if raw == "-":
            return None
        if raw == "":
            return ()
        return tuple(int(p) for p in raw.split(","))

    def update_and_fetch(self, keys, values):
        prev = self._stored
        n_new = int(keys.shape[2])
        start_pos = self.offset
        required = prev + n_new
        if self.keys is None or required > self.keys.shape[2]:
            bsz, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            grow = ((required - prev + self.step - 1) // self.step) * self.step
            new_k = mx.zeros((bsz, n_kv_heads, grow, k_head_dim), keys.dtype)
            new_v = mx.zeros((bsz, n_kv_heads, grow, v_head_dim), values.dtype)
            if self.keys is not None:
                self.keys = mx.concatenate([self.keys[..., :prev, :], new_k], axis=2)
                self.values = mx.concatenate(
                    [self.values[..., :prev, :], new_v], axis=2
                )
            else:
                self.keys, self.values = new_k, new_v

        self.keys[..., prev:required, :] = keys
        self.values[..., prev:required, :] = values
        self._stored = required
        self.offset += n_new
        if self._positions is not None:
            self._positions = self._positions + tuple(
                range(start_pos, start_pos + n_new)
            )
        if self._speculating and n_new:
            self._speculative_appends.append(n_new)
        return (
            self.keys[..., : self._stored, :],
            self.values[..., : self._stored, :],
        )

    def size(self):
        return self._stored

    @property
    def state(self):
        if self.keys is None:
            return None
        return (
            self.keys[..., : self._stored, :],
            self.values[..., : self._stored, :],
        )

    @state.setter
    def state(self, v):
        if v is None:
            self.keys = None
            self.values = None
            self._stored = 0
            self.offset = 0
            self._protected_stored = 0
            self._positions = ()
            self._speculating = False
            self._speculative_appends = deque()
            return
        self.keys, self.values = v
        self._stored = int(self.keys.shape[2])
        self.offset = self._stored
        self._protected_stored = self._stored
        self._positions = tuple(range(self._stored))
        self._speculating = False
        self._speculative_appends = deque()

    @property
    def meta_state(self):
        return tuple(
            map(
                str,
                (
                    self.offset,
                    self._stored,
                    self._protected_stored,
                    self._encode_positions(self._positions),
                ),
            )
        )

    @meta_state.setter
    def meta_state(self, v):
        vals = tuple(v)
        ints = tuple(map(int, vals[:3]))
        if len(ints) == 2:
            self.offset, self._stored = ints
            self._protected_stored = self._stored
        else:
            self.offset, self._stored, self._protected_stored = ints
        if len(vals) >= 4:
            self._positions = self._decode_positions(str(vals[3]))
            if self._positions is not None and len(self._positions) != self._stored:
                raise ValueError("stored row count does not match position metadata")
        elif self.offset == self._stored:
            self._positions = tuple(range(self._stored))
        else:
            self._positions = None
        self._speculating = False
        self._speculative_appends = deque()

    def is_trimmable(self):
        return self._speculating or self._positions is not None

    @property
    def protected_stored(self):
        return self._protected_stored

    @property
    def positions(self):
        return self._positions

    def start_speculation(self):
        self._speculating = True
        self._speculative_appends.clear()

    def stop_speculation(self):
        self._speculating = False
        self._speculative_appends.clear()

    def _trim_speculative_suffix(self, n):
        if not self._speculative_appends:
            raise RuntimeError("No speculative append is available to trim")
        latest = self._speculative_appends.pop()
        if n > latest:
            self._speculative_appends.append(latest)
            raise RuntimeError(
                f"Cannot trim {n} tokens from PositionPreservingKVCache: "
                f"latest speculative append has {latest} tokens."
            )
        removable = self._stored - self._protected_stored
        if n > removable:
            self._speculative_appends.append(latest)
            raise RuntimeError(
                f"Cannot trim {n} tokens from PositionPreservingKVCache: "
                f"only {removable} appended rows are removable."
            )
        self._stored -= n
        self.offset -= n
        if self._positions is not None and n:
            self._positions = self._positions[:-n]
        if n < latest:
            self._speculative_appends.append(latest - n)
        if self.offset < 0:
            raise ValueError("trim would make true offset negative")
        return n

    def _trim_logical_prefix(self, n):
        if self._positions is None:
            raise RuntimeError(
                "PositionPreservingKVCache has no position metadata for prefix trim"
            )
        n = min(int(n), self.offset)
        if n <= 0:
            return 0
        new_offset = self.offset - n
        old_positions = self._positions
        keep_physical = [i for i, pos in enumerate(old_positions) if pos < new_offset]
        new_positions = tuple(old_positions[i] for i in keep_physical)
        if len(keep_physical) != self._stored:
            if keep_physical:
                self.keys = _take_positions(
                    self.keys[..., : self._stored, :], keep_physical
                )
                self.values = _take_positions(
                    self.values[..., : self._stored, :], keep_physical
                )
            else:
                self.keys = self.keys[..., :0, :]
                self.values = self.values[..., :0, :]
        self._protected_stored = sum(
            1 for pos in old_positions[: self._protected_stored] if pos < new_offset
        )
        self._stored = len(new_positions)
        self._positions = new_positions
        self.offset = new_offset
        return n

    def trim(self, n):
        if n <= 0:
            return 0
        n = int(n)
        if self._speculating:
            return self._trim_speculative_suffix(n)
        return self._trim_logical_prefix(n)

    def make_mask(self, n_tokens, window_size=None, return_array: bool = False):
        if window_size is not None:
            raise ValueError(
                "PositionPreservingKVCache does not support sliding-window masks"
            )
        if n_tokens == 1 and not return_array:
            return None
        prefix = mx.ones((n_tokens, self._stored), dtype=mx.bool_)
        causal = mx.tril(mx.ones((n_tokens, n_tokens), dtype=mx.bool_))
        return mx.concatenate([prefix, causal], axis=1)

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return (
            self.keys[..., : self._stored, :].nbytes
            + self.values[..., : self._stored, :].nbytes
        )


def _take_positions(x, keep: Sequence[int]):
    return mx.take(x, mx.array(keep, dtype=mx.int32), axis=2)


@dataclass(frozen=True)
class SnapKVEvictionResult:
    cache: list
    evicted: bool
    true_offset: int
    retained_tokens: int
    original_tokens: int
    kv_layers: int
    compact_cache_nbytes: int


def evict_prompt_cache(
    prompt_cache: List[Any],
    keep_indices: Sequence[int],
    *,
    true_offset: int,
) -> SnapKVEvictionResult:
    """Replace plain ``KVCache`` layers with position-preserving compact caches.

    ``keep_indices`` are the sorted prompt positions to retain (see
    ``snapkv_keep_indices``); every other layer type is left untouched.
    """
    keep = tuple(int(i) for i in keep_indices)
    if any(i < 0 for i in keep):
        raise ValueError("keep_indices must be non-negative")
    if tuple(sorted(keep)) != keep:
        raise ValueError("keep_indices must be sorted")

    out = list(prompt_cache)
    original_tokens = int(true_offset)
    kv_layers = 0
    for idx, cache in enumerate(prompt_cache):
        if type(cache) is not KVCache or cache.keys is None:
            continue
        if keep and keep[-1] >= cache.offset:
            raise ValueError("keep index exceeds KV cache offset")
        keys, values = cache.state
        new_keys = _take_positions(keys, keep) if keep else keys[..., :0, :]
        new_values = _take_positions(values, keep) if keep else values[..., :0, :]
        out[idx] = PositionPreservingKVCache(
            new_keys,
            new_values,
            offset=true_offset,
            positions=keep,
        )
        original_tokens = max(original_tokens, int(cache.offset))
        kv_layers += 1

    retained = len(keep)
    compact_cache_nbytes = sum(int(getattr(cache, "nbytes", 0) or 0) for cache in out)
    return SnapKVEvictionResult(
        cache=out,
        evicted=kv_layers > 0 and retained < original_tokens,
        true_offset=int(true_offset),
        retained_tokens=retained,
        original_tokens=original_tokens,
        kv_layers=kv_layers,
        compact_cache_nbytes=compact_cache_nbytes,
    )


class SnapKVAttentionCapture:
    """Capture windowed attention scores by wrapping ``mx.fast`` SDPA.

    Used as a context manager around a prefill. The model output still uses the
    original fused kernel; this hook separately scores only the final
    ``window`` observation-window query rows, in small chunks reduced
    immediately to a per-key vector, so it never retains a prompt-sized dense
    attention matrix. ``snap_scores`` then returns a per-prompt-position score.

    The patch is process-global for the duration of the ``with`` block, so a
    prefill scored this way should not run concurrently with unscored prefills.
    """

    def __init__(self, window: int = 24, score_chunk_size: int = 8):
        if window <= 0:
            raise ValueError("window must be positive")
        if score_chunk_size <= 0:
            raise ValueError("score_chunk_size must be positive")
        self.window = int(window)
        self.score_chunk_size = int(score_chunk_size)
        self._orig = None
        self.snap = None

    def __enter__(self):
        self._orig = mx.fast.scaled_dot_product_attention
        mx.fast.scaled_dot_product_attention = self._capture
        return self

    def __exit__(self, *_exc):
        mx.fast.scaled_dot_product_attention = self._orig

    def _align_add(self, current, vec):
        if current is None:
            return vec
        if current.shape[0] < vec.shape[0]:
            current = mx.concatenate(
                [current, mx.zeros((vec.shape[0] - current.shape[0],), current.dtype)]
            )
        elif vec.shape[0] < current.shape[0]:
            vec = mx.concatenate(
                [vec, mx.zeros((current.shape[0] - vec.shape[0],), vec.dtype)]
            )
        return current + vec

    @staticmethod
    def _causal_chunk_mask(q_start, q_end, q_len, key_len):
        key_positions = mx.arange(key_len)
        query_positions = key_len - q_len + mx.arange(q_start, q_end)
        return key_positions[None, :] <= query_positions[:, None]

    @staticmethod
    def _slice_mask(mask, q_start, q_end, q_len):
        if mask is None or isinstance(mask, str):
            return mask
        if len(mask.shape) >= 2 and mask.shape[-2] == q_len:
            prefix = (slice(None),) * (len(mask.shape) - 2)
            return mask[prefix + (slice(q_start, q_end), slice(None))]
        return mask

    @staticmethod
    def _slice_query_heads(mask, start, end):
        if mask is None or isinstance(mask, str):
            return mask
        if len(mask.shape) >= 4 and mask.shape[-3] >= end:
            prefix = (slice(None),) * (len(mask.shape) - 3)
            return mask[prefix + (slice(start, end), slice(None), slice(None))]
        return mask

    @staticmethod
    def _apply_mask(scores, mask):
        if mask is None or isinstance(mask, str):
            return scores
        return mx.where(mask, scores, -1e9) if mask.dtype == mx.bool_ else scores + mask

    def _score_chunk(self, q_chunk, k, scale, chunk_mask):
        _bsz, q_heads, _chunk, _dim = q_chunk.shape
        kv_heads = k.shape[1]
        if kv_heads == q_heads:
            scores = (q_chunk @ k.swapaxes(-1, -2)) * scale
            return self._apply_mask(scores, chunk_mask)
        if q_heads % kv_heads != 0:
            raise ValueError("query heads must be a multiple of KV heads")
        repeats = q_heads // kv_heads
        groups = []
        for kv_head in range(kv_heads):
            q_start = kv_head * repeats
            q_end = q_start + repeats
            q_group = q_chunk[:, q_start:q_end, :, :]
            k_group = k[:, kv_head : kv_head + 1, :, :]
            scores = (q_group @ k_group.swapaxes(-1, -2)) * scale
            scores = self._apply_mask(
                scores, self._slice_query_heads(chunk_mask, q_start, q_end)
            )
            groups.append(scores)
        return mx.concatenate(groups, axis=1)

    def _capture(self, q, k, v, *, scale, mask=None, **kwargs):
        out = self._orig(q, k, v, scale=scale, mask=mask, **kwargs)
        _bsz, q_heads, q_len, _dim = q.shape
        key_len = k.shape[2]
        snap_window = min(self.window, q_len)
        q_start = q_len - snap_window
        for chunk_start in range(q_start, q_len, self.score_chunk_size):
            chunk_end = min(q_len, chunk_start + self.score_chunk_size)
            q_chunk = q[:, :, chunk_start:chunk_end, :]
            chunk_mask = self._slice_mask(mask, chunk_start, chunk_end, q_len)
            if isinstance(chunk_mask, str) and chunk_mask == "causal":
                chunk_mask = self._causal_chunk_mask(
                    chunk_start, chunk_end, q_len, key_len
                )
            scores = self._score_chunk(q_chunk, k, scale, chunk_mask)
            weights = mx.softmax(scores.astype(mx.float32), axis=-1)
            snap = weights.sum(axis=(0, 1, 2))
            mx.eval(snap)
            self.snap = self._align_add(self.snap, snap)
            mx.eval(self.snap)
        return out

    def snap_scores(self, seq_len: int) -> List[float]:
        if self.snap is None:
            return [0.0] * seq_len
        mx.eval(self.snap)
        vals = self.snap.tolist()
        if len(vals) < seq_len:
            vals.extend([0.0] * (seq_len - len(vals)))
        return [float(v) for v in vals[:seq_len]]


def compact_prompt_cache(
    model,
    prompt,
    *,
    budget: int,
    window: int = 24,
    sink_tokens: int = 4,
    recent_tokens: Optional[int] = None,
    min_tokens: int = 128,
    score_chunk_size: int = 8,
) -> SnapKVEvictionResult:
    """Prefill ``prompt`` and return a SnapKV-D-compacted prompt cache.

    Convenience wrapper: prefills ``prompt`` (a 1-D token sequence) under a
    ``SnapKVAttentionCapture``, scores it, and evicts every full-attention
    ``KVCache`` layer down to ``budget`` retained rows. The returned
    ``SnapKVEvictionResult.cache`` is ready to decode from at the true prompt
    offset. No-op (all rows kept) for prompts at or below ``min_tokens``.
    """
    prompt = list(int(t) for t in prompt)
    cache = make_prompt_cache(model)
    with SnapKVAttentionCapture(
        window=window, score_chunk_size=score_chunk_size
    ) as capture:
        logits = model(mx.array([prompt]), cache=cache)
        mx.eval(logits, [c.state for c in cache])
    seq_len = len(prompt)
    scores = capture.snap_scores(seq_len)
    keep = snapkv_keep_indices(
        seq_len,
        budget,
        scores,
        sink_tokens=sink_tokens,
        recent_tokens=recent_tokens,
        min_tokens=min_tokens,
    )
    return evict_prompt_cache(cache, keep, true_offset=seq_len)


# DuoAttention: head-partitioned KV eviction (retrieval vs streaming heads)
#
# DuoAttention (arXiv:2410.10819) observes that only a subset of attention
# heads ("retrieval heads") need the full long context; the remaining
# ("streaming heads") attend well with just attention sinks plus a recent
# window. Keeping the full KV only for retrieval heads and a sink+recent slice
# for streaming heads cuts the decode-time KV read further than a uniform
# SnapKV-D budget, at no quality cost on the streaming heads.
#
# This is the SnapKV-D second stage: after the caller decides which prompt
# positions each KV head keeps, ``HeadPartitionedKVCache`` stores the union of
# retained positions once and exposes a per-head prefix mask so each head
# attends only to its own retained rows. ``offset`` still tracks the true
# sequence position for future RoPE. The head-CLASSIFICATION policy (which heads
# are retrieval vs streaming) is intentionally out of scope here: this module
# only provides the cache representation and the eviction op that applies a
# caller-supplied ``head_keep_indices``.
# ---------------------------------------------------------------------------


class HeadPartitionedKVCache(_BaseCache):
    """Sparse KV cache with independent retained positions per KV head.

    This is the representation DuoAttention eviction needs. Storage uses the
    union of all retained prefix positions, but ``make_mask`` exposes a per-head
    prefix mask so retrieval heads can attend long-range retained rows while
    streaming heads attend only their own recency rows. ``offset`` remains the
    true logical sequence position for future RoPE.
    """

    def __init__(
        self,
        keys=None,
        values=None,
        *,
        offset: int = 0,
        positions: Optional[Sequence[int]] = None,
        head_positions: Optional[Sequence[Sequence[int]]] = None,
        query_heads: Optional[int] = None,
        protected_stored: Optional[int] = None,
    ):
        self.keys = keys
        self.values = values
        self.offset = int(offset)
        self._stored = 0 if keys is None else int(keys.shape[2])
        self._positions = self._coerce_positions(positions)
        self._head_positions = self._coerce_head_positions(head_positions)
        self.query_heads = None if query_heads is None else int(query_heads)
        self._head_position_mask = self._build_head_position_mask()
        self._protected_stored = (
            self._stored if protected_stored is None else int(protected_stored)
        )
        if self._protected_stored < 0 or self._protected_stored > self._stored:
            raise ValueError("protected_stored must be between 0 and stored rows")
        self._speculating = False
        self._speculative_appends = deque()

    def _coerce_positions(self, positions: Optional[Sequence[int]]):
        if positions is None:
            if self._stored == 0:
                return ()
            if self.offset == self._stored:
                return tuple(range(self._stored))
            raise ValueError("positions are required for sparse head-partitioned KV")
        out = tuple(int(p) for p in positions)
        if len(out) != self._stored:
            raise ValueError("positions length must match stored rows")
        if any(p < 0 for p in out):
            raise ValueError("positions must be non-negative")
        if tuple(sorted(out)) != out:
            raise ValueError("positions must be sorted")
        return out

    def _coerce_head_positions(self, head_positions):
        if self.keys is None:
            n_heads = 0
        else:
            n_heads = int(self.keys.shape[1])
        if head_positions is None:
            return tuple(self._positions for _ in range(n_heads))
        out = []
        valid = set(self._positions)
        for row in head_positions:
            vals = tuple(int(p) for p in row)
            if tuple(sorted(vals)) != vals:
                raise ValueError("head positions must be sorted")
            if any(p not in valid for p in vals):
                raise ValueError("head positions must be a subset of positions")
            out.append(vals)
        if n_heads and len(out) != n_heads:
            raise ValueError("head_positions length must match KV heads")
        return tuple(out)

    def _build_head_position_mask(self):
        rows = []
        for row in self._head_positions:
            keep = set(row)
            rows.append([pos in keep for pos in self._positions])
        return tuple(tuple(row) for row in rows)

    @staticmethod
    def _encode_positions(positions):
        return ",".join(map(str, positions))

    @staticmethod
    def _decode_positions(raw: str):
        if raw == "":
            return ()
        return tuple(int(p) for p in raw.split(","))

    @staticmethod
    def _encode_head_positions(head_positions):
        return "|".join(",".join(map(str, positions)) for positions in head_positions)

    @classmethod
    def _decode_head_positions(cls, raw: str):
        if raw == "":
            return ()
        return tuple(cls._decode_positions(part) for part in raw.split("|"))

    @property
    def positions(self):
        return self._positions

    @property
    def head_positions(self):
        return self._head_positions

    @property
    def protected_stored(self):
        return self._protected_stored

    def size(self):
        return self._stored

    @property
    def state(self):
        if self.keys is None:
            return None
        return self.keys[..., : self._stored, :], self.values[..., : self._stored, :]

    @state.setter
    def state(self, v):
        if v is None:
            self.keys = None
            self.values = None
            self._stored = 0
            self.offset = 0
            self._positions = ()
            self._head_positions = ()
            self.query_heads = None
            self._head_position_mask = ()
            self._protected_stored = 0
            self._speculating = False
            self._speculative_appends = deque()
            return
        self.keys, self.values = v
        self._stored = int(self.keys.shape[2])
        self.offset = self._stored
        self._positions = tuple(range(self._stored))
        self._head_positions = tuple(
            self._positions for _ in range(int(self.keys.shape[1]))
        )
        self.query_heads = None
        self._head_position_mask = self._build_head_position_mask()
        self._protected_stored = self._stored
        self._speculating = False
        self._speculative_appends = deque()

    @property
    def meta_state(self):
        return tuple(
            map(
                str,
                (
                    self.offset,
                    self._stored,
                    self._protected_stored,
                    self._encode_positions(self._positions),
                    self._encode_head_positions(self._head_positions),
                    "" if self.query_heads is None else self.query_heads,
                ),
            )
        )

    @meta_state.setter
    def meta_state(self, v):
        vals = tuple(v)
        if len(vals) < 5:
            raise ValueError("HeadPartitionedKVCache meta_state is incomplete")
        self.offset, self._stored, self._protected_stored = map(int, vals[:3])
        self._positions = self._decode_positions(str(vals[3]))
        self._head_positions = self._decode_head_positions(str(vals[4]))
        self.query_heads = None if len(vals) < 6 or str(vals[5]) == "" else int(vals[5])
        if len(self._positions) != self._stored:
            raise ValueError("stored row count does not match position metadata")
        if self.keys is not None and len(self._head_positions) != int(
            self.keys.shape[1]
        ):
            raise ValueError("head position count does not match KV heads")
        self._head_position_mask = self._build_head_position_mask()
        self._speculating = False
        self._speculative_appends = deque()

    def update_and_fetch(self, keys, values):
        n_new = int(keys.shape[2])
        start_pos = self.offset
        if self.keys is None:
            self.keys = keys
            self.values = values
            self._stored = n_new
            self.offset = n_new
            self._positions = tuple(range(n_new))
            self._head_positions = tuple(
                self._positions for _ in range(int(keys.shape[1]))
            )
            self.query_heads = None
            self._head_position_mask = self._build_head_position_mask()
            self._protected_stored = self._stored
            return self.state

        if int(keys.shape[1]) != int(self.keys.shape[1]):
            raise ValueError("new keys must have the same KV head count")
        self.keys = mx.concatenate([self.keys[..., : self._stored, :], keys], axis=2)
        self.values = mx.concatenate(
            [self.values[..., : self._stored, :], values], axis=2
        )
        new_positions = tuple(range(start_pos, start_pos + n_new))
        self._stored += n_new
        self.offset += n_new
        self._positions = self._positions + new_positions
        self._head_positions = tuple(
            row + new_positions for row in self._head_positions
        )
        self._head_position_mask = self._build_head_position_mask()
        if self._speculating and n_new:
            self._speculative_appends.append(n_new)
        return self.state

    def make_mask(
        self,
        n_tokens: int,
        window_size=None,
        return_array: bool = False,
        *,
        query_heads: Optional[int] = None,
    ):
        if window_size is not None:
            raise ValueError(
                "HeadPartitionedKVCache does not support sliding-window masks"
            )
        if n_tokens <= 0:
            raise ValueError("n_tokens must be positive")
        prefix = mx.array(self._head_position_mask, dtype=mx.bool_)
        if query_heads is not None:
            effective_query_heads = int(query_heads)
        elif self.query_heads is not None:
            effective_query_heads = int(self.query_heads)
        else:
            effective_query_heads = None
        if effective_query_heads is not None:
            kv_heads = prefix.shape[0]
            if effective_query_heads % kv_heads != 0:
                raise ValueError("query_heads must be a multiple of KV heads")
            prefix = mx.repeat(prefix, effective_query_heads // kv_heads, axis=0)
        prefix = mx.broadcast_to(
            prefix[:, None, :],
            (prefix.shape[0], n_tokens, self._stored),
        )
        causal = mx.tril(mx.ones((n_tokens, n_tokens), dtype=mx.bool_))
        causal = mx.broadcast_to(
            causal[None, :, :],
            (prefix.shape[0], n_tokens, n_tokens),
        )
        return mx.concatenate([prefix, causal], axis=2)[None, ...]

    def is_trimmable(self):
        return self._speculating or self._positions is not None

    def start_speculation(self):
        self._speculating = True
        self._speculative_appends.clear()

    def stop_speculation(self):
        self._speculating = False
        self._speculative_appends.clear()

    def trim(self, n):
        if n <= 0:
            return 0
        n = int(n)
        old_positions = self._positions
        if self._speculating:
            if not self._speculative_appends:
                raise RuntimeError("No speculative append is available to trim")
            latest = self._speculative_appends.pop()
            if n > latest:
                self._speculative_appends.append(latest)
                raise RuntimeError("trim exceeds latest speculative append")
            removable = self._stored - self._protected_stored
            if n > removable:
                self._speculative_appends.append(latest)
                raise RuntimeError("trim exceeds removable speculative rows")
            if n < latest:
                self._speculative_appends.append(latest - n)
        n = min(n, self.offset)
        new_offset = self.offset - n
        keep_physical = [i for i, pos in enumerate(self._positions) if pos < new_offset]
        if len(keep_physical) != self._stored:
            if keep_physical:
                self.keys = _take_positions(
                    self.keys[..., : self._stored, :], keep_physical
                )
                self.values = _take_positions(
                    self.values[..., : self._stored, :], keep_physical
                )
            else:
                self.keys = self.keys[..., :0, :]
                self.values = self.values[..., :0, :]
        self._positions = tuple(old_positions[i] for i in keep_physical)
        self._head_positions = tuple(
            tuple(pos for pos in row if pos < new_offset)
            for row in self._head_positions
        )
        self._stored = len(self._positions)
        self._protected_stored = sum(
            1 for pos in old_positions[: self._protected_stored] if pos < new_offset
        )
        self.offset = new_offset
        self._head_position_mask = self._build_head_position_mask()
        return n

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return (
            self.keys[..., : self._stored, :].nbytes
            + self.values[..., : self._stored, :].nbytes
        )


def _union_positions(
    head_keep_indices: Sequence[Sequence[int]],
) -> Tuple[int, ...]:
    out = sorted({int(pos) for row in head_keep_indices for pos in row})
    return tuple(out)


def _validate_head_keep_indices(head_keep_indices, n_heads: int, offset: int):
    if len(head_keep_indices) != n_heads:
        raise ValueError("head_keep_indices length must match KV heads")
    out = []
    for row in head_keep_indices:
        vals = tuple(int(pos) for pos in row)
        if tuple(sorted(vals)) != vals:
            raise ValueError("head keep indices must be sorted")
        if any(pos < 0 or pos >= offset for pos in vals):
            raise ValueError("head keep index exceeds KV cache offset")
        out.append(vals)
    return tuple(out)


def _take_head_partitioned(x, union_keep, head_keep_indices):
    bsz, n_heads, _tokens, dim = x.shape
    zero = mx.zeros((bsz, 1, 1, dim), dtype=x.dtype)
    head_chunks = []
    for head, keep in enumerate(head_keep_indices):
        keep_set = set(keep)
        rows = []
        for pos in union_keep:
            rows.append(
                x[:, head : head + 1, pos : pos + 1, :] if pos in keep_set else zero
            )
        if rows:
            head_chunks.append(mx.concatenate(rows, axis=2))
        else:
            head_chunks.append(x[:, head : head + 1, :0, :])
    return mx.concatenate(head_chunks, axis=1) if head_chunks else x[:, :0, :0, :]


@dataclass(frozen=True)
class HeadPartitionedEvictionResult:
    cache: list
    evicted: bool
    true_offset: int
    union_retained_tokens: int
    per_head_retained_tokens: Tuple[int, ...]
    original_tokens: int
    kv_layers: int
    compact_cache_nbytes: int


def evict_prompt_cache_by_head(
    prompt_cache: List[Any],
    head_keep_indices: Sequence[Sequence[int]],
    *,
    true_offset: int,
    query_heads: Optional[int] = None,
) -> HeadPartitionedEvictionResult:
    """Replace plain ``KVCache`` layers with a DuoAttention-ready cache.

    ``head_keep_indices`` gives, per KV head, the sorted prompt positions that
    head retains (retrieval heads typically keep everything; streaming heads
    keep only sinks + a recent window). Every other layer type is left
    untouched. The head-classification decision is the caller's responsibility.
    """
    out = list(prompt_cache)
    original_tokens = int(true_offset)
    kv_layers = 0
    union_keep = _union_positions(head_keep_indices)
    per_head_counts = tuple(len(row) for row in head_keep_indices)
    for idx, cache in enumerate(prompt_cache):
        if type(cache) is not KVCache or cache.keys is None:
            continue
        keys, values = cache.state
        n_heads = int(keys.shape[1])
        head_keep = _validate_head_keep_indices(
            head_keep_indices, n_heads, int(cache.offset)
        )
        union_keep = _union_positions(head_keep)
        new_keys = _take_head_partitioned(keys, union_keep, head_keep)
        new_values = _take_head_partitioned(values, union_keep, head_keep)
        out[idx] = HeadPartitionedKVCache(
            new_keys,
            new_values,
            offset=true_offset,
            positions=union_keep,
            head_positions=head_keep,
            query_heads=query_heads,
        )
        original_tokens = max(original_tokens, int(cache.offset))
        kv_layers += 1

    compact_cache_nbytes = sum(int(getattr(cache, "nbytes", 0) or 0) for cache in out)
    return HeadPartitionedEvictionResult(
        cache=out,
        evicted=kv_layers > 0
        and any(count < original_tokens for count in per_head_counts),
        true_offset=int(true_offset),
        union_retained_tokens=len(union_keep),
        per_head_retained_tokens=per_head_counts,
        original_tokens=original_tokens,
        kv_layers=kv_layers,
        compact_cache_nbytes=compact_cache_nbytes,
    )
