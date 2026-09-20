"""KV cache that holds only one rank's shard of the global context.

Two positions matter here and they are *not* the same thing:

- ``offset`` (used for RoPE) is the true global position of the token(s)
  currently being processed. It must be identical across every rank, because
  it describes one shared token stream, not any single rank's local shard.
  It defaults to ``shard_start + local_offset`` (correct when this shard
  always holds the most recently generated token), but can be set explicitly
  -- e.g. by a decode loop that knows the true global step count -- when a
  different shard owns the newest token.
- ``local_offset`` / ``shard_start`` describe where in *this rank's* local
  buffer data is stored; only used for storage indexing.

``owns_new_token`` controls whether ``update_and_fetch`` actually stores the
given keys/values. During distributed decode only one shard should grow per
step (typically the shard holding the most recent context); other ranks
still need to compute Q/K for the new token (for RoPE / the local matmul)
but must not add it to their own storage, or it would be double-counted
across shards during the attention merge.
"""

from typing import Optional

import mlx.core as mx
from mlx.utils import tree_map, tree_reduce

from ..distributed_attention import (
    sharded_prefill_attention,
    sharded_query_attention,
    sharded_scaled_dot_product_attention,
)
from .cache import _BaseCache


class ShardedKVCache(_BaseCache):
    step = 256

    def __init__(
        self,
        shard_start: int = 0,
        owns_new_token: bool = True,
        shard_lengths=None,
        kv_bits: Optional[int] = None,
        group_size: int = 64,
        group=None,
    ):
        # The mx.distributed group this shard belongs to. While it is set, the
        # shared ``scaled_dot_product_attention`` routes attention here.
        self.group = group
        # kv_bits None: keys/values stored as-is. Otherwise they are stored
        # quantized as (packed, scales, biases) tuples (mx.quantize) and the
        # attention reads them without making a full-precision copy.
        self.kv_bits = kv_bits
        self.group_size = group_size
        self.keys = None
        self.values = None
        self.local_offset = 0
        self.shard_start = shard_start
        self.owns_new_token = owns_new_token
        # Per-rank prefill shard lengths when shards are unequal; None = equal.
        self.shard_lengths = shard_lengths
        # True: multi-token inputs are replicated queries against the sharded
        # past (no ring); False: multi-token inputs are a prefill of new shards.
        self.query_mode = False
        # One id per token of the block being prefilled (-1 = text): tokens of
        # the same image attend to each other in both directions.
        self.image_groups = None
        self._query_offset: Optional[int] = None

    @property
    def offset(self):
        if self._query_offset is not None:
            return self._query_offset
        return self.shard_start + self.local_offset

    @offset.setter
    def offset(self, value: int):
        self._query_offset = value

    def update_and_fetch(self, keys, values):
        if not self.owns_new_token:
            return self.keys_and_values()
        if self.kv_bits is not None:
            return self._update_and_fetch_quantized(keys, values)

        prev = self.local_offset
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

        self.local_offset += keys.shape[2]
        self.keys[..., prev : self.local_offset, :] = keys
        self.values[..., prev : self.local_offset, :] = values
        return self.keys_and_values()

    def _update_and_fetch_quantized(self, keys, values):
        B, n_kv_heads, num_steps, k_dim = keys.shape
        v_dim = values.shape[-1]
        prev = self.local_offset
        if self.keys is None or (prev + num_steps) > self.keys[0].shape[-2]:
            el_per_int = 8 * mx.uint32.size // self.kv_bits
            new_steps = (self.step + num_steps - 1) // self.step * self.step
            shape = (B, n_kv_heads, new_steps)

            def init_quant(dim):
                return (
                    mx.zeros((*shape, dim // el_per_int), dtype=mx.uint32),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                )

            def expand_quant(x):
                return mx.concatenate(
                    [x, mx.zeros((*shape, x.shape[-1]), dtype=x.dtype)], axis=-2
                )

            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys, self.values = tree_map(
                        lambda x: x[..., :prev, :], (self.keys, self.values)
                    )
                self.keys, self.values = tree_map(expand_quant, (self.keys, self.values))
            else:
                self.keys, self.values = init_quant(k_dim), init_quant(v_dim)

        self.local_offset += num_steps
        qk = mx.quantize(keys, group_size=self.group_size, bits=self.kv_bits)
        qv = mx.quantize(values, group_size=self.group_size, bits=self.kv_bits)
        for i in range(3):
            self.keys[i][..., prev : self.local_offset, :] = qk[i]
            self.values[i][..., prev : self.local_offset, :] = qv[i]
        return self.keys_and_values()

    def keys_and_values(self):
        if self.keys is None:
            return self.keys, self.values
        if self.kv_bits is not None:
            n = self.local_offset
            if n == self.keys[0].shape[2]:
                return self.keys, self.values
            return tree_map(lambda x: x[..., :n, :], (self.keys, self.values))
        if self.local_offset < self.keys.shape[2]:
            return (
                self.keys[..., : self.local_offset, :],
                self.values[..., : self.local_offset, :],
            )
        return self.keys, self.values

    def attend(self, queries, keys, values, scale, softcap=None):
        """Attention of ``queries`` against the sharded context (this rank's
        ``keys``/``values`` are what ``update_and_fetch`` just returned)."""
        L = queries.shape[2]
        if L == 1:
            # Decode: every cached key is in the past, so no mask is needed.
            return sharded_scaled_dot_product_attention(
                queries, keys, values, scale=scale, group=self.group, softcap=softcap
            )
        if self.query_mode:
            # Prepared corpus: replicated question tokens against the sharded past.
            return sharded_query_attention(
                queries,
                keys,
                values,
                scale=scale,
                group=self.group,
                new_len=L,
                owns_new=self.owns_new_token,
                softcap=softcap,
                image_groups=self.image_groups,
            )
        # Prefill: each rank's chunk of queries, ring-combined over all shards.
        return sharded_prefill_attention(
            queries,
            keys,
            values,
            scale=scale,
            group=self.group,
            shard_lengths=self.shard_lengths,
            softcap=softcap,
        )

    def size(self):
        return self.local_offset

    @property
    def state(self):
        return (
            self.keys,
            self.values,
            self.local_offset,
            self.shard_start,
            self.owns_new_token,
        )

    @state.setter
    def state(self, v):
        (
            self.keys,
            self.values,
            self.local_offset,
            self.shard_start,
            self.owns_new_token,
        ) = v

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.local_offset, n)
        self.local_offset -= n
        return n

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return tree_reduce(lambda a, x: a + x.nbytes, (self.keys, self.values), 0)
