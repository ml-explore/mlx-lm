# Copyright © 2025 Apple Inc.

"""SpeculativeArraysCache - ArraysCache variant supporting checkpoint/rollback for speculative decoding.

This cache is designed for use with speculative decoding (e.g., DFlash) where draft tokens
may be rejected and need to be rolled back. Unlike ArraysCache which uses a rolling buffer
that cannot selectively remove tokens, SpeculativeArraysCache tracks committed vs speculative
tokens and supports O(1) rollback via checkpoint/restore operations.

Key differences from ArraysCache:
- Tracks committed_up_to position (all tokens <= committed_up_to are "finalized")
- Supports save_checkpoint() before draft, rollback() on rejection
- Uses valid flag for O(1) rollback (no need to copy arrays)
- Matches ArraysCache interface for drop-in compatibility
"""

import logging
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .cache import _BaseCache, ArraysCache, KVCache

logger = logging.getLogger(__name__)


class SpeculativeArraysCache(_BaseCache):
    """ArraysCache variant supporting checkpoint/rollback for speculative decoding.

    This cache is designed for models with linear attention layers (e.g., Qwen3.5)
    that use a rolling buffer cache (ArraysCache). When used with speculative
    decoding, draft tokens may be rejected and need to be rolled back.

    The key innovation is tracking which tokens are "committed" (verified by
    target model) vs "speculative" (proposed by draft model). On rejection,
    we rollback to the last checkpoint, invalidating all speculative tokens.

    Usage pattern in speculative decoding:
        1. cache.save_checkpoint() - before running draft model
        2. ... run draft model, update cache with speculative tokens ...
        3. ... verify draft tokens with target model ...
        4. If tokens rejected: cache.rollback() - discard speculative tokens
        5. If tokens accepted: cache.commit(up_to_position) - mark as committed

    Args:
        size: Number of cache entries (typically 2 for Qwen3.5 linear attention)
        conv_kernel_size: Convolution kernel size (default 4 for Qwen3.5)
        max_size: Maximum sequence length for pre-allocation (default 8192)
    """

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.left_padding = None
        instance.lengths = None
        instance._offset = 0
        instance._recording = False
        instance._verify_qkvgb = None
        instance._conv_input = None
        return instance

    def __init__(
        self,
        size: int,
        conv_kernel_size: int = 4,
        max_size: int = 8192,
        left_padding: Optional[List[int]] = None,
    ):
        """Initialize SpeculativeArraysCache.

        Args:
            size: Number of cache entries (typically 2 for Qwen3.5 linear attention)
            conv_kernel_size: Convolution kernel size (default 4 for Qwen3.5)
            max_size: Maximum sequence length for pre-allocation (default 8192)
            left_padding: Optional left padding for batch processing
        """
        # Use standard ArraysCache storage (list of arrays)
        self.cache = [None] * size
        self.size = size

        # Linear attention parameters
        self.conv_kernel_size = conv_kernel_size
        self.n_keep = conv_kernel_size - 1  # Number of tokens to keep in rolling buffer

        # Speculative decoding state
        self.max_size = max_size
        self._committed_up_to = 0  # All positions <= committed_up_to are finalized
        self._checkpoint_committed_up_to = 0  # Checkpoint state for rollback

        # Batch processing support
        if left_padding:
            self.left_padding = mx.array(left_padding)

    @property
    def offset(self):
        """Get current offset based on actual array size."""
        if self.cache[0] is not None:
            return self.cache[0].shape[1] if len(self.cache[0].shape) >= 2 else 0
        return 0

    @offset.setter
    def offset(self, value):
        """Set offset manually (for testing or special cases)."""
        self._offset = value

    @property
    def committed_up_to(self):
        """Get the position up to which tokens are committed."""
        return self._committed_up_to

    def __getitem__(self, idx):
        """Return cache entry directly (same as ArraysCache)."""
        return self.cache[idx]

    def __setitem__(self, idx, value):
        """Set cache entry at index."""
        self.cache[idx] = value

    @property
    def state(self):
        """Return cache state for serialization."""
        return self.cache

    @state.setter
    def state(self, v):
        """Set cache state from serialization."""
        self.cache = v

    def save_checkpoint(self):
        """Save current state before running draft model.

        This should be called BEFORE running the draft model to save a checkpoint.
        If draft tokens are rejected, call rollback() to restore this state.
        """
        # Save the committed_up_to position
        self._checkpoint_committed_up_to = self._committed_up_to

        # Save snapshots of cache arrays at checkpoint time
        self._checkpoint_cache = []
        for i in range(len(self.cache)):
            if self.cache[i] is not None:
                # Create a copy of the cache array at checkpoint
                self._checkpoint_cache.append(self.cache[i])
            else:
                self._checkpoint_cache.append(None)

        logger.debug(f"SpeculativeArraysCache: checkpoint saved at position {self._committed_up_to}")

    def rollback(self):
        """Rollback cache to last checkpoint, discarding all speculative tokens.

        This restores the cache arrays to the exact state they were in at checkpoint time.
        """
        old_committed = self._committed_up_to
        self._committed_up_to = self._checkpoint_committed_up_to

        # Restore cache arrays from checkpoint snapshots
        for i in range(len(self.cache)):
            if self._checkpoint_cache[i] is not None:
                # Restore the saved cache array
                self.cache[i] = self._checkpoint_cache[i]
            else:
                self.cache[i] = None

        logger.debug(f"SpeculativeArraysCache: rollback from {old_committed} to {self._committed_up_to}")

    def start_recording(self):
        """Enable q/k/v/g/beta saving for the next forward pass.

        When recording is enabled, the linear attention layer saves the
        intermediate tensors needed to replay the recurrent state update
        to any position. After the forward pass, call restore_to_position()
        to set the cache to the correct state via a cheap replay (no full
        model rebuild needed).
        """
        self._recording = True
        self._verify_qkvgb = None
        self._conv_input = None
        # Save checkpoint so replay has the correct starting state
        self.save_checkpoint()

    def restore_to_position(self, position):
        """Restore cache to the state at the given block position via replay.

        Replays the cheap recurrent state update for positions 0..position
        using the q/k/v/g/beta tensors saved during the verify forward pass.
        This avoids a full model rebuild — just iterates the simple state
        update loop for acceptance_length+1 positions.

        Args:
            position: Block position to restore to (0-indexed).
                The state is restored to what it was after processing
                tokens 0 through `position`.
        """
        from .gated_delta import _gated_delta_step_ops

        if self._verify_qkvgb is not None:
            q, k, v, g, beta = self._verify_qkvgb
            # Handle q/k repeat for Hv > Hk (same as gated_delta_ops)
            Hk = q.shape[-2]
            Hv = v.shape[-2]
            if (repeat_factor := Hv // Hk) > 1:
                q = mx.repeat(q, repeat_factor, -2)
                k = mx.repeat(k, repeat_factor, -2)
            # Start from pre-verify checkpoint state
            state = self._checkpoint_cache[1]
            # Replay only accepted positions
            for t in range(position + 1):
                _, state = _gated_delta_step_ops(
                    q[:, t], k[:, t], v[:, t], g[:, t], beta[:, t], state
                )
            self.cache[1] = state

        if self._conv_input is not None:
            n_keep = self.conv_kernel_size - 1
            # conv_input = [old_conv_state(n_keep), block_tokens(T)]
            # We want the n_keep tokens ending at block position `position`
            start_idx = position + 1  # offset for old conv_state prefix
            end_idx = start_idx + n_keep
            if end_idx <= self._conv_input.shape[1]:
                self.cache[0] = mx.contiguous(
                    self._conv_input[:, start_idx:end_idx, :]
                )

        self._recording = False
        self._verify_qkvgb = None
        self._conv_input = None

    def commit(self, up_to: int):
        """Mark tokens up to position as committed.

        Args:
            up_to: Position up to which tokens should be committed (inclusive)

        This should be called after target model verification when tokens are accepted.
        """
        if up_to > self._committed_up_to:
            self._committed_up_to = up_to
            logger.debug(f"SpeculativeArraysCache: committed up to position {up_to}")

    def filter(self, batch_indices):
        """In-place filter to keep just the given indices in the cache."""
        self.cache = [c[batch_indices] if c is not None else None for c in self.cache]
        if self.lengths is not None:
            self.lengths = self.lengths[batch_indices]

    def extend(self, other):
        """In-place extend this cache with the other cache."""
        def cat(a, b):
            if a is None:
                return b
            if b is None:
                return a
            return mx.concatenate([a, b])

        old_offset = self.offset
        self.cache = [cat(c, o) for c, o in zip(self.cache, other.cache)]
        if self.cache[0] is not None:
            self.offset = self.cache[0].shape[1]
        else:
            self.offset = old_offset

    def extract(self, idx):
        """Extract cache for a single batch index."""
        cache = SpeculativeArraysCache(
            size=self.size,
            conv_kernel_size=self.conv_kernel_size,
            max_size=self.max_size,
        )
        cache.cache = [c[idx : idx + 1] for c in self.cache]
        return cache

    def prepare(self, lengths=None, **kwargs):
        """Prepare cache for batch processing."""
        self.lengths = mx.array(lengths)

    def finalize(self):
        """Finalize cache after batch processing."""
        self.lengths = None
        self.left_padding = None

    def size(self):
        """Return the current valid size based on actual arrays."""
        return self.offset

    def trim(self, n: int) -> int:
        """Trim n tokens by slicing arrays from the beginning.

        Note: This is different from rollback() which discards speculative tokens.
        trim() is used for removing prefix tokens (e.g., in prefill-to-decode transition).

        Args:
            n: Number of tokens to trim from the beginning

        Returns:
            Actual number of tokens trimmed
        """
        if self.cache[0] is None:
            return 0
        current_len = self.cache[0].shape[1] if len(self.cache[0].shape) >= 2 else 0
        n = min(n, current_len)
        keep_len = current_len - n
        if keep_len > 0:
            for i in range(len(self.cache)):
                if self.cache[i] is not None:
                    self.cache[i] = self.cache[i][:, keep_len:, ...]
            # Also adjust committed_up_to
            self._committed_up_to = max(0, self._committed_up_to - n)
        return n

    def is_trimmable(self):
        """Return True - this cache supports trim operations."""
        return True

    def crop(self, target_len):
        """Crop cache to keep only the first target_len tokens.

        Unlike trim() which removes from the beginning, this removes
        tokens from the END — used for discarding rejected speculative
        tokens while keeping verified context.

        Args:
            target_len: Number of tokens to keep from the beginning
        """
        if self.cache[0] is None:
            return
        current_len = self.cache[0].shape[1] if len(self.cache[0].shape) >= 2 else 0
        if current_len > target_len:
            for i in range(len(self.cache)):
                if self.cache[i] is not None:
                    self.cache[i] = self.cache[i][:, :target_len, ...]
            self._committed_up_to = min(self._committed_up_to, target_len)

    def advance(self, N):
        """Advance cache position by N tokens."""
        if self.lengths is not None:
            self.lengths -= N
        if self.left_padding is not None:
            self.left_padding -= N

    def make_mask(self, N: int):
        """Make attention mask for batch processing."""
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
        """Merge multiple caches into a single batched cache."""
        n_state = len(caches[0].cache)
        B = len(caches)
        cache = cls(n_state)

        if all(c.empty() for c in caches):
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
        """Return True if cache is empty."""
        return self.cache[0] is None

    @property
    def nbytes(self):
        """Return the size of this cache in bytes."""
        return sum(c.nbytes for c in self.cache if c is not None)


def make_speculative_cache(
    model: nn.Module,
    max_kv_size: Optional[int] = None,
    conv_kernel_size: int = 4,
    max_size: int = 8192,
) -> List[Any]:
    """Create speculative cache for models with linear attention layers.

    This function creates a SpeculativeArraysCache for linear attention layers
    and standard KVCache for regular attention layers.

    Args:
        model: The language model
        max_kv_size: Maximum KV cache size (not used for SpeculativeArraysCache)
        conv_kernel_size: Convolution kernel size (default 4 for Qwen3.5)
        max_size: Maximum sequence length for SpeculativeArraysCache

    Returns:
        List of cache objects (SpeculativeArraysCache for linear layers, KVCache for others)
    """
    if hasattr(model, "make_speculative_cache"):
        return model.make_speculative_cache()

    num_layers = len(model.layers)
    from .cache import KVCache

    return [
        SpeculativeArraysCache(size=2, conv_kernel_size=conv_kernel_size, max_size=max_size)
        if l.is_linear
        else KVCache()
        for l in model.layers
    ]
