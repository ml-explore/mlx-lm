# DeepSeek V4 model implementation for MLX.
#
# Ported from deepseek-ai/DeepSeek-V4-Flash/inference/model.py
#
# Architecture:
#   - Compressed Sparse Attention (CSA, ratio=4) with Lightning Indexer
#   - Heavily Compressed Attention (HCA, ratio=128)
#   - Sliding window (128 tokens) for local context
#   - Hyper-Connections (HC) replacing standard residuals
#   - Hash routing for first N MoE layers
#   - Grouped output projection (o_groups)

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask
from .cache import KVCache, RotatingKVCache, _BaseCache, dynamic_roll
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU

# Register with transformers so AutoTokenizer/AutoConfig work
try:
    from transformers import AutoConfig, PretrainedConfig

    class _DeepseekV4Config(PretrainedConfig):
        model_type = "deepseek_v4"

        def __init__(self, **kw):
            self.rope_scaling = kw.pop("rope_scaling", None)
            super().__init__(**kw)

    AutoConfig.register("deepseek_v4", _DeepseekV4Config)
except Exception:
    pass


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "deepseek_v4"
    vocab_size: int = 129280
    hidden_size: int = 4096
    num_hidden_layers: int = 43
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    head_dim: int = 512
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    o_groups: int = 8
    qk_rope_head_dim: int = 64
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    attention_bias: bool = False
    attention_dropout: float = 0.0
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6
    moe_intermediate_size: int = 2048
    scoring_func: str = "sqrtsoftplus"
    routed_scaling_factor: float = 1.5
    norm_topk_prob: bool = True
    topk_method: str = "noaux_tc"
    swiglu_limit: float = 10.0
    num_hash_layers: int = 3
    compress_ratios: List[int] = field(default_factory=list)
    compress_rope_theta: float = 160000.0
    sliding_window: int = 128
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    num_nextn_predict_layers: int = 1
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict] = None
    tie_word_embeddings: bool = False


# ---------------------------------------------------------------------------
# Sparse KV Cache
# ---------------------------------------------------------------------------

class SparseKVCache(_BaseCache):
    """Cache for compressed layers: stores window + compressed buffers
    plus compressor/indexer decode state. Survives cache save/load."""

    step = 256

    # Extra state attrs beyond keys/values (order matters for serialization)
    _SPARSE_ATTRS = (
        'win_buf', 'comp_buf',
        'comp_kv_state', 'comp_score_state',
        'idx_kv', 'idx_comp_kv_state', 'idx_comp_score_state',
    )

    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0
        for attr in self._SPARSE_ATTRS:
            setattr(self, attr, None)

    def update_and_fetch(self, keys, values):
        prev = self.offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            needed = prev + keys.shape[2]
            n_steps = (needed + self.step - 1) // self.step
            new_k = mx.zeros((B, n_kv_heads, n_steps * self.step, k_head_dim), keys.dtype)
            new_v = mx.zeros((B, n_kv_heads, n_steps * self.step, v_head_dim), values.dtype)
            if self.keys is not None:
                new_k[..., :prev, :] = self.keys[..., :prev, :]
                new_v[..., :prev, :] = self.values[..., :prev, :]
            self.keys = new_k
            self.values = new_v
        self.offset += keys.shape[2]
        self.keys[..., prev:self.offset, :] = keys
        self.values[..., prev:self.offset, :] = values
        return self.keys[..., :self.offset, :], self.values[..., :self.offset, :]

    def empty(self):
        return self.keys is None and self.win_buf is None

    @property
    def state(self):
        if self.keys is None:
            return (None, None)
        parts = [self.keys[..., :self.offset, :],
                 self.values[..., :self.offset, :]]
        # Always include ALL attrs (None if absent) to maintain positional alignment
        for attr in self._SPARSE_ATTRS:
            parts.append(getattr(self, attr, None))
        return tuple(parts)

    @state.setter
    def state(self, v):
        if v is None or v[0] is None:
            return
        self.keys, self.values = v[0], v[1]
        self.offset = self.keys.shape[2]
        for i, attr in enumerate(self._SPARSE_ATTRS):
            idx = i + 2
            if idx < len(v):
                setattr(self, attr, v[idx])

    @property
    def meta_state(self):
        n = 2 + sum(1 for a in self._SPARSE_ATTRS
                     if getattr(self, a, None) is not None)
        return {"n_parts": str(n)}

    @classmethod
    def from_state(cls, state, meta_state):
        cache = cls()
        cache.state = state
        return cache

    @property
    def nbytes(self):
        total = 0
        if self.keys is not None:
            total += self.keys.nbytes + self.values.nbytes
        for attr in self._SPARSE_ATTRS:
            val = getattr(self, attr, None)
            if val is not None:
                total += val.nbytes
        return total

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        # Invalidate sparse state on trim (stale after position change)
        self.win_buf = None
        self.comp_buf = None
        self.comp_kv_state = None
        self.comp_score_state = None
        self.idx_kv = None
        self.idx_comp_kv_state = None
        self.idx_comp_score_state = None
        return n

    def is_trimmable(self):
        return True

    def size(self):
        return self.offset

    @classmethod
    def merge(cls, caches):
        return BatchSparseKVCache.merge(caches)


class BatchSparseKVCache(_BaseCache):
    """Batched version of SparseKVCache for concurrent request handling.

    Wraps multiple SparseKVCache entries into a single batched cache.
    Tracks per-entry offsets and batched sparse state (window buffers,
    compressed buffers, compressor/indexer state).

    During decode, the attention module processes sparse layers per-entry
    because the compressor state machine has entry-dependent modular
    arithmetic (offset % ratio). Dense layers (ratio=0) use
    BatchRotatingKVCache and are fully batched.
    """

    step = 256

    _SPARSE_ATTRS = SparseKVCache._SPARSE_ATTRS

    def __init__(self, left_padding):
        self.keys = None
        self.values = None
        self.left_padding = mx.array(left_padding)
        self.offset = mx.array([-l for l in left_padding])
        self._idx = 0
        self._right_padding = None
        for attr in self._SPARSE_ATTRS:
            setattr(self, attr, None)
        # Track per-entry sparse buffer counts for variable-length comp_buf
        self._comp_ns = None  # mx.array of per-entry comp counts

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

    def empty(self):
        return self.keys is None and self.win_buf is None

    def size(self):
        return self._idx

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        if left_padding is not None:
            if self.keys is not None:
                raise ValueError(
                    "Left padding can only be added to an empty BatchSparseKVCache"
                )
            left_padding = mx.array(left_padding)
            self.left_padding += left_padding
            self.offset -= left_padding

        if right_padding is not None and max(right_padding) > 0:
            self._right_padding = mx.array(right_padding)

    def finalize(self):
        if self._right_padding is not None:
            padding = self._right_padding
            if self.keys is not None:
                self.keys = dynamic_roll(self.keys, padding[:, None], axis=2)
                self.values = dynamic_roll(self.values, padding[:, None], axis=2)
            self.offset -= padding
            self.left_padding += padding
            self._right_padding = None

    def make_mask(self, N: int, return_array: bool = False, **kwargs):
        from .base import create_causal_mask
        return create_causal_mask(
            N, offset=self._idx, left_padding=self.left_padding, **kwargs
        )

    @property
    def state(self):
        k, v = self.keys, self.values
        if k is not None and self._idx < k.shape[2]:
            k = k[..., : self._idx, :]
            v = v[..., : self._idx, :]
        parts = [k, v, self.offset, self.left_padding]
        for attr in self._SPARSE_ATTRS:
            parts.append(getattr(self, attr, None))
        return tuple(parts)

    @state.setter
    def state(self, v):
        if v is None or v[0] is None:
            return
        self.keys, self.values, self.offset, self.left_padding = v[:4]
        self._idx = self.keys.shape[2] if self.keys is not None else 0
        for i, attr in enumerate(self._SPARSE_ATTRS):
            idx = i + 4
            if idx < len(v):
                setattr(self, attr, v[idx])

    @property
    def meta_state(self):
        return {"_idx": str(self._idx)}

    @meta_state.setter
    def meta_state(self, v):
        self._idx = int(v.get("_idx", 0))

    @classmethod
    def from_state(cls, state, meta_state):
        obj = cls.__new__(cls)
        obj._right_padding = None
        obj._comp_ns = None
        for attr in cls._SPARSE_ATTRS:
            setattr(obj, attr, None)
        obj.state = state
        obj.meta_state = meta_state
        return obj

    @property
    def nbytes(self):
        total = 0
        if self.keys is not None:
            total += self.keys.nbytes + self.values.nbytes
        for attr in self._SPARSE_ATTRS:
            val = getattr(self, attr, None)
            if val is not None:
                total += val.nbytes
        return total

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self._idx, n)
        self._idx -= n
        self.offset -= n
        for attr in self._SPARSE_ATTRS:
            setattr(self, attr, None)
        self._comp_ns = None
        return n

    def filter(self, batch_indices):
        """In-place filter to keep just the given indices in the cache."""
        if self.keys is not None:
            self.keys = self.keys[batch_indices]
            self.values = self.values[batch_indices]
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]

        for attr in self._SPARSE_ATTRS:
            val = getattr(self, attr, None)
            if val is not None:
                setattr(self, attr, val[batch_indices])
        if self._comp_ns is not None:
            self._comp_ns = self._comp_ns[batch_indices]

        # Reduce padding
        min_left_pad = self.left_padding.min().item()
        if min_left_pad > 0:
            if self.keys is not None:
                self.keys = self.keys[..., min_left_pad:, :]
                self.values = self.values[..., min_left_pad:, :]
            self._idx -= min_left_pad
            self.left_padding -= min_left_pad

    def extend(self, other):
        """In-place extend this cache with another BatchSparseKVCache."""
        if self.keys is None and other.keys is None:
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            self.offset = mx.concatenate([self.offset, other.offset])
            self._extend_sparse_attrs(other)
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

        def pad_kv(c):
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
                p = [(0, 0), (0, 0), (left, right), (0, 0)]
                k = mx.pad(k, p)
                v = mx.pad(v, p)
            left_padding = c.left_padding + left
            return k, v, c.offset, left_padding

        self.keys, self.values, self.offset, self.left_padding = map(
            mx.concatenate, zip(*(pad_kv(self), pad_kv(other)))
        )
        self._idx = max_idx
        self._extend_sparse_attrs(other)

    def _extend_sparse_attrs(self, other):
        """Concatenate sparse attrs along batch dim, padding as needed."""
        self_B = self.offset.shape[0]
        other_B = other.offset.shape[0]

        for attr in self._SPARSE_ATTRS:
            a = getattr(self, attr, None)
            b = getattr(other, attr, None)
            if a is None and b is None:
                continue
            if a is None:
                shape_a = list(b.shape)
                shape_a[0] = self_B - b.shape[0] if self_B > b.shape[0] else self_B
                a = mx.zeros(shape_a, dtype=b.dtype)
            if b is None:
                shape_b = list(a.shape)
                shape_b[0] = other_B
                b = mx.zeros(shape_b, dtype=a.dtype)
            # Pad along non-batch dims if shapes differ
            if a.shape[1:] != b.shape[1:]:
                max_shape = [max(sa, sb) for sa, sb in zip(a.shape[1:], b.shape[1:])]
                if list(a.shape[1:]) != max_shape:
                    pad_widths = [(0, 0)] + [(0, ms - s) for s, ms in zip(a.shape[1:], max_shape)]
                    a = mx.pad(a, pad_widths)
                if list(b.shape[1:]) != max_shape:
                    pad_widths = [(0, 0)] + [(0, ms - s) for s, ms in zip(b.shape[1:], max_shape)]
                    b = mx.pad(b, pad_widths)
            setattr(self, attr, mx.concatenate([a, b], axis=0))

        # Extend comp_ns
        a_ns = self._comp_ns
        b_ns = getattr(other, '_comp_ns', None)
        if a_ns is not None or b_ns is not None:
            if a_ns is None:
                a_ns = mx.zeros((self_B,), dtype=mx.int32)
            if b_ns is None:
                b_ns = mx.zeros((other_B,), dtype=mx.int32)
            self._comp_ns = mx.concatenate([a_ns, b_ns])

    def extract(self, idx):
        """Extract a single cache entry back to a SparseKVCache."""
        mx.eval(self.left_padding, self.offset)
        cache = SparseKVCache()
        padding = max(0, self.left_padding.tolist()[idx])
        offset_val = self.offset.tolist()[idx]

        if self.keys is not None:
            cache.keys = mx.contiguous(self.keys[idx : idx + 1, :, padding : self._idx])
            cache.values = mx.contiguous(self.values[idx : idx + 1, :, padding : self._idx])
        cache.offset = offset_val

        for attr in self._SPARSE_ATTRS:
            val = getattr(self, attr, None)
            if val is not None:
                setattr(cache, attr, mx.contiguous(val[idx : idx + 1]))
            else:
                setattr(cache, attr, None)

        return cache

    @classmethod
    def merge(cls, caches):
        """Merge multiple SparseKVCache instances into a BatchSparseKVCache."""
        lengths = [c.size() for c in caches]
        max_length = max(lengths)

        if max_length == 0:
            return cls([0] * len(caches))

        padding = [max_length - l for l in lengths]
        B = len(caches)

        # Merge keys/values (these are dummy offset trackers in sparse layers)
        has_keys = any(c.keys is not None for c in caches)
        if has_keys:
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
        else:
            keys = None
            values = None

        batch_cache = cls(padding)
        batch_cache.keys = keys
        batch_cache.values = values
        if keys is not None:
            batch_cache.offset += keys.shape[2]
            batch_cache._idx = keys.shape[2]

        # Merge sparse attrs: pad + concatenate along batch dim
        for attr in SparseKVCache._SPARSE_ATTRS:
            vals = [getattr(c, attr, None) for c in caches]
            if all(v is None for v in vals):
                setattr(batch_cache, attr, None)
                continue
            # Find max shape along non-batch dims
            shapes = [v.shape for v in vals if v is not None]
            ndim = len(shapes[0])
            max_shape = list(shapes[0])
            for s in shapes[1:]:
                for d in range(1, ndim):
                    max_shape[d] = max(max_shape[d], s[d])
            dt = next(v.dtype for v in vals if v is not None)
            # Pad None entries and mismatched shapes
            padded = []
            for v in vals:
                if v is None:
                    padded.append(mx.zeros([1] + max_shape[1:], dtype=dt))
                elif list(v.shape[1:]) != max_shape[1:]:
                    pw = [(0, 0)] + [(0, ms - s) for s, ms in zip(v.shape[1:], max_shape[1:])]
                    padded.append(mx.pad(v, pw))
                else:
                    padded.append(v)
            setattr(batch_cache, attr, mx.concatenate(padded, axis=0))

        # Track per-entry compressed buffer counts
        comp_ns = []
        for c in caches:
            cb = getattr(c, 'comp_buf', None)
            comp_ns.append(cb.shape[1] if cb is not None else 0)
        batch_cache._comp_ns = mx.array(comp_ns, dtype=mx.int32)

        return batch_cache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _buf_append(buf, buf_n, data, step=256):
    """Append to pre-allocated buffer with step-based growth.

    Returns (buf, new_count). Grows by `step` when capacity exceeded,
    avoiding O(n^2) concatenation on every append.
    """
    new_n = data.shape[1]
    if buf is None:
        alloc = max(step, new_n)
        buf = mx.zeros((data.shape[0], alloc, data.shape[2]), dtype=data.dtype)
        buf[:, :new_n] = data
        return buf, new_n
    needed = buf_n + new_n
    if needed > buf.shape[1]:
        ext = mx.zeros(
            (buf.shape[0], max(step, new_n), buf.shape[2]), dtype=buf.dtype)
        buf = mx.concatenate([buf, ext], axis=1)
    buf[:, buf_n:buf_n + new_n] = data
    return buf, needed


def _apply_rope_at_positions(rope_obj, x, positions):
    """Vectorized RoPE at arbitrary positions (no loop).

    rope_obj: RoPE module (nn.RoPE, YarnRoPE, etc.)
    x: [..., T, rd]
    positions: [T] int array of position indices
    """
    rd = x.shape[-1]
    if hasattr(rope_obj, '_freqs'):
        freqs = rope_obj._freqs
    elif hasattr(rope_obj, 'base'):
        freqs = rope_obj.base ** (mx.arange(0, rd, 2, dtype=mx.float32) / rd)
    else:
        freqs = 10000.0 ** (mx.arange(0, rd, 2, dtype=mx.float32) / rd)

    # Apply position scaling: nn.RoPE uses scale as a divisor on positions
    # (mx.fast.rope computes positions / scale), so replicate that here.
    scale = getattr(rope_obj, 'scale', 1.0)
    t = positions.astype(mx.float32)
    if scale != 1.0:
        t = t / scale
    angles = t[:, None] / freqs[None, :]  # [T, rd//2]
    cos_a = mx.cos(angles)
    sin_a = mx.sin(angles)

    # Amplitude scaling: YarnRoPE uses mscale, SuScaledRoPE uses _scale
    mscale = getattr(rope_obj, 'mscale', 1.0)
    if hasattr(rope_obj, '_scale'):
        mscale = rope_obj._scale
    if mscale != 1.0:
        x = x * mscale

    x_pairs = x.reshape(*x.shape[:-1], -1, 2)
    x0, x1 = x_pairs[..., 0], x_pairs[..., 1]
    out_0 = x0 * cos_a - x1 * sin_a
    out_1 = x0 * sin_a + x1 * cos_a
    return mx.stack([out_0, out_1], axis=-1).reshape(x.shape)


# ---------------------------------------------------------------------------
# Compressor
# ---------------------------------------------------------------------------

class Compressor(nn.Module):
    """Learned softmax-gated pooling for KV cache compression."""

    def __init__(self, args: ModelArgs, compress_ratio: int, head_dim: int):
        super().__init__()
        self.head_dim = head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4

        coff = 1 + int(self.overlap)
        self.wkv = nn.Linear(args.hidden_size, coff * head_dim, bias=False)
        self.wgate = nn.Linear(args.hidden_size, coff * head_dim, bias=False)
        self.norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.ape = mx.zeros((compress_ratio, coff * head_dim))

        # Internal state for decode
        self._kv_state = None
        self._score_state = None

    def reset_state(self, B: int):
        coff = 1 + int(self.overlap)
        ratio = self.compress_ratio
        self._kv_state = mx.zeros((B, coff * ratio, coff * self.head_dim))
        self._score_state = mx.full(
            (B, coff * ratio, coff * self.head_dim), float("-inf")
        )

    def __call__(
        self, x: mx.array, start_pos: int, rope_fn,
    ) -> Optional[mx.array]:
        """Compress input tokens via learned gated pooling.

        Returns compressed KV [B, n_compressed, head_dim] or None.
        """
        B, S, _ = x.shape
        ratio = self.compress_ratio
        d = self.head_dim
        rd = self.rope_head_dim
        coff = 1 + int(self.overlap)
        out_dtype = x.dtype

        kv_raw = self.wkv(x)      # [B, S, coff*d]
        score_raw = self.wgate(x)  # [B, S, coff*d]

        if start_pos == 0:
            # Prefill
            self.reset_state(B)

            if S < ratio:
                # Too few tokens -- save for decode continuity
                offset_idx = ratio if self.overlap else 0
                for j in range(S):
                    self._kv_state[:B, offset_idx + j] = kv_raw[:, j]
                    self._score_state[:B, offset_idx + j] = (
                        score_raw[:, j] + self.ape[j])
                return None

            remainder = S % ratio
            cutoff = S - remainder

            # Save overlap state from last window (decode continuity)
            if self.overlap and cutoff >= ratio:
                self._kv_state[:B, :ratio] = kv_raw[:, cutoff - ratio:cutoff]
                self._score_state[:B, :ratio] = (
                    score_raw[:, cutoff - ratio:cutoff] + self.ape)

            # Save remainder tokens for decode continuity
            if remainder > 0:
                offset_idx = ratio if self.overlap else 0
                rem_kv = kv_raw[:, cutoff:]
                rem_sc = score_raw[:, cutoff:]
                for j in range(remainder):
                    self._kv_state[:B, offset_idx + j] = rem_kv[:, j]
                    self._score_state[:B, offset_idx + j] = (
                        rem_sc[:, j] + self.ape[j])

            # Reshape to compression windows and add positional encoding
            kv = kv_raw[:, :cutoff].reshape(B, -1, ratio, coff * d)
            score = score_raw[:, :cutoff].reshape(B, -1, ratio, coff * d) + self.ape

            if self.overlap:
                n_win = kv.shape[1]
                # Overlap transform: extend each window with prev window data
                kv_ov = mx.zeros((B, n_win, 2 * ratio, d))
                sc_ov = mx.full((B, n_win, 2 * ratio, d), float("-inf"))
                # Second-half dims from current window
                kv_ov[:, :, ratio:] = kv[:, :, :, d:]
                sc_ov[:, :, ratio:] = score[:, :, :, d:]
                # First-half dims from previous window
                if n_win > 1:
                    kv_ov[:, 1:, :ratio] = kv[:, :-1, :, :d]
                    sc_ov[:, 1:, :ratio] = score[:, :-1, :, :d]
                kv = kv_ov
                score = sc_ov

            weights = mx.softmax(score, axis=2)
            compressed = (kv * weights).sum(axis=2)  # [B, n_comp, d]
            compressed = self.norm(compressed)

            # Apply RoPE at correct positions (vectorized, no loop)
            n_comp = compressed.shape[1]
            positions = mx.arange(n_comp) * ratio
            compressed[:, :, -rd:] = _apply_rope_at_positions(
                rope_fn, compressed[:, :, -rd:], positions)

            return compressed.astype(out_dtype)

        else:
            # Decode: accumulate tokens, compress when ratio reached
            if self._kv_state is None:
                self.reset_state(B)

            should_compress = (start_pos + 1) % ratio == 0
            kv_tok = kv_raw
            score_tok = score_raw + self.ape[start_pos % ratio]

            compressed = None
            if self.overlap:
                idx = ratio + start_pos % ratio
                self._kv_state[:B, idx] = kv_tok.squeeze(1)
                self._score_state[:B, idx] = score_tok.squeeze(1)
                if should_compress:
                    kv_s = mx.concatenate([
                        self._kv_state[:B, :ratio, :d],
                        self._kv_state[:B, ratio:, d:]
                    ], axis=1)
                    sc_s = mx.concatenate([
                        self._score_state[:B, :ratio, :d],
                        self._score_state[:B, ratio:, d:]
                    ], axis=1)
                    compressed = (kv_s * mx.softmax(sc_s, axis=1)).sum(
                        axis=1, keepdims=True)
                    self._kv_state[:B, :ratio] = self._kv_state[:B, ratio:]
                    self._score_state[:B, :ratio] = self._score_state[:B, ratio:]
            else:
                self._kv_state[:B, start_pos % ratio] = kv_tok.squeeze(1)
                self._score_state[:B, start_pos % ratio] = score_tok.squeeze(1)
                if should_compress:
                    compressed = (
                        self._kv_state[:B]
                        * mx.softmax(self._score_state[:B], axis=1)
                    ).sum(axis=1, keepdims=True)

            if not should_compress:
                return None

            compressed = self.norm(compressed)
            comp_pe = rope_fn(
                compressed[..., -rd:].reshape(B, 1, 1, rd),
                offset=start_pos + 1 - ratio,
            )
            compressed = mx.concatenate(
                [compressed[..., :-rd], comp_pe.reshape(B, 1, rd)], axis=-1
            )
            return compressed.astype(out_dtype)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class Indexer(nn.Module):
    """Lightning Indexer for CSA layers. Scores compressed positions
    and selects top-k for sparse attention."""

    def __init__(self, args: ModelArgs, compress_ratio: int = 4):
        super().__init__()
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        self.compress_ratio = compress_ratio
        self.softmax_scale = self.head_dim ** -0.5

        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(args.hidden_size, self.n_heads, bias=False)
        self.compressor = Compressor(args, compress_ratio, self.head_dim)
        self._index_kv = None  # [B, n_comp, head_dim]


class DeepseekV4Attention(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.n_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.n_groups = args.o_groups
        self.window_size = args.sliding_window
        self.scale = args.head_dim ** -0.5
        self.compress_ratio = (
            args.compress_ratios[layer_id]
            if layer_id < len(args.compress_ratios)
            else 0
        )

        # Q: low-rank
        self.wq_a = nn.Linear(args.hidden_size, args.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(args.q_lora_rank, eps=args.rms_norm_eps)
        self.wq_b = nn.Linear(args.q_lora_rank, self.n_heads * self.head_dim, bias=False)

        # KV: single head (MQA)
        self.wkv = nn.Linear(args.hidden_size, self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        # O: grouped low-rank
        self.wo_a = [
            nn.Linear(
                self.n_heads * self.head_dim // self.n_groups,
                self.o_lora_rank, bias=False,
            )
            for _ in range(self.n_groups)
        ]
        self.wo_b = nn.Linear(self.n_groups * self.o_lora_rank, args.hidden_size, bias=False)

        self.attn_sink = mx.zeros((self.n_heads,))

        # Compressor + Indexer for CSA/HCA layers
        if self.compress_ratio > 0:
            self.compressor = Compressor(args, self.compress_ratio, self.head_dim)
            if self.compress_ratio == 4:
                self.indexer = Indexer(args, self.compress_ratio)

        # RoPE
        rope_theta = args.compress_rope_theta if self.compress_ratio > 0 else args.rope_theta
        rope_scaling = args.rope_scaling if self.compress_ratio > 0 else None
        self.rope = initialize_rope(
            dims=args.qk_rope_head_dim,
            base=rope_theta,
            traditional=True,
            max_position_embeddings=args.max_position_embeddings,
            scaling_config=rope_scaling,
        )

    def _dense_attn(self, q, kv_all, mask, L):
        """Standard dense attention for prefill."""
        scores = (q @ kv_all[:, None, :, :].transpose(0, 1, 3, 2)) * self.scale
        scores = scores + self.attn_sink[:, None, None]
        if mask is not None and not isinstance(mask, str):
            scores = mx.where(mask, scores, -1e9)
        elif L > 1:
            T = kv_all.shape[1]
            causal = mx.triu(mx.full((L, T), -1e9), k=T - L + 1)
            scores = scores + causal
        weights = mx.softmax(scores, axis=-1)
        return weights @ kv_all[:, None, :, :]

    def _init_win_buf(self, kv, B, L):
        """Initialize circular window buffer after prefill."""
        win = self.window_size
        D = self.head_dim
        dtype = kv.dtype
        if L <= win:
            buf = mx.zeros((B, win, D), dtype=dtype)
            buf[:, :L] = kv
            self._win_buf = buf
        else:
            cutoff = L % win
            last_win = kv[:, -win:]
            if cutoff == 0:
                self._win_buf = last_win
            else:
                buf = mx.zeros((B, win, D), dtype=dtype)
                buf[:, cutoff:] = last_win[:, :win - cutoff]
                buf[:, :cutoff] = last_win[:, win - cutoff:]
                self._win_buf = buf

    def _sparse_prefill(self, q, kv, x, B, L):
        """Sparse prefill: sliding window + compressed context.

        Uses chunked processing for long prompts to limit peak memory.
        """
        win = self.window_size
        ratio = self.compress_ratio

        # Run main compressor
        self._comp_buf = self.compressor(x, 0, self.rope)
        self._comp_n = self._comp_buf.shape[1] if self._comp_buf is not None else 0

        # Run indexer compressor to keep state in sync
        if hasattr(self, 'indexer'):
            idx_comp = self.indexer.compressor(x, 0, self.rope)
            self.indexer._index_kv = idx_comp
            self.indexer._idx_n = idx_comp.shape[1] if idx_comp is not None else 0

        if self._comp_buf is None:
            # No compressed context (prompt too short)
            s = mx.arange(L)[:, None]
            t = mx.arange(L)[None, :]
            causal = t <= s
            scores = (q @ kv[:, None, :, :].transpose(0, 1, 3, 2)) * self.scale
            scores = scores + self.attn_sink[:, None, None]
            scores = mx.where(causal, scores, -1e9)
            weights = mx.softmax(scores, axis=-1)
            return weights @ kv[:, None, :, :]

        n_comp = self._comp_buf.shape[1]
        all_kv = mx.concatenate([kv, self._comp_buf], axis=1)

        CHUNK = 256
        if L <= CHUNK:
            # Small enough for single pass
            return self._sparse_prefill_chunk(q, all_kv, L, n_comp, 0, L)

        # Chunked: process CHUNK queries at a time (16x less peak memory)
        outputs = []
        for s0 in range(0, L, CHUNK):
            s1 = min(s0 + CHUNK, L)
            q_c = q[:, :, s0:s1]
            out_c = self._sparse_prefill_chunk(
                q_c, all_kv, L, n_comp, s0, s1)
            outputs.append(out_c)
        return mx.concatenate(outputs, axis=2)

    def _sparse_prefill_chunk(self, q_c, all_kv, L, n_comp, s0, s1):
        """One chunk of sparse prefill attention."""
        win = self.window_size
        ratio = self.compress_ratio

        s = mx.arange(s0, s1)[:, None]
        t_raw = mx.arange(L)[None, :]
        raw_mask = (t_raw <= s) & (t_raw >= mx.maximum(s - win + 1, 0))

        c = mx.arange(n_comp)[None, :]
        comp_mask = c < ((s + 1) // ratio)

        sparse_mask = mx.concatenate([raw_mask, comp_mask], axis=1)

        scores = (q_c @ all_kv[:, None, :, :].transpose(0, 1, 3, 2)) * self.scale
        scores = scores + self.attn_sink[:, None, None]
        scores = mx.where(sparse_mask, scores, -1e9)
        weights = mx.softmax(scores, axis=-1)
        return weights @ all_kv[:, None, :, :]

    def _continuation_prefill(self, q, kv, x, B, L, offset):
        """Handle continuation prefill chunks (chunked prefill support).

        When the server splits a long prompt into chunks, subsequent chunks
        arrive as L>1 but buffers already exist from the first chunk.
        Dense attention within chunk + update buffers.
        """
        win = self.window_size
        comp_n = getattr(self, '_comp_n', 0)

        # Attend within chunk + existing compressed context
        if self._comp_buf is not None and comp_n > 0:
            comp_valid = self._comp_buf[:, :comp_n]
            all_kv = mx.concatenate([kv, comp_valid], axis=1)
            T = all_kv.shape[1]
            # Causal within chunk + all compressed visible
            s = mx.arange(L)[:, None]
            t_raw = mx.arange(L)[None, :]
            raw_mask = (t_raw <= s) & (t_raw >= mx.maximum(s - win + 1, 0))
            comp_mask = mx.ones((L, comp_n), dtype=mx.bool_)
            mask_full = mx.concatenate([raw_mask, comp_mask], axis=1)
            scores = (q @ all_kv[:, None, :, :].transpose(0, 1, 3, 2)) * self.scale
            scores = scores + self.attn_sink[:, None, None]
            scores = mx.where(mask_full, scores, -1e9)
            weights = mx.softmax(scores, axis=-1)
            output = weights @ all_kv[:, None, :, :]
        else:
            # No compressed context, dense causal within chunk
            s = mx.arange(L)[:, None]
            t = mx.arange(L)[None, :]
            causal = t <= s
            scores = (q @ kv[:, None, :, :].transpose(0, 1, 3, 2)) * self.scale
            scores = scores + self.attn_sink[:, None, None]
            scores = mx.where(causal, scores, -1e9)
            weights = mx.softmax(scores, axis=-1)
            output = weights @ kv[:, None, :, :]

        # Extend compressed buffer: process chunk token-by-token
        # (compressor decode mode expects L=1)
        for i in range(L):
            comp = self.compressor(x[:, i:i+1], offset + i, self.rope)
            if comp is not None:
                self._comp_buf, self._comp_n = _buf_append(
                    self._comp_buf, getattr(self, '_comp_n', 0), comp)
            if hasattr(self, 'indexer'):
                idx_comp = self.indexer.compressor(
                    x[:, i:i+1], offset + i, self.rope)
                if idx_comp is not None:
                    self.indexer._index_kv, self.indexer._idx_n = _buf_append(
                        self.indexer._index_kv,
                        getattr(self.indexer, '_idx_n', 0), idx_comp)
            # Flush Metal buffers to avoid resource limit
            if (i + 1) % 32 == 0:
                mx.eval(self.compressor._kv_state)

        # Update window buffer incrementally (don't reinitialize)
        win = self.window_size
        D = self.head_dim
        for t in range(L):
            pos = (offset + t) % win
            self._win_buf[:, pos:pos+1] = kv[:, t:t+1]

        return output

    def _sparse_decode(self, q, kv, x, B, offset, qr):
        """Sparse decode: window + compressed with Indexer selection."""
        win = self.window_size

        # Safety: init buffers if missing (single-token prompt edge case)
        if getattr(self, '_win_buf', None) is None:
            self._init_win_buf(kv, B, 1)
            self._comp_buf = None
            self._comp_n = 0

        # Update window buffer
        pos = offset % win
        self._win_buf[:, pos:pos + 1] = kv

        # Run main compressor (step-based growth)
        comp = self.compressor(x, offset, self.rope)
        if comp is not None:
            self._comp_buf, self._comp_n = _buf_append(
                self._comp_buf, getattr(self, '_comp_n', 0), comp)

        # Run indexer compressor (CSA layers)
        if hasattr(self, 'indexer'):
            idx_comp = self.indexer.compressor(x, offset, self.rope)
            if idx_comp is not None:
                self.indexer._index_kv, self.indexer._idx_n = _buf_append(
                    self.indexer._index_kv,
                    getattr(self.indexer, '_idx_n', 0), idx_comp)

        # Gather window
        n_win = min(offset + 1, win)
        win_kv = self._win_buf if offset + 1 >= win else self._win_buf[:, :n_win]

        # Gather compressed (with Indexer top-k for CSA layers)
        comp_n = getattr(self, '_comp_n', 0)
        parts = [win_kv]
        if self._comp_buf is not None and comp_n > 0:
            comp_valid = self._comp_buf[:, :comp_n]
            if (hasattr(self, 'indexer')
                    and self.indexer._index_kv is not None
                    and comp_n > self.indexer.index_topk):
                parts.append(self._indexer_select(x, qr, offset, B))
            else:
                parts.append(comp_valid)

        kv_ctx = mx.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]

        # MQA attention with per-head attn_sink bias
        k = kv_ctx[:, None, :, :]  # [B, 1, T, D]
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        scores = scores + self.attn_sink[:, None, None]
        weights = mx.softmax(scores, axis=-1)
        return weights @ k

    def _indexer_select(self, x, qr, offset, B):
        """Indexer: score compressed positions, return top-k from main buffer."""
        idx = self.indexer
        rd = self.rope_head_dim
        comp_n = self._comp_n
        idx_n = getattr(idx, '_idx_n', 0)
        n = min(comp_n, idx_n)
        k = min(idx.index_topk, n)

        # Project Q for indexing
        q = idx.wq_b(qr).reshape(B, 1, idx.n_heads, idx.head_dim)
        q_pe = self.rope(q[..., -rd:], offset=offset)
        q = mx.concatenate([q[..., :-rd], q_pe], axis=-1)

        # Score: multi-head Q @ single-head index_KV, weighted by projection
        w = idx.weights_proj(x) * (idx.softmax_scale * idx.n_heads ** -0.5)
        scores = mx.einsum("bshd,btd->bsht", q, idx._index_kv[:B, :n])
        scores = (mx.maximum(scores, 0) * w[:, :, :, None]).sum(axis=2)
        scores = scores.squeeze(1)  # [B, n]

        topk = mx.argpartition(-scores, kth=k - 1, axis=-1)[:, :k]

        D = self._comp_buf.shape[-1]
        topk_exp = mx.broadcast_to(topk[:, :, None], (B, k, D))
        return mx.take_along_axis(
            self._comp_buf[:, :comp_n], topk_exp, axis=1)

    def _batched_sparse_decode(self, q, kv, x, cache, qr):
        """Sparse decode for BatchSparseKVCache: per-entry processing.

        The compressor state machine uses offset-dependent modular arithmetic
        (offset % ratio for APE indexing, (offset+1) % ratio == 0 for compression
        triggers), so different batch entries at different offsets cannot be
        trivially vectorized. Instead we process each entry independently and
        stack results.
        """
        B = x.shape[0]
        mx.eval(cache.offset)
        offsets = cache.offset.tolist()

        outputs = []
        for i in range(B):
            # Temporarily load per-entry sparse state into module
            self._win_buf = cache.win_buf[i:i+1] if cache.win_buf is not None else None
            cb = cache.comp_buf
            if cb is not None:
                cn = cache._comp_ns[i].item() if cache._comp_ns is not None else cb.shape[1]
                self._comp_buf = cb[i:i+1, :cn]
                self._comp_n = cn
            else:
                self._comp_buf = None
                self._comp_n = 0

            if hasattr(self, 'compressor'):
                if cache.comp_kv_state is not None:
                    self.compressor._kv_state = cache.comp_kv_state[i:i+1]
                    self.compressor._score_state = cache.comp_score_state[i:i+1]
                else:
                    self.compressor.reset_state(1)

            if hasattr(self, 'indexer'):
                if cache.idx_kv is not None:
                    idx_n = cache.idx_kv.shape[1]
                    self.indexer._index_kv = cache.idx_kv[i:i+1]
                    self.indexer._idx_n = idx_n
                else:
                    self.indexer._index_kv = None
                    self.indexer._idx_n = 0
                if cache.idx_comp_kv_state is not None:
                    self.indexer.compressor._kv_state = cache.idx_comp_kv_state[i:i+1]
                    self.indexer.compressor._score_state = cache.idx_comp_score_state[i:i+1]
                else:
                    self.indexer.compressor.reset_state(1)

            # Run sparse decode for this entry
            out_i = self._sparse_decode(
                q[i:i+1], kv[i:i+1], x[i:i+1], 1, offsets[i],
                qr[i:i+1] if qr is not None else None,
            )
            outputs.append(out_i)

            # Save per-entry sparse state back to cache (in-place slicing)
            if cache.win_buf is not None:
                cache.win_buf[i:i+1] = self._win_buf
            elif self._win_buf is not None:
                # First entry initializes the buffer; allocate for full batch
                win = self.window_size
                D = self.head_dim
                cache.win_buf = mx.zeros(
                    (B, win, D), dtype=self._win_buf.dtype)
                cache.win_buf[i:i+1] = self._win_buf

            # Sync comp_buf back: collect per-entry buffers for later merge
            cn = getattr(self, '_comp_n', 0)
            if not hasattr(self, '_batch_comp_bufs'):
                self._batch_comp_bufs = [None] * B
                self._batch_comp_ns = [0] * B
            self._batch_comp_bufs[i] = self._comp_buf[:, :cn] if self._comp_buf is not None and cn > 0 else None
            self._batch_comp_ns[i] = cn

            # Sync compressor state back
            if hasattr(self, 'compressor'):
                if cache.comp_kv_state is None and self.compressor._kv_state is not None:
                    sh = list(self.compressor._kv_state.shape)
                    sh[0] = B
                    cache.comp_kv_state = mx.zeros(sh, dtype=self.compressor._kv_state.dtype)
                    cache.comp_score_state = mx.full(sh, float("-inf"))
                if cache.comp_kv_state is not None:
                    cache.comp_kv_state[i:i+1] = self.compressor._kv_state
                    cache.comp_score_state[i:i+1] = self.compressor._score_state

            # Sync indexer state back
            if hasattr(self, 'indexer'):
                if cache.idx_kv is None and self.indexer._index_kv is not None:
                    sh = list(self.indexer._index_kv.shape)
                    sh[0] = B
                    cache.idx_kv = mx.zeros(sh, dtype=self.indexer._index_kv.dtype)
                if cache.idx_kv is not None and self.indexer._index_kv is not None:
                    idx_n = self.indexer._idx_n
                    # May need to grow batch cache idx_kv
                    if idx_n > cache.idx_kv.shape[1]:
                        ext = mx.zeros(
                            (B, idx_n - cache.idx_kv.shape[1], cache.idx_kv.shape[2]),
                            dtype=cache.idx_kv.dtype)
                        cache.idx_kv = mx.concatenate([cache.idx_kv, ext], axis=1)
                    cache.idx_kv[i:i+1, :idx_n] = self.indexer._index_kv[:, :idx_n]
                if cache.idx_comp_kv_state is None and hasattr(self.indexer, 'compressor') and self.indexer.compressor._kv_state is not None:
                    sh = list(self.indexer.compressor._kv_state.shape)
                    sh[0] = B
                    cache.idx_comp_kv_state = mx.zeros(sh, dtype=self.indexer.compressor._kv_state.dtype)
                    cache.idx_comp_score_state = mx.full(sh, float("-inf"))
                if cache.idx_comp_kv_state is not None:
                    cache.idx_comp_kv_state[i:i+1] = self.indexer.compressor._kv_state
                    cache.idx_comp_score_state[i:i+1] = self.indexer.compressor._score_state

        # Merge comp_buf back to cache
        bufs = getattr(self, '_batch_comp_bufs', [None] * B)
        ns = getattr(self, '_batch_comp_ns', [0] * B)
        max_cn = max(ns) if ns else 0
        if max_cn > 0:
            D = next(b.shape[2] for b in bufs if b is not None)
            dt = next(b.dtype for b in bufs if b is not None)
            merged_comp = mx.zeros((B, max_cn, D), dtype=dt)
            for i in range(B):
                if bufs[i] is not None and ns[i] > 0:
                    merged_comp[i:i+1, :ns[i]] = bufs[i]
            cache.comp_buf = merged_comp
            cache._comp_ns = mx.array(ns, dtype=mx.int32)
        else:
            cache.comp_buf = None
            cache._comp_ns = mx.zeros((B,), dtype=mx.int32)

        # Clean up temp state
        if hasattr(self, '_batch_comp_bufs'):
            del self._batch_comp_bufs
            del self._batch_comp_ns

        # Advance cache offset without per-call allocation (avoids Metal
        # resource leak from mx.zeros being called every layer every step).
        cache._idx += 1
        if hasattr(cache.offset, "shape"):
            cache.offset = cache.offset + 1
        else:
            cache.offset += 1

        return mx.concatenate(outputs, axis=0)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        rd = self.rope_head_dim
        ratio = self.compress_ratio
        is_batch_sparse = isinstance(cache, BatchSparseKVCache)

        # Reset stale sparse state when a new conversation starts
        if cache is not None and not is_batch_sparse and cache.offset == 0:
            self._win_buf = None
            self._comp_buf = None
            self._comp_n = 0

        # Restore sparse state from cache (after cache load / multi-turn)
        if (ratio > 0 and cache is not None
                and isinstance(cache, SparseKVCache)
                and cache.win_buf is not None
                and getattr(self, '_win_buf', None) is None):
            self._win_buf = cache.win_buf
            self._comp_buf = cache.comp_buf
            self._comp_n = cache.comp_buf.shape[1] if cache.comp_buf is not None else 0
            if hasattr(self, 'compressor') and cache.comp_kv_state is not None:
                self.compressor._kv_state = cache.comp_kv_state
                self.compressor._score_state = cache.comp_score_state
            if hasattr(self, 'indexer'):
                if cache.idx_kv is not None:
                    self.indexer._index_kv = cache.idx_kv
                    self.indexer._idx_n = cache.idx_kv.shape[1]
                if cache.idx_comp_kv_state is not None:
                    self.indexer.compressor._kv_state = cache.idx_comp_kv_state
                    self.indexer.compressor._score_state = cache.idx_comp_score_state

        # Fused Q+KV first projection (1 dispatch instead of 2)
        if B == 1 and L == 1 and hasattr(self.wq_a, 'bits'):
            if not hasattr(self, '_fused_qkv_w'):
                assert self.wq_a.group_size == self.wkv.group_size and self.wq_a.bits == self.wkv.bits
                self._fused_qkv_w = mx.concatenate([self.wq_a.weight, self.wkv.weight], axis=0)
                self._fused_qkv_s = mx.concatenate([self.wq_a.scales, self.wkv.scales], axis=0)
                self._fused_qkv_b = mx.concatenate([self.wq_a.biases, self.wkv.biases], axis=0)
                self._qr_split = self.wq_a.weight.shape[0]
                mx.eval(self._fused_qkv_w, self._fused_qkv_s, self._fused_qkv_b)
            combined = mx.quantized_matmul(
                x.reshape(1, -1), self._fused_qkv_w, self._fused_qkv_s, self._fused_qkv_b,
                transpose=True, group_size=self.wq_a.group_size, bits=self.wq_a.bits)
            qr_raw = combined[:, :self._qr_split]
            kv_raw = combined[:, self._qr_split:]
        else:
            qr_raw = self.wq_a(x).reshape(1, -1) if B == 1 else self.wq_a(x)
            kv_raw = self.wkv(x).reshape(1, -1) if B == 1 else self.wkv(x)

        # Q chain
        qr = self.q_norm(qr_raw.reshape(B, L, -1))
        q = self.wq_b(qr).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = q * mx.rsqrt(mx.mean(q * q, axis=-1, keepdims=True) + self.args.rms_norm_eps)

        offset = cache.offset if cache is not None else 0
        q_pe = self.rope(q[..., -rd:], offset=offset)
        q = mx.concatenate([q[..., :-rd], q_pe], axis=-1)

        # KV chain
        kv = self.kv_norm(kv_raw.reshape(B, L, -1))
        kv_pe = self.rope(kv[..., -rd:].reshape(B, 1, L, rd), offset=offset)
        kv = mx.concatenate([kv[..., :-rd], kv_pe.squeeze(1)], axis=-1)

        if ratio == 0 or cache is None:
            # Dense path for non-compressed layers
            if cache is not None:
                kv_exp = kv.reshape(B, 1, L, self.head_dim)
                kv_cached, _ = cache.update_and_fetch(
                    kv_exp, mx.zeros((B, 1, L, 0)))
                kv_all = kv_cached.squeeze(1)
            else:
                kv_all = kv
            if L > 1 and L > self.window_size and ratio == 0:
                # Sliding window prefill (matches V4 training pattern)
                win = self.window_size
                T = kv_all.shape[1]
                s = mx.arange(L)[:, None]
                t = mx.arange(T)[None, :]
                off = T - L
                win_mask = (t <= s + off) & (
                    t >= mx.maximum(s + off - win + 1, 0))
                scores = (q @ kv_all[:, None, :, :].transpose(
                    0, 1, 3, 2)) * self.scale
                scores = scores + self.attn_sink[:, None, None]
                scores = mx.where(win_mask, scores, -1e9)
                weights = mx.softmax(scores, axis=-1)
                output = weights @ kv_all[:, None, :, :]
            else:
                output = self._dense_attn(q, kv_all, mask, L)
        elif is_batch_sparse and L == 1:
            # Batched sparse decode: process per-entry
            output = self._batched_sparse_decode(q, kv, x, cache, qr)
        elif L > 1:
            if is_batch_sparse:
                # For batch sparse prefill, extract scalar offset from first
                # entry (all entries in a prompt batch start at same position)
                mx.eval(cache.offset)
                offset = cache.offset[0].item()
            if offset == 0:
                # First prefill chunk (new conversation)
                output = self._sparse_prefill(q, kv, x, B, L)
                self._init_win_buf(kv, B, L)
            else:
                # Continuation prefill chunk (chunked prefill)
                output = self._continuation_prefill(q, kv, x, B, L, offset)
            cache.update_and_fetch(
                mx.zeros((B, 1, L, 1)), mx.zeros((B, 1, L, 1)))
        else:
            # Sparse decode with Indexer selection
            output = self._sparse_decode(q, kv, x, B, offset, qr)
            cache.update_and_fetch(
                mx.zeros((B, 1, 1, 1)), mx.zeros((B, 1, 1, 1)))

        # Inverse RoPE = RoPE with negated angle
        if L == 1:
            o_inv = self.rope(output[..., -rd:], offset=-offset)
        else:
            positions = -(mx.arange(L) + offset)
            o_inv = _apply_rope_at_positions(self.rope, output[..., -rd:].reshape(-1, L, rd), positions)
            o_inv = o_inv.reshape(output[..., -rd:].shape)
        output = mx.concatenate([output[..., :-rd], o_inv], axis=-1)

        # Grouped output projection
        output = output.transpose(0, 2, 1, 3)
        heads_per_group = self.n_heads // self.n_groups
        output = output.reshape(B, L, self.n_groups, heads_per_group * self.head_dim)
        if not isinstance(self.wo_a, list):
            # Single wo_a linear (Thump604 format): per-group matmul with row slicing
            if hasattr(self.wo_a, 'bits'):
                pieces = []
                for g in range(self.n_groups):
                    rows = slice(g * self.o_lora_rank, (g + 1) * self.o_lora_rank)
                    biases = self.wo_a.biases[rows] if self.wo_a.biases is not None else None
                    pieces.append(mx.quantized_matmul(
                        output[:, :, g, :], self.wo_a.weight[rows], self.wo_a.scales[rows],
                        biases, transpose=True, group_size=self.wo_a.group_size, bits=self.wo_a.bits,
                    ))
                output = mx.concatenate(pieces, axis=-1)
            else:
                pieces = []
                for g in range(self.n_groups):
                    rows = slice(g * self.o_lora_rank, (g + 1) * self.o_lora_rank)
                    pieces.append(output[:, :, g, :] @ self.wo_a.weight[rows].T)
                output = mx.concatenate(pieces, axis=-1)
        elif B == 1 and L == 1 and hasattr(self.wo_a[0], 'bits') and self.wo_a[0].bits in (4, 8):
            from .fused_moe_kernel import fused_grouped_wo
            x_flat = output.reshape(self.n_groups, -1)
            output = fused_grouped_wo(x_flat, self.wo_a).astype(output.dtype)
            output = output.reshape(1, 1, -1)
        else:
            group_outputs = []
            for g in range(self.n_groups):
                group_outputs.append(self.wo_a[g](output[:, :, g, :]))
            output = mx.concatenate(group_outputs, axis=-1)

        # Sync all sparse state to cache for serialization.
        # For BatchSparseKVCache during decode (L==1), state is synced
        # in _batched_sparse_decode. For prefill (L>1), sync here.
        if ratio > 0 and cache is not None and isinstance(cache, (SparseKVCache, BatchSparseKVCache)):
            if not (is_batch_sparse and L == 1):
                cache.win_buf = getattr(self, '_win_buf', None)
                comp_n = getattr(self, '_comp_n', 0)
                buf = getattr(self, '_comp_buf', None)
                cache.comp_buf = buf[:, :comp_n] if buf is not None and comp_n > 0 else None
                if is_batch_sparse and cache.comp_buf is not None:
                    cache._comp_ns = mx.array(
                        [comp_n] * cache.comp_buf.shape[0], dtype=mx.int32)
                if hasattr(self, 'compressor'):
                    cache.comp_kv_state = self.compressor._kv_state
                    cache.comp_score_state = self.compressor._score_state
                if hasattr(self, 'indexer'):
                    idx_n = getattr(self.indexer, '_idx_n', 0)
                    idx_buf = self.indexer._index_kv
                    cache.idx_kv = idx_buf[:, :idx_n] if idx_buf is not None and idx_n > 0 else None
                    cache.idx_comp_kv_state = self.indexer.compressor._kv_state
                    cache.idx_comp_score_state = self.indexer.compressor._score_state

        return self.wo_b(output)


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

class DeepseekV4Gate(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.topk = args.num_experts_per_tok
        self.score_func = args.scoring_func
        self.route_scale = args.routed_scaling_factor
        self.norm_topk_prob = args.norm_topk_prob
        self.is_hash = layer_id < args.num_hash_layers

        self.weight = mx.zeros((args.n_routed_experts, args.hidden_size))
        if self.is_hash:
            self.tid2eid = mx.zeros((args.vocab_size, args.num_experts_per_tok), dtype=mx.int32)
        else:
            self.bias = mx.zeros((args.n_routed_experts,))

    def __call__(self, x: mx.array, input_ids: Optional[mx.array] = None):
        scores = (x @ self.weight.T).astype(mx.float32)
        if self.score_func == "softmax":
            scores = mx.softmax(scores, axis=-1)
        elif self.score_func == "sigmoid":
            scores = mx.sigmoid(scores)
        else:
            scores = mx.sqrt(mx.log1p(mx.exp(scores)))

        original_scores = scores
        if hasattr(self, "bias") and self.bias is not None:
            scores = scores + self.bias

        if self.is_hash and input_ids is not None:
            indices = self.tid2eid[input_ids.reshape(-1)]
            indices = indices.reshape(x.shape[0], x.shape[1], self.topk)
        else:
            indices = mx.argpartition(-scores, kth=self.topk - 1, axis=-1)[..., :self.topk]

        weights = mx.take_along_axis(original_scores, indices, axis=-1)
        if self.score_func != "softmax" and self.norm_topk_prob:
            weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-8)
        weights = weights * self.route_scale
        return weights, indices


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------

class DeepseekV4SharedExpert(nn.Module):
    def __init__(self, dim: int, inter_dim: int, swiglu_limit: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)
        self.swiglu_limit = swiglu_limit

    def __call__(self, x: mx.array) -> mx.array:
        gate = self.w1(x)
        up = self.w3(x)
        if self.swiglu_limit > 0:
            up = mx.clip(up, -self.swiglu_limit, self.swiglu_limit)
            gate = mx.minimum(gate, self.swiglu_limit)
        return self.w2(nn.silu(gate) * up)


class DeepseekV4MoE(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.num_experts_per_tok = args.num_experts_per_tok
        self.experts = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.n_routed_experts,
        )
        self.gate = DeepseekV4Gate(layer_id, args)
        if args.n_shared_experts and args.n_shared_experts > 0:
            inter = args.moe_intermediate_size * args.n_shared_experts
            self.shared_experts = DeepseekV4SharedExpert(args.hidden_size, inter, args.swiglu_limit)
        else:
            self.shared_experts = None

    def __call__(self, x: mx.array, input_ids: Optional[mx.array] = None) -> mx.array:
        weights, indices = self.gate(x, input_ids)
        y = self.experts(x, indices)
        y = (y * weights[..., None]).sum(axis=-2).astype(y.dtype)
        if self.shared_experts is not None:
            y = y + self.shared_experts(x)
        return y


# ---------------------------------------------------------------------------
# Hyper-Connection Block
# ---------------------------------------------------------------------------

class HyperConnectionBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        # Layers where MoE is skipped during decode (19% skip, quality-validated).
        # Only apply for the 43-layer config this was tuned for.
        if args.num_hidden_layers == 43:
            self._skip_moe_layers = frozenset(range(3, 41, 5))  # {3,8,13,18,23,28,33,38}
        else:
            self._skip_moe_layers = frozenset()
        self.hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        self.norm_eps = args.rms_norm_eps

        self.attn = DeepseekV4Attention(layer_id, args)
        self.ffn = DeepseekV4MoE(layer_id, args)
        self.attn_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        hc = args.hc_mult
        mix_hc = (2 + hc) * hc
        hc_dim = hc * args.hidden_size
        self.hc_attn_fn = mx.zeros((mix_hc, hc_dim))
        self.hc_ffn_fn = mx.zeros((mix_hc, hc_dim))
        self.hc_attn_base = mx.zeros((mix_hc,))
        self.hc_ffn_base = mx.zeros((mix_hc,))
        self.hc_attn_scale = mx.zeros((3,))
        self.hc_ffn_scale = mx.zeros((3,))

    def _hc_pre(self, x, hc_fn, hc_scale, hc_base):
        B, S, M, D = x.shape

        hc = self.hc_mult
        x_flat = x.reshape(B, S, M * D).astype(mx.float32)
        rsqrt = mx.rsqrt(mx.mean(x_flat * x_flat, axis=-1, keepdims=True) + self.norm_eps)
        mixes = (x_flat @ hc_fn.T) * rsqrt

        pre_raw = mixes[..., :hc] * hc_scale[0] + hc_base[:hc]
        post_raw = mixes[..., hc:2*hc] * hc_scale[1] + hc_base[hc:2*hc]
        comb_raw = mixes[..., 2*hc:] * hc_scale[2] + hc_base[2*hc:]

        pre = mx.sigmoid(pre_raw) + self.hc_eps
        post = 2.0 * mx.sigmoid(post_raw)

        comb = comb_raw.reshape(B, S, hc, hc)
        comb = mx.softmax(comb, axis=-1) + self.hc_eps
        # Cap Sinkhorn iterations: 4x4 matrix converges in ~8 iterations.
        # Full 20 iterations add ~12% decode latency with negligible quality gain.
        n_iters = min(self.hc_sinkhorn_iters, 8)
        for _ in range(n_iters):
            comb = comb / comb.sum(axis=-2, keepdims=True)
            comb = comb / comb.sum(axis=-1, keepdims=True)

        y = mx.sum(pre[..., None] * x, axis=2)
        return y.astype(x.dtype), post, comb

    def _hc_post(self, x, residual, post, comb):
        y = post[..., None] * x[:, :, None, :] + mx.einsum("bsji,bsjd->bsid", comb, residual)
        return y.astype(x.dtype)

    def __call__(self, x, mask=None, cache=None, input_ids=None):
        residual = x
        y, post, comb = self._hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        y = self.attn(self.attn_norm(y), mask, cache)
        x = self._hc_post(y, residual, post, comb)

        # Skip MoE on selected layers during decode (saves ~25% MoE compute)
        if self.layer_id in self._skip_moe_layers and x.shape[1] == 1:
            return x

        residual = x
        y, post, comb = self._hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        y = self.ffn(self.ffn_norm(y), input_ids)
        x = self._hc_post(y, residual, post, comb)
        return x


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DeepseekV4Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.hc_mult = args.hc_mult
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [HyperConnectionBlock(i, args) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        hc_dim = args.hc_mult * args.hidden_size
        self.hc_head_fn = mx.zeros((args.hc_mult, hc_dim))
        self.hc_head_base = mx.zeros((args.hc_mult,))
        self.hc_head_scale = mx.zeros((1,))

    def _hc_head(self, x):
        B, S, M, D = x.shape
        x_flat = x.reshape(B, S, M * D).astype(mx.float32)
        rsqrt = mx.rsqrt(mx.mean(x_flat * x_flat, axis=-1, keepdims=True) + self.args.rms_norm_eps)
        mixes = (x_flat @ self.hc_head_fn.T) * rsqrt
        pre = mx.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.args.hc_eps
        y = mx.sum(pre[..., None] * x, axis=2)
        return y.astype(x.dtype)

    def __call__(self, x, cache=None):
        h = self.embed_tokens(x)
        h = mx.repeat(h[:, :, None, :], self.hc_mult, axis=2)
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(h[:, :, 0, :], cache[0])
        for i, layer in enumerate(self.layers):
            h = layer(h, mask, cache[i], input_ids=x)
        h = self._hc_head(h)
        return self.norm(h)


class _ShallowV4(nn.Module):
    """Lightweight wrapper: runs first N layers of V4 as draft model
    for self-speculative decoding. Shares weights (zero extra memory)."""

    def __init__(self, full_model, n_layers):
        super().__init__()
        self._full = full_model
        self._n_layers = n_layers

    def __call__(self, inputs, cache=None):
        m = self._full.model
        h = m.embed_tokens(inputs)
        h = mx.repeat(h[:, :, None, :], m.hc_mult, axis=2)
        if cache is None:
            cache = [None] * self._n_layers
        mask = create_attention_mask(h[:, :, 0, :], cache[0])
        for i in range(self._n_layers):
            h = m.layers[i](h, mask, cache[i], input_ids=inputs)
        h = m._hc_head(h)
        h = m.norm(h)
        return self._full.lm_head(h)

    @property
    def layers(self):
        return self._full.model.layers[:self._n_layers]

    @property
    def args(self):
        return self._full.args

    def make_cache(self):
        win = self.args.sliding_window
        caches = []
        for layer in self.layers:
            ratio = layer.attn.compress_ratio
            if ratio == 0:
                caches.append(RotatingKVCache(max_size=win))
            else:
                caches.append(SparseKVCache())
        return caches


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = DeepseekV4Model(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None):
        # Compile modules on first call (after weights are loaded)
        # Skip compilation when expert offloading is active since the
        # offloaded code path uses Python-level LRU mutations and lazy I/O.
        if not getattr(self, '_compiled', False):
            skip_compile = getattr(self, '_expert_offloading', False)
            for layer in self.model.layers:
                if not skip_compile:
                    layer.ffn = mx.compile(layer.ffn)
                layer._hc_pre = mx.compile(layer._hc_pre)
                layer._hc_post = mx.compile(layer._hc_post)
            self._compiled = True
        out = self.model(inputs, cache)
        return self.lm_head(out)

    def sanitize(self, weights):
        n_layers = self.args.num_hidden_layers

        # Detect original HF checkpoint format:
        #   - has `.scale` tensors (FP8 block scaling)
        #   - has `mtp.` prefix (multi-token prediction weights)
        #   - has `gate.bias` instead of `gate.e_score_correction_bias`
        is_hf_original = any(
            k.endswith(".scale") or k.startswith("mtp.")
            for k in weights
        )

        # Detect Thump604 MLX conversion format:
        #   - has `hc_attn.base` (dot-separated HC attrs vs our `hc_attn_base`)
        #   - has `e_score_correction_bias` (vs our `gate.bias`)
        #   - has `switch_mlp.` (vs our `ffn.experts.`)
        #   - has `shared_experts.gate_proj` (vs our `shared_experts.w1`)
        is_thump604 = any(
            "hc_attn.base" in k or "hc_ffn.base" in k or "hc_head.base" in k
            or "attn_hc.base" in k or "ffn_hc.base" in k or "head_hc.base" in k
            or ".e_score_correction_bias" in k
            or ".switch_mlp." in k
            for k in weights
        )

        # --- Step 0: Drop MTP weights and layers beyond num_hidden_layers ---
        def _is_excess_layer(key):
            """Check if key belongs to a layer index >= n_layers.
            Handles both 'layers.N.x' (HF original) and 'model.layers.N.x'."""
            parts = key.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    return int(parts[i + 1]) >= n_layers
            return False

        weights = {
            k: v for k, v in weights.items()
            if not k.startswith("mtp.") and not _is_excess_layer(k)
        }

        # --- Step 1: FP8/FP4 block dequantization (original HF only) ---
        if is_hf_original:
            weights = self._dequant_scaled_weights(weights)

        # --- Step 1b: Thump604 MLX conversion remapping ---
        if is_thump604:
            weights = self._remap_thump604(weights)

        # --- Step 2: Top-level key remapping ---
        renames = {}
        for k in list(weights.keys()):
            new_k = k
            if k.startswith("embed."):
                new_k = k.replace("embed.", "model.embed_tokens.", 1)
            elif k.startswith("head."):
                new_k = k.replace("head.", "lm_head.", 1)
            elif k.startswith("norm."):
                new_k = "model." + k
            elif k.startswith("hc_head_"):
                new_k = "model." + k
            elif k.startswith("layers."):
                new_k = "model." + k
            if new_k != k:
                renames[k] = new_k
        for old, new in renames.items():
            weights[new] = weights.pop(old)

        # --- Step 3: Routed expert w1/w2/w3 rename (pre-stacked mlx-community) ---
        new_weights = {}
        for k, v in weights.items():
            nk = k
            if ".ffn.experts.w1." in nk:
                nk = nk.replace(".ffn.experts.w1.", ".ffn.experts.gate_proj.")
            elif ".ffn.experts.w2." in nk:
                nk = nk.replace(".ffn.experts.w2.", ".ffn.experts.down_proj.")
            elif ".ffn.experts.w3." in nk:
                nk = nk.replace(".ffn.experts.w3.", ".ffn.experts.up_proj.")
            new_weights[nk] = v
        weights = new_weights

        # --- Step 4: Stack per-expert weights for SwitchGLU (HF original) ---
        # HF original has per-expert: model.layers.N.ffn.experts.E.w{1,2,3}.weight
        # We need stacked: model.layers.N.ffn.experts.{gate,down,up}_proj.weight
        _expert_w_map = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
        for l in range(n_layers):
            prefix = f"model.layers.{l}.ffn.experts"
            for src, dst in _expert_w_map.items():
                key0 = f"{prefix}.0.{src}.weight"
                if key0 in weights:
                    stack = []
                    for e in range(self.args.n_routed_experts):
                        ek = f"{prefix}.{e}.{src}.weight"
                        stack.append(weights.pop(ek))
                    weights[f"{prefix}.{dst}.weight"] = mx.stack(stack)

        return weights

    def _remap_thump604(self, weights):
        """Remap weight keys from Thump604 MLX conversion naming to ours.

        Thump604 (e.g. Thump604/DeepSeek-V4-Flash-MLX-Q2-mixed-gs128-affine)
        uses a different model class with these naming differences:

          Thump604                          Ours
          --------                          ----
          hc_attn.base / .fn / .scale       hc_attn_base / _fn / _scale
          hc_ffn.base / .fn / .scale        hc_ffn_base / _fn / _scale
          hc_head.base / .fn / .scale       hc_head_base / _fn / _scale
          gate.e_score_correction_bias      gate.bias
          shared_experts.gate_proj          shared_experts.w1
          shared_experts.up_proj            shared_experts.w3
          shared_experts.down_proj          shared_experts.w2
          switch_mlp.gate_proj/up_proj/     ffn.experts.gate_proj/up_proj/
            down_proj                         down_proj
          mlp.switch_mlp.*                  ffn.experts.*
          mlp.shared_experts.*              ffn.shared_experts.*
          mlp.gate.*                        ffn.gate.*
          self_attn.*                       attn.*
          input_layernorm                   attn_norm
          post_attention_layernorm          ffn_norm
          attn.wo_a (single QuantizedLinear) attn.wo_a.{0..N} (grouped list)
        """
        n_groups = self.args.o_groups

        new_weights = {}
        # Collect wo_a keys that need splitting: {layer_idx: {suffix: tensor}}
        wo_a_singles = {}

        for k, v in weights.items():
            nk = k

            # --- Hyper-connection dot notation -> underscore ---
            # hc_attn.base -> hc_attn_base (etc for fn, scale)
            # Some uploads use reversed naming (attn_hc.base) -- handle both.
            for hc_prefix, target in (
                ("hc_attn", "hc_attn"), ("hc_ffn", "hc_ffn"), ("hc_head", "hc_head"),
                ("attn_hc", "hc_attn"), ("ffn_hc", "hc_ffn"), ("head_hc", "hc_head"),
            ):
                for hc_attr in ("base", "fn", "scale"):
                    dot_form = f"{hc_prefix}.{hc_attr}"
                    underscore_form = f"{target}_{hc_attr}"
                    if dot_form in nk:
                        nk = nk.replace(dot_form, underscore_form)

            # --- Layer norm renames ---
            nk = nk.replace(".input_layernorm.", ".attn_norm.")
            nk = nk.replace(".post_attention_layernorm.", ".ffn_norm.")

            # --- self_attn -> attn ---
            nk = nk.replace(".self_attn.", ".attn.")

            # --- Gate bias rename ---
            nk = nk.replace(
                ".e_score_correction_bias", ".bias"
            )

            # --- MLP wrapper -> ffn ---
            # mlp.switch_mlp.* -> ffn.experts.*
            nk = nk.replace(".mlp.switch_mlp.", ".ffn.experts.")
            # mlp.shared_experts.* -> ffn.shared_experts.*
            nk = nk.replace(".mlp.shared_experts.", ".ffn.shared_experts.")
            # mlp.gate.* -> ffn.gate.*
            nk = nk.replace(".mlp.gate.", ".ffn.gate.")
            # ffn.switch_mlp -> ffn.experts (Thump604 format, already under ffn.)
            nk = nk.replace(".ffn.switch_mlp.", ".ffn.experts.")
            # Bare switch_mlp (no ffn. or mlp. wrapper)
            if ".switch_mlp." in nk:
                nk = nk.replace(".switch_mlp.", ".ffn.experts.")

            # --- Shared experts: gate_proj/up_proj/down_proj -> w1/w3/w2 ---
            nk = nk.replace(".shared_experts.gate_proj.", ".shared_experts.w1.")
            nk = nk.replace(".shared_experts.up_proj.", ".shared_experts.w3.")
            nk = nk.replace(".shared_experts.down_proj.", ".shared_experts.w2.")

            new_weights[nk] = v

        # --- wo_a: replace grouped list with single Linear if needed ---
        # Thump604 stores wo_a as a single QuantizedLinear, our model inits
        # it as a list. Replace self.wo_a with a single Linear so load_weights
        # can assign the weights.
        has_single_wo_a = any(
            ".attn.wo_a.weight" in k and not any(
                f".attn.wo_a.{g}." in k for g in range(n_groups)
            ) for k in new_weights
        )
        if has_single_wo_a:
            for layer in self.model.layers:
                group_feat = layer.attn.n_heads * layer.attn.head_dim // layer.attn.n_groups
                layer.attn.wo_a = nn.Linear(
                    group_feat,
                    layer.attn.n_groups * layer.attn.o_lora_rank,
                    bias=False,
                )

        return new_weights

    @staticmethod
    def _dequant_scaled_weights(weights):
        """Dequantize FP8 e4m3 block-scaled and FP4 packed weights.

        Original HF checkpoint stores:
          - Most weight matrices as FP8 e4m3 (uint8) with ue8m0 128x128 block scales
          - Routed expert weights as FP4 packed (int8, 2 values per byte) with 32-element block scales
          - Scale tensors have `.scale` suffix matching the `.weight` tensor

        After dequant, `.scale` keys are consumed and only `.weight` keys remain.
        """

        def _scale_to_float(scale):
            """Convert ue8m0 scale (uint8 encoding of fp32 exponent) to float."""
            if scale.dtype == mx.uint8:
                return mx.exp((scale.astype(mx.float32) - 127.0) * math.log(2.0))
            return scale.astype(mx.float32)

        def _dequant_fp8_block(weight, scale, block_size=128):
            """Dequantize FP8 e4m3 weight with ue8m0 128x128 block scaling."""
            weight = mx.from_fp8(weight, dtype=mx.bfloat16)
            scale = _scale_to_float(scale)
            m, n = weight.shape
            # Pad to block_size boundary
            pad_m = (-m) % block_size
            pad_n = (-n) % block_size
            if pad_m or pad_n:
                weight = mx.pad(weight, ((0, pad_m), (0, pad_n)))
            mb = (m + pad_m) // block_size
            nb = (n + pad_n) // block_size
            weight = weight.reshape(mb, block_size, nb, block_size)
            weight = (weight * scale[:, None, :, None]).reshape(
                m + pad_m, n + pad_n)
            return weight[:m, :n].astype(mx.bfloat16)

        def _dequant_fp4_block(weight, scale, block_size=32):
            """Dequantize FP4 packed expert weights (2 nibbles per byte)."""
            table = mx.array(
                [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                dtype=mx.float32,
            )
            packed = weight.astype(mx.uint8)
            low = packed & 0x0F
            high = (packed >> 4) & 0x0F
            unpacked = mx.stack(
                [mx.take(table, low), mx.take(table, high)], axis=-1)
            unpacked = unpacked.reshape(weight.shape[0], weight.shape[1] * 2)
            scale = mx.repeat(_scale_to_float(scale), block_size, axis=-1)
            return (unpacked * scale).astype(mx.bfloat16)

        new = {}
        for k, v in weights.items():
            if k.endswith(".scale"):
                wk = k[:-len(".scale")] + ".weight"
                w = weights.get(wk)
                if w is None:
                    # Orphan scale (no matching weight), keep it
                    new[k] = v
                    continue
                # FP4 packed routed experts: int8/uint8 weight where
                # scale covers 2x more columns (each byte = 2 values)
                if (w.dtype in (mx.int8, mx.uint8)
                        and ".ffn.experts." in wk
                        and "shared_experts" not in wk
                        and v.shape[-1] * 16 == w.shape[-1]):
                    new[wk] = _dequant_fp4_block(w, v)
                # FP8 e4m3: uint8 weight with block scale
                elif w.dtype == mx.uint8:
                    new[wk] = _dequant_fp8_block(w, v)
                else:
                    # Non-FP8 scale (keep both)
                    new[k] = v
                    if wk not in new:
                        new[wk] = w
            elif k not in new:
                new[k] = v
        return new

    @property
    def layers(self):
        return self.model.layers

    def draft_model(self, n_layers=10):
        """Create a shallow draft model for self-speculative decoding.

        Shares weights (zero extra memory). Uses first n_layers
        out of 43 for fast draft predictions.
        """
        return _ShallowV4(self, n_layers)

    def make_cache(self):
        caches = []
        win = self.args.sliding_window
        for layer in self.layers:
            ratio = layer.attn.compress_ratio
            if ratio == 0:
                # Pure sliding window layer
                caches.append(RotatingKVCache(max_size=win))
            else:
                # Compressed layer with sparse state serialization
                caches.append(SparseKVCache())
        return caches
