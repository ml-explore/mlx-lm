# ABOUTME: Compiles chunked prefill graphs for LLaMA-style transformer blocks.
# ABOUTME: Emits per-layer K/V tensors plus timing metrics for paged prefill.

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import mlx.core as mx

from ..paged_slot_kv_cache import PagedSlotKVCache
from .llama_arrays import LlamaArrayGraph


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.lower() not in {"", "0", "false", "no", "off"}


def _infer_attn_head_dim(attn) -> int:
    head_dim = getattr(attn, "head_dim", None)
    if isinstance(head_dim, int) and head_dim > 0:
        return head_dim
    hidden_size = getattr(attn, "hidden_size", None)
    num_heads = getattr(attn, "n_heads", None)
    if isinstance(hidden_size, int) and isinstance(num_heads, int) and num_heads > 0:
        return hidden_size // num_heads
    q_proj = getattr(attn, "q_proj", None)
    if q_proj is not None:
        weight = getattr(q_proj, "weight", None)
        if (
            weight is not None
            and weight.shape
            and isinstance(num_heads, int)
            and num_heads > 0
        ):
            return int(weight.shape[0]) // num_heads
    k_proj = getattr(attn, "k_proj", None)
    num_kv = getattr(attn, "n_kv_heads", None)
    if k_proj is not None and isinstance(num_kv, int) and num_kv > 0:
        weight = getattr(k_proj, "weight", None)
        if weight is not None and weight.shape:
            return int(weight.shape[0]) // num_kv
    raise ValueError("Unable to infer head_dim for attention module")


@dataclass
class _ChunkInputs:
    tokens: mx.array
    base_lens: mx.array
    start_positions: mx.array
    block_tables: mx.array
    k_cache: mx.array
    v_cache: mx.array
    kv_head_mapping: Optional[mx.array]


class LlamaPrefillGraph:
    """Produces per-layer K/V chunks for paged caches using MX graphs."""

    def __init__(self, model) -> None:
        self._graph = LlamaArrayGraph(model)
        self._layers = list(self._graph.layers)
        if not self._layers:
            raise ValueError(
                "LlamaPrefillGraph requires at least one transformer layer"
            )
        attn = self._layers[0].self_attn
        self._head_dim = _infer_attn_head_dim(attn)
        self._cache: Dict[Tuple, mx.Function] = {}
        self._profile_enabled = _env_flag("MLXLM_PREFILL_PROFILE")
        self._metrics = self._reset_metrics()

    @property
    def head_dim(self) -> int:
        return self._head_dim

    def _reset_metrics(self) -> Dict[str, object]:
        layer_count = len(self._layers)
        return {
            "array_prefill_qkv_s": 0.0,
            "array_prefill_rope_s": 0.0,
            "array_prefill_attn_s": 0.0,
            "array_prefill_mlp_s": 0.0,
            "array_prefill_overlay_s": 0.0,
            "array_prefill_layer_attn_s": [0.0 for _ in range(layer_count)],
            "array_prefill_layer_mlp_s": [0.0 for _ in range(layer_count)],
            "array_prefill_layer_overlay_s": [0.0 for _ in range(layer_count)],
        }

    def get_compiled(
        self,
        *,
        batch_size: int,
        chunk_len: int,
        block_tables_shape: Tuple[int, int],
        k_cache_shape: Tuple[int, ...],
        v_cache_shape: Tuple[int, ...],
        kv_map_shape: Optional[Tuple[int, ...]],
        dtype,
        pending_flag: int,
    ):
        key = (
            batch_size,
            chunk_len,
            block_tables_shape,
            k_cache_shape,
            v_cache_shape,
            kv_map_shape,
            str(dtype),
            int(bool(pending_flag)),
            self._profile_enabled,
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        def _fn(
            tokens,
            base_lens,
            block_tables,
            k_cache,
            v_cache,
            pending_k,
            pending_v,
            kv_map=None,
        ):
            chunk_inputs = _ChunkInputs(
                tokens=tokens,
                base_lens=base_lens,
                start_positions=mx.zeros((tokens.shape[0],), dtype=mx.int32),
                block_tables=block_tables,
                k_cache=k_cache,
                v_cache=v_cache,
                kv_head_mapping=kv_map,
            )
            hidden_last, k_layers, v_layers = self._run_eager_chunk(chunk_inputs, dtype)
            if pending_flag:
                k_layers = mx.concatenate([pending_k, k_layers], axis=1)
                v_layers = mx.concatenate([pending_v, v_layers], axis=1)
            return hidden_last, k_layers, v_layers

        compiled = _fn if self._profile_enabled else mx.compile(_fn)
        self._cache[key] = compiled
        return compiled

    def _run_eager_chunk(
        self,
        inputs: _ChunkInputs,
        kv_dtype,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        chunk_len = inputs.tokens.shape[1]
        batch = inputs.tokens.shape[0]
        layers = len(self._layers)
        attn = self._layers[0].self_attn
        num_kv_heads = attn.n_kv_heads
        head_dim = self._head_dim

        token_matrix = inputs.tokens
        if token_matrix.ndim != 2:
            raise ValueError("tokens must have shape [batch, chunk_len]")
        embeddings = self._graph.embed(token_matrix)
        if (
            embeddings.ndim == 3
            and embeddings.shape[0] == batch
            and embeddings.shape[1] == chunk_len
        ):
            hidden = embeddings
        elif (
            embeddings.ndim == 3
            and embeddings.shape[0] == chunk_len
            and embeddings.shape[1] == batch
        ):
            hidden = mx.transpose(embeddings, (1, 0, 2))
        else:
            raise ValueError("Unexpected embedding shape for prefill chunk")

        k_chunk = mx.zeros(
            (layers, chunk_len, batch, num_kv_heads, head_dim), dtype=kv_dtype
        )
        v_chunk = mx.zeros_like(k_chunk)

        for layer_idx, layer in enumerate(self._layers):
            hidden, k_out, v_out = self._run_layer(
                layer_idx,
                layer,
                hidden,
                inputs,
                kv_dtype=kv_dtype,
            )
            k_chunk[layer_idx] = k_out
            v_chunk[layer_idx] = v_out

        return hidden[:, -1, :], k_chunk, v_chunk

    def _run_layer(
        self,
        layer_idx: int,
        layer,
        hidden: mx.array,
        chunk_inputs: _ChunkInputs,
        *,
        kv_dtype,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        attn = layer.self_attn
        profile = self._profile_enabled

        norm_hidden = layer.input_layernorm(hidden)

        qkv_start = time.perf_counter()
        proj = self._graph.project_qkv(attn, norm_hidden)
        q = proj.q  # [B, Hq, T, Dh]
        k_tokens = proj.k
        v_tokens = proj.v
        if profile:
            self._metrics["array_prefill_qkv_s"] += time.perf_counter() - qkv_start

        rope_start = time.perf_counter()
        q = self._apply_rope(attn, q, chunk_inputs.base_lens)
        k_tokens = self._apply_rope(attn, k_tokens, chunk_inputs.base_lens)
        if profile:
            self._metrics["array_prefill_rope_s"] += time.perf_counter() - rope_start

        k_tokens = mx.transpose(
            k_tokens.astype(kv_dtype), (2, 0, 1, 3)
        )  # [T, B, KvH, Dh]
        v_tokens = mx.transpose(v_tokens.astype(kv_dtype), (2, 0, 1, 3))

        attn_start = time.perf_counter()
        hidden = hidden + self._attention_block(
            layer_idx,
            layer,
            q,
            mx.transpose(k_tokens, (1, 2, 0, 3)),
            mx.transpose(v_tokens, (1, 2, 0, 3)),
            chunk_inputs,
        )
        attn_dur = time.perf_counter() - attn_start
        if profile:
            self._metrics["array_prefill_attn_s"] += attn_dur
            self._metrics["array_prefill_layer_attn_s"][layer_idx] += attn_dur

        mlp_start = time.perf_counter()
        hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        mlp_dur = time.perf_counter() - mlp_start
        if profile:
            self._metrics["array_prefill_mlp_s"] += mlp_dur
            self._metrics["array_prefill_layer_mlp_s"][layer_idx] += mlp_dur

        return hidden, k_tokens, v_tokens

    def _apply_rope(self, attn, tensor: mx.array, base_lens: mx.array) -> mx.array:
        batch, _, tokens, _ = tensor.shape
        positions = mx.arange(tokens, dtype=mx.int32)
        absolute = mx.expand_dims(base_lens, axis=0) + positions[:, None]  # [T, B]
        transposed = mx.transpose(tensor, (2, 0, 1, 3))  # [T, B, H, Dh]
        flattened = mx.reshape(
            transposed, (tokens * batch, tensor.shape[1], 1, tensor.shape[3])
        )
        offsets = mx.reshape(mx.transpose(absolute, (0, 1)), (tokens * batch,))
        rotated = attn.rope(flattened, offset=offsets)
        restored = mx.reshape(
            rotated, (tokens, batch, tensor.shape[1], tensor.shape[3])
        )
        return mx.transpose(restored, (1, 2, 0, 3))

    def _attention_block(
        self,
        layer_idx: int,
        layer,
        q,
        k_chunk,
        v_chunk,
        chunk_inputs: _ChunkInputs,
    ) -> mx.array:
        block_tables = chunk_inputs.block_tables
        context_lens = chunk_inputs.base_lens
        block_size = getattr(PagedSlotKVCache, "block_size", 16)
        max_prefix = int(mx.max(context_lens).item()) if context_lens.size > 0 else 0
        token_count = k_chunk.shape[2]
        total_k = max_prefix + token_count

        overlay_start = time.perf_counter()
        k_prefix = self._gather_prefix(
            chunk_inputs.k_cache,
            layer_idx,
            block_tables,
            max_prefix,
            block_size,
        )
        v_prefix = self._gather_prefix(
            chunk_inputs.v_cache,
            layer_idx,
            block_tables,
            max_prefix,
            block_size,
        )
        overlay_dur = time.perf_counter() - overlay_start
        if self._profile_enabled:
            self._metrics["array_prefill_overlay_s"] += overlay_dur
            self._metrics["array_prefill_layer_overlay_s"][layer_idx] += overlay_dur

        k_all = mx.concatenate([k_prefix, k_chunk], axis=2)
        v_all = mx.concatenate([v_prefix, v_chunk], axis=2)

        k_all, v_all = self._maybe_map_kv_heads(
            q.shape[1],
            k_all,
            v_all,
            chunk_inputs.kv_head_mapping,
        )

        attn_mask = self._build_mask(context_lens, token_count, max_prefix)
        scale = layer.self_attn.scale
        attn_ctx = mx.fast.scaled_dot_product_attention(
            q,
            k_all,
            v_all,
            mask=attn_mask,
            scale=scale,
        )
        return self._graph.attention_output(layer.self_attn, attn_ctx)

    def _maybe_map_kv_heads(
        self,
        num_heads: int,
        k_all: mx.array,
        v_all: mx.array,
        kv_head_mapping: Optional[mx.array],
    ) -> Tuple[mx.array, mx.array]:
        num_kv_heads = k_all.shape[1]
        if num_heads == num_kv_heads:
            return k_all, v_all
        if kv_head_mapping is not None:
            mapping = [int(x) for x in kv_head_mapping.tolist()]
        elif num_kv_heads == 1:
            mapping = [0 for _ in range(num_heads)]
        else:
            mapping = [(h * num_kv_heads) // num_heads for h in range(num_heads)]
        idx = mx.array(mapping, dtype=mx.int32)
        k_mapped = mx.take(k_all, idx, axis=1)
        v_mapped = mx.take(v_all, idx, axis=1)
        return k_mapped, v_mapped

    def _build_mask(
        self, base_lengths: mx.array, chunk_len: int, prefix_len: int
    ) -> mx.array:
        batch = base_lengths.shape[0]
        total_k = prefix_len + chunk_len
        mask = mx.full((batch, 1, chunk_len, total_k), -mx.inf, dtype=mx.float32)

        limits = mx.reshape(base_lengths + chunk_len, (batch, 1, 1, 1))
        positions = mx.reshape(mx.arange(total_k, dtype=mx.int32), (1, 1, 1, total_k))
        zeros = mx.zeros_like(mask)
        allowed = positions < limits
        mask = mx.where(allowed, zeros, mask)

        causal = mx.tril(mx.ones((chunk_len, chunk_len), dtype=mx.float32))
        causal = mx.where(causal > 0, 0.0, -mx.inf)
        left = mx.zeros((1, 1, chunk_len, prefix_len), dtype=mx.float32)
        right_len = max(0, total_k - prefix_len - chunk_len)
        right = mx.zeros((1, 1, chunk_len, right_len), dtype=mx.float32)
        causal_full = mx.concatenate(
            [left, mx.expand_dims(mx.expand_dims(causal, 0), 0), right], axis=3
        )
        mask = mask + causal_full
        return mask

    def _gather_prefix(
        self,
        cache: mx.array,
        layer_idx: int,
        block_tables: mx.array,
        prefix_len: int,
        block_size: int,
    ) -> mx.array:
        batch = block_tables.shape[0]
        kv_heads = cache.shape[1]
        head_dim = cache.shape[-1]
        if prefix_len <= 0:
            return mx.zeros((batch, kv_heads, 0, head_dim), dtype=cache.dtype)

        prefix_rows = []
        tables_np = block_tables.tolist()
        for b in range(batch):
            rows = []
            remaining = prefix_len
            for table_entry in tables_np[b]:
                block_id = int(table_entry)
                if block_id < 0:
                    break
                block = cache[layer_idx, :, block_id]  # [KvH, block_size, Dh]
                take = min(block_size, remaining)
                rows.append(block[:, :take])
                remaining -= take
                if remaining <= 0:
                    break
            if rows:
                stacked = mx.concatenate(rows, axis=1)
                if stacked.shape[1] < prefix_len:
                    pad = mx.zeros(
                        (kv_heads, prefix_len - stacked.shape[1], head_dim),
                        dtype=cache.dtype,
                    )
                    stacked = mx.concatenate([stacked, pad], axis=1)
                else:
                    stacked = stacked[:, :prefix_len]
            else:
                stacked = mx.zeros((kv_heads, prefix_len, head_dim), dtype=cache.dtype)
            prefix_rows.append(mx.expand_dims(stacked, axis=0))
        return mx.concatenate(prefix_rows, axis=0)

    def consume_metrics(self) -> Dict[str, object]:
        if not self._profile_enabled:
            return {}
        snapshot = {}
        for key, value in self._metrics.items():
            if isinstance(value, list):
                snapshot[key] = list(value)
            else:
                snapshot[key] = float(value)
        self._metrics = self._reset_metrics()
        return snapshot


__all__ = ["LlamaPrefillGraph", "_infer_attn_head_dim"]
