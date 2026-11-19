# Copyright © 2026 Apple Inc.

import math
from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear

from mlx_lm.layers.distributed import PipelineMixin
from mlx_lm.models.deepseek_v41 import checkpoint
from mlx_lm.models.deepseek_v41.config import ModelArgs
from mlx_lm.models.deepseek_v41.layers import (
    DecoderLayer,
    LayerCache,
    append,
    collapse,
)

__all__ = [
    "Model",
    "TextModel",
]

# The shared state that grows each step, kept in the LayerCache field of that name.
_CACHE_FIELDS = ("compress_kv", "index_k")

# A view at a byte offset that is not a multiple of its item size reads the
# wrong bytes, so each part of a message starts at a multiple of 8 bytes.
_ALIGN = 8


def _shared_reads(args: ModelArgs, layer: int) -> dict[str, int]:
    """The shared state a layer reads, as {key: layer that writes it}."""
    ratios = args.compress_ratios
    if layer >= len(ratios) or not ratios[layer]:
        return {}
    kv, index = set(args.kv_source_layer_ids), set(args.index_source_layer_ids)

    def last(match: Callable[[int], bool]) -> Optional[int]:
        return max((j for j in range(layer + 1) if match(j)), default=None)

    reads = {"compress_kv": last(lambda j: j in kv and ratios[j])}
    if layer in index:
        reads["index_k"] = last(lambda j: j in kv and j in index)
        source = args.candidate_source_layer_id
        if 0 <= source < layer and args.candidate_topk_blocks > 0:
            reads["candidates"] = source
    else:
        reads["topk_idx"] = last(lambda j: j in index)
    return {key: source for key, source in reads.items() if source is not None}


def _entries(args: ModelArgs, source: Optional[int], position: int) -> int:
    """The compressed entries of kv source `source` after `position` tokens."""
    return position // args.compress_ratios[source] if source is not None else 0


def _index_keys(args: ModelArgs, layer: int, position: int) -> int:
    """The index keys that index layer `layer` scores after `position` tokens."""
    return _entries(args, _shared_reads(args, layer).get("index_k"), position)


def _window_indices(*, previous: int, length: int, window: int) -> mx.array:
    """The window rows each query attends to, -1 for none. The window holds
    the previous rows, then the new ones."""
    width = min(window, previous + length)
    end = previous + mx.arange(length)
    start = mx.maximum(end - window + 1, 0)
    indices = start[:, None] + mx.arange(width)[None, :]
    return mx.where(indices > end[:, None], -1, indices)


def _layout(
    args: ModelArgs,
    crossing: dict[str, int],
    batch: int,
    length: int,
    start_pos: int,
    dtype: mx.Dtype,
) -> list[tuple[str, tuple[int, ...], mx.Dtype]]:
    """The parts of the message at a stage boundary, as (key, shape, dtype).

    The compressed KV and the index keys have the dtype of the streams, because
    the next stage cannot know the dtype the stage before makes them in.
    """
    end = start_pos + length
    layout = [
        ("streams", (batch, length, args.hc_mult, args.hidden_size), dtype),
        ("pre_mix", (batch, length, args.hc_mult), mx.float32),
    ]
    for key, source in crossing.items():
        if key in _CACHE_FIELDS:
            # Only the new entries: the next stage keeps the earlier ones.
            new = _entries(args, source, end) - _entries(args, source, start_pos)
            width = args.head_dim if key == "compress_kv" else args.index_head_dim
            layout.append((key, (batch, new, width), dtype))
        elif key == "candidates":
            # One flag for each candidate block.
            block = args.candidate_block_size
            blocks = (_index_keys(args, source, end) + block - 1) // block
            layout.append((key, (batch, length, blocks), mx.bool_))
        else:
            topk = min(args.index_topk, _index_keys(args, source, end))
            layout.append((key, (batch, length, topk), mx.int32))
    return layout


def _nbytes(shape: tuple[int, ...], dtype: mx.Dtype) -> int:
    """The bytes of one part in a message, with its padding."""
    size = math.prod(shape) * dtype.size
    return (size + _ALIGN - 1) // _ALIGN * _ALIGN


def _unpack(
    packed: mx.array, layout: list[tuple[str, tuple[int, ...], mx.Dtype]]
) -> dict[str, mx.array]:
    parts = {}
    start = 0
    for key, shape, dtype in layout:
        data = packed[start : start + math.prod(shape) * dtype.size]
        parts[key] = data.reshape(*shape[:-1], shape[-1] * dtype.size).view(dtype)
        start += _nbytes(shape, dtype)
    return parts


def _pack(parts: list[mx.array]) -> mx.array:
    """The bytes of all parts in one array: two sends can run in a different
    order than the two receives on the other rank."""
    data = []
    for part in parts:
        flat = part.view(mx.uint8).reshape(-1)
        pad = _nbytes(part.shape, part.dtype) - flat.size
        data.append(mx.pad(flat, (0, pad)) if pad else flat)
    return mx.concatenate(data)


def _crossing(args: ModelArgs, boundary: int) -> dict[str, int]:
    """The shared state that layers from `boundary` on read from layers before
    it, as {key: layer that writes it}."""
    crossing = {}
    for layer in range(boundary, args.num_hidden_layers):
        for key, source in _shared_reads(args, layer).items():
            if source < boundary:
                crossing[key] = source
    return crossing


class TextModel(PipelineMixin, nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.args = args

    def __call__(
        self, inputs: mx.array, cache: Optional[list[LayerCache]] = None
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        batch, length, _ = h.shape
        layers = self.pipeline_layers
        cache = cache or self.make_cache()
        start_pos = cache[0].offset
        # Every layer's window holds the same rows, so all use the same indices.
        previous = 0 if cache[0].window is None else cache[0].window.shape[1]
        window_idx = _window_indices(
            previous=previous, length=length, window=self.args.sliding_window
        )
        window_idx = mx.broadcast_to(window_idx, (batch, *window_idx.shape))

        streams = mx.broadcast_to(
            h[:, :, None, :], (batch, length, self.args.hc_mult, self.args.hidden_size)
        )
        pre_mix = mx.zeros((batch, length, self.args.hc_mult), dtype=mx.float32)
        pre_mix[..., 0] = 1.0
        rank, size = self.pipeline_rank, self.pipeline_size
        shared = {}
        if rank < size - 1:
            streams, pre_mix = self._receive(streams, start_pos, cache[0], shared)

        for layer, layer_cache in zip(layers, cache, strict=True):
            streams, pre_mix = layer(
                streams,
                pre_mix,
                start_pos=start_pos,
                window_idx=window_idx,
                cache=layer_cache,
                shared=shared,
            )

        if rank > 0:
            sent = self._send(streams, pre_mix, start_pos, shared)
            streams, pre_mix = mx.depends([streams, pre_mix], sent)
            # Prefill only evaluates the cache, which must still send.
            cache[-1].window = mx.depends(cache[-1].window, sent)
        out = collapse(streams, pre_mix)
        if size > 1:
            # Rank 0 runs the last layers, and every rank needs its output.
            out = mx.distributed.all_gather(out)[:batch]
        return self.norm(out)

    def make_cache(self) -> list[LayerCache]:
        return [LayerCache(self.args.sliding_window) for _ in self.pipeline_layers]

    def _receive(
        self,
        like: mx.array,
        start_pos: int,
        cache: LayerCache,
        shared: dict[str, Optional[mx.array]],
    ) -> tuple[mx.array, mx.array]:
        """Receive the streams and pre_mix, and put the shared state in `shared`."""
        batch, length = like.shape[:2]
        layout = _layout(
            self.args, self._shared_in, batch, length, start_pos, like.dtype
        )
        size = sum(_nbytes(shape, dtype) for _, shape, dtype in layout)
        packed = mx.distributed.recv((size,), mx.uint8, self.pipeline_rank + 1)
        parts = _unpack(packed, layout)
        streams = parts.pop("streams")
        pre_mix = parts.pop("pre_mix")
        for key, part in parts.items():
            if key in _CACHE_FIELDS:
                # The first layer never writes this field, so it keeps the
                # entries of the source in the stage before.
                if part.shape[1]:
                    setattr(cache, key, append(getattr(cache, key), part))
                shared[key] = getattr(cache, key)
            elif key == "candidates":
                keys = _index_keys(self.args, self._shared_in[key], start_pos + length)
                mask = mx.repeat(part, self.args.candidate_block_size, axis=-1)
                shared[key] = mask[..., :keys] if keys else None
            else:
                shared[key] = part
        return streams, pre_mix

    def _send(
        self,
        streams: mx.array,
        pre_mix: mx.array,
        start_pos: int,
        shared: dict[str, Optional[mx.array]],
    ) -> mx.array:
        """Send the streams, pre_mix and the shared state the next stages read."""
        batch, length = streams.shape[:2]
        layout = _layout(
            self.args, self._shared_out, batch, length, start_pos, streams.dtype
        )
        state = {**shared, "streams": streams, "pre_mix": pre_mix}
        parts = []
        for key, shape, dtype in layout:
            part = state.get(key)
            if part is None:
                # Not written yet, so the layout has no entries either.
                part = mx.zeros(shape, dtype)
            elif key in _CACHE_FIELDS:
                # It holds all entries so far; send only the new ones.
                part = part[:, part.shape[1] - shape[1] :]
            elif key == "candidates":
                part = part[..., :: self.args.candidate_block_size]
            if part.shape != shape or part.dtype != dtype:
                raise ValueError(
                    f"The next pipeline stage expects {key} as {dtype} {shape}, "
                    f"not {part.dtype} {part.shape}."
                )
            parts.append(part)
        return mx.distributed.send(_pack(parts), self.pipeline_rank - 1)

    def pipeline(
        self, group: mx.distributed.Group, split: Optional[list[int]] = None
    ) -> None:
        super().pipeline(group, split)
        self._shared_in = _crossing(self.args, self.start_idx)
        self._shared_out = _crossing(self.args, self.end_idx)


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = TextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self, inputs: mx.array, cache: Optional[list[LayerCache]] = None
    ) -> mx.array:
        return self.lm_head(self.model(inputs, cache))

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        return checkpoint.sanitize(self, weights)

    @property
    def cast_predicate(self) -> Callable[[str], bool]:
        # The reference runtime keeps these in float32.
        keep = (
            "attn_hc.",
            "ffn_hc.",
            "sinks",
            "e_score_correction_bias",
            "compressor.",
        )

        def predicate(k: str) -> bool:
            return not any(s in k for s in keep)

        return predicate

    def shard(self, group: Optional[mx.distributed.Group] = None) -> None:
        group = group or mx.distributed.init()
        size, rank = group.size(), group.rank()
        heads, groups = self.args.num_attention_heads, self.args.o_groups
        if heads % size or groups % size:
            raise ValueError(
                f"Tensor parallelism over {size} ranks needs {size} to divide the "
                f"{heads} attention heads and the {groups} output groups."
            )
        for layer in self.layers:
            attn = layer.self_attn
            attn.q_b_proj = shard_linear(attn.q_b_proj, "all-to-sharded", group=group)
            # An output group reads only its own heads, so a rank keeps whole groups.
            shard_inplace(attn.o_a_proj, lambda path, weight: 0, group=group)
            attn.o_b_proj = shard_linear(attn.o_b_proj, "sharded-to-all", group=group)
            attn.num_heads //= size
            attn.o_groups //= size
            attn._group_ids = mx.arange(attn.o_groups)[None]
            attn.sinks = attn.sinks[rank * attn.num_heads : (rank + 1) * attn.num_heads]

            layer.mlp.sharding_group = group
            for mlp in (layer.mlp.experts.switch_mlp, layer.mlp.shared_experts):
                shard_inplace(mlp.gate_proj, "all-to-sharded", group=group)
                shard_inplace(mlp.up_proj, "all-to-sharded", group=group)
                shard_inplace(mlp.down_proj, "sharded-to-all", group=group)

    @property
    def layers(self) -> list[DecoderLayer]:
        return self.model.pipeline_layers

    def make_cache(self) -> list[LayerCache]:
        return self.model.make_cache()
