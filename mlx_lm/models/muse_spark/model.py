# Copyright © 2026 Apple Inc.

import re
from typing import Any, Callable

import mlx.core as mx
from mlx import nn
from mlx.nn.layers.distributed import shard_linear

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.layers.common import splice
from mlx_lm.layers.distributed import PipelineMixin, gather_last_axis
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.muse_spark import checkpoint
from mlx_lm.models.muse_spark.config import ModelArgs, TextArgs
from mlx_lm.models.muse_spark.kernels import pre_norm
from mlx_lm.models.muse_spark.layers import TransformerBlock, WeightlessRMSNorm
from mlx_lm.models.muse_spark.vision import VisionAdapter, VisionTower

__all__ = [
    "Model",
    "MuseSparkModel",
    "MuseSparkTextModel",
    "shard_experts_inplace",
]


# TODO: mx.contiguous of a slice fails on arrays this large; copy in chunks.
_MAX_COPY_ELEMENTS = 1 << 31


class MuseSparkTextModel(PipelineMixin, nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.args = args
        self.sliding_window = args.sliding_window
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(
                args,
                use_rope=bool(rope_theta),
                use_sliding=layer_type == "sliding_attention",
            )
            for rope_theta, layer_type in zip(args.layer_rope_theta, args.layer_types)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        *,
        cache: list[Any] | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        # input_embeddings are already normalized.
        h = self.embed(inputs) if input_embeddings is None else input_embeddings
        h = h.astype(mx.float32)

        layers = self.pipeline_layers
        if cache is None:
            cache = [None] * len(layers)

        fa_mask = create_attention_mask(h, cache[self.fa_idx])
        if self.swa_idx is not None:
            swa_mask = create_attention_mask(
                h, cache[self.swa_idx], window_size=self.sliding_window
            )

        rank, size = self.pipeline_rank, self.pipeline_size
        attn_input = None
        if rank < size - 1:
            h = mx.distributed.recv_like(h, rank + 1)
            # Equal to the fused norm of the previous stage, not to input_layernorm.
            norm = layers[0].input_layernorm
            attn_input = pre_norm(h, norm, dtype=norm.weight.dtype)

        # Each block also makes the next block's attention input, so that the
        # norm is fused with its add.
        for i, (layer, c) in enumerate(zip(layers, cache)):
            mask = swa_mask if layer.use_sliding else fa_mask
            nxt = layers[i + 1].input_layernorm if i + 1 < len(layers) else None
            h, attn_input = layer(
                h, mask=mask, cache=c, attn_input=attn_input, next_norm=nxt
            )

        # Send to the next process in the pipeline
        if rank != 0:
            h = mx.distributed.send(h, (rank - 1) % size)
            if cache[-1] is not None:
                cache[-1].keys = mx.depends(cache[-1].keys, h)

        # Broadcast h while keeping it in the graph
        if size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        # The norm reads the fp32 stream.
        return self.norm(h).astype(mx.bfloat16)

    def embed(self, inputs: mx.array) -> mx.array:
        """The normalized token embeddings."""
        return mx.fast.rms_norm(self.embed_tokens(inputs), None, self.args.rms_norm_eps)

    @property
    def fa_idx(self) -> int:
        layers = self.pipeline_layers
        return next((i for i, l in enumerate(layers) if not l.use_sliding), 0)

    @property
    def swa_idx(self) -> int | None:
        layers = self.pipeline_layers
        return next((i for i, l in enumerate(layers) if l.use_sliding), None)


class MuseSparkModel(nn.Module):
    """The language model and the vision tower, named as in the checkpoint."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.language_model = MuseSparkTextModel(args.text)
        # The vision tower is not sharded.
        if args.vision is not None:
            self.vision_tower = VisionTower(args.vision)
            self.vision_adapter = VisionAdapter(
                args.out_hidden_size, args.projector_hidden_size
            )
            self.vision_projection = nn.Linear(
                args.projector_hidden_size, args.text.hidden_size, bias=False
            )

    # mlx_lm.utils pipelines model.model with these.
    @property
    def layers(self) -> list[TransformerBlock | None]:
        return self.language_model.layers

    @property
    def start_idx(self) -> int:
        return self.language_model.start_idx

    @property
    def end_idx(self) -> int | None:
        return self.language_model.end_idx

    def pipeline(self, group, split=None) -> None:
        self.language_model.pipeline(group, split)


def shard_experts_inplace(
    module: nn.Module, sharding: str, group: mx.distributed.Group
) -> None:
    N, r = group.size(), group.rank()
    updates = {}
    for name, w in module.parameters().items():
        if not isinstance(w, mx.array):
            continue
        if w.ndim < 3:
            # we don't shard scales for the nvfp4 weights
            continue
        axis = w.ndim - 2 if sharding == "all-to-sharded" else w.ndim - 1
        n = w.shape[axis] // N
        lo, hi = r * n, (r + 1) * n
        per_expert = w.size // w.shape[0]
        step = max(1, _MAX_COPY_ELEMENTS // (2 * per_expert))
        parts = []
        for i in range(0, w.shape[0], step):
            chunk = w[i : i + step]
            chunk = chunk[:, lo:hi] if axis == 1 else chunk[..., lo:hi]
            chunk = mx.contiguous(chunk)
            mx.eval(chunk)
            parts.append(chunk)
        updates[name] = mx.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
    module.update(updates)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = MuseSparkModel(args)
        text = args.text
        self.lm_head = nn.Linear(text.hidden_size, text.vocab_size, bias=False)
        self.sharding_group = None
        if args.vision is not None:
            self.perception_emb_norm = WeightlessRMSNorm(eps=text.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        *,
        cache: list[Any] | None = None,
        input_embeddings: mx.array | None = None,
        pixel_values: mx.array | list[mx.array] | None = None,
    ) -> mx.array:
        if pixel_values is not None:
            assert input_embeddings is None, "Pass pixel_values or input_embeddings."
            input_embeddings = self.embed_inputs(inputs, pixel_values=pixel_values)
        out = self.model.language_model(
            inputs, cache=cache, input_embeddings=input_embeddings
        )
        out = self.lm_head(out.astype(self.lm_head.weight.dtype))
        if self.sharding_group is not None:
            out = gather_last_axis(out, self.sharding_group)
        out = out.astype(mx.float32) * self.args.text.output_multiplier
        cap = self.args.text.final_logit_softcapping
        return mx.tanh(out / cap) * cap

    def embed_inputs(
        self,
        inputs: mx.array,
        *,
        pixel_values: mx.array | list[mx.array] | None = None,
    ) -> mx.array:
        """The normalized input embeddings, with the vision rows at the patch tokens."""
        h = self.model.language_model.embed(inputs)
        # The vision rows take the dtype of the text embeddings.
        if pixel_values is not None:
            h = splice(
                h,
                inputs,
                rows=self.encode_image(pixel_values),
                token_id=self.args.image_token_id,
            )
        return h

    def encode_image(self, pixel_values: mx.array | list[mx.array]) -> mx.array:
        """pixel_values: one image (num_channels, H, W), a list of images of
        any size, or images of one size (num_images, num_channels, H, W)."""
        if isinstance(pixel_values, mx.array) and pixel_values.ndim == 3:
            pixel_values = [pixel_values]
        assert len(pixel_values), "No images."
        assert all(p.ndim == 3 for p in pixel_values), "Images must be (C, H, W)."
        tower, adapter = self.model.vision_tower, self.model.vision_adapter
        dtype = tower.patch_embedder.patch_embedding.weight.dtype
        # TODO: consider packing the images into one sequence, with attention
        # per image (MLX has no varlen attention).
        h = mx.concatenate([adapter(tower(p.astype(dtype))) for p in pixel_values])
        h = self.model.vision_projection(h)
        return self.perception_emb_norm(h)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        return checkpoint.sanitize(self, weights)

    @property
    def cast_predicate(self) -> Callable[[str], bool]:
        """Keep the gate weights in fp32 when the model is converted."""

        def predicate(k: str) -> bool:
            return not k.endswith(("_alpha", "_beta"))

        return predicate

    @property
    def quant_predicate(self) -> Callable[[str, nn.Module], bool]:
        """Quantize only MoE layers except the first one and the last one."""
        experts_re = re.compile(
            r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(gate|up|down)_proj$"
        )
        full_precision = {0, self.args.text.num_hidden_layers - 1}

        def predicate(path: str, module: nn.Module) -> bool:
            m = experts_re.match(path)
            return m is not None and int(m.group(1)) not in full_precision

        return predicate

    def shard(self, group: mx.distributed.Group | None = None) -> None:
        group = group or mx.distributed.init()
        N = group.size()
        n_kv_heads = self.args.text.num_key_value_heads
        assert (
            n_kv_heads % N == 0
        ), f"group size {N} must divide num_key_value_heads {n_kv_heads}"
        vocab = self.args.text.vocab_size
        assert vocab % N == 0, f"group size {N} must divide vocab_size {vocab}"
        self.lm_head = shard_linear(self.lm_head, "all-to-sharded", group=group)
        self.sharding_group = group
        for layer in self.layers:
            attn = layer.self_attn
            q_dim = attn.n_heads * attn.head_dim
            kv_dim = attn.n_kv_heads * attn.head_dim
            attn.qkvg_proj = shard_linear(
                attn.qkvg_proj,
                "all-to-sharded",
                segments=[q_dim, q_dim + kv_dim, q_dim + 2 * kv_dim],
                group=group,
            )
            attn.o_proj = shard_linear(attn.o_proj, "sharded-to-all", group=group)
            attn.n_heads //= N
            attn.n_kv_heads //= N

            layer.mlp.sharding_group = group
            experts = layer.mlp.experts
            shard_experts_inplace(experts.gate_proj, "all-to-sharded", group)
            shard_experts_inplace(experts.down_proj, "sharded-to-all", group)
            shard_experts_inplace(experts.up_proj, "all-to-sharded", group)

    @property
    def layers(self) -> list[TransformerBlock]:
        return self.model.language_model.pipeline_layers

    def make_cache(self) -> list[KVCache | RotatingKVCache]:
        return [
            (
                RotatingKVCache(max_size=self.model.language_model.sliding_window)
                if layer.use_sliding
                else KVCache()
            )
            for layer in self.layers
        ]
