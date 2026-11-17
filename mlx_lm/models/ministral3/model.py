# Copyright © 2023 Apple Inc.

from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_linear

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.layers.distributed import PipelineMixin, gather_last_axis
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.ministral3 import checkpoint
from mlx_lm.models.ministral3.config import ModelArgs
from mlx_lm.models.ministral3.layers import (
    Fp8AllToShardedLinear,
    Fp8Linear,
    Fp8ShardedToAllLinear,
    FusedGLU,
    TransformerBlock,
    _get_llama_4_attn_scale,
)

__all__ = [
    "LanguageModel",
    "Model",
    "shard_layers",
]


def _shard_linear(layer, sharding, group, segments=1):
    if not isinstance(layer, Fp8Linear):
        return shard_linear(layer, sharding, segments=segments, group=group)
    if sharding == "all-to-sharded":
        cls = Fp8AllToShardedLinear
    else:
        cls = Fp8ShardedToAllLinear
    scale = layer.pop("output_scale")
    sharded = cls.from_quantized_linear(layer, segments=segments, group=group)
    # A per-row scale follows the output rows. A scalar scale applies to each
    # shard and to each partial sum, so every rank keeps all of it.
    if scale.ndim == 1:
        N, r = group.size(), group.rank()
        parts = mx.split(scale, segments)
        scale = mx.concatenate([mx.split(p, N)[r] for p in parts])
    sharded.output_scale = scale
    sharded.freeze()
    return sharded


class LanguageModel(PipelineMixin, nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.layer_types = args.layer_types
        self.sliding_window = args.sliding_window
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args, use_sliding=layer_type == "sliding_attention")
            for layer_type in self.layer_types
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.fa_idx = self.layer_types.index("full_attention")
        self.swa_idx = None
        for e, l in enumerate(self.layers):
            if l.use_sliding:
                self.swa_idx = e
                break

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * len(self.pipeline_layers)
            offset = 0
        else:
            offset = cache[0].offset

        swa_mask = fa_mask = None
        if self.fa_idx is not None:
            fa_mask = create_attention_mask(h, cache[self.fa_idx])
        if self.swa_idx is not None:
            swa_mask = create_attention_mask(
                h, cache[self.swa_idx], window_size=self.sliding_window
            )

        attn_scale = _get_llama_4_attn_scale(
            inputs.shape[1],
            offset,
            self.args.rope_parameters["llama_4_scaling_beta"],
            self.args.rope_parameters["original_max_position_embeddings"],
        ).astype(h.dtype)

        # Receive from the previous process in the pipeline
        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        for l, c in zip(self.pipeline_layers, cache):
            mask = swa_mask if l.use_sliding else fa_mask
            h = l(h, attn_scale, mask, cache=c)

        # Send to the next process in the pipeline
        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            if cache[-1] is not None:
                cache[-1].keys = mx.depends(cache[-1].keys, h)

        # Broadcast h while keeping it in the graph
        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        return self.norm(h)

    def pipeline(self, group, split=None):
        super().pipeline(group, split=split)
        self.fa_idx = None
        self.swa_idx = None
        for e, l in enumerate(self.pipeline_layers):
            if self.swa_idx is None and l.use_sliding:
                self.swa_idx = e
            elif self.fa_idx is None and not l.use_sliding:
                self.fa_idx = e
            if self.fa_idx is not None and self.swa_idx is not None:
                break


def shard_layers(layers, group: mx.distributed.Group):
    """Shard the decoder layers for tensor parallelism."""
    N = group.size()
    for layer in layers:
        # Shard the self attention
        attn = layer.self_attn
        if "qkv_proj" in attn:
            # Each rank takes its heads of q, k and v.
            q = attn.n_heads * attn.head_dim
            kv = attn.n_kv_heads * attn.head_dim
            attn.qkv_proj = _shard_linear(
                attn.qkv_proj, "all-to-sharded", group, segments=[q, q + kv]
            )
        else:
            attn.q_proj = _shard_linear(attn.q_proj, "all-to-sharded", group)
            attn.k_proj = _shard_linear(attn.k_proj, "all-to-sharded", group)
            attn.v_proj = _shard_linear(attn.v_proj, "all-to-sharded", group)
        attn.o_proj = _shard_linear(attn.o_proj, "sharded-to-all", group)
        attn.n_heads //= N
        attn.n_kv_heads //= N

        # Shard the MLP
        mlp = layer.mlp
        if isinstance(mlp, FusedGLU):
            mlp.gate_up_proj = _shard_linear(
                mlp.gate_up_proj, "all-to-sharded", group, segments=2
            )
        else:
            mlp.gate_proj = _shard_linear(mlp.gate_proj, "all-to-sharded", group)
            mlp.up_proj = _shard_linear(mlp.up_proj, "all-to-sharded", group)
        mlp.down_proj = _shard_linear(mlp.down_proj, "sharded-to-all", group)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LanguageModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.sharding_group = None

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
            if self.sharding_group is not None:
                out = gather_last_axis(out, self.sharding_group)
        return out

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()

        if not self.args.tie_word_embeddings:
            vocab = self.args.vocab_size
            assert vocab % N == 0, f"group size {N} must divide vocab_size {vocab}"
            self.lm_head = _shard_linear(self.lm_head, "all-to-sharded", group)
            self.sharding_group = group

        shard_layers(self.model.layers, group)

    @property
    def layers(self):
        return self.model.pipeline_layers

    def make_cache(self):
        return [
            (
                RotatingKVCache(max_size=self.model.sliding_window)
                if layer.use_sliding
                else KVCache()
            )
            for layer in self.layers
        ]
