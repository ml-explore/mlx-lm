# Copyright © 2026 Apple Inc.

from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.layers.distributed import PipelineMixin
from mlx_lm.models.cache import CacheList, KVCache
from mlx_lm.models.mistral_large4 import checkpoint
from mlx_lm.models.mistral_large4.config import ModelArgs, TextArgs
from mlx_lm.models.mistral_large4.layers import DecoderLayer, LeRoPE, MoE

__all__ = [
    "MistralLarge4TextModel",
    "Model",
]


def _log_positions(offset, length):
    if isinstance(offset, mx.array) and offset.ndim > 0:
        offset = offset[:, None]
    # Left padding of a batch has negative positions.
    positions = mx.maximum(mx.arange(length) + offset, 0)
    return mx.log(positions.astype(mx.float32) + 1)


class MistralLarge4TextModel(PipelineMixin, nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.rope = LeRoPE(args.qk_rope_head_dim, args.rope_parameters)
        # KV source layers whose latent rows cross into or out of this stage.
        self.recv_sources = []
        self.send_sources = []

    def pipeline(self, group, split=None):
        if split is None:
            split = self._stage_sizes(group.size())[::-1]
        super().pipeline(group, split)
        start, end = self.start_idx, self.end_idx
        if self.args.indexer_types[start] != "full":
            raise ValueError(
                f"A pipeline stage cannot start at layer {start}, "
                "which reuses the top-k of the layer before it."
            )
        sources = self.args.kv_source_layers
        self.recv_sources = sorted({s for c, s in enumerate(sources) if s < start <= c})
        self.send_sources = sorted({s for c, s in enumerate(sources) if s < end <= c})

    def _stage_sizes(self, n):
        # Even stages, each moved to start at a layer that runs its own indexer.
        num_layers, kinds = len(self.layers), self.args.indexer_types
        bounds = [0]
        for i in range(1, n):
            b = max(round(i * num_layers / n), bounds[-1] + 1)
            while b < num_layers and kinds[b] != "full":
                b += 1
            bounds.append(b)
        bounds.append(num_layers)
        return [e - s for s, e in zip(bounds, bounds[1:])]

    def _latent_width(self, layer):
        rope = self.args.qk_rope_head_dim if self.args.no_rope_layers[layer] else 0
        return self.args.kv_lora_rank + rope

    def __call__(self, inputs, cache=None, input_embeddings=None):
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)
        layers = self.pipeline_layers
        if cache is None:
            cache = [None] * len(layers)
        # Every cache of a layer has the same offset.
        first = cache[0][0] if cache[0] is not None else None
        offset = first.offset if first is not None else 0
        mask = create_attention_mask(h, first, return_array=True)

        dims, freqs = self.rope.dims, self.rope.freqs()

        def rope(x):
            return mx.fast.rope(
                x,
                dims,
                traditional=True,
                base=None,
                scale=1.0,
                offset=offset,
                freqs=freqs,
            )

        B, L, D = h.shape
        log_pos = _log_positions(offset, L)
        rank, size = self.pipeline_rank, self.pipeline_size
        shared_kv = {}

        # Receive the hidden states, and the new latent rows of KV source layers
        # that run on an earlier stage. A replica cache holds those rows.
        if rank < size - 1:
            widths = [self._latent_width(s) for s in self.recv_sources]
            x = mx.distributed.recv_like(
                mx.zeros((B, L, D + sum(widths)), h.dtype), rank + 1
            )
            cuts = [D + sum(widths[:i]) for i in range(len(widths))]
            h, *rows = mx.split(x, cuts, axis=-1)
            replicas = []
            if cache[0] is not None and rows:
                replicas = cache[0].caches[-len(rows) :]
            for i, (s, r) in enumerate(zip(self.recv_sources, rows)):
                latent, pe = mx.split(r[:, None], [self.args.kv_lora_rank], axis=-1)
                if replicas:
                    latent, pe = replicas[i].update_and_fetch(latent, pe)
                shared_kv[s] = (latent, pe)

        topk = None
        for layer, c in zip(layers, cache):
            h, topk = layer(h, mask, rope, log_pos, c, shared_kv, topk)

        if rank > 0:
            # The new rows are the last L rows of the full latent.
            rows = [
                mx.concatenate(shared_kv[s], axis=-1)[:, 0, -L:]
                for s in self.send_sources
            ]
            sent = mx.distributed.send(mx.concatenate([h, *rows], axis=-1), rank - 1)
            h = mx.depends(h, [sent])
            if cache[-1] is not None:
                cache[-1][0].keys = mx.depends(cache[-1][0].keys, [sent])

        # Every rank needs the output of rank 0, which runs the last layers.
        if size > 1:
            h = mx.distributed.all_gather(h)[:B]
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = MistralLarge4TextModel(args.text)
        self.lm_head = nn.Linear(
            args.text.hidden_size, args.text.vocab_size, bias=False
        )

    def __call__(self, inputs, cache=None, input_embeddings=None):
        out = self.lm_head(self.model(inputs, cache, input_embeddings))
        if cap := self.args.text.final_logit_softcapping:
            out = mx.tanh(out / cap) * cap
        return out

    @property
    def layers(self):
        return self.model.pipeline_layers

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()
        for layer in self.model.layers:
            # Heads are sharded. The latent cache and the indexer are replicated,
            # so every rank selects the same top-k.
            attn = layer.self_attn
            attn.q_proj = shard_linear(attn.q_proj, "all-to-sharded", group=group)
            attn.o_proj = shard_linear(attn.o_proj, "sharded-to-all", group=group)
            attn.num_heads //= N
            sh = group.rank() * attn.num_heads
            eh = sh + attn.num_heads

            def shard_heads(w):
                return w[sh:eh]

            attn.embed_q.apply(shard_heads)
            attn.unembed_out.apply(shard_heads)
            attn.softmax_scaling.apply(shard_heads)

            mlp = layer.mlp
            if isinstance(mlp, MoE):
                # Shard in place: the MoE aggregates the partial sums itself.
                mlp.sharding_group = group
                for m in (mlp.shared_experts, mlp.switch_mlp):
                    shard_inplace(m.gate_proj, "all-to-sharded", group=group)
                    shard_inplace(m.up_proj, "all-to-sharded", group=group)
                    shard_inplace(m.down_proj, "sharded-to-all", group=group)
            else:
                mlp.gate_proj = shard_linear(
                    mlp.gate_proj, "all-to-sharded", group=group
                )
                mlp.up_proj = shard_linear(mlp.up_proj, "all-to-sharded", group=group)
                mlp.down_proj = shard_linear(
                    mlp.down_proj, "sharded-to-all", group=group
                )

    def make_cache(self):
        caches = []
        for layer in self.layers:
            attn = layer.self_attn
            parts = []
            if attn.owns_kv:
                parts.append(KVCache())
            if attn.indexer is not None:
                parts.append(KVCache())
            caches.append(CacheList(*parts))
        # The first layer of a pipeline stage also holds the replica caches of
        # KV source layers that run on an earlier stage.
        if self.model.recv_sources:
            replicas = [KVCache() for _ in self.model.recv_sources]
            caches[0] = CacheList(*caches[0].caches, *replicas)
        return caches

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    @property
    def quant_predicate(self):
        def predicate(path, _):
            # The checkpoint keeps these unquantized, so they stay in bf16.
            return not path.endswith(("mlp.gate", "indexer.wk", "indexer.weights_proj"))

        return predicate
