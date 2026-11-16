# Copyright © 2026 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear

from mlx_lm.layers.masks import create_attention_mask
from mlx_lm.layers.pipeline import PipelineMixin
from mlx_lm.models.deepseek_v3.layers import DeepseekV3MLP
from mlx_lm.models.deepseek_v3.model import DeepseekV3Model
from mlx_lm.models.ministral3.layers import _get_llama_4_attn_scale
from mlx_lm.models.mistral4 import checkpoint
from mlx_lm.models.mistral4.config import ModelArgs
from mlx_lm.models.mistral4.layers import Mistral4DecoderLayer

__all__ = [
    "Mistral4Model",
    "Model",
]


class Mistral4Model(DeepseekV3Model, PipelineMixin, nn.Module):
    def __init__(self, args: ModelArgs):
        PipelineMixin.__init__(self)
        self.args = args
        self.vocab_size = args.vocab_size
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Mistral4DecoderLayer(args, idx) for idx in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(x)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * len(self.pipeline_layers)

        offset = cache[0].offset if cache[0] is not None else 0
        # "causal" suits prefill; at L == 1 it is None, which decode wants.
        mask = create_attention_mask(h, cache[0])

        attn_scale = _get_llama_4_attn_scale(
            h.shape[1],
            offset,
            self.args.rope_parameters["llama_4_scaling_beta"],
            self.args.rope_parameters["original_max_position_embeddings"],
        ).astype(h.dtype)

        # Receive from the previous process in the pipeline
        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        for l, c in zip(self.pipeline_layers, cache):
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


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Mistral4Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        out = self.model(inputs, cache=cache, input_embeddings=input_embeddings)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    @property
    def layers(self):
        return self.model.pipeline_layers

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()

        for layer in self.model.layers:
            if layer.self_attn.q_lora_rank is None:
                layer.self_attn.q_proj = shard_linear(
                    layer.self_attn.q_proj, "all-to-sharded", group=group
                )
            else:
                layer.self_attn.q_b_proj = shard_linear(
                    layer.self_attn.q_b_proj, "all-to-sharded", group=group
                )

            layer.self_attn.num_heads //= N
            num_heads = layer.self_attn.num_heads
            sh = group.rank() * num_heads
            eh = sh + num_heads

            def shard_heads(w):
                return w[sh:eh]

            layer.self_attn.embed_q.apply(shard_heads)
            layer.self_attn.unembed_out.apply(shard_heads)

            layer.self_attn.o_proj = shard_linear(
                layer.self_attn.o_proj, "sharded-to-all", group=group
            )

            if isinstance(layer.mlp, DeepseekV3MLP):
                layer.mlp.gate_proj = shard_linear(
                    layer.mlp.gate_proj, "all-to-sharded", group=group
                )
                layer.mlp.down_proj = shard_linear(
                    layer.mlp.down_proj, "sharded-to-all", group=group
                )
                layer.mlp.up_proj = shard_linear(
                    layer.mlp.up_proj, "all-to-sharded", group=group
                )

            else:
                # Shard in place: the MoE aggregates the partial sums itself.
                layer.mlp.sharding_group = group
                if hasattr(layer.mlp, "shared_experts"):
                    shard_inplace(
                        layer.mlp.shared_experts.gate_proj,
                        "all-to-sharded",
                        group=group,
                    )
                    shard_inplace(
                        layer.mlp.shared_experts.down_proj,
                        "sharded-to-all",
                        group=group,
                    )
                    shard_inplace(
                        layer.mlp.shared_experts.up_proj, "all-to-sharded", group=group
                    )
                shard_inplace(
                    layer.mlp.switch_mlp.gate_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.down_proj, "sharded-to-all", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.up_proj, "all-to-sharded", group=group
                )
