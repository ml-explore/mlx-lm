# Copyright © 2026 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.layers.distributed import PipelineMixin
from mlx_lm.models.glm4_moe_lite import checkpoint
from mlx_lm.models.glm4_moe_lite.config import ModelArgs
from mlx_lm.models.glm4_moe_lite.layers import Glm4MoeLiteDecoderLayer, Glm4MoeLiteMLP

__all__ = [
    "Glm4MoeLiteModel",
    "Model",
]


class Glm4MoeLiteModel(PipelineMixin, nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Glm4MoeLiteDecoderLayer(config, idx)
            for idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.embed_tokens(x)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * len(self.pipeline_layers)
        mask = create_attention_mask(h, cache[0], return_array=True)

        # Receive from the previous process in the pipeline
        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        for l, c in zip(self.pipeline_layers, cache):
            h = l(h, mask, cache=c)

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
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = Glm4MoeLiteModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ):
        out = self.model(inputs, cache)
        return self.lm_head(out)

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        rank = group.rank()
        N = group.size()
        for layer in self.model.layers:
            # Shard the self attention
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
            sh = rank * num_heads
            eh = sh + num_heads

            def shard_heads(w):
                return w[sh:eh]

            layer.self_attn.embed_q.apply(shard_heads)
            layer.self_attn.unembed_out.apply(shard_heads)

            layer.self_attn.o_proj = shard_linear(
                layer.self_attn.o_proj, "sharded-to-all", group=group
            )

            # Shard the MLP
            if isinstance(layer.mlp, Glm4MoeLiteMLP):
                layer.mlp.gate_proj = shard_linear(
                    layer.mlp.gate_proj, "all-to-sharded", group=group
                )
                layer.mlp.down_proj = shard_linear(
                    layer.mlp.down_proj, "sharded-to-all", group=group
                )
                layer.mlp.up_proj = shard_linear(
                    layer.mlp.up_proj, "all-to-sharded", group=group
                )

            # Shard the MoE. Shard in place since the MoE should be responsible
            # for aggregating the results.
            else:
                layer.mlp.sharding_group = group
                if getattr(layer.mlp, "shared_experts", None) is not None:
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
                        layer.mlp.shared_experts.up_proj,
                        "all-to-sharded",
                        group=group,
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

    @property
    def layers(self):
        return self.model.pipeline_layers

    @property
    def cast_predicate(self):
        def predicate(k):
            return "e_score_correction_bias" not in k

        return predicate
