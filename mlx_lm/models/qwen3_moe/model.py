# Copyright © 2026 Apple Inc.

from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_linear

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.layers.distributed import PipelineMixin
from mlx_lm.layers.moe import shard_switch_linear
from mlx_lm.models.qwen3_moe import checkpoint
from mlx_lm.models.qwen3_moe.config import ModelArgs
from mlx_lm.models.qwen3_moe.layers import MLP, Qwen3MoeDecoderLayer

__all__ = [
    "Model",
    "Qwen3MoeModel",
    "moe_quant_predicate",
    "shard_layers",
]


class Qwen3MoeModel(PipelineMixin, nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Qwen3MoeDecoderLayer(args=args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * len(self.pipeline_layers)

        mask = create_attention_mask(h, cache[0])

        # Receive from the previous process in the pipeline
        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        for layer, c in zip(self.pipeline_layers, cache):
            h = layer(h, mask, c)

        # Send to the next process in the pipeline
        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            if cache[-1] is not None:
                cache[-1].keys = mx.depends(cache[-1].keys, h)

        # Broadcast h while keeping it in the graph
        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        return self.norm(h)


def moe_quant_predicate(path, _):
    """Keep the routers in 8 bits."""
    if path.endswith("mlp.gate"):
        return {"group_size": 64, "bits": 8}
    return True


def shard_layers(layers, group: mx.distributed.Group):
    """Shard the decoder layers for tensor parallelism."""
    N = group.size()
    for layer in layers:
        # Shard the self attention
        layer.self_attn.q_proj = shard_linear(
            layer.self_attn.q_proj, "all-to-sharded", group=group
        )
        layer.self_attn.k_proj = shard_linear(
            layer.self_attn.k_proj, "all-to-sharded", group=group
        )
        layer.self_attn.v_proj = shard_linear(
            layer.self_attn.v_proj, "all-to-sharded", group=group
        )
        layer.self_attn.o_proj = shard_linear(
            layer.self_attn.o_proj, "sharded-to-all", group=group
        )
        layer.self_attn.n_heads //= N
        layer.self_attn.n_kv_heads //= N

        # Shard the dense MLP layers
        if isinstance(layer.mlp, MLP):
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
        # for aggregating the results. The router (gate) stays replicated.
        else:
            layer.mlp.sharding_group = group
            shard_switch_linear(
                layer.mlp.switch_mlp.gate_proj, "all-to-sharded", group=group
            )
            shard_switch_linear(
                layer.mlp.switch_mlp.down_proj, "sharded-to-all", group=group
            )
            shard_switch_linear(
                layer.mlp.switch_mlp.up_proj, "all-to-sharded", group=group
            )


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen3MoeModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    @property
    def quant_predicate(self):
        return moe_quant_predicate

    def shard(self, group: Optional[mx.distributed.Group] = None):
        shard_layers(self.model.layers, group or mx.distributed.init())

    @property
    def layers(self):
        return self.model.pipeline_layers
