# Copyright © 2026 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear
from mlx.utils import tree_map

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.layers.common import ZeroCenteredRMSNorm
from mlx_lm.layers.distributed import PipelineMixin, gather_last_axis
from mlx_lm.layers.recurrent import create_ssm_mask
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.qwen3_5 import checkpoint
from mlx_lm.models.qwen3_5.config import ModelArgs, TextModelArgs
from mlx_lm.models.qwen3_5.layers import DecoderLayer
from mlx_lm.models.qwen3_next.layers import Qwen3NextMLP as MLP

__all__ = [
    "Model",
    "Qwen3_5Model",
    "Qwen3_5TextModel",
]


class Qwen3_5TextModel(PipelineMixin, nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args=args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]
        self.norm = ZeroCenteredRMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ssm_idx = 0
        self.fa_idx = args.full_attention_interval - 1

    def pipeline(self, group, split=None):
        super().pipeline(group, split=split)
        self.ssm_idx = None
        self.fa_idx = None
        for e, l in enumerate(self.pipeline_layers):
            if self.ssm_idx is None and l.is_linear:
                self.ssm_idx = e
            elif self.fa_idx is None and not l.is_linear:
                self.fa_idx = e
            if self.ssm_idx is not None and self.fa_idx is not None:
                break

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if input_embeddings is not None:
            hidden_states = input_embeddings
        else:
            hidden_states = self.embed_tokens(inputs)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * len(self.pipeline_layers)

        fa_mask = None
        ssm_mask = None
        if self.fa_idx is not None:
            fa_mask = create_attention_mask(hidden_states, cache[self.fa_idx])
        if self.ssm_idx is not None:
            ssm_mask = create_ssm_mask(hidden_states, cache[self.ssm_idx])

        # Receive from the previous process in the pipeline
        if pipeline_rank < pipeline_size - 1:
            hidden_states = mx.distributed.recv_like(hidden_states, (pipeline_rank + 1))

        for layer, c in zip(self.pipeline_layers, cache):
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden_states = layer(hidden_states, mask=mask, cache=c)

        # Send to the next process in the pipeline
        if pipeline_rank != 0:
            hidden_states = mx.distributed.send(
                hidden_states, (pipeline_rank - 1) % pipeline_size
            )
            if cache[-1] is not None:
                if hasattr(cache[-1], "keys"):
                    cache[-1].keys = mx.depends(cache[-1].keys, hidden_states)
                else:
                    cache[-1][0] = mx.depends(cache[-1][0], hidden_states)

        # Broadcast h while keeping it in the graph
        if pipeline_size > 1:
            hidden_states = mx.distributed.all_gather(hidden_states)[
                : hidden_states.shape[0]
            ]

        return self.norm(hidden_states)


class Qwen3_5Model(nn.Module):
    """The language model, named as in the checkpoint."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.language_model = Qwen3_5TextModel(args.text)

    # mlx_lm.utils pipelines model.model with these.
    @property
    def layers(self):
        return self.language_model.layers

    @property
    def start_idx(self):
        return self.language_model.start_idx

    @property
    def end_idx(self):
        return self.language_model.end_idx

    def pipeline(self, group, split=None):
        self.language_model.pipeline(group, split)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen3_5Model(args)
        if not args.text.tie_word_embeddings:
            self.lm_head = nn.Linear(
                args.text.hidden_size, args.text.vocab_size, bias=False
            )
        self.sharding_group = None

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        language_model = self.model.language_model
        out = language_model(inputs, cache, input_embeddings=input_embeddings)
        if self.args.text.tie_word_embeddings:
            return language_model.embed_tokens.as_linear(out)
        out = self.lm_head(out)
        if self.sharding_group is not None:
            out = gather_last_axis(out, self.sharding_group)
        return out

    @property
    def layers(self):
        return self.model.language_model.pipeline_layers

    def make_cache(self):
        return [ArraysCache(size=2) if l.is_linear else KVCache() for l in self.layers]

    def sanitize(self, weights):
        return checkpoint.sanitize(self, weights)

    def shard(self, group=None):
        group = group or mx.distributed.init()
        N = group.size()
        rank = group.rank()

        # Each rank computes a slice of the vocabulary
        if not self.args.text.tie_word_embeddings:
            vocab = self.args.text.vocab_size
            assert vocab % N == 0, f"group size {N} must divide vocab_size {vocab}"
            self.lm_head = shard_linear(self.lm_head, "all-to-sharded", group=group)
            self.sharding_group = group

        # A sharding factory for the convolution in gated delta net
        def conv_sharding(key_dim):
            return lambda p, w: (0, [key_dim, 2 * key_dim])

        def repeat_kv_layer_inplace(layer, h):
            # No repeat needed cause we have more heads than nodes
            if N <= h:
                return

            # Repeat function to apply to the layer weights
            def _repeat(p):
                s = p.shape
                p = p.reshape(h, s[0] // h, *s[1:])
                p = mx.repeat(p, N // h, axis=0)
                p = p.reshape(-1, *s[1:])
                return p

            layer.update(tree_map(_repeat, layer.parameters()))

        for layer in self.layers:
            # Linear attention
            if layer.is_linear:
                kd = layer.linear_attn.key_dim
                layer.linear_attn.sharding_group = group
                shard_inplace(layer.linear_attn.conv1d, conv_sharding(kd), group=group)
                layer.linear_attn.conv1d.groups //= N
                shard_inplace(
                    layer.linear_attn.in_proj_qkv,
                    "all-to-sharded",
                    segments=[kd, 2 * kd],
                    group=group,
                )
                shard_inplace(
                    layer.linear_attn.in_proj_z, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.linear_attn.in_proj_b, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.linear_attn.in_proj_a, "all-to-sharded", group=group
                )
                layer.linear_attn.dt_bias = mx.contiguous(
                    mx.split(layer.linear_attn.dt_bias, N)[rank]
                )
                layer.linear_attn.A_log = mx.contiguous(
                    mx.split(layer.linear_attn.A_log, N)[rank]
                )
                shard_inplace(layer.linear_attn.out_proj, "sharded-to-all", group=group)
                layer.linear_attn.num_k_heads //= N
                layer.linear_attn.num_v_heads //= N
                layer.linear_attn.key_dim //= N
                layer.linear_attn.value_dim //= N
                layer.linear_attn.conv_dim //= N

            # Softmax attention
            else:
                layer.self_attn.o_proj = shard_linear(
                    layer.self_attn.o_proj, "sharded-to-all", group=group
                )
                layer.self_attn.q_proj = shard_linear(
                    layer.self_attn.q_proj, "all-to-sharded", group=group
                )
                repeat_kv_layer_inplace(
                    layer.self_attn.k_proj, layer.self_attn.num_key_value_heads
                )
                repeat_kv_layer_inplace(
                    layer.self_attn.v_proj, layer.self_attn.num_key_value_heads
                )
                layer.self_attn.k_proj = shard_linear(
                    layer.self_attn.k_proj, "all-to-sharded", group=group
                )
                layer.self_attn.v_proj = shard_linear(
                    layer.self_attn.v_proj, "all-to-sharded", group=group
                )
                layer.self_attn.num_attention_heads //= N
                layer.self_attn.num_key_value_heads = max(
                    1, layer.self_attn.num_key_value_heads // N
                )

            # MLP
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

            # MoE
            else:
                layer.mlp.sharding_group = group
                shard_inplace(
                    layer.mlp.shared_expert.gate_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.shared_expert.down_proj, "sharded-to-all", group=group
                )
                shard_inplace(
                    layer.mlp.shared_expert.up_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.experts.gate_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.experts.down_proj, "sharded-to-all", group=group
                )
                shard_inplace(layer.mlp.experts.up_proj, "all-to-sharded", group=group)

    @property
    def quant_predicate(self):
        if self.args.text.num_experts <= 0:
            return None

        def predicate(path, _):
            if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def cast_predicate(self):
        def predicate(path: str):
            if path.endswith("A_log"):
                return False
            return True

        return predicate
