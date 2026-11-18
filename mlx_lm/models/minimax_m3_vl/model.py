# Copyright © 2025 Apple Inc.

from typing import Any, Callable, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.layers.common import ZeroCenteredRMSNorm
from mlx_lm.layers.distributed import PipelineMixin
from mlx_lm.models.cache import CacheList, KVCache
from mlx_lm.models.minimax_m3_vl import checkpoint
from mlx_lm.models.minimax_m3_vl.config import ModelArgs, TextArgs
from mlx_lm.models.minimax_m3_vl.layers import DecoderLayer

__all__ = [
    "MiniMaxM3VLModel",
    "Model",
    "TextModel",
]


class TextModel(PipelineMixin, nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        NormClass = ZeroCenteredRMSNorm if args.use_gemma_norm else nn.RMSNorm
        self.norm = NormClass(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * len(self.pipeline_layers)
        mask = create_attention_mask(h, cache[0][0] if cache[0] else None)

        # Receive from the previous process in the pipeline
        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        for layer, c in zip(self.pipeline_layers, cache):
            h = layer(h, mask, c)

        # Send to the next process in the pipeline
        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            if cache[-1] is not None:
                cache[-1][0].keys = mx.depends(cache[-1][0].keys, h)

        # Broadcast h while keeping it in the graph
        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        return self.norm(h)


class MiniMaxM3VLModel(nn.Module):
    """Puts the language model at model.language_model, as the checkpoint names it."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.language_model = TextModel(args.text)

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
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = MiniMaxM3VLModel(config)
        if not config.text.tie_word_embeddings:
            self.lm_head = nn.Linear(
                config.text.hidden_size, config.text.vocab_size, bias=False
            )

    def __call__(self, inputs: mx.array, cache: Optional[Any] = None) -> mx.array:
        language_model = self.model.language_model
        out = language_model(inputs, cache=cache)
        if self.args.text.tie_word_embeddings:
            return language_model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        return checkpoint.sanitize(self, weights)

    @property
    def cast_predicate(self) -> Callable[[str], bool]:
        def predicate(k):
            # Keep the router in float32 as the original
            return (
                "e_score_correction_bias" not in k and "block_sparse_moe.gate." not in k
            )

        return predicate

    @property
    def quant_predicate(self) -> Callable[[str, nn.Module], Any]:
        def predicate(path, _):
            if path.endswith("block_sparse_moe.gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()
        # pipeline() blanks the layers it does not hold, so take the live ones.
        for layer in self.layers:
            attn = layer.self_attn
            # Check before the first mutation, or a failure leaves the layer
            # half sharded.
            heads = [attn.num_attention_heads, attn.num_key_value_heads]
            if attn.is_sparse_attn:
                heads.append(attn.index_n_heads)
            assert all(h % N == 0 for h in heads), (
                f"head counts {heads} are not all divisible by {N} ranks; "
                "is the model already sharded?"
            )
            attn.q_proj = shard_linear(attn.q_proj, "all-to-sharded", group=group)
            attn.k_proj = shard_linear(attn.k_proj, "all-to-sharded", group=group)
            attn.v_proj = shard_linear(attn.v_proj, "all-to-sharded", group=group)
            attn.o_proj = shard_linear(attn.o_proj, "sharded-to-all", group=group)
            attn.num_attention_heads //= N
            attn.num_key_value_heads //= N
            if attn.is_sparse_attn:
                # Each index head serves one KV group, so it follows the query heads.
                attn.index_q_proj = shard_linear(
                    attn.index_q_proj, "all-to-sharded", group=group
                )
                attn.index_n_heads //= N

            if layer.is_moe:
                moe = layer.block_sparse_moe
                # In place, so the block aggregates both experts with one sum.
                for mlp in (moe.switch_mlp, moe.shared_experts):
                    shard_inplace(mlp.gate_proj, "all-to-sharded", group=group)
                    shard_inplace(mlp.up_proj, "all-to-sharded", group=group)
                    shard_inplace(mlp.down_proj, "sharded-to-all", group=group)
                moe.sharding_group = group
            else:
                mlp = layer.mlp
                mlp.gate_proj = shard_linear(
                    mlp.gate_proj, "all-to-sharded", group=group
                )
                mlp.up_proj = shard_linear(mlp.up_proj, "all-to-sharded", group=group)
                mlp.down_proj = shard_linear(
                    mlp.down_proj, "sharded-to-all", group=group
                )

    @property
    def layers(self) -> List[nn.Module]:
        return self.model.language_model.pipeline_layers

    def make_cache(self) -> List[Any]:
        # Only sparse layers have indexer keys; an unused KVCache breaks .state.
        return [
            (
                CacheList(KVCache(), KVCache())
                if layer.self_attn.is_sparse_attn
                else CacheList(KVCache())
            )
            for layer in self.layers
        ]
