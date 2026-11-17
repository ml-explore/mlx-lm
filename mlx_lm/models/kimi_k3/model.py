# Copyright © 2026 Apple Inc.

from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear

from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.layers.recurrent import create_ssm_mask
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.kimi_k3 import checkpoint
from mlx_lm.models.kimi_k3.config import ModelArgs, TextArgs
from mlx_lm.models.kimi_k3.layers import (
    KimiK3DecoderLayer,
    KimiK3SparseMoE,
    ResidualBlocks,
    _attn_res_mix,
)

__all__ = [
    "KimiK3Model",
    "KimiK3TextModel",
    "Model",
]


class KimiK3TextModel(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            KimiK3DecoderLayer(args, i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.use_attn_res = args.attn_res_block_size is not None
        if self.use_attn_res:
            self.output_attn_res_proj = nn.Linear(args.hidden_size, 1, bias=False)
            self.output_attn_res_norm = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self._output_res_w_eff = None

        self.pipeline_rank = 0
        self.pipeline_size = 1
        self.start_idx = 0
        self.end_idx = len(self.layers)
        self.num_layers = len(self.layers)
        self.in_blocks = 0
        self._set_cache_indices()

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        boundary_dtype = h.dtype
        if cache is None:
            cache = [None] * self.num_layers

        ssm_mask = (
            create_ssm_mask(h, cache[self.ssm_idx])
            if self.ssm_idx is not None
            else None
        )
        if self.attn_idx is not None:
            attn_mask = create_attention_mask(
                h, cache[self.attn_idx], return_array=True
            )
        else:
            attn_mask = None

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        blocks = ResidualBlocks(self.args.rms_norm_eps) if self.use_attn_res else None

        if pipeline_rank < pipeline_size - 1:
            src = pipeline_rank + 1
            if blocks is not None:
                packed = mx.distributed.recv(
                    (self.in_blocks + 1, *h.shape), h.dtype, src
                )
                h = packed[-1]
                blocks.raw = packed[:-1]
                xf = blocks.raw.astype(mx.float32)
                blocks.inv_rms = mx.rsqrt(
                    (xf * xf).mean(axis=-1) + self.args.rms_norm_eps
                )
            else:
                h = mx.distributed.recv_like(h, src)

        for i in range(self.num_layers):
            layer = self.layers[self.start_idx + i]
            mask = ssm_mask if layer.is_linear else attn_mask
            h, blocks = layer(h, mask=mask, cache=cache[i], blocks=blocks)

        if pipeline_rank != 0:
            dst = pipeline_rank - 1
            if blocks is not None:
                packed = mx.concatenate([blocks.raw, h[None]]).astype(boundary_dtype)
                packed = mx.distributed.send(packed, dst)
                h = packed[-1]
            else:
                h = mx.distributed.send(h.astype(boundary_dtype), dst)
            if cache[-1] is not None:
                if hasattr(cache[-1], "keys"):
                    cache[-1].keys = mx.depends(cache[-1].keys, h)
                else:
                    cache[-1][0] = mx.depends(cache[-1][0], h)
        elif blocks is not None:
            if self.training or self._output_res_w_eff is None:
                self._output_res_w_eff = self.output_attn_res_norm.weight.astype(
                    mx.float32
                ) * self.output_attn_res_proj.weight.reshape(-1)
            h = _attn_res_mix(
                blocks,
                h,
                self._output_res_w_eff,
                self.args.rms_norm_eps,
                not self.training,
            )

        if pipeline_size > 1:
            h = mx.distributed.all_gather(h.astype(boundary_dtype))[: h.shape[0]]

        return self.norm(h)

    def _set_cache_indices(self):
        self.ssm_idx = None
        self.attn_idx = None
        for i, layer in enumerate(self.layers[self.start_idx : self.end_idx]):
            if layer.is_linear:
                if self.ssm_idx is None:
                    self.ssm_idx = i
            elif self.attn_idx is None:
                self.attn_idx = i
            if self.ssm_idx is not None and self.attn_idx is not None:
                break

    def pipeline(self, group):
        self.pipeline_rank = group.rank()
        self.pipeline_size = group.size()
        base, extra = divmod(len(self.layers), self.pipeline_size)
        seg = self.pipeline_size - self.pipeline_rank - 1
        self.start_idx = seg * base + min(seg, extra)
        self.end_idx = self.start_idx + base + (1 if seg < extra else 0)
        self.num_layers = self.end_idx - self.start_idx
        self.layers = self.layers[: self.end_idx]
        self.layers[: self.start_idx] = [None] * self.start_idx
        if self.use_attn_res:
            block_size = self.args.attn_res_block_size
            self.in_blocks = (self.start_idx + block_size - 1) // block_size
        self._set_cache_indices()


class KimiK3Model(nn.Module):
    """The language model, named as in the checkpoint."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.language_model = KimiK3TextModel(args.text)

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

    def pipeline(self, group):
        self.language_model.pipeline(group)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = KimiK3Model(args)
        if not args.text.tie_word_embeddings:
            self.lm_head = nn.Linear(
                args.text.hidden_size, args.text.vocab_size, bias=False
            )

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        language_model = self.model.language_model
        out = language_model(inputs, cache)
        if self.args.text.tie_word_embeddings:
            return language_model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        return checkpoint.sanitize(self, weights)

    @property
    def cast_predicate(self):
        def predicate(path: str):
            if "e_score_correction_bias" in path:
                return False
            if path.endswith("A_log") or path.endswith("dt_bias"):
                return False
            return True

        return predicate

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("mlp.gate"):
                return {"group_size": 64, "bits": 8}
            if path.endswith("res_proj"):
                return False
            return True

        return predicate

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()
        if N == 1:
            return
        rank = group.rank()

        for layer in self.layers:
            attn = layer.self_attn

            if layer.is_linear:
                D = attn.head_dim
                P = attn.projection_dim
                num_heads = attn.num_heads // N
                sh = rank * num_heads
                eh = sh + num_heads

                attn.qkv_proj = shard_linear(
                    attn.qkv_proj, "all-to-sharded", segments=3, group=group
                )
                attn.f_b_proj = shard_linear(
                    attn.f_b_proj, "all-to-sharded", group=group
                )
                if attn.use_full_rank_gate:
                    attn.g_proj = shard_linear(
                        attn.g_proj, "all-to-sharded", group=group
                    )
                else:
                    attn.g_b_proj = shard_linear(
                        attn.g_b_proj, "all-to-sharded", group=group
                    )
                attn.b_proj = shard_linear(attn.b_proj, "all-to-sharded", group=group)
                attn.o_proj = shard_linear(attn.o_proj, "sharded-to-all", group=group)

                w = attn.qkv_conv.conv.weight
                attn.qkv_conv.conv.weight = mx.concatenate(
                    [w[seg * P + sh * D : seg * P + eh * D] for seg in range(3)],
                    axis=0,
                )
                attn.qkv_conv.conv.groups = 3 * num_heads * D

                attn.A_log = attn.A_log.reshape(-1)[sh:eh]
                attn.dt_bias = attn.dt_bias.reshape(-1)[sh * D : eh * D]
                attn.num_heads = num_heads
                attn.projection_dim = num_heads * D
            else:
                if attn.q_lora_rank is not None:
                    attn.q_b_proj = shard_linear(
                        attn.q_b_proj, "all-to-sharded", group=group
                    )
                else:
                    attn.q_proj = shard_linear(
                        attn.q_proj, "all-to-sharded", group=group
                    )
                if attn.use_gate:
                    attn.g_proj = shard_linear(
                        attn.g_proj, "all-to-sharded", group=group
                    )
                attn.o_proj = shard_linear(attn.o_proj, "sharded-to-all", group=group)

                attn.num_heads //= N
                num_heads = attn.num_heads
                sh = rank * num_heads
                eh = sh + num_heads

                def shard_heads(w):
                    return w[sh:eh]

                attn.embed_q.apply(shard_heads)
                attn.unembed_out.apply(shard_heads)

            if isinstance(layer.mlp, KimiK3SparseMoE):
                layer.mlp.sharding_group = group
                shard_inplace(
                    layer.mlp.switch_mlp.gate_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.up_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.down_proj, "sharded-to-all", group=group
                )
                if layer.mlp.shared_experts is not None:
                    shard_inplace(
                        layer.mlp.shared_experts.gate_proj,
                        "all-to-sharded",
                        group=group,
                    )
                    shard_inplace(
                        layer.mlp.shared_experts.up_proj,
                        "all-to-sharded",
                        group=group,
                    )
                    shard_inplace(
                        layer.mlp.shared_experts.down_proj,
                        "sharded-to-all",
                        group=group,
                    )
            else:
                layer.mlp.gate_proj = shard_linear(
                    layer.mlp.gate_proj, "all-to-sharded", group=group
                )
                layer.mlp.up_proj = shard_linear(
                    layer.mlp.up_proj, "all-to-sharded", group=group
                )
                layer.mlp.down_proj = shard_linear(
                    layer.mlp.down_proj, "sharded-to-all", group=group
                )

    @property
    def layers(self):
        language_model = self.model.language_model
        return language_model.layers[language_model.start_idx : language_model.end_idx]

    def make_cache(self):
        caches: List[Any] = []
        for layer in self.layers:
            if layer.is_linear:
                caches.append(ArraysCache(size=2))
            else:
                caches.append(KVCache())
        return caches
