# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear

from .base import BaseModelArgs, create_attention_mask
from .cache import CacheList, KVCache
from .deepseek_v32 import DeepseekV32Attention
from .glm4_moe_lite import Glm4MoeLiteMLP, Glm4MoeLiteMoE
from .pipeline import PipelineMixin


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "glm4_moe_dsa"
    vocab_size: int = 154880
    hidden_size: int = 2048
    intermediate_size: int = 10240
    moe_intermediate_size: int = 1536
    num_hidden_layers: int = 47
    num_attention_heads: int = 20
    num_key_value_heads: int = 20
    n_shared_experts: Optional[int] = 1
    n_routed_experts: Optional[int] = 64
    routed_scaling_factor: float = 1.8
    kv_lora_rank: int = 512
    q_lora_rank: int = 768
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 192
    v_head_dim: int = 256
    topk_method: str = "noaux_tc"
    scoring_func: str = "sigmoid"
    norm_topk_prob: bool = True
    n_group: int = 1
    topk_group: int = 1
    num_experts_per_tok: int = 4
    moe_layer_freq: int = 1
    first_k_dense_replace: int = 1
    max_position_embeddings: int = 202752
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1_000_000.0
    rope_scaling: Optional[Dict] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    partial_rotary_factor: float = 1.0
    tie_word_embeddings: bool = False
    num_nextn_predict_layers: int = 1
    index_head_dim: int = 128
    index_n_heads: int = 32
    index_topk: int = 2048
    quantization: Optional[Dict[str, Any]] = None


class Glm4MoeDSADecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = DeepseekV32Attention(config)
        use_moe = (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        )
        self.mlp = Glm4MoeLiteMoE(config) if use_moe else Glm4MoeLiteMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class Glm4MoeDSAModel(PipelineMixin, nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Glm4MoeDSADecoderLayer(config, idx)
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
        mask = create_attention_mask(
            h, cache[0][0] if cache[0] else None, return_array=True
        )

        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        for l, c in zip(self.pipeline_layers, cache):
            h = l(h, mask, cache=c)

        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            if cache[-1] is not None:
                cache[-1][0].keys = mx.depends(cache[-1][0].keys, h)

        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = Glm4MoeDSAModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        out = self.model(inputs, cache)
        return self.lm_head(out)

    def sanitize(self, weights):
        # Remove multi-token prediction layers
        mpt_layer = self.args.num_hidden_layers
        new_weights = {}
        for k, v in weights.items():
            parts = k.split(".")
            if len(parts) >= 3 and parts[1] == "layers" and int(parts[2]) >= mpt_layer:
                continue
            new_weights[k] = v
        weights = new_weights

        # Stack experts and absorb MLA weights
        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"
            for n, m in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]:
                for k in ["weight", "scales", "biases"]:
                    if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                            for e in range(self.args.n_routed_experts)
                        ]
                        weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)

            # MLA absorption: split kv_b_proj into embed_q and unembed_out
            attn_prefix = f"{prefix}.self_attn"
            if f"{attn_prefix}.kv_b_proj.weight" in weights:
                quantized = f"{attn_prefix}.kv_b_proj.scales" in weights
                v = weights.pop(f"{attn_prefix}.kv_b_proj.weight")
                head_dim = self.args.qk_nope_head_dim + self.args.v_head_dim

                if quantized:
                    dims = self.args.kv_lora_rank
                    scales = weights.pop(f"{attn_prefix}.kv_b_proj.scales")
                    biases = weights.pop(f"{attn_prefix}.kv_b_proj.biases")
                    bits = (v.shape[-1] * 32) // dims
                    group_size = dims // scales.shape[-1]
                    v = mx.dequantize(
                        v, scales, biases, bits=bits, group_size=group_size
                    )
                num_heads = self.args.num_attention_heads
                v = v.reshape(num_heads, head_dim, -1)
                wk = mx.contiguous(
                    v[:, : self.args.qk_nope_head_dim, :].swapaxes(-1, -2)
                )
                wv = mx.contiguous(v[:, self.args.qk_nope_head_dim :, :])
                if quantized:
                    wk, wk_scales, wk_biases = mx.quantize(
                        wk, bits=bits, group_size=group_size
                    )
                    wv, wv_scales, wv_biases = mx.quantize(
                        wv, bits=bits, group_size=group_size
                    )
                    weights[f"{attn_prefix}.embed_q.scales"] = wk_scales
                    weights[f"{attn_prefix}.unembed_out.scales"] = wv_scales
                    weights[f"{attn_prefix}.embed_q.biases"] = wk_biases
                    weights[f"{attn_prefix}.unembed_out.biases"] = wv_biases
                weights[f"{attn_prefix}.embed_q.weight"] = wk
                weights[f"{attn_prefix}.unembed_out.weight"] = wv

        return weights

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()
        rank = group.rank()
        for layer in self.model.layers:
            # Shard the self attention
            layer.self_attn.q_b_proj = shard_linear(
                layer.self_attn.q_b_proj, "all-to-sharded", group=group
            )
            layer.self_attn.o_proj = shard_linear(
                layer.self_attn.o_proj, "sharded-to-all", group=group
            )
            layer.self_attn.num_heads //= N
            num_heads = layer.self_attn.num_heads
            sh = rank * num_heads
            eh = sh + num_heads

            def shard_heads(w):
                return w[sh:eh]

            layer.self_attn.embed_q.apply(shard_heads)
            layer.self_attn.unembed_out.apply(shard_heads)

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
                shard_inplace(
                    layer.mlp.shared_experts.gate_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.shared_experts.down_proj, "sharded-to-all", group=group
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

    @property
    def layers(self):
        return self.model.pipeline_layers

    @property
    def cast_predicate(self):
        def predicate(k):
            return "e_score_correction_bias" not in k

        return predicate

    def make_cache(self):
        return [CacheList(KVCache(), KVCache()) for _ in self.layers]
