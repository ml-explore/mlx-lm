# Copyright © 2026 Apple Inc.

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import CacheList, KVCache
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    ffn_hidden_size: int
    moe_topk: int
    expert_ffn_hidden_size: int
    n_routed_experts: int
    zero_expert_num: int
    num_layers: int
    vocab_size: int
    max_position_embeddings: int
    num_attention_heads: int
    kv_lora_rank: int
    q_lora_rank: int
    qk_rope_head_dim: int
    qk_nope_head_dim: int
    v_head_dim: int
    routed_scaling_factor: float
    rms_norm_eps: float
    rope_theta: float
    mla_scale_q_lora: bool
    mla_scale_kv_lora: bool
    attention_bias: bool = False
    ngram_vocab_size_ratio: int = 78
    emb_neighbor_num: int = 4
    emb_split_num: int = 4
    norm_topk_prob: bool = False
    router_bias: bool = False
    rope_scaling: Optional[Dict] = None


class NgramCache:
    def __init__(self, max_context_len: int):
        self.ngram_context = None
        self.max_context_len = max_context_len

    def update(self, new_tokens: mx.array) -> None:
        if self.ngram_context is None:
            self.ngram_context = new_tokens
        else:
            self.ngram_context = mx.concatenate(
                [self.ngram_context, new_tokens], axis=-1
            )

        if self.ngram_context.shape[-1] > self.max_context_len:
            self.ngram_context = self.ngram_context[..., -self.max_context_len :]


class NgramEmbedding(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.hidden_size = args.hidden_size

        self.m = args.ngram_vocab_size_ratio * args.vocab_size
        self.k = args.emb_split_num
        self.n = args.emb_neighbor_num

        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)

        num_embedders = self.k * (self.n - 1)
        emb_dim = args.hidden_size // num_embedders

        self.embedders = []
        self.post_projs = []

        for i in range(num_embedders):
            emb_vocab_size = int(self.m + i * 2 + 1)
            self.embedders.append(nn.Embedding(emb_vocab_size, emb_dim))
            self.post_projs.append(nn.Linear(emb_dim, args.hidden_size, bias=False))

        self._vocab_mods_cache: Optional[Dict[Tuple[int, int], List[int]]] = None

    def _precompute_vocab_mods(self) -> Dict[Tuple[int, int], List[int]]:
        if self._vocab_mods_cache is not None:
            return self._vocab_mods_cache

        vocab_mods = {}
        for i in range(2, self.n + 1):
            for j in range(self.k):
                index = (i - 2) * self.k + j
                emb_vocab_dim = int(self.m + index * 2 + 1)

                mods = []
                power_mod = 1
                for _ in range(i - 1):
                    power_mod = (power_mod * self.vocab_size) % emb_vocab_dim
                    mods.append(power_mod)

                vocab_mods[(i, j)] = mods

        self._vocab_mods_cache = vocab_mods
        return vocab_mods

    def _shift_right(self, tensor: mx.array, n: int) -> mx.array:
        if n <= 0:
            return tensor

        batch_size, seq_len = tensor.shape
        if seq_len <= n:
            return mx.zeros_like(tensor)

        return mx.concatenate(
            [mx.zeros((batch_size, n), dtype=tensor.dtype), tensor[..., :-n]], axis=-1
        )

    def _get_ngram_ids(
        self,
        input_ids: mx.array,
        shifted_ids: Dict[int, mx.array],
        vocab_mods: List[int],
        ngram: int,
    ) -> mx.array:
        ngram_ids = input_ids
        for k in range(2, ngram + 1):
            ngram_ids = ngram_ids + shifted_ids[k] * vocab_mods[k - 2]
        return ngram_ids

    def __call__(
        self,
        input_ids: mx.array,
        ngram_context: Optional[mx.array] = None,
    ) -> mx.array:
        seq_len = input_ids.shape[-1]

        if ngram_context is not None:
            context = mx.concatenate(
                [ngram_context[..., -(self.n - 1) :], input_ids], axis=-1
            )
        else:
            context = input_ids

        x = self.word_embeddings(input_ids)
        vocab_mods = self._precompute_vocab_mods()

        shifted_ids = {}
        for i in range(2, self.n + 1):
            shifted_ids[i] = self._shift_right(context, i - 1)

        for i in range(2, self.n + 1):
            for j in range(self.k):
                index = (i - 2) * self.k + j
                emb_vocab_dim = int(self.m + index * 2 + 1)

                ngram_ids = self._get_ngram_ids(
                    context, shifted_ids, vocab_mods[(i, j)], ngram=i
                )
                new_ids = (ngram_ids % emb_vocab_dim)[..., -seq_len:]

                x_ngram = self.embedders[index](new_ids)
                x_proj = self.post_projs[index](x_ngram)
                x = x + x_proj

        return x / (1 + self.k * (self.n - 1))


class LongcatFlashLiteMLA(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.kv_lora_rank = args.kv_lora_rank
        self.q_lora_rank = args.q_lora_rank
        self.v_head_dim = args.v_head_dim

        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.scale = self.qk_head_dim**-0.5

        self.mla_scale_q_lora = None
        self.mla_scale_kv_lora = None

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                args.hidden_size,
                self.num_attention_heads * self.qk_head_dim,
                bias=False,
            )
        else:
            self.q_a_proj = nn.Linear(
                args.hidden_size, self.q_lora_rank, bias=args.attention_bias
            )
            self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank,
                self.num_attention_heads * self.qk_head_dim,
                bias=False,
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            args.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=args.attention_bias,
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_attention_heads * (self.qk_nope_head_dim + args.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_attention_heads * args.v_head_dim,
            args.hidden_size,
            bias=args.attention_bias,
        )

        if args.mla_scale_q_lora:
            self.mla_scale_q_lora = (args.hidden_size / self.q_lora_rank) ** 0.5
        if args.mla_scale_kv_lora:
            self.mla_scale_kv_lora = (args.hidden_size / self.kv_lora_rank) ** 0.5

        if args.rope_scaling is not None:
            mscale_all_dim = args.rope_scaling.get("mscale_all_dim", 0)
            if mscale_all_dim:
                scaling_factor = args.rope_scaling["factor"]
                if scaling_factor > 1:
                    s = 0.1 * mscale_all_dim * math.log(scaling_factor) + 1.0
                    self.scale = self.scale * s * s

        self.rope = initialize_rope(
            dims=self.qk_rope_head_dim,
            base=args.rope_theta,
            traditional=True,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        if self.q_lora_rank is None:
            q_states = self.q_proj(x)
        else:
            q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))

        q_states = q_states.reshape(B, L, -1, self.qk_head_dim).transpose(0, 2, 1, 3)

        if self.mla_scale_q_lora is not None:
            q_states = q_states * self.mla_scale_q_lora

        q_pass, q_rot = mx.split(q_states, [self.qk_nope_head_dim], axis=-1)

        compressed_kv = self.kv_a_proj_with_mqa(x)
        k_pass, k_rot = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pass = self.kv_a_layernorm(k_pass)

        if self.mla_scale_kv_lora is not None:
            k_pass = k_pass * self.mla_scale_kv_lora

        key_shape = (B, L, -1, self.qk_nope_head_dim + self.v_head_dim)
        k_pass = self.kv_b_proj(k_pass).reshape(*key_shape).transpose(0, 2, 1, 3)
        k_pass, value_states = mx.split(k_pass, [self.qk_nope_head_dim], axis=-1)

        k_rot = k_rot.reshape(B, 1, L, self.qk_rope_head_dim)

        if cache is not None:
            q_rot = self.rope(q_rot, cache.offset)
            k_rot = self.rope(k_rot, cache.offset)
        else:
            q_rot = self.rope(q_rot)
            k_rot = self.rope(k_rot)

        k_rot = mx.broadcast_to(k_rot, (*k_pass.shape[:-1], k_rot.shape[-1]))

        query_states = mx.concatenate([q_pass, q_rot], axis=-1)
        key_states = mx.concatenate([k_pass, k_rot], axis=-1)

        if cache is not None:
            key_states, value_states = cache.update_and_fetch(key_states, value_states)

        attn_output = scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(attn_output)


class LongcatFlashLiteMLP(nn.Module):
    def __init__(self, args: ModelArgs, is_expert: bool = False):
        super().__init__()
        hidden_size = args.expert_ffn_hidden_size if is_expert else args.ffn_hidden_size

        self.gate_proj = nn.Linear(args.hidden_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class LongcatFlashLiteTopkRouter(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.config = args
        self.top_k = args.moe_topk
        self.n_routed_experts = args.n_routed_experts + args.zero_expert_num
        self.routed_scaling_factor = args.routed_scaling_factor
        self.norm_topk_prob = args.norm_topk_prob
        self.router_bias = args.router_bias

        self.classifier = nn.Linear(
            args.hidden_size, self.n_routed_experts, bias=self.router_bias
        )
        self.e_score_correction_bias = mx.zeros((self.n_routed_experts,))

    def __call__(self, hidden_states: mx.array) -> Tuple[mx.array, mx.array]:

        dtype = hidden_states.dtype
        router_logits = self.classifier(hidden_states)
        scores = mx.softmax(router_logits, axis=-1)

        corrected_scores = scores + self.e_score_correction_bias
        topk_indices = mx.argpartition(corrected_scores, kth=-self.top_k, axis=-1)[
            ..., -self.top_k :
        ]
        topk_weights = mx.take_along_axis(scores, topk_indices, axis=-1)

        if self.norm_topk_prob:
            denominator = mx.sum(topk_weights, axis=-1, keepdims=True) + 1e-20
            topk_weights = topk_weights / denominator

        topk_weights = topk_weights * self.routed_scaling_factor

        return topk_indices, topk_weights.astype(dtype)


class LongcatFlashLiteMoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.config = args
        self.num_experts_per_tok = args.moe_topk
        self.n_routed_experts = args.n_routed_experts
        self.zero_expert_num = args.zero_expert_num

        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.expert_ffn_hidden_size,
            args.n_routed_experts,
        )

        self.router = LongcatFlashLiteTopkRouter(args)
        self.sharding_group = None

    def __call__(self, hidden_states):
        if self.sharding_group is not None:
            hidden_states = sum_gradients(self.sharding_group)(hidden_states)

        topk_indices, topk_weights = self.router(hidden_states)

        mask = topk_indices >= self.n_routed_experts
        topk_indices = mx.where(mask, 0, topk_indices)
        regular_weights = mx.where(mask, 0.0, topk_weights)

        regular_outputs = self.switch_mlp(hidden_states, topk_indices)

        weighted_outputs = regular_outputs * regular_weights[..., None]
        final_output = mx.sum(weighted_outputs, axis=-2)

        if self.sharding_group is not None:
            final_output = mx.distributed.all_sum(
                final_output, group=self.sharding_group
            )

        identity_weights_sum = mx.sum(
            mx.where(mask, topk_weights, 0.0), axis=-1, keepdims=True
        )
        final_output = final_output + hidden_states * identity_weights_sum

        return final_output


class LongcatFlashLiteDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.mlp = LongcatFlashLiteMoE(args)

        self.self_attn = [LongcatFlashLiteMLA(args) for _ in range(2)]
        self.mlps = [LongcatFlashLiteMLP(args, False) for _ in range(2)]
        self.input_layernorm = [
            nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps) for _ in range(2)
        ]
        self.post_attention_layernorm = [
            nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps) for _ in range(2)
        ]

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        hidden_states = x
        shortcut_mlp_output = None

        if cache is None:
            cache = (None, None)

        for i in range(2):
            residual = hidden_states

            hidden_states = self.input_layernorm[i](hidden_states)
            hidden_states = self.self_attn[i](hidden_states, mask=mask, cache=cache[i])
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = self.post_attention_layernorm[i](hidden_states)

            if i == 0:
                shortcut_mlp_output = self.mlp(hidden_states)

            hidden_states = self.mlps[i](hidden_states)
            hidden_states = residual + hidden_states

            if i == 1:
                hidden_states = hidden_states + shortcut_mlp_output

        return hidden_states


class LongcatFlashLiteModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_layers = args.num_layers
        self.ngram_embeddings = NgramEmbedding(args)
        self.layers = [
            LongcatFlashLiteDecoderLayer(args) for _ in range(args.num_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, args.rms_norm_eps)

    def __call__(
        self,
        input_ids: mx.array,
        cache: Optional[Any] = None,
        ngram_cache: Optional[NgramCache] = None,
    ) -> mx.array:
        ngram_context = None
        if ngram_cache is not None and ngram_cache.ngram_context is not None:
            ngram_context = ngram_cache.ngram_context

        h = self.ngram_embeddings(input_ids, ngram_context=ngram_context)

        if ngram_cache is not None:
            ngram_cache.update(input_ids)

        if cache is None:
            cache = [(None, None)] * self.num_layers

        mask = create_attention_mask(h, cache[0][0])

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LongcatFlashLiteModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self._ngram_cache: Optional[NgramCache] = None

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ):
        out = self.model(inputs, cache, ngram_cache=self._ngram_cache)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("classifier"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def cast_predicate(self):
        def predicate(k):
            return "e_score_correction_bias" not in k

        return predicate

    def sanitize(self, weights):
        for l in range(self.args.num_layers):
            prefix = f"model.layers.{l}"
            for n, m in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]:
                for k in ["weight", "scales", "biases"]:
                    if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                            for e in range(self.args.n_routed_experts)
                        ]
                        weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)

        if "model.embed_tokens.weight" in weights:
            weights["model.ngram_embeddings.word_embeddings.weight"] = weights.pop(
                "model.embed_tokens.weight"
            )

        return {k: v for k, v in weights.items() if not k.startswith("model.mtp")}

    def make_cache(self):
        self._ngram_cache = NgramCache(max_context_len=self.args.emb_neighbor_num - 1)
        return [CacheList(KVCache(), KVCache()) for _ in self.model.layers]

    def reset_cache(self):
        self._ngram_cache = NgramCache(max_context_len=self.args.emb_neighbor_num - 1)

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()

        for layer in self.model.layers:
            for attn in layer.self_attn:
                if attn.q_lora_rank is None:
                    attn.q_proj = shard_linear(
                        attn.q_proj, "all-to-sharded", group=group
                    )
                else:
                    attn.q_b_proj = shard_linear(
                        attn.q_b_proj, "all-to-sharded", group=group
                    )
                attn.kv_b_proj = shard_linear(
                    attn.kv_b_proj, "all-to-sharded", group=group
                )
                attn.o_proj = shard_linear(attn.o_proj, "sharded-to-all", group=group)
                attn.num_attention_heads //= N

            for mlp in layer.mlps:
                mlp.gate_proj = shard_linear(
                    mlp.gate_proj, "all-to-sharded", group=group
                )
                mlp.up_proj = shard_linear(mlp.up_proj, "all-to-sharded", group=group)
                mlp.down_proj = shard_linear(
                    mlp.down_proj, "sharded-to-all", group=group
                )

            layer.mlp.sharding_group = group
            shard_inplace(layer.mlp.switch_mlp.gate_proj, "all-to-sharded", group=group)
            shard_inplace(layer.mlp.switch_mlp.up_proj, "all-to-sharded", group=group)
            shard_inplace(layer.mlp.switch_mlp.down_proj, "sharded-to-all", group=group)
