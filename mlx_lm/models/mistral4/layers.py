# Copyright © 2026 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import sum_gradients

from mlx_lm.layers.attention import scaled_dot_product_attention
from mlx_lm.layers.mla import MultiLinear
from mlx_lm.layers.rope import apply_yarn_mscale, initialize_rope
from mlx_lm.layers.switch import SwitchGLU
from mlx_lm.models.deepseek_v3.layers import DeepseekV3MLP
from mlx_lm.models.mistral4.config import ModelArgs

__all__ = [
    "Mistral4Attention",
    "Mistral4DecoderLayer",
    "Mistral4MoE",
    "mistral4_expert_select",
]


class Mistral4Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.max_position_embeddings = args.max_position_embeddings
        self.rope_theta = args.rope_theta
        self.q_lora_rank = args.q_lora_rank
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.kv_lora_rank = args.kv_lora_rank
        self.v_head_dim = args.v_head_dim
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.q_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim

        self.scale = apply_yarn_mscale(self.q_head_dim**-0.5, args.rope_parameters)

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                self.hidden_size, self.num_heads * self.q_head_dim, bias=False
            )
        else:
            self.q_a_proj = nn.Linear(
                self.hidden_size, self.q_lora_rank, bias=args.attention_bias
            )
            self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=1e-6)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=args.attention_bias,
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=1e-6)
        # kv_b_proj absorbed, so the cache holds the compressed latent.
        self.embed_q = MultiLinear(
            self.qk_nope_head_dim, self.kv_lora_rank, self.num_heads
        )
        self.unembed_out = MultiLinear(
            self.kv_lora_rank, self.v_head_dim, self.num_heads
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=args.attention_bias,
        )

        self.rope = initialize_rope(
            dims=self.qk_rope_head_dim,
            base=self.rope_theta,
            traditional=(
                args.rope_interleave if args.rope_interleave is not None else True
            ),
            max_position_embeddings=self.max_position_embeddings,
            scaling_config=self.args.rope_parameters,
        )

    def __call__(
        self,
        x: mx.array,
        attn_scale: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))

        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_rope = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed_kv = self.kv_a_proj_with_mqa(x)
        k_latent, k_rope = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_rope = k_rope.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = mx.expand_dims(self.kv_a_layernorm(k_latent), axis=1)

        offset = cache.offset if cache is not None else 0
        q_rope = self.rope(q_rope, offset)
        k_rope = self.rope(k_rope, offset)

        # The llama-4 scale applies to the whole query, so scale both halves.
        q_nope = q_nope * attn_scale
        q_rope = q_rope * attn_scale

        if cache is not None:
            kv_latent, k_rope = cache.update_and_fetch(kv_latent, k_rope)

        if L == 1:
            # Decode: attend to the latent directly. pe_scores is [B, H, 1, L].
            pe_scores = (q_rope * self.scale) @ k_rope.swapaxes(-1, -2)
            if mask is not None:
                pe_scores = mx.where(
                    mask,
                    pe_scores,
                    mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
                )
            output = scaled_dot_product_attention(
                self.embed_q(q_nope),
                kv_latent,
                kv_latent,
                cache=cache,
                scale=self.scale,
                mask=pe_scores,
            )
            output = self.unembed_out(output)
        else:
            k_nope = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)
            k_rope = mx.broadcast_to(
                k_rope, [B, self.num_heads, k_rope.shape[2], self.qk_rope_head_dim]
            )
            output = scaled_dot_product_attention(
                mx.concatenate([q_nope, q_rope], axis=-1),
                mx.concatenate([k_nope, k_rope], axis=-1),
                v,
                cache=cache,
                scale=self.scale,
                mask=mask,
            )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


@mx.compile
def mistral4_expert_select(
    gates,
    top_k,
    n_group,
    topk_group,
    routed_scaling_factor,
    norm_topk_prob,
):
    scores = mx.softmax(gates.astype(mx.float32), axis=-1)

    if n_group > 1:
        scores_grouped = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores_grouped, 2, axis=-1).sum(axis=-1, keepdims=True)
        # Zero out bottom (n_group - topk_group) groups
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        scores_grouped = mx.put_along_axis(
            scores_grouped, mx.stop_gradient(group_idx), mx.array(0.0), axis=-2
        )
        scores_for_choice = mx.flatten(scores_grouped, -2, -1)
    else:
        scores_for_choice = scores

    inds = mx.argpartition(-scores_for_choice, kth=top_k - 1, axis=-1)[..., :top_k]
    inds = mx.stop_gradient(inds)

    selected_scores = mx.take_along_axis(scores, inds, axis=-1)
    if norm_topk_prob:
        denominator = selected_scores.sum(axis=-1, keepdims=True) + 1e-20
        selected_scores = selected_scores / denominator
    selected_scores = selected_scores * routed_scaling_factor

    return inds, selected_scores


class Mistral4MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.num_experts_per_tok = args.num_experts_per_tok
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.n_routed_experts,
        )

        self.gate = nn.Linear(args.hidden_size, args.n_routed_experts, bias=False)
        if args.n_shared_experts is not None:
            intermediate_size = args.moe_intermediate_size * args.n_shared_experts
            self.shared_experts = DeepseekV3MLP(
                args, intermediate_size=intermediate_size
            )

        self.sharding_group = None

    def __call__(self, x):
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)

        inds, scores = mistral4_expert_select(
            self.gate(x),
            self.num_experts_per_tok,
            self.args.n_group,
            self.args.topk_group,
            self.args.routed_scaling_factor,
            self.args.norm_topk_prob,
        )
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        if self.args.n_shared_experts is not None:
            y = y + self.shared_experts(x)

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)

        return y


class Mistral4DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Mistral4Attention(args)
        self.mlp = (
            Mistral4MoE(args)
            if (
                args.n_routed_experts is not None
                and layer_idx >= args.first_k_dense_replace
            )
            else DeepseekV3MLP(args)
        )
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        attn_scale: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), attn_scale, mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r
