# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from typing import Union, Dict, Optional, List, Any

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear

from .base import BaseModelArgs, scaled_dot_product_attention
from .rope_utils import initialize_rope
from .deepseek_v3 import (
    DeepseekV3MLP,
    DeepseekV3Model,
)
from .switch_layers import SwitchGLU


def _get_llama_4_attn_scale(size, offset, beta: float, max_position_embeddings: int):
    if isinstance(offset, mx.array) and offset.ndim > 0:
        offset = offset[:, None]

    scaling = 1 + beta * mx.log(
        1 + mx.floor((mx.arange(size) + offset) / max_position_embeddings)
    )
    if scaling.ndim == 2:
        return scaling[:, None, :, None]
    else:
        return scaling[:, None]

@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    n_shared_experts: int
    n_routed_experts: int
    routed_scaling_factor: float
    kv_lora_rank: int
    q_lora_rank: int
    norm_topk_prob: bool
    max_position_embeddings: int
    rms_norm_eps: float
    topk_group: int
    num_experts_per_tok: int
    first_k_dense_replace: int
    n_group: int
    qk_rope_head_dim: int
    qk_nope_head_dim: int
    v_head_dim: int
    head_dim: Optional[int] = None
    qk_head_dim: Optional[int] = None
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False
    rope_interleave: Optional[bool] = None
    attention_bias: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    rope_parameters: Optional[Dict[str, Union[float, str, bool, List[int]]]] = None

    def __post_init__(self, **kwargs):
        if self.qk_head_dim is None:
            self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        
        if self.head_dim is None:
            self.head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        
        if self.rope_parameters is None:
            self.rope_parameters = {
                "type": "yarn",
                "rope_theta": 10000.0,
                "factor": 128.0,
                "original_max_position_embeddings": 8192,
                "max_position_embeddings": self.max_position_embeddings,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "mscale_all_dim": 1.0,
                "mscale": 1.0,
                "llama_4_scaling_beta": 0.1,
                "partial_rotary_factor": self.qk_rope_head_dim / self.head_dim,
            }


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

        self.scale = self.q_head_dim**-0.5

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
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=args.attention_bias,
        )

        self.rope = initialize_rope(
            dims=self.qk_rope_head_dim,
            base=self.rope_theta,
            traditional=args.rope_interleave if args.rope_interleave is not None else True,
            max_position_embeddings=self.max_position_embeddings,
            scaling_config=self.args.rope_parameters,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        # Query projection
        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))

        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_rope = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        # KV projection
        compressed_kv = self.kv_a_proj_with_mqa(x)
        k_latent, k_rope = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)

        # Project latent to K and V
        kv = self.kv_b_proj(self.kv_a_layernorm(k_latent))
        kv = kv.reshape(B, L, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        kv = kv.transpose(0, 2, 1, 3)
        k_nope, v = mx.split(kv, [self.qk_nope_head_dim], axis=-1)

        # Reshape k_rope to match k_nope shape
        k_rope = k_rope.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)

        # Apply RoPE
        offset = cache.offset if cache is not None else 0
        q_rope = self.rope(q_rope, offset)
        k_rope = self.rope(k_rope, offset)

        # Expand k_rope to all heads
        k_rope = mx.broadcast_to(k_rope, [B, self.num_heads, k_rope.shape[2], self.qk_rope_head_dim])

        # Concatenate to form full query and key states
        query_states = mx.concatenate([q_nope, q_rope], axis=-1)
        key_states = mx.concatenate([k_nope, k_rope], axis=-1)

        # Apply Llama-4 attention scaling
        if self.args.rope_parameters is not None:
            llama_4_beta = self.args.rope_parameters.get("llama_4_scaling_beta", 0.1)
            original_max_pos = self.args.rope_parameters.get(
                "original_max_position_embeddings", 8192
            )
            attn_scale = _get_llama_4_attn_scale(
                L, offset, llama_4_beta, original_max_pos
            )
            query_states = query_states * attn_scale

        # Update cache
        if cache is not None:
            key_states, v = cache.update_and_fetch(key_states, v)

        # Standard attention
        output = scaled_dot_product_attention(
            query_states, key_states, v, scale=self.scale, mask=mask
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
    """Mistral-4 routing using softmax."""
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

    selected_scores = mx.take_along_axis(scores, inds, axis=-1)
    if norm_topk_prob:
        denominator = selected_scores.sum(axis=-1, keepdims=True) + 1e-20
        selected_scores = selected_scores / denominator
    selected_scores = selected_scores * routed_scaling_factor

    return inds, selected_scores


class Mistral4MoEGate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.n_routed_experts = args.n_routed_experts
        self.routed_scaling_factor = args.routed_scaling_factor
        self.n_group = args.n_group
        self.topk_group = args.topk_group
        self.weight = mx.zeros((self.n_routed_experts, args.hidden_size))

    def __call__(self, x):
        gates = x @ self.weight.T
        return mistral4_expert_select(
            gates,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )


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

        self.gate = Mistral4MoEGate(args)
        if args.n_shared_experts is not None:
            intermediate_size = args.moe_intermediate_size * args.n_shared_experts
            self.shared_experts = DeepseekV3MLP(
                args, intermediate_size=intermediate_size
            )

    def __call__(self, x):
        inds, scores = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        if self.args.n_shared_experts is not None:
            y = y + self.shared_experts(x)
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
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class Mistral4Model(DeepseekV3Model):
    def __init__(self, args: ModelArgs):
        nn.Module.__init__(self)
        self.vocab_size = args.vocab_size
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Mistral4DecoderLayer(args, idx)
            for idx in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)


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
    ) -> mx.array:
        out = self.model(inputs, cache=cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    @property
    def layers(self):
        return self.model.layers

    def sanitize(self, weights):
        def dequant(weight, scale_inv):
            dtype = mx.bfloat16
            weight = mx.from_fp8(weight, dtype=mx.bfloat16)
            bs = 128
            m, n = weight.shape
            pad_bottom = (-m) % bs
            pad_side = (-n) % bs
            weight = mx.pad(weight, ((0, pad_bottom), (0, pad_side)))
            weight = weight.reshape(
                ((m + pad_bottom) // bs, bs, (n + pad_side) // bs, bs)
            )
            weight = (weight * scale_inv[:, None, :, None]).reshape(
                m + pad_bottom, n + pad_side
            )
            return weight[:m, :n].astype(dtype)

        # Remap for int4
        new_weights = {}
        for k, v in weights.items():
            if k.endswith("weight_shape"):
                base = k.replace("weight_shape", "")
                new_weights[base + "weight"] = weights[base + "weight_packed"].view(
                    mx.uint32
                )
                s = weights[base + "weight_scale"]
                new_weights[base + "scales"] = s
                new_weights[base + "biases"] = -8 * s
            elif not (k.endswith("weight_scale") or k.endswith("weight_packed")):
                new_weights[k] = v
        weights = new_weights

        # Dequantize fp8
        new_weights = {}
        for k, v in weights.items():
            if "weight_scale_inv" in k:
                scale_inv = v
                wk = k.replace("_scale_inv", "")
                weight = weights[wk]
                weight = dequant(weight, scale_inv)
                new_weights[wk] = weight
            elif k not in new_weights:
                new_weights[k] = v
        weights = new_weights

        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"

            # Handle fused gate_up_proj format (Mistral4NaiveMoe)
            gup_key = f"{prefix}.mlp.experts.gate_up_proj"
            if gup_key in weights:
                gate_up = weights.pop(gup_key)
                gate, up = mx.split(gate_up, 2, axis=1)
                weights[f"{prefix}.mlp.switch_mlp.gate_proj.weight"] = gate
                weights[f"{prefix}.mlp.switch_mlp.up_proj.weight"] = up
            down_key = f"{prefix}.mlp.experts.down_proj"
            if down_key in weights:
                weights[f"{prefix}.mlp.switch_mlp.down_proj.weight"] = weights.pop(down_key)

            # Handle per-expert weights format
            for m in ["gate_proj", "down_proj", "up_proj"]:
                for k in ["weight", "scales", "biases"]:
                    if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                            for e in range(self.args.n_routed_experts)
                        ]
                        weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)

        return {
            k: v
            for k, v in weights.items()
            if "rotary_emb.inv_freq" not in k
        }

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()
        rank = group.rank()
        
        for layer in self.model.layers:
            if layer.self_attn.q_lora_rank is None:
                layer.self_attn.q_proj = shard_linear(
                    layer.self_attn.q_proj, "all-to-sharded", group=group
                )
            else:
                layer.self_attn.q_b_proj = shard_linear(
                    layer.self_attn.q_b_proj, "all-to-sharded", group=group
                )
            
            layer.self_attn.kv_b_proj = shard_linear(
                layer.self_attn.kv_b_proj, "all-to-sharded", group=group
            )
            
            layer.self_attn.num_heads //= N

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
                if hasattr(layer.mlp, 'shared_experts'):
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