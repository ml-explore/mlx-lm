from typing import Optional, Any, Tuple, Dict
from dataclasses import dataclass
import math

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, scaled_dot_product_attention, create_attention_mask
from .switch_layers import SwitchGLU
from .deepseek_v3 import DeepseekV3YarnRotaryEmbedding


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    attention_method: str
    zero_expert_type: str
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
    attention_bias: bool
    norm_topk_prob: bool = False
    router_bias: bool = False
    rope_scaling: Dict = None


class LongcatFlashMLA(nn.Module):
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

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(args.hidden_size, self.num_attention_heads * self.qk_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(args.hidden_size, self.q_lora_rank, bias=args.attention_bias)
            self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank)
            self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_attention_heads * self.qk_head_dim, bias=False)

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
            scaling_factor = args.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.scale = self.scale * mscale * mscale

            rope_kwargs = {
                key: args.rope_scaling[key]
                for key in [
                    "original_max_position_embeddings",
                    "beta_fast",
                    "beta_slow",
                    "mscale",
                    "mscale_all_dim",
                ]
                if key in args.rope_scaling
            }
            self.rope = DeepseekV3YarnRotaryEmbedding(
                dim=self.qk_rope_head_dim,
                max_position_embeddings=args.max_position_embeddings,
                scaling_factor=scaling_factor,
                base=args.rope_theta,
                **rope_kwargs,
            )
        else:
            self.rope = nn.RoPE(
                dims=self.qk_rope_head_dim,
                base=args.rope_theta,
                traditional=True
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
        q_pass, q_rot = mx.split(q_states, [self.qk_nope_head_dim], axis=-1)

        if self.mla_scale_q_lora is not None:
            q_pass = q_pass * self.mla_scale_q_lora
            q_rot = q_rot * self.mla_scale_q_lora

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
            mask=mask
        )

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(attn_output)


class LongcatFlashMLP(nn.Module):
    def __init__(self, args: ModelArgs, is_expert: bool = False):
        super().__init__()
        hidden_size = args.expert_ffn_hidden_size if is_expert else args.ffn_hidden_size
        
        self.gate_proj = nn.Linear(args.hidden_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class LongcatFlashTopkRouter(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.config = args
        self.top_k = args.moe_topk
        self.n_routed_experts = (
            args.n_routed_experts
            if args.zero_expert_num is None
            else args.n_routed_experts + args.zero_expert_num
        )
        self.routed_scaling_factor = args.routed_scaling_factor
        self.norm_topk_prob = args.norm_topk_prob
        self.router_bias = args.router_bias

        self.classifier = nn.Linear(args.hidden_size, self.n_routed_experts, bias=self.router_bias)
        self.e_score_correction_bias = mx.zeros((self.n_routed_experts,))

    def __call__(self, hidden_states: mx.array) -> Tuple[mx.array, mx.array]:
        hidden_states = hidden_states.reshape(-1, self.config.hidden_size)
        router_logits = self.classifier(hidden_states.astype(mx.float32))
        scores = mx.softmax(router_logits, axis=-1)
        
        # Use argsort in descending order to get indices
        # argsort gives ascending order, so we negate the scores to get descending order
        sorted_indices = mx.argsort(-scores, axis=-1)  # Negative scores for descending order
        topk_indices = sorted_indices[..., :self.top_k]  # Take first k (which are the largest due to negation)
        
        # Gather the corresponding weights using advanced indexing
        batch_indices = mx.arange(topk_indices.shape[0])[:, None]
        topk_weights = scores[batch_indices, topk_indices]
        
        if self.norm_topk_prob:
            denominator = mx.sum(topk_weights, axis=-1, keepdims=True) + 1e-20
            topk_weights = topk_weights / denominator
            
        topk_weights = topk_weights * self.routed_scaling_factor
        
        return topk_indices, topk_weights


class LongcatFlashMoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.config = args
        self.num_experts_per_tok = args.moe_topk
        self.n_routed_experts = args.n_routed_experts
        self.zero_expert_num = args.zero_expert_num
        self.zero_expert_type = args.zero_expert_type
        
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.expert_ffn_hidden_size,
            args.n_routed_experts,
        )
        
        self.router = LongcatFlashTopkRouter(args)
        
    def __call__(self, hidden_states):
        orig_shape = hidden_states.shape
        hidden_states_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        
        topk_indices, topk_weights = self.router(hidden_states_flat)
        
        # Initialize output
        final_output = mx.zeros_like(hidden_states_flat).astype(topk_weights.dtype)
        
        # Process token by token to handle mixed expert types
        batch_size, top_k = topk_indices.shape
        
        # Collect outputs for each token
        token_outputs = []
        
        for token_idx in range(batch_size):
            token_input = hidden_states_flat[token_idx:token_idx+1]  # Keep batch dimension
            token_output = mx.zeros_like(token_input).astype(topk_weights.dtype)
            
            for k in range(top_k):
                expert_idx = topk_indices[token_idx, k]
                weight = topk_weights[token_idx, k]
                
                if expert_idx < self.n_routed_experts:
                    # Regular expert - use SwitchGLU
                    expert_indices = mx.array([[expert_idx]], dtype=mx.int32)
                    expert_output = self.switch_mlp(token_input, expert_indices)
                    token_output = token_output + weight * expert_output.squeeze(-2)
                    
                elif (self.zero_expert_num is not None and 
                      expert_idx >= self.n_routed_experts and
                      self.zero_expert_type == "identity"):
                    # Zero expert with identity
                    token_output = token_output + weight * token_input.squeeze(0)
            
            token_outputs.append(token_output.squeeze(0))
        
        # Stack all token outputs
        final_output = mx.stack(token_outputs, axis=0)
        
        return final_output.reshape(*orig_shape).astype(hidden_states.dtype)


class LongcatFlashDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.mlp = LongcatFlashMoE(args)

        self.self_attn = [LongcatFlashMLA(args) for _ in range(2)]
        self.mlps = [LongcatFlashMLP(args, False) for _ in range(2)]
        self.input_layernorm = [
            nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps) 
            for _ in range(2)
        ]
        self.post_attention_layernorm = [
            nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps) 
            for _ in range(2)
        ]

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        hidden_states = x
        shortcut_mlp_output = None
        
        for i in range(2):
            residual = hidden_states

            hidden_states = self.input_layernorm[i](hidden_states)
            hidden_states = self.self_attn[i](hidden_states, mask=mask, cache=cache)
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


class LongcatFlashModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_layers = args.num_layers
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            LongcatFlashDecoderLayer(args)
            for idx in range(args.num_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.embed_tokens(x)

        if mask is None:
            mask = create_attention_mask(h, cache)

        if cache is None:
            cache = [None] * self.num_layers
        
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)
        
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LongcatFlashModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ):
        out = self.model(inputs, mask, cache)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers
    
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
        return weights