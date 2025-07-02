# Copyright © 2023-2024 Apple Inc.

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope


# python -m mlx_lm.generate --model baidu/ERNIE-4.5-0.3B-PT --prompt "wrtie a long storry about a AI machine helping a human" -m 20000
# python -m mlx_lm.convert --hf-path baidu/ERNIE-4.5-21B-A3B-PT -q --mlx-path /Users/gokdenizgulmez/Desktop/ERNIE-4.5-21B-A3B-PT-4bit


@dataclass
class ModelArgs(BaseModelArgs):
    hidden_size: int
    intermediate_size: int
    model_type: str
    max_position_embeddings: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    rms_norm_eps: float
    vocab_size: int
    rope_theta: float
    use_bias: bool
    tie_word_embeddings: bool
    moe_num_experts: int
    moe_num_shared_experts: int
    moe_layer_start_index: int
    moe_intermediate_size: int
    moe_capacity: list[int]
    moe_k: int
    moe_layer_interval: int
    moe_use_aux_free: bool
    moe_layer_end_index: Optional[int] = None
    head_dim: Optional[int] = None
    moe_gate_act: str = "softmax"


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        self.head_dim = head_dim = args.head_dim or dim // n_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=args.use_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.use_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.use_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=args.use_bias)

        self.rope = initialize_rope(
            head_dim,
            base=args.rope_theta,
            traditional=True,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class Ernie4_5_MLP(nn.Module):
    def __init__(self, dim, hidden_dim, use_bias=False):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=use_bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=use_bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=use_bias)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Ernie4_5_MoeStatics(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        num_experts = args.moe_num_experts
        num_experts_groups = 1
        self.e_score_correction_bias = mx.zeros((num_experts_groups, num_experts), dtype=mx.float32)
    
    def __call__(self, x):
        return x
    

class Ernie4_5_MoeMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.k = args.moe_k
        self.moe_intermediate_size = args.moe_intermediate_size if args.moe_intermediate_size else args.intermediate_size

        self.gate = nn.Linear(args.hidden_size, args.moe_num_experts, bias=False)
        self.experts = [Ernie4_5_MLP(dim=args.hidden_size, hidden_dim=self.moe_intermediate_size) for _ in range(args.moe_num_experts)]

        if getattr(args, "moe_num_shared_experts", 0) > 0:
            args.intermediate_size = (
                args.moe_intermediate_size * args.moe_num_shared_experts
                if getattr(args, "moe_intermediate_size", None)
                else args.intermediate_size * args.moe_num_shared_experts
            )
            self.shared_experts = Ernie4_5_MLP(args.hidden_size, args.intermediate_size)
        else:
            self.shared_experts = None

        if args.moe_use_aux_free:
            self.moe_statics = Ernie4_5_MoeStatics(args)

        if args.moe_gate_act == "softmax":
            self.gate_act = nn.Softmax()
        elif args.moe_gate_act == "sigmoid":
            self.gate_act = nn.Sigmoid()
        else:
            raise ValueError(f"{args.moe_gate_act} is not supported.")

    def __call__(self, input: mx.array) -> mx.array:
        """Forward pass through MoE layer."""
        # Handle 3D input
        orig_shape = None
        if input.ndim == 3:
            orig_shape = input.shape
            input = input.reshape(-1, input.shape[-1])
        
        assert input.ndim == 2, f"Input must be 2D, got shape: {input.shape}"
        
        # Route and process through experts
        output = self._route_and_process(input)
        
        # Add shared expert output if available
        if self.shared_experts is not None:
            output = output + self.shared_experts(input)
        
        # Restore original shape
        if orig_shape:
            output = output.reshape(orig_shape[:-1] + (output.shape[-1],))
        
        return output
    
    def _route_and_process(self, input: mx.array) -> mx.array:
        """Combined routing and expert processing."""
        # Gate computation
        gate_logits = self.gate(input.astype(mx.float32))
        gate_probs = self.gate_act(gate_logits)
        topk_probs, topk_indices = mx.topk(gate_probs, self.k, axis=-1)
        
        # Normalize weights
        combine_weights = topk_probs / mx.maximum(
            topk_probs.sum(axis=-1, keepdims=True), 1e-12
        )
        
        # Process through experts and combine
        return self._process_experts(input, topk_indices, combine_weights)
    
    def _process_experts(self, input: mx.array, expert_indices: mx.array, weights: mx.array) -> mx.array:
        """Process input through selected experts and combine outputs."""
        batch_size, hidden_size = input.shape
        output_dim = self.experts[0](input[:1]).shape[-1]  # Get output dimension
        
        # Initialize combined output
        combined_output = mx.zeros((batch_size, output_dim))
        
        # Process each token
        for i in range(batch_size):
            token_input = input[i:i+1]  # Keep batch dimension
            token_output = mx.zeros((1, output_dim))
            
            # Process through top-k experts for this token
            for k in range(self.k):
                expert_idx = expert_indices[i, k].item()
                weight = weights[i, k]
                
                if weight > 1e-12:  # Skip if weight is negligible
                    expert_output = self.experts[expert_idx](token_input)
                    token_output = token_output + weight * expert_output
            
            combined_output = combined_output.at[i].set(token_output.squeeze(0))
        
        return combined_output


class Ernie4_5_DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args)

        moe_layer_start_index = (
            min(args.moe_layer_start_index)
            if isinstance(args.moe_layer_start_index, (tuple, list))
            else args.moe_layer_start_index
        )
        
        if args.moe_layer_end_index is None:
            moe_layer_end_index = args.num_hidden_layers - 1
        else:
            moe_layer_end_index = (
                max(args.moe_layer_end_index)
                if isinstance(args.moe_layer_end_index, (tuple, list))
                else args.moe_layer_end_index
            )

        # Decide whether to use MoE or regular MLP
        if (
            ((layer_idx + 1) % args.moe_layer_interval == 0)
            and layer_idx >= moe_layer_start_index
            and layer_idx <= moe_layer_end_index
        ):
            self.mlp = Ernie4_5_MoeMLP(args)
        else:
            self.mlp = Ernie4_5_MLP(args.hidden_size, args.intermediate_size, args.use_bias)

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


class Ernie45Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [Ernie4_5_DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
    ):
        h = self.embed_tokens(inputs)

        if mask is None:
            mask = create_attention_mask(h, cache)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Ernie45Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
    ):
        out = self.model(inputs, mask, cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    @property
    def layers(self):
        return self.model.layers
    
    def sanitize(self, weights):
        remove_patterns = [
            "mtp_block.",
            "mtp_linear_proj.",
            "mtp_hidden_norm.",
            "mtp_emb_norm.",
        ]
        
        # Filter out unwanted parameters
        sanitized_weights = {}
        for key, value in weights.items():
            should_remove = any(pattern in key for pattern in remove_patterns)
            if not should_remove:
                sanitized_weights[key] = value
        
        return sanitized_weights
