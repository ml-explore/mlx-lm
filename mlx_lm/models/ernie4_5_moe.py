# Copyright © 2023-2024 Apple Inc.

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


# python -m mlx_lm.generate --model baidu/ERNIE-4.5-0.3B-PT --prompt "wrtie a long storry about a AI machine helping a human" -m 20000
# python -m mlx_lm.convert --hf-path baidu/ERNIE-4.5-21B-A3B-PT -q --mlx-path /Users/gokdenizgulmez/Desktop/ERNIE-4.5-21B-A3B-PT-4bit
# python -m mlx_lm.generate --model /Users/gokdenizgulmez/Desktop/ERNIE-4.5-21B-A3B-PT-4bit --prompt "wrtie a long storry about a AI machine helping a human" -m 20000


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
        
        # Use SwitchGLU instead of custom experts
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            self.moe_intermediate_size,
            args.moe_num_experts,
            bias=args.use_bias,
        )

        # Keep shared experts functionality
        if getattr(args, "moe_num_shared_experts", 0) > 0:
            shared_intermediate_size = (
                args.moe_intermediate_size * args.moe_num_shared_experts
                if getattr(args, "moe_intermediate_size", None)
                else args.intermediate_size * args.moe_num_shared_experts
            )
            self.shared_experts = Ernie4_5_MLP(args.hidden_size, shared_intermediate_size, args.use_bias)
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

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass through MoE layer."""
        # Gate computation
        gates = self.gate(x)
        gates = self.gate_act(gates)
        
        # Get top-k indices
        k = self.k
        inds = mx.stop_gradient(mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k])
        scores = mx.take_along_axis(gates, inds, axis=-1)
        
        # Normalize scores
        scores = scores / mx.maximum(scores.sum(axis=-1, keepdims=True), 1e-12)
        
        # Process through switch MLP
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        
        # Add shared expert output if available
        if self.shared_experts is not None:
            y = y + self.shared_experts(x)
        
        return y


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
        # First remove unwanted patterns
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
        
        # Handle MoE expert weight mapping
        # Dynamically discover expert indices for each layer
        expert_keys = {}
        for key in list(sanitized_weights.keys()):
            if ".mlp.experts." in key:
                # Extract layer index and expert index
                parts = key.split(".")
                layer_idx = None
                expert_idx = None
                
                for i, part in enumerate(parts):
                    if part == "layers" and i + 1 < len(parts):
                        layer_idx = int(parts[i + 1])
                    elif part == "experts" and i + 1 < len(parts):
                        expert_idx = int(parts[i + 1])
                
                if layer_idx is not None and expert_idx is not None:
                    if layer_idx not in expert_keys:
                        expert_keys[layer_idx] = set()
                    expert_keys[layer_idx].add(expert_idx)
        
        # Map individual expert weights to SwitchGLU format
        for layer_idx, expert_indices in expert_keys.items():
            prefix = f"model.layers.{layer_idx}"
            expert_indices = sorted(expert_indices)
            
            # Handle the three projection types
            for n in ["up_proj", "down_proj", "gate_proj"]:
                # Handle weights
                to_join_weights = []
                for e in expert_indices:
                    key = f"{prefix}.mlp.experts.{e}.{n}.weight"
                    if key in sanitized_weights:
                        to_join_weights.append(sanitized_weights.pop(key))
                if to_join_weights:
                    sanitized_weights[f"{prefix}.mlp.switch_mlp.{n}.weight"] = mx.stack(to_join_weights)
                
                # Handle biases if they exist
                to_join_biases = []
                for e in expert_indices:
                    key = f"{prefix}.mlp.experts.{e}.{n}.bias"
                    if key in sanitized_weights:
                        to_join_biases.append(sanitized_weights.pop(key))
                if to_join_biases:
                    sanitized_weights[f"{prefix}.mlp.switch_mlp.{n}.bias"] = mx.stack(to_join_biases)
                
                # Handle scales if they exist (for quantization)
                to_join_scales = []
                for e in expert_indices:
                    key = f"{prefix}.mlp.experts.{e}.{n}.scales"
                    if key in sanitized_weights:
                        to_join_scales.append(sanitized_weights.pop(key))
                if to_join_scales:
                    sanitized_weights[f"{prefix}.mlp.switch_mlp.{n}.scales"] = mx.stack(to_join_scales)
                
                # Handle biases for scales if they exist (for quantization)
                to_join_biases_scales = []
                for e in expert_indices:
                    key = f"{prefix}.mlp.experts.{e}.{n}.biases"
                    if key in sanitized_weights:
                        to_join_biases_scales.append(sanitized_weights.pop(key))
                if to_join_biases_scales:
                    sanitized_weights[f"{prefix}.mlp.switch_mlp.{n}.biases"] = mx.stack(to_join_biases_scales)
        
        return sanitized_weights