# Copyright © 2023-2024 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope

@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    logits_scaling: float
    attention_multiplier: float
    embedding_multiplier: float
    residual_multiplier: float
    max_position_embeddings: int
    num_key_value_heads: int
    attention_bias: bool
    rope_theta: float
    num_local_experts: int
    num_experts_per_tok: int
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = True


class GraniteMoeAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        self.head_dim = head_dim = args.hidden_size // n_heads

        self.scale = args.attention_multiplier
        attention_bias = args.attention_bias
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=attention_bias)

        self.rope = initialize_rope(
            self.head_dim,
            args.rope_theta,
            False,
            args.rope_scaling,
            args.max_position_embeddings,
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
    

class GraniteMoeParallelExperts(nn.Module):
    def __init__(self, num_experts: int, input_size: int, output_size: int):
        super().__init__()
        self.weight = mx.random.normal((num_experts, output_size, input_size))
        self.num_experts = num_experts
        self.input_size = input_size
        self.output_size = output_size
    
    def __call__(self, inputs: mx.array, expert_indices: mx.array) -> mx.array:
        all_expert_outputs = mx.einsum('ti,eoi->teo', inputs, self.weight)
        expert_one_hot = mx.eye(self.num_experts)[expert_indices]
        return mx.sum(all_expert_outputs * mx.expand_dims(expert_one_hot, -1), axis=1)


class GraniteMoeTopKGating(nn.Module):
    def __init__(self, input_size: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.input_size = input_size
        self.top_k = top_k
        self.layer = nn.Linear(input_size, num_experts, bias=False)
    
    def __call__(self, hidden_states: mx.array):
        logits = self.layer(hidden_states).astype(mx.float32)
        top_k_indices = mx.stop_gradient(mx.argsort(logits, axis=1)[:, -self.top_k:])
        row_ids = mx.arange(logits.shape[0])[:, None]
        top_k_logits = logits[row_ids, top_k_indices]
        top_k_gates = mx.softmax(top_k_logits, axis=1)
        top_k_experts = top_k_indices.reshape((-1,))
        top_k_gates_flat = top_k_gates.reshape((-1,))
        token_ids = mx.repeat(mx.arange(hidden_states.shape[0]), self.top_k)
        sorted_indices = mx.stop_gradient(mx.argsort(top_k_experts, axis=0))
        sorted_expert_ids = top_k_experts[sorted_indices]
        sorted_token_ids = token_ids[sorted_indices] 
        sorted_gates = top_k_gates_flat[sorted_indices]
        expert_sizes = mx.array([mx.sum(sorted_expert_ids == i) for i in range(self.num_experts)])
        
        return (
            mx.stop_gradient(sorted_token_ids), 
            sorted_gates,
            mx.stop_gradient(expert_sizes), 
            mx.stop_gradient(sorted_expert_ids)
        )


class GraniteMoeMoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        self.input_size = args.hidden_size
        self.hidden_size = args.intermediate_size
        self.input_linear = GraniteMoeParallelExperts(args.num_local_experts, self.input_size, self.hidden_size * 2)
        self.output_linear = GraniteMoeParallelExperts(args.num_local_experts, self.hidden_size, self.input_size)
        
        self.router = GraniteMoeTopKGating(
            input_size=self.input_size,
            num_experts=args.num_local_experts,
            top_k=args.num_experts_per_tok,
        )

    def __call__(self, layer_input: mx.array) -> mx.array:
        bsz, length, emb_size = layer_input.shape
        layer_input = layer_input.reshape(-1, emb_size)
        token_ids, gates, expert_sizes, expert_ids = self.router(layer_input)
        selected_tokens = layer_input[token_ids] 
        hidden_states = self.input_linear(selected_tokens, expert_ids)
        chunk_size = hidden_states.shape[-1] // 2
        chunked_hidden_states = [
            hidden_states[..., :chunk_size],
            hidden_states[..., chunk_size:]
        ]
        hidden_states = nn.silu(chunked_hidden_states[0]) * chunked_hidden_states[1]
        expert_outputs = self.output_linear(hidden_states, expert_ids)
        expert_outputs = expert_outputs * gates[:, None]
        if len(expert_outputs) == len(layer_input):
            return expert_outputs.reshape(bsz, length, self.input_size)
        else:
            zeros = mx.zeros((bsz * length, self.input_size), dtype=expert_outputs.dtype)
            zeros = zeros.at[token_ids].add(expert_outputs)
            return zeros.reshape(bsz, length, self.input_size)

class GraniteMoeDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = GraniteMoeAttention(args)
        if args.num_local_experts > 0:
            self.block_sparse_moe = GraniteMoeMoE(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.residual_multiplier = args.residual_multiplier

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r * self.residual_multiplier
        r = self.block_sparse_moe(self.post_attention_layernorm(h))
        out = h + r * self.residual_multiplier
        return out


class GraniteMoEModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            GraniteMoeDecoderLayer(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.embedding_multiplier = args.embedding_multiplier

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
    ):
        h = self.embed_tokens(inputs) * self.embedding_multiplier

        if mask is None:
            mask = create_attention_mask(h, cache)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = GraniteMoEModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.logits_scaling = args.logits_scaling

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
        return out / self.logits_scaling

    @property
    def layers(self):
        return self.model.layers
