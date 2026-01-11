# Copyright © 2023-2024 Apple Inc.


from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import mlx.core as mx
import mlx.nn as nn


from .base import BaseModelArgs, scaled_dot_product_attention
from .switch_layers import SwitchGLU


@dataclass
class SarvamMoEModelOutputWithPast:
    last_hidden_state: mx.array = None
    past_key_values: Optional[List[mx.array]] = None
    hidden_states: Optional[Tuple[mx.array]] = None
    attentions: Optional[Tuple[mx.array]] = None
    router_logits: Optional[Tuple[mx.array]] = None


@dataclass
class SarvamMoECausalLMOutputWithPast:
    loss: Optional[mx.array] = None
    logits: Optional[mx.array] = None
    past_key_values: Optional[List[mx.array]] = None
    hidden_states: Optional[Tuple[mx.array]] = None
    attentions: Optional[Tuple[mx.array]] = None

    z_loss: Optional[mx.array] = None
    aux_loss: Optional[mx.array] = None
    router_logits: Optional[Tuple[mx.array]] = None


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(axis=-1).astype(mx.int32)
    # create indices from mask
    # MLX boolean indexing is limited, use numpy/cpu fallback for this helper
    mask_np = np.array(attention_mask.flatten())
    indices_np = np.nonzero(mask_np)[0]
    indices = mx.array(indices_np)
    
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = mx.cumsum(seqlens_in_batch, axis=0).astype(mx.int32)
    # Pad with 0
    cu_seqlens = mx.concatenate([mx.array([0], dtype=mx.int32), cu_seqlens])
    return indices, cu_seqlens, max_seqlen_in_batch


def _make_causal_mask(
    input_ids_shape: Tuple[int, int],
    dtype: mx.Dtype,
    past_key_values_length: int = 0,
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = mx.full((tgt_len, tgt_len), -1e9, dtype=dtype)
    mask = mx.triu(mask, k=1)
    
    if past_key_values_length > 0:
        past_mask = mx.zeros((tgt_len, past_key_values_length), dtype=dtype)
        mask = mx.concatenate([past_mask, mask], axis=-1)
        
    mask = mx.expand_dims(mask, axis=[0, 1])  # (1, 1, tgt_len, tgt_len + past_len)
    return mask


def _expand_mask(mask: mx.array, dtype: mx.Dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.shape
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].astype(dtype)
    
    # Inverted mask: 1.0 where we want to attend, 0.0 where we don't.
    # Usually in transformers: 1 is keep, 0 is mask. 
    # And then (1.0 - mask) * min_value.
    # However, create_attention_mask in MLX often expects additive mask (0 for keep, -inf for mask).
    
    # If the input mask is 1 for keep, 0 for discard:
    inverted_mask = 1.0 - expanded_mask
    
    # If direct usage of min_dtype
    # Check what SarvamMoE expected: It uses _prepare_4d_attention_mask which returns additive mask.
    
    return inverted_mask * -1e9







@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    hidden_act: str = "silu"
    use_qkv_bias: bool = False
    use_bias: bool = False
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    embedding_dropout: float = 0.0
    attention_dropout: float = 0.0
    output_dropout: float = 0.0
    initializer_range: float = 0.02
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    use_cache: bool = True
    max_window_layers: int = 19
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    pad_token_id: int = 0
    eos_token_id: int = 1
    num_experts: int = 128
    num_shared_experts: int = 1
    num_experts_per_tok: int = 6
    n_group: int = 1
    topk_group: int = 1
    moe_intermediate_size: int = 1024
    first_k_dense_replace: int = 1
    head_dim: int = 256
    output_router_logits: bool = False
    use_qk_norm: bool = True
    moe_router_enable_expert_bias: bool = True
    routed_scaling_factor: float = 2.5
    attn_implementation: str = "eager"
    partial_rotary_factor: float = 0.5


class SarvamMoERotaryEmbedding(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.max_position_embeddings = args.max_position_embeddings
        self.rope_theta = args.rope_theta
        
        # Calculate rope dimension
        # Note: head_dim is usually set in args if not from hidden/heads
        dim = args.head_dim or (args.hidden_size // args.num_attention_heads)
        
        inv_freq = 1.0 / (
            self.rope_theta
            ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim)
        )
        self.inv_freq = inv_freq
        self.attention_scaling = 1.0

    def __call__(self, x: mx.array, position_ids: mx.array) -> Tuple[mx.array, mx.array]:
        # position_ids: (1, L) or (B, L)
        
        inv_freq_expanded = self.inv_freq[None, :, None] # (1, D/2, 1)
        position_ids_expanded = position_ids[:, None, :].astype(mx.float32) # (B, 1, L)
        
        # (1, D/2, 1) * (B, 1, L) -> (B, D/2, L)
        freqs = inv_freq_expanded * position_ids_expanded
        
        # Transpose to (B, L, D/2)
        freqs = freqs.transpose(0, 2, 1)
        
        # emb = cat(freqs, freqs) -> (B, L, D)
        emb = mx.concatenate((freqs, freqs), axis=-1)
        
        cos = mx.cos(emb) * self.attention_scaling
        sin = mx.sin(emb) * self.attention_scaling
        
        return cos, sin
    num_experts_per_tok: int = 6
    n_group: int = 1
    topk_group: int = 1
    moe_intermediate_size: int = 1024
    first_k_dense_replace: int = 1
    head_dim: Optional[int] = None
    output_router_logits: bool = False
    use_qk_norm: bool = True
    moe_router_enable_expert_bias: bool = True
    routed_scaling_factor: float = 2.5
    partial_rotary_factor: float = 0.5
    attn_implementation: str = "eager"

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.rope_scaling:
            if not isinstance(self.rope_scaling, dict):
                self.rope_scaling = None


class SarvamMoERMSNorm(nn.RMSNorm):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__(dims, eps)

# rotate_half logic
def rotate_half(x):
    D = x.shape[-1]
    x1 = x[..., : D // 2]
    x2 = x[..., D // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: (B, H, L, D_full) - full head dimension
    # cos, sin: (B, L, D_full) - full head dimension from SarvamMoERotaryEmbedding
    
    # Reshape for broadcasting: (B, L, D) -> (B, 1, L, D)
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    
    # Get rotary dimension from cos (matching reference behavior)
    rotary_dim = cos.shape[-1]
    
    # Split q and k into rotary and pass-through parts
    q_rot = q[..., :rotary_dim]
    q_pass = q[..., rotary_dim:]
    k_rot = k[..., :rotary_dim]
    k_pass = k[..., rotary_dim:]
        
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    
    # Concatenate back
    q_embed = mx.concatenate([q_embed, q_pass], axis=-1)
    k_embed = mx.concatenate([k_embed, k_pass], axis=-1)
    
    return q_embed, k_embed


def repeat_kv(x: mx.array, n_rep: int) -> mx.array:
    if n_rep == 1:
        return x
    B, n_kv_heads, L, D = x.shape
    x = mx.expand_dims(x, axis=2)
    x = mx.repeat(x, n_rep, axis=2)
    return x.reshape(B, n_kv_heads * n_rep, L, D)



class SarvamMoEMLP(nn.Module):
    def __init__(self, args: ModelArgs, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))

class SarvamMoEAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim or (dim // self.n_heads)
        self.scale = self.head_dim**-0.5
        self.partial_rotary_factor = args.partial_rotary_factor

        # Merged QKV projection
        self.query_key_value = nn.Linear(
            dim,
            (self.n_heads + 2 * self.n_kv_heads) * self.head_dim,
            bias=args.use_qkv_bias,
        )

        if args.use_qk_norm:
            self.query_layernorm = SarvamMoERMSNorm(self.head_dim, eps=args.rms_norm_eps)
            self.key_layernorm = SarvamMoERMSNorm(self.head_dim, eps=args.rms_norm_eps)
        else:
            self.query_layernorm = None
            self.key_layernorm = None

        self.dense = nn.Linear(self.n_heads * self.head_dim, dim, bias=args.use_bias)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_embeddings: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        qkv = self.query_key_value(x)
        # Split Q, K, V
        q_size = self.n_heads * self.head_dim
        kv_size = self.n_kv_heads * self.head_dim
        
        queries, keys, values = mx.split(
            qkv, [q_size, q_size + kv_size], axis=-1
        )
        
        # Transpose immediately to (B, H, L, D) for easier processing with RoPE and SDPA
        queries = queries.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if self.query_layernorm is not None:
            queries = self.query_layernorm(queries)
            keys = self.key_layernorm(keys)

        # Apply RoPE using position_embeddings if provided
        if position_embeddings is not None:
             cos, sin = position_embeddings
             queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        n_rep = self.n_heads // self.n_kv_heads
        keys = repeat_kv(keys, n_rep)
        values = repeat_kv(values, n_rep)

        # Output: (B, H, L, D)
        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        
        # Transpose back: (B, H, L, D) -> (B, L, H, D) -> (B, L, Hidden)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.dense(output)



class SarvamMoEGate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.num_experts = args.num_experts
        self.n_group = args.n_group
        self.topk_group = args.topk_group
        self.routed_scaling_factor = args.routed_scaling_factor
        
        # Use direct parameter for weight to match checkpoint structure (nn.Parameter in Torch)
        # Shape: (num_experts, hidden_size)
        scale = args.hidden_size ** -0.5
        self.weight = mx.random.uniform(
            low=-scale, high=scale,
            shape=(args.num_experts, args.hidden_size)
        )
        
        if args.moe_router_enable_expert_bias:
            self.expert_bias = mx.zeros((args.num_experts,))
        else:
            self.expert_bias = None

    def _topk(self, x: mx.array, k: int):
        inds = mx.argpartition(x, kth=-k, axis=-1)[..., -k:]
        vals = mx.take_along_axis(x, inds, axis=-1)
        # Sort by score desc
        order = mx.argsort(vals, axis=-1)[..., ::-1]
        inds = mx.take_along_axis(inds, order, axis=-1)
        vals = mx.take_along_axis(vals, order, axis=-1)
        return inds, vals

    def group_limited_topk(self, scores: mx.array):
        if self.n_group == 1:
            return self._topk(scores, self.top_k)
        
        num_tokens, num_experts = scores.shape
        group_scores = scores.reshape(num_tokens, self.n_group, -1)
        # Sum of top 2 in each group
        # mx.topk returns values directly (unsorted, but sum doesn't care)
        top2_vals = mx.topk(group_scores, k=2, axis=-1)
        group_score_sums = top2_vals.sum(axis=-1)
        
        # Select topk groups
        # We need indices here, so we use _topk (which wraps argpartition+sort)
        group_idx, _ = self._topk(group_score_sums, k=self.topk_group)
        
        # Create mask
        # Reference: group_mask.scatter_(1, group_idx, 1)
        # MLX supports scatter via advanced indexing
        group_mask = mx.zeros((num_tokens, self.n_group), dtype=scores.dtype)
        # [B, Kg] indices
        batch_col = mx.arange(num_tokens)[:, None]
        group_mask[batch_col, group_idx] = 1 # Set selected groups to 1
        
        # Expand mask to experts: [B, n_group] -> [B, n_group, 1] -> broadcast
        # experts_per_group implied by reshape
        experts_per_group = num_experts // self.n_group
        
        # We can reshape mask to [B, num_experts] by repeating elements?
        # group_mask: [B, G]. We want [B, G * E_per_G] where each expert in group G gets mask[g]
        score_mask = mx.repeat(group_mask[:, :, None], experts_per_group, axis=2)
        score_mask = score_mask.reshape(num_tokens, num_experts)
        
        # Apply mask
        neg_inf = -1e9
        masked_scores = mx.where(score_mask > 0, scores, neg_inf)
        
        # Final global topk
        return self._topk(masked_scores, self.top_k)

    def __call__(self, x: mx.array):
        # x: [B, L, H], weight: [E, H]
        # logits: [B, L, E] = x @ weight.T
        logits = x @ self.weight.T
        scores = mx.sigmoid(logits)
        
        scores_for_routing = scores
        if self.expert_bias is not None:
             scores_for_routing = scores_for_routing + self.expert_bias
        
        # Reshape for routing logic which typically expects 2D [Tokens, Experts] 
        # (though our implementation handles broadcasting, group_limited_topk assumes 2D)
        
        B, L, E = scores_for_routing.shape
        scores_flat = scores_for_routing.reshape(-1, E)
        
        # Routing
        inds_flat, _ = self.group_limited_topk(scores_flat)
        
        # Reshape back to [B, L, k]
        inds = inds_flat.reshape(B, L, self.top_k)
        
        # Gather weights from original scores (sigmoid, no bias)
        gathered_scores = mx.take_along_axis(scores, inds, axis=-1)
        
        if self.top_k > 1:
            denom = gathered_scores.sum(axis=-1, keepdims=True) + 1e-20
            topk_weight = gathered_scores / denom
        else:
            topk_weight = gathered_scores
            
        topk_weight = topk_weight * self.routed_scaling_factor
        
        return inds, topk_weight, logits


class SarvamMoEExperts(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # We use SwitchGLU for efficient MoE execution in MLX
        self.switch_mlp = SwitchGLU(
            args.hidden_size, 
            args.moe_intermediate_size, 
            args.num_experts, 
            bias=False
        )
        
    def __call__(self, x: mx.array, topk_inds: mx.array, topk_weights: mx.array) -> mx.array:
        # switch_mlp expects (x, indices) and returns expert outputs
        # We then need to weight them.
        
        # SwitchGLU in mlx_lm typically takes (x, indices). 
        # But wait, SwitchGLU returns the aggregated output if it handles the gathering?
        # Let's check SwitchGLU implementation via memory or assumption.
        # Usually: y = switch_mlp(x, inds) -> returns result for each token?
        # If it returns [B, L, TopK, Dim], we sum.
        # Checking previous sarvam_moe.py:
        # y = self.switch_mlp(x, inds)
        # y = (y * scores[..., None]).sum(axis=-2)
        
        y = self.switch_mlp(x, topk_inds)
        y = (y * topk_weights[..., None]).sum(axis=-2)
        return y


class SarvamMoESparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate = SarvamMoEGate(args)
        self.experts = SarvamMoEExperts(args)
        
        if args.num_shared_experts > 0:
            shared_inter_size = args.moe_intermediate_size * args.num_shared_experts
            self.shared_experts = SarvamMoEMLP(args, shared_inter_size)
        else:
            self.shared_experts = None

    def __call__(self, x: mx.array) -> mx.array:
        identity = x
        topk_inds, topk_weights, router_logits = self.gate(x)
        
        y = self.experts(x, topk_inds, topk_weights)
        
        if self.shared_experts is not None:
             y = y + self.shared_experts(identity)
             
        return y, router_logits


class SarvamMoEDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.attention = SarvamMoEAttention(args) # Renamed to match reference and checkpoint

        self.input_layernorm = SarvamMoERMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = SarvamMoERMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        first_k_dense = args.first_k_dense_replace
        
        # Condition from reference: (config.num_experts is not None and layer_idx >= config.first_k_dense_replace)
        is_moe = (args.num_experts > 0) and (layer_idx >= first_k_dense)
        
        if is_moe:
            self.mlp = SarvamMoESparseMoeBlock(args)
        else:
            self.mlp = SarvamMoEMLP(args, args.intermediate_size)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_embeddings: Optional[Tuple[mx.array, mx.array]] = None,
        output_attentions: bool = False,
        output_router_logits: bool = False,
        **kwargs,
    ) -> mx.array:
        r = self.attention(self.input_layernorm(x), mask, cache, position_embeddings)
        h = x + r
        
        r = self.mlp(self.post_attention_layernorm(h))
        
        router_logits = None
        if isinstance(r, tuple):
            r, rl = r
            if output_router_logits:
                router_logits = rl
            
        out = h + r
        return out, router_logits


class SarvamMoEModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            SarvamMoEDecoderLayer(args, i) for i in range(args.num_hidden_layers)
        ]
        self.norm = SarvamMoERMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.rotary_emb = SarvamMoERotaryEmbedding(args)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        output_router_logits: bool = False,
    ):
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = None
        if h.shape[1] > 1:
             # create mask for sequence using helpers
             if cache is not None and cache[0] is not None:
                 past_key_values_length = cache[0].offset
             else:
                 past_key_values_length = 0
             
             mask = _make_causal_mask(
                 h.shape[:2], 
                 dtype=h.dtype, 
                 past_key_values_length=past_key_values_length
             )
             # Expand to batch size if needed by broadcasting, but _make_causal_mask returns (1, 1, L, L) usually
             # or we might need to handle specific logic if we want to match exact torch behavior.
             # _make_causal_mask above creates (1, 1, L, L), so it broadcasts.

        all_router_logits = [] if output_router_logits else None

        # position_ids: (1, L)
        start = 0
        if cache and cache[0] is not None:
             start = cache[0].offset
        L = h.shape[1]
        position_ids = mx.arange(start, start + L).reshape(1, -1)
        
        cos, sin = self.rotary_emb(h, position_ids)
        position_embeddings = (cos, sin)

        for layer, c in zip(self.layers, cache):
            h, router_logits = layer(
                h, 
                mask, 
                c, 
                position_embeddings, 
                output_router_logits=output_router_logits
            )
            if output_router_logits and router_logits is not None:
                all_router_logits.append(router_logits)

        out = self.norm(h)
        
        if output_router_logits:
             return SarvamMoEModelOutputWithPast(
                 last_hidden_state=out,
                 past_key_values=cache,
                 router_logits=tuple(all_router_logits) if all_router_logits else None
             )

        return out


class SarvamMoEForCausalLM(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model = SarvamMoEModel(args)
        self.vocab_size = args.vocab_size
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
    
    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        output_router_logits: bool = False,
    ):
        outputs = self.model(inputs, cache, input_embeddings, output_router_logits=output_router_logits)
        
        if output_router_logits and isinstance(outputs, SarvamMoEModelOutputWithPast):
            hidden_state = outputs.last_hidden_state
            
            if self.args.tie_word_embeddings:
                logits = self.model.embed_tokens.as_linear(hidden_state)
            else:
                logits = self.lm_head(hidden_state)
            
            return SarvamMoECausalLMOutputWithPast(
                logits=logits,
                past_key_values=outputs.past_key_values,
                router_logits=outputs.router_logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        
        # Simple tensor output
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(outputs)
        else:
            out = self.lm_head(outputs)
        return out


class Model(SarvamMoEForCausalLM):
    def __init__(self, args: ModelArgs):
        super().__init__(args)
        self.model_type = args.model_type

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.head_dim or (self.args.hidden_size // self.args.num_attention_heads)

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads



    def sanitize(self, weights):
        # Remove unused keys (like FP8 scales) to avoid strict load errors
        keys_to_remove = [k for k in weights.keys() if "input_scale" in k or "weight_scale" in k]
        for k in keys_to_remove:
            weights.pop(k, None)

        # Add inv_freq if missing (it's often not in checkpoints but is a parameter in MLX RoPE)
        if "model.rotary_emb.inv_freq" not in weights:
            weights["model.rotary_emb.inv_freq"] = self.model.rotary_emb.inv_freq

        # Remove unused weights
        # Reference uses 'word_embeddings' but we use 'embed_tokens' in MLX standard
        # So we might need to map 'model.word_embeddings.weight' -> 'model.embed_tokens.weight'
        if "model.word_embeddings.weight" in weights:
            weights["model.embed_tokens.weight"] = weights.pop("model.word_embeddings.weight")

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        def split_qkv(qkv, n_heads, n_kv_heads, head_dim):
            # This logic was used when we had separate projections.
            # Now we use merged query_key_value, so we might NOT need to split if the weights are already merged.
            # If the weights come from HF, they are typically 'q_proj', 'k_proj', 'v_proj'.
            pass
        
        # Helper to merge QKV if they are separate in weights
        def merge_qkv(prefix):
            q = weights.get(f"{prefix}.q_proj.weight")
            k = weights.get(f"{prefix}.k_proj.weight")
            v = weights.get(f"{prefix}.v_proj.weight")
            if q is not None and k is not None and v is not None:
                # Remove originals
                del weights[f"{prefix}.q_proj.weight"]
                del weights[f"{prefix}.k_proj.weight"]
                del weights[f"{prefix}.v_proj.weight"]
                
                # Stack/Concatenate logic
                # HF definition: 
                # q: [n_heads * h, dim]
                # k: [n_kv * h, dim]
                # v: [n_kv * h, dim]
                # merged: split(..., [q, k, v] sizes)
                # So we just concat them along axis 0
                val = mx.concatenate([q, k, v], axis=0)
                weights[f"{prefix}.query_key_value.weight"] = val

        for layer_idx in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}"
            
            # Handle Attention
            # Use 'attention' as in reference/checkpoint. (Was 'self_attn')
            attn_prefix = f"{prefix}.attention"
            
            # If resulting weights use 'self_attn' (e.g. from some other conversion), mapping might be needed.
            # But error log says checkpoint has 'attention'.
            
            # If weights have q_proj, k_proj, v_proj, merge them into query_key_value
            merge_qkv(attn_prefix)
            
            # Handle MLP
            mlp_prefix = f"{prefix}.mlp"

            # Check if it is MoE or Dense
            # If Dense: gate_proj, up_proj, down_proj
            # If MoE: experts, gate, shared_experts
            
            # Shared Experts mapping
            # Reference: layers.*.mlp.shared_experts.gate_proj etc.
            # Our model: Same structure.
            
            # Experts mapping
            # We need to stack expert weights for SwitchGLU
            # experts.{e}.gate_proj.weight -> stack -> switch_mlp.gate_proj.weight
            
            # Check for expert weights
            if f"{mlp_prefix}.experts.0.gate_proj.weight" in weights:
                for n in ["gate_proj", "up_proj", "down_proj"]:
                    # Collect from experts 0..N
                    w_list = []
                    for e in range(self.args.num_experts):
                        key = f"{mlp_prefix}.experts.{e}.{n}.weight"
                        if key in weights:
                            w = weights.pop(key)
                            w_list.append(w)
                    
                    if w_list:
                        stacked = mx.stack(w_list)
                        # Assign to switch_mlp.{n}.weight
                        # Warning: SwitchGLU in mlx_lm.models.switch_layers expects specific names?
                        # Usually SwitchGLU has .gate_proj, .up_proj, .down_proj
                        weights[f"{mlp_prefix}.experts.switch_mlp.{n}.weight"] = stacked

            # Rename 'gate' weights if needed. 
            # Reference: mlp.gate.weight
            # Our code: mlp.gate.weight (SarvamMoEGate -> self.weight)
            # So naming is consistent.
            
            # Rename gate_up_proj if it exists (some formats)
            # But the provided reference sarvam_moe_transformers.py shows separate gate_proj and up_proj.

        return weights


