# Copyright © 2023-2024 Apple Inc.

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_linear

from .base import BaseModelArgs, scaled_dot_product_attention
from .switch_layers import SwitchGLU


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
    head_dim: Optional[int] = None
    output_router_logits: bool = False
    use_qk_norm: bool = True
    moe_router_enable_expert_bias: bool = True
    routed_scaling_factor: float = 2.5
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


class SarvamMoEAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim or (dim // self.n_heads)
        self.scale = self.head_dim**-0.5

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

        self.rope = nn.RoPE(
            self.head_dim,
            traditional=False,
            base=args.rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        qkv = self.query_key_value(x)
        # Split Q, K, V
        # qkv shape: [B, L, (n_heads + 2*n_kv_heads) * head_dim]
        # We need to split carefully.
        # Construct splits for split_einsum-like behavior or manual slice.
        # Since merged, the layout typically assumes concatenation along the last dim.
        
        q_size = self.n_heads * self.head_dim
        kv_size = self.n_kv_heads * self.head_dim
        
        queries, keys, values = mx.split(
            qkv, [q_size, q_size + kv_size], axis=-1
        )
        
        queries = queries.reshape(B, L, self.n_heads, self.head_dim)
        keys = keys.reshape(B, L, self.n_kv_heads, self.head_dim)
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim)

        if self.query_layernorm is not None:
            queries = self.query_layernorm(queries)
            keys = self.key_layernorm(keys)

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
        output = output.reshape(B, L, -1)
        return self.dense(output)


class SarvamMoEMLP(nn.Module):
    def __init__(self, args: ModelArgs, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


def _topk_lastdim(x: mx.array, k: int):
    inds = mx.argpartition(x, kth=-k, axis=-1)[..., -k:]
    vals = mx.take_along_axis(x, inds, axis=-1)
    # Sort by score desc
    order = mx.argsort(vals, axis=-1)[..., ::-1]
    inds = mx.take_along_axis(inds, order, axis=-1)
    vals = mx.take_along_axis(vals, order, axis=-1)
    return inds, vals


def _grouped_topk(scores: mx.array, top_k: int, n_group: int, topk_group: int):
    # scores: [..., E]
    *prefix, E = scores.shape
    assert E % n_group == 0
    gsz = E // n_group
    
    s = scores.reshape(*prefix, n_group, gsz)
    
    # topk within each group
    gi, gv = _topk_lastdim(s, topk_group) # [..., G, topk_group]
    
    # map to global expert ids
    group_offsets = mx.arange(n_group).reshape(*([1] * len(prefix)), n_group, 1) * gsz
    gi_global = gi + group_offsets
    
    # flatten candidates
    cand_inds = gi_global.reshape(*prefix, n_group * topk_group)
    cand_vals = gv.reshape(*prefix, n_group * topk_group)
    
    # pick final top_k
    ci, cv = _topk_lastdim(cand_vals, top_k)
    final_inds = mx.take_along_axis(cand_inds, ci, axis=-1)
    final_vals = cv
    return final_inds, final_vals


class SarvamMoEGate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.n_group = args.n_group
        self.topk_group = args.topk_group
        self.routed_scaling_factor = args.routed_scaling_factor
        
        self.weight = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        
        if args.moe_router_enable_expert_bias:
            self.expert_bias = mx.zeros((args.num_experts,))
        else:
            self.expert_bias = None

    def __call__(self, x: mx.array):
        logits = self.weight(x)
        scores = mx.sigmoid(logits)
        
        scores_for_routing = scores
        if self.expert_bias is not None:
             scores_for_routing = scores_for_routing + self.expert_bias

        if self.n_group > 1:
            inds, final_scores = _grouped_topk(scores_for_routing, self.top_k, self.n_group, self.topk_group)
            
            # Re-gather original scores (without bias? Reference uses gathered scores)
            # Reference: scores = torch.gather(scores, dim=1, index=topk_idx)
            # So we use the 'inds' to gather from the RAW 'scores' (sigmoid output)
            gathered_scores = mx.take_along_axis(scores, inds, axis=-1)
            
        else:
            # Standard top-k
            inds, gathered_scores = _topk_lastdim(scores_for_routing, self.top_k)
            # If standard, we also want to gather from original scores if correct?
            # Reference actually uses scores_for_routing for topk logic, but gathers from 'scores'.
            gathered_scores = mx.take_along_axis(scores, inds, axis=-1)

        # Normalize logic from reference:
        # topk_weight = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if self.top_k > 1 else scores
        # topk_weight = topk_weight * self.routed_scaling_factor
        
        if self.top_k > 1:
            denom = gathered_scores.sum(axis=-1, keepdims=True) + 1e-20
            topk_weight = gathered_scores / denom
        else:
            topk_weight = gathered_scores
            
        topk_weight = topk_weight * self.routed_scaling_factor
        
        return inds, topk_weight


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
        topk_inds, topk_weights = self.gate(x)
        
        y = self.experts(x, topk_inds, topk_weights)
        
        if self.shared_experts is not None:
             y = y + self.shared_experts(identity)
             
        return y


class SarvamMoEDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.self_attn = SarvamMoEAttention(args) # Renamed from Attention
        
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
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


class SarvamMoEModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            SarvamMoEDecoderLayer(args, i) for i in range(args.num_hidden_layers)
        ]
        self.norm = SarvamMoERMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = None
        if h.shape[1] > 1:
             # create mask for sequence
             # We can rely on scaled_dot_product_attention to handle causal masking if mask is None?
             # mlx.nn.layers.base.create_attention_mask is usually used.
             from .base import create_attention_mask
             mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = SarvamMoEModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights):
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

        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"
            
            # Handle Attention
            # Use 'self_attn' as in reference
            attn_prefix = f"{prefix}.self_attn"
            
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

    @property
    def layers(self):
        return self.model.layers
