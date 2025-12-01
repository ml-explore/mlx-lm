# Copyright © 2024 Apple Inc.

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .pipeline import PipelineMixin
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "deepseek_v3"
    vocab_size: int = 102400
    hidden_size: int = 4096
    intermediate_size: int = 11008
    moe_intermediate_size: int = 1407
    num_hidden_layers: int = 30
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    n_shared_experts: Optional[int] = None
    n_routed_experts: Optional[int] = None
    routed_scaling_factor: float = 1.0
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    qk_nope_head_dim: int = 128
    topk_method: str = "noaux_tc"
    scoring_func: str = "sigmoid"
    norm_topk_prob: bool = True
    n_group: int = 1
    topk_group: int = 1
    num_experts_per_tok: int = 1
    moe_layer_freq: int = 1
    first_k_dense_replace: int = 0
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: Dict = None
    attention_bias: bool = False
    # Indexer parameters
    use_indexer: bool = True
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 2048


def yarn_find_correction_dim(
    num_rotations, dim, base=10000, max_position_embeddings=2048
):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def yarn_find_correction_range(
    low_rot, high_rot, dim, base=10000, max_position_embeddings=2048
):
    low = math.floor(
        yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(min_val, max_val, dim):
    if min_val == max_val:
        max_val += 0.001  # Prevent singularity

    linear_func = (mx.arange(dim, dtype=mx.float32) - min_val) / (max_val - min_val)
    return mx.clip(linear_func, 0, 1)


class DeepseekV3YarnRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        scaling_factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1,
        mscale_all_dim=0,
    ):
        super().__init__()
        self.mscale = yarn_get_mscale(scaling_factor, mscale) / yarn_get_mscale(
            scaling_factor, mscale_all_dim
        )
        freq_extra = base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim)
        freq_inter = scaling_factor * freq_extra
        low, high = yarn_find_correction_range(
            beta_fast,
            beta_slow,
            dim,
            base,
            original_max_position_embeddings,
        )
        freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2)
        self._freqs = (freq_inter * freq_extra) / (
            freq_inter * freq_mask + freq_extra * (1 - freq_mask)
        )

    def __call__(self, x, offset=0):
        if self.mscale != 1.0:
            x = self.mscale * x
        return mx.fast.rope(
            x,
            x.shape[-1],
            traditional=True,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )


class Indexer(nn.Module):
    """
    Indexer module for selecting top-k positions in attention.
    This implements a fast attention indexing mechanism.
    """
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.q_lora_rank = config.q_lora_rank
        
        self.wq_b = nn.Linear(self.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        
        self.softmax_scale = self.head_dim ** -0.5
        
        # Use standard RoPE for indexer (non-interleaved in PyTorch version)
        self.rope = nn.RoPE(
            dims=self.rope_head_dim,
            base=config.rope_theta,
            traditional=False,  # Non-interleaved
        )

    def __call__(
        self,
        x: mx.array,
        qr: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape
        
        # Query projection
        q = self.wq_b(qr)
        q = q.reshape(B, L, self.num_heads, self.head_dim)
        q_pe, q_nope = mx.split(q, [self.rope_head_dim], axis=-1)
        
        # Apply RoPE to query (non-interleaved)
        if cache is not None:
            q_pe = self.rope(q_pe, cache.offset)
        else:
            q_pe = self.rope(q_pe)
        
        q = mx.concatenate([q_pe, q_nope], axis=-1)
        
        # Key projection
        k = self.wk(x)
        k = self.k_norm(k)
        k_pe, k_nope = mx.split(k, [self.rope_head_dim], axis=-1)
        
        # Apply RoPE to key (non-interleaved)
        k_pe_expanded = k_pe.reshape(B, L, 1, self.rope_head_dim)
        if cache is not None:
            k_pe_expanded = self.rope(k_pe_expanded, cache.offset)
        else:
            k_pe_expanded = self.rope(k_pe_expanded)
        k_pe = k_pe_expanded.reshape(B, L, self.rope_head_dim)
        
        k = mx.concatenate([k_pe, k_nope], axis=-1)
        
        # Compute attention weights for indexing
        # Shape: (B, L, num_heads, head_dim)
        q = q.transpose(0, 2, 1, 3)  # (B, num_heads, L, head_dim)
        
        # Compute weights projection
        weights = self.weights_proj(x)  # (B, L, num_heads)
        weights = weights * (self.num_heads ** -0.5)
        
        # In the reference implementation, they cache keys and compute scores
        # For MLX, we'll compute attention scores directly
        if cache is not None:
            # Update cache with new keys
            k_expanded = k.reshape(B, L, 1, self.head_dim)
            keys = cache.update_and_fetch(k_expanded, k_expanded)[0]
            keys = keys.squeeze(2)  # (B, seq_len, head_dim)
        else:
            keys = k
        
        # Compute index scores: (B, num_heads, L, seq_len)
        keys_expanded = keys[:, None, :, :]  # (B, 1, seq_len, head_dim)
        index_scores = (q @ keys_expanded.transpose(0, 1, 3, 2)) * self.softmax_scale
        
        # Apply weights
        index_scores = index_scores * weights[:, None, :, None]
        
        # Average over query positions and heads
        index_scores = index_scores.mean(axis=(1, 2))  # (B, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            # Mask should be (B, L, seq_len) or broadcastable
            index_scores = index_scores + mask
        
        # Select top-k indices
        seq_len = keys.shape[1]
        k = min(self.index_topk, seq_len)
        topk_indices = mx.argpartition(-index_scores, kth=k-1, axis=-1)[..., :k]
        
        return topk_indices


class DeepseekV3Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.use_indexer = config.use_indexer

        self.scale = self.q_head_dim**-0.5

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                self.hidden_size, self.num_heads * self.q_head_dim, bias=False
            )
        else:
            self.q_a_proj = nn.Linear(
                self.hidden_size, self.q_lora_rank, bias=config.attention_bias
            )
            self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=1e-6)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=1e-6)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads
            * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )

        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.scale = self.scale * mscale * mscale

            rope_kwargs = {
                key: self.config.rope_scaling[key]
                for key in [
                    "original_max_position_embeddings",
                    "beta_fast",
                    "beta_slow",
                    "mscale",
                    "mscale_all_dim",
                ]
                if key in self.config.rope_scaling
            }
            self.rope = DeepseekV3YarnRotaryEmbedding(
                dim=self.qk_rope_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                scaling_factor=scaling_factor,
                base=self.rope_theta,
                **rope_kwargs,
            )
        else:
            self.rope = nn.RoPE(
                dims=self.qk_rope_head_dim,
                base=self.rope_theta,
                traditional=True,
            )
        
        # Initialize indexer if enabled
        if self.use_indexer:
            self.indexer = Indexer(config)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        if self.q_lora_rank is None:
            q = self.q_proj(x)
            qr = None
        else:
            qr = self.q_a_layernorm(self.q_a_proj(x))
            q = self.q_b_proj(qr)

        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)
        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        kv = kv.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)

        k_nope, values = mx.split(kv, [self.qk_nope_head_dim], axis=-1)

        if cache is not None:
            q_pe = self.rope(q_pe, cache.offset)
            k_pe = self.rope(k_pe, cache.offset)
            k_pe = mx.repeat(k_pe, self.num_heads, axis=1)
            keys, values = cache.update_and_fetch(
                mx.concatenate([k_nope, k_pe], axis=-1), values
            )
        else:
            q_pe = self.rope(q_pe)
            k_pe = self.rope(k_pe)
            k_pe = mx.repeat(k_pe, self.num_heads, axis=1)
            keys = mx.concatenate([k_nope, k_pe], axis=-1)

        queries = mx.concatenate([q_nope, q_pe], axis=-1)

        # Apply indexer if enabled and we have qr (for LoRA-based queries)
        index_mask = None
        if self.use_indexer and qr is not None and mask is not None:
            # Get top-k indices from indexer
            topk_indices = self.indexer(x, qr, mask, cache)
            
            # Create index mask: set non-selected positions to -inf
            seq_len = keys.shape[2]
            index_mask = mx.full((B, L, seq_len), float("-inf"))
            
            # Scatter zero values at top-k positions
            # This is a simplified approach; in practice, you'd need a more efficient scatter
            for b in range(B):
                for i in range(L):
                    indices = topk_indices[b]
                    index_mask[b, i, indices] = 0.0
            
            # Combine with original mask
            if mask is not None:
                index_mask = index_mask + mask.reshape(B, L, seq_len)
            
            # Reshape for attention
            index_mask = index_mask[:, None, :, :]  # (B, 1, L, seq_len)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, 
            mask=index_mask if index_mask is not None else mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class DeepseekV3MLP(nn.Module):
    def __init__(
        self, config: ModelArgs, hidden_size: int = None, intermediate_size: int = None
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def __call__(self, x):
        down_proj = self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


@mx.compile
def group_expert_select(
    gates,
    e_score_correction_bias,
    top_k,
    n_group,
    topk_group,
    routed_scaling_factor,
    norm_topk_prob,
):

    scores = mx.sigmoid(gates.astype(mx.float32))
    orig_scores = scores
    scores = scores + e_score_correction_bias
    if n_group > 1:
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        scores = mx.put_along_axis(
            scores, mx.stop_gradient(group_idx), mx.array(0.0), axis=-2
        )
        scores = mx.flatten(scores, -2, -1)

    k = top_k
    inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)
    if top_k > 1 and norm_topk_prob:
        denominator = scores.sum(axis=-1, keepdims=True)
        scores = scores / denominator
    scores = scores * routed_scaling_factor

    return inds, scores


class MoEGate(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.weight = mx.zeros((self.n_routed_experts, config.hidden_size))
        self.e_score_correction_bias = mx.zeros((self.n_routed_experts,))
        assert config.topk_method == "noaux_tc", "Unsupported topk method."

    def __call__(self, x):
        return group_expert_select(
            x @ self.weight.T,
            self.e_score_correction_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )


class DeepseekV3MoE(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            config.moe_intermediate_size,
            config.n_routed_experts,
        )

        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV3MLP(
                config=config, intermediate_size=intermediate_size
            )

    def __call__(self, x):
        inds, scores = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(x)

        return y


class DeepseekV3DecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = DeepseekV3Attention(config)
        self.mlp = (
            DeepseekV3MoE(config)
            if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0
            )
            else DeepseekV3MLP(config)
        )
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


class DeepseekV3Model(PipelineMixin, nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            DeepseekV3DecoderLayer(config, idx)
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
        mask = create_attention_mask(h, cache[0])

        # Receive from the previous process in the pipeline
        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        for l, c in zip(self.pipeline_layers, cache):
            h = l(h, mask, cache=c)

        # Send to the next process in the pipeline
        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            if cache[-1] is not None:
                cache[-1].keys = mx.depends(cache[-1].keys, h)

        # Broadcast h while keeping it in the graph
        h = mx.distributed.all_gather(h)[: h.shape[0]]

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = DeepseekV3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ):
        out = self.model(inputs, cache)
        return self.lm_head(out)

    def sanitize(self, weights):
        def dequant(weight, scale_inv):
            dtype = weight.dtype
            bs = 128  # block size
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

        # Stack experts
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

        # Remove multi-token prediction layer and any unused precomputed rotary freqs
        return {
            k: v
            for k, v in weights.items()
            if not k.startswith("model.layers.61") and "rotary_emb.inv_freq" not in k
        }

    @property
    def layers(self):
        return self.model.pipeline_layers

    @property
    def cast_predicate(self):
        def predicate(k):
            return "e_score_correction_bias" not in k

        return predicate