# Copyright © 2025 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import sum_gradients

from mlx_lm.layers.activations import SwigluOAI
from mlx_lm.layers.attention import scaled_dot_product_attention
from mlx_lm.layers.norms import ZeroCenteredRMSNorm
from mlx_lm.layers.rope import initialize_rope
from mlx_lm.layers.switch import SwitchGLU
from mlx_lm.models.minimax_m3_vl.config import TextArgs

__all__ = [
    "Attention",
    "DecoderLayer",
    "DenseMLP",
    "SparseMoeBlock",
]


class DenseMLP(nn.Module):
    def __init__(self, hidden: int, inter: int, *, activation):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)
        self.activation = activation

    def __call__(self, x: mx.array) -> mx.array:
        x_up = self.up_proj(x)
        x_gate = self.gate_proj(x)
        return self.down_proj(self.activation(x_up, x_gate))


class SparseMoeBlock(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.num_experts_per_tok = args.num_experts_per_tok
        self.routed_scaling_factor = args.routed_scaling_factor

        self.gate = nn.Linear(args.hidden_size, args.num_local_experts, bias=False)
        self.e_score_correction_bias = mx.zeros((args.num_local_experts,))

        activation = SwigluOAI(alpha=args.swiglu_alpha, limit=args.swiglu_limit)
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.intermediate_size,
            args.num_local_experts,
            activation=activation,
        )

        self.shared_experts = DenseMLP(
            args.hidden_size,
            args.shared_intermediate_size,
            activation=activation,
        )

        self.sharding_group = None

    def __call__(self, x: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)

        ne = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])

        shared_out = self.shared_experts(x_flat)

        # The reference routes in the weight dtype and upcasts only the logits.
        gates = self.gate(x_flat)
        scores = mx.sigmoid(gates.astype(mx.float32))
        orig_scores = scores
        scores = scores + self.e_score_correction_bias

        k = self.num_experts_per_tok
        inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
        inds = mx.stop_gradient(inds)
        scores = mx.take_along_axis(orig_scores, inds, axis=-1)
        scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-20)
        scores = (scores * self.routed_scaling_factor).astype(x_flat.dtype)

        y = self.switch_mlp(x_flat, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        y = y + shared_out

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)

        return y.reshape(*ne, -1)


class Attention(nn.Module):
    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__()
        self.hidden_dim = args.hidden_size
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        NormClass = ZeroCenteredRMSNorm if args.use_gemma_norm else nn.RMSNorm

        self.q_proj = nn.Linear(
            args.hidden_size, self.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim, args.hidden_size, bias=False
        )

        self.use_qk_norm = args.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = NormClass(self.head_dim, eps=args.rms_norm_eps)
            self.k_norm = NormClass(self.head_dim, eps=args.rms_norm_eps)

        self.rope = initialize_rope(
            dims=args.rotary_dim,
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_parameters,
            max_position_embeddings=args.max_position_embeddings,
        )

        # The indexer picks this layer's key blocks; its weights sit on self_attn.
        self.is_sparse_attn = args.is_sparse_attn(layer_idx)
        if self.is_sparse_attn:
            self.index_n_heads = args.index_n_heads
            self.index_head_dim = args.index_head_dim
            self.index_block_size = args.index_block_size
            self.index_topk_blocks = args.index_topk_blocks
            self.index_local_blocks = args.index_local_blocks
            self.index_q_proj = nn.Linear(
                args.hidden_size, self.index_n_heads * self.index_head_dim, bias=False
            )
            self.index_k_proj = nn.Linear(
                args.hidden_size, self.index_head_dim, bias=False
            )
            self.index_q_norm = NormClass(self.index_head_dim, eps=args.rms_norm_eps)
            self.index_k_norm = NormClass(self.index_head_dim, eps=args.rms_norm_eps)
            # The reference truncates cos/sin to the index head.
            self.index_rope = initialize_rope(
                dims=min(args.rotary_dim, self.index_head_dim),
                base=args.rope_theta,
                traditional=False,
                scaling_config=args.rope_parameters,
                max_position_embeddings=args.max_position_embeddings,
            )

    def _block_mask(
        self,
        x: mx.array,
        mask: Optional[mx.array],
        offset: int,
        cache: Optional[Any],
    ) -> Optional[mx.array]:
        """Bool mask keeping only the top-k key blocks per query."""
        B, L, _ = x.shape
        H, D = self.index_n_heads, self.index_head_dim

        # Cache the index keys even when no block is dropped: later steps need them.
        k = self.index_k_proj(x).reshape(B, L, 1, D)
        k = self.index_k_norm(k).transpose(0, 2, 1, 3)
        k = self.index_rope(k, offset=offset)
        if cache is not None:
            k, _ = cache.update_and_fetch(k, mx.zeros((B, 1, L, 0), k.dtype))

        block_size = self.index_block_size
        topk = self.index_topk_blocks
        S = k.shape[2]
        full_blocks, tail_keys = divmod(S, block_size)
        n_blocks = full_blocks + bool(tail_keys)
        # Every block fits the budget, so nothing would be dropped.
        if topk >= n_blocks:
            return mask

        q = self.index_q_proj(x).reshape(B, L, H, D)
        q = self.index_q_norm(q).transpose(0, 2, 1, 3)
        q = self.index_rope(q, offset=offset)

        # Blocks are cut over cache slots, so place queries by slot, not content.
        qpos = mx.arange(S - L, S)
        causal = mx.arange(S)[None] <= qpos[:, None]
        scores = q.astype(mx.float32) @ k.astype(mx.float32).swapaxes(-1, -2)
        # A future key must not win a block, so mask before pooling.
        per_key = mask if isinstance(mask, mx.array) else causal
        scores = mx.where(per_key, scores, -mx.inf)
        split = full_blocks * block_size
        block_scores = scores[..., :split].reshape(B, H, L, full_blocks, block_size)
        block_scores = block_scores.max(axis=-1)
        if tail_keys:
            tail_score = scores[..., split:].max(axis=-1, keepdims=True)
            block_scores = mx.concatenate([block_scores, tail_score], axis=-1)
        if self.index_local_blocks > 0:
            q_block = (qpos // block_size)[:, None]
            ids = mx.arange(n_blocks)
            is_local = (ids <= q_block) & (ids > q_block - self.index_local_blocks)
            block_scores = mx.where(is_local, mx.inf, block_scores)

        inds = mx.argpartition(-block_scores, kth=topk - 1, axis=-1)[..., :topk]
        inds = mx.stop_gradient(inds)
        keep = mx.put_along_axis(
            mx.zeros(block_scores.shape, dtype=mx.bool_),
            inds,
            mx.array(True),
            axis=-1,
        )
        # A kept block still holds future keys, so mask again.
        keep = mx.repeat(keep, block_size, axis=-1)[..., :S] & per_key
        # One selection per KV group, so widen it to every query head in the group.
        return mx.repeat(keep, self.num_attention_heads // H, axis=1)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        kv_cache = cache[0] if cache is not None else None
        offset = kv_cache.offset if kv_cache is not None else 0

        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = q.reshape(B, L, self.num_attention_heads, self.head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_key_value_heads, self.head_dim)
        k = k.transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_key_value_heads, self.head_dim)
        v = v.transpose(0, 2, 1, 3)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        # Select before the cache advances: it moves its offset array in place.
        if self.is_sparse_attn:
            mask = self._block_mask(
                x, mask, offset, cache[1] if cache is not None else None
            )
        if kv_cache is not None:
            k, v = kv_cache.update_and_fetch(k, v)
            # Keep the indexer cache in the graph: below the block budget nothing
            # consumes its keys, and nothing ever consumes its values.
            # Metal buffers are exhausted on long generations.
            if self.is_sparse_attn:
                index_kv = [cache[1].keys, cache[1].values]
                kv_cache.keys = mx.depends(kv_cache.keys, index_kv)

        out = scaled_dot_product_attention(
            q, k, v, cache=kv_cache, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class DecoderLayer(nn.Module):
    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args, layer_idx)

        self.is_moe = args.is_moe(layer_idx)
        if self.is_moe:
            self.block_sparse_moe = SparseMoeBlock(args)
        else:
            activation = SwigluOAI(alpha=args.swiglu_alpha, limit=args.swiglu_limit)
            self.mlp = DenseMLP(
                args.hidden_size,
                args.dense_intermediate_size,
                activation=activation,
            )

        NormClass = ZeroCenteredRMSNorm if args.use_gemma_norm else nn.RMSNorm
        self.input_layernorm = NormClass(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = NormClass(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = x + self.self_attn(self.input_layernorm(x), mask, cache)
        if self.is_moe:
            r = r + self.block_sparse_moe(self.post_attention_layernorm(r))
        else:
            r = r + self.mlp(self.post_attention_layernorm(r))
        return r
