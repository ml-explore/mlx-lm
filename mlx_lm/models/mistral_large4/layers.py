# Copyright © 2026 Apple Inc.

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import sum_gradients

from mlx_lm.layers.attention import MultiLinear, scaled_dot_product_attention
from mlx_lm.layers.moe import QuantizedSwitchLinear, SwitchGLU, SwitchLinear
from mlx_lm.layers.rope import YarnRoPE
from mlx_lm.models.deepseek_v3.layers import DeepseekV3MLP
from mlx_lm.models.mistral4.layers import mistral4_expert_select
from mlx_lm.models.mistral_large4.config import TextArgs

__all__ = [
    "Attention",
    "DecoderLayer",
    "Fp8Linear",
    "Fp8SwitchLinear",
    "Indexer",
    "LeRoPE",
    "MoE",
    "SoftmaxScaling",
]


def _mxfp8_dequantize(weight, scales):
    return mx.dequantize(
        weight, scales, group_size=32, bits=8, mode="mxfp8", dtype=mx.bfloat16
    )


class Fp8Linear(nn.QuantizedLinear):
    """FP8 checkpoint weights, run as mxfp8.

    Unlike other quantized layers, ``convert -q`` quantizes it again, since a
    bf16 copy of this model does not fit in memory or on disk.
    """

    def __init__(self, input_dims: int, output_dims: int):
        super().__init__(input_dims, output_dims, False, 32, 8, "mxfp8")

    def to_quantized(self, group_size=None, bits=None, mode="affine", **kwargs):
        out_dims, packed = self.weight.shape
        linear = nn.Linear(packed * 4, out_dims, bias=False)
        linear.weight = _mxfp8_dequantize(self.weight, self.scales)
        return linear.to_quantized(
            group_size=group_size, bits=bits, mode=mode, **kwargs
        )


class Fp8SwitchLinear(QuantizedSwitchLinear):
    """FP8 checkpoint experts, run as mxfp8. See Fp8Linear."""

    def __init__(self, input_dims: int, output_dims: int, num_experts: int):
        super().__init__(input_dims, output_dims, num_experts, False, 32, 8, "mxfp8")

    def to_quantized(self, group_size=64, bits=4, mode="affine", **kwargs):
        num_experts, out_dims, packed = self.weight.shape
        linear = SwitchLinear(packed * 4, out_dims, num_experts, bias=False)
        linear.weight = _mxfp8_dequantize(self.weight, self.scales)
        return linear.to_quantized(
            group_size=group_size, bits=bits, mode=mode, **kwargs
        )


class LeRoPE(YarnRoPE):
    """Interleaved YaRN RoPE with a learned scale for each frequency."""

    def __init__(self, dims: int, rope_parameters: dict):
        if rope_parameters["rope_type"] != "yarn":
            raise ValueError("Only yarn RoPE is supported.")
        if rope_parameters["attention_factor"] != 1.0:
            raise ValueError("Only yarn without magnitude scaling is supported.")
        super().__init__(
            dims,
            traditional=True,
            base=rope_parameters["rope_theta"],
            scaling_factor=rope_parameters["factor"],
            original_max_position_embeddings=rope_parameters[
                "original_max_position_embeddings"
            ],
            beta_fast=rope_parameters["beta_fast"],
            beta_slow=rope_parameters["beta_slow"],
            mscale=0,
        )
        self.alpha = mx.zeros((dims // 2,))

    def freqs(self):
        # exp(alpha) scales the inverse frequencies, so it divides the wavelengths.
        return self._freqs * mx.exp(-self.alpha.astype(mx.float32))


class SoftmaxScaling(nn.Module):
    """Per-head query scale s * log(position + 1) + b."""

    def __init__(self, num_heads: int):
        super().__init__()
        self.scale = mx.zeros((num_heads,))
        self.bias = mx.ones((num_heads,))

    def __call__(self, q, log_pos):
        # q is [B, H, L, D] and log_pos is [L] or [B, L].
        scale = self.scale[:, None] * log_pos[..., None, :] + self.bias[:, None]
        return q * scale[..., None].astype(q.dtype)


class Indexer(nn.Module):
    def __init__(self, args: TextArgs, use_rope: bool):
        super().__init__()
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.index_topk = args.index_topk
        self.use_rope = use_rope
        # No q-LoRA, so the queries come from the hidden states.
        self.q_input_norm = nn.LayerNorm(args.hidden_size, eps=1e-6)
        self.wq_b = nn.Linear(
            args.hidden_size, self.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(args.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(args.hidden_size, self.n_heads, bias=False)
        self.softmax_scaling = SoftmaxScaling(self.n_heads)

    def __call__(self, x, mask, rope, log_pos, cache=None, block=4096):
        B, L, _ = x.shape
        q = self.wq_b(self.q_input_norm(x))
        q = q.reshape(B, L, self.n_heads, self.head_dim).swapaxes(1, 2)
        q = self.softmax_scaling(q, log_pos)
        k = self.k_norm(self.wk(x)).reshape(B, 1, L, self.head_dim)
        if self.use_rope:
            # The rope dims come first in the indexer heads.
            q = rope(q)
            k = rope(k)
        if cache is not None:
            k, _ = cache.update_and_fetch(k, mx.zeros((B, 1, L, 0), k.dtype))
        N = k.shape[2]
        if N <= self.index_topk:
            return None

        w = self.weights_proj(x.astype(mx.float32))
        w = (w * (self.n_heads**-0.5 * self.head_dim**-0.5))[:, :, None, :]
        q = q.swapaxes(1, 2).astype(mx.float32)  # [B, L, H, D]
        k = k.astype(mx.float32)

        best_v = best_i = None
        for s0 in range(0, N, block):
            s1 = min(N, s0 + block)
            sc = q @ k[:, :, s0:s1].swapaxes(-1, -2)  # [B, L, H, blk]
            sc = (w @ mx.maximum(sc, 0)).swapaxes(1, 2)
            if mask is not None:
                sc = mx.where(mask[..., s0:s1], sc, -mx.inf)
            ids = mx.broadcast_to(mx.arange(s0, s1), sc.shape)
            if best_v is not None:
                sc = mx.concatenate([best_v, sc], axis=-1)
                ids = mx.concatenate([best_i, ids], axis=-1)
            if sc.shape[-1] > self.index_topk:
                pos = mx.argpartition(sc, kth=-self.index_topk, axis=-1)
                pos = pos[..., -self.index_topk :]
                sc = mx.take_along_axis(sc, pos, axis=-1)
                ids = mx.take_along_axis(ids, pos, axis=-1)
            best_v, best_i = sc, ids
        return mx.stop_gradient(best_i)  # [B, 1, L, topk]


class Attention(nn.Module):
    """Absorbed MLA with DSA top-k selection, NoPE layers and KV sharing."""

    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = args.num_attention_heads
        self.use_rope = bool(args.no_rope_layers[layer_idx])
        self.kv_source = args.kv_source_layers[layer_idx]
        self.owns_kv = self.kv_source == layer_idx
        self.kv_lora_rank = args.kv_lora_rank
        self.q_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        # NoPE layers use the rope dims as nope dims.
        self.rope_dim = args.qk_rope_head_dim if self.use_rope else 0
        self.nope_dim = self.q_head_dim - self.rope_dim
        self.scale = self.q_head_dim**-0.5

        self.q_proj = nn.Linear(
            args.hidden_size, self.num_heads * self.q_head_dim, bias=False
        )
        if self.owns_kv:
            self.kv_a_proj_with_mqa = nn.Linear(
                args.hidden_size, self.kv_lora_rank + self.rope_dim, bias=False
            )
            self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=args.rms_norm_eps)
        # kv_b_proj is absorbed, so the cache holds the compressed latent.
        self.embed_q = MultiLinear(self.nope_dim, self.kv_lora_rank, self.num_heads)
        self.unembed_out = MultiLinear(
            self.kv_lora_rank, args.v_head_dim, self.num_heads
        )
        self.o_proj = nn.Linear(
            self.num_heads * args.v_head_dim, args.hidden_size, bias=False
        )
        self.softmax_scaling = SoftmaxScaling(self.num_heads)
        if args.indexer_types[layer_idx] == "full":
            self.indexer = Indexer(args, self.use_rope)
        else:
            self.indexer = None

    def __call__(self, x, mask, rope, log_pos, cache, shared_kv, prev_topk):
        B, L, _ = x.shape
        has_indexer = self.indexer is not None
        mla_cache = cache[0] if cache is not None and self.owns_kv else None
        index_cache = None
        if cache is not None and has_indexer:
            index_cache = cache[1 if self.owns_kv else 0]

        q = self.q_proj(x).reshape(B, L, self.num_heads, self.q_head_dim)
        q = self.softmax_scaling(q.transpose(0, 2, 1, 3), log_pos)
        q_nope, q_pe = mx.split(q, [self.nope_dim], axis=-1)
        if self.use_rope:
            q_pe = rope(q_pe)

        if self.owns_kv:
            kv = self.kv_a_proj_with_mqa(x)
            kv_latent, k_pe = mx.split(kv, [self.kv_lora_rank], axis=-1)
            kv_latent = mx.expand_dims(self.kv_a_layernorm(kv_latent), axis=1)
            k_pe = k_pe.reshape(B, L, 1, self.rope_dim).transpose(0, 2, 1, 3)
            if self.use_rope:
                k_pe = rope(k_pe)
            if mla_cache is not None:
                kv_latent, k_pe = mla_cache.update_and_fetch(kv_latent, k_pe)
            shared_kv[self.layer_idx] = (kv_latent, k_pe)
        else:
            kv_latent, k_pe = shared_kv[self.kv_source]

        if has_indexer:
            topk = self.indexer(x, mask, rope, log_pos, index_cache)
        else:
            topk = prev_topk

        if topk is not None:
            if L == 1:
                idx = topk[:, :, 0, :, None]

                def gather(a):
                    shape = idx.shape[:-1] + (a.shape[-1],)
                    return mx.take_along_axis(a, mx.broadcast_to(idx, shape), axis=2)

                kv_latent = gather(kv_latent)
                if self.rope_dim:
                    k_pe = gather(k_pe)
                if mask is not None:
                    mask = mx.take_along_axis(mask, topk, axis=-1)
            else:
                shape = topk.shape[:-1] + (kv_latent.shape[2],)
                sparse = mx.put_along_axis(
                    mx.zeros(shape, dtype=mx.bool_), topk, mx.array(True), axis=-1
                )
                mask = sparse if mask is None else sparse & mask

        if L == 1:
            # Decode attends to the latent directly.
            if self.rope_dim:
                pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
                if mask is not None:
                    min_value = mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype)
                    pe_scores = mx.where(mask, pe_scores, min_value)
                mask = pe_scores
            output = scaled_dot_product_attention(
                self.embed_q(q_nope),
                kv_latent,
                kv_latent,
                cache=None,
                scale=self.scale,
                mask=mask,
            )
            output = self.unembed_out(output)
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)
            if self.rope_dim:
                k_pe = mx.broadcast_to(k_pe, k.shape[:-1] + (self.rope_dim,))
                q = mx.concatenate([q_nope, q_pe], axis=-1)
                k = mx.concatenate([k, k_pe], axis=-1)
            else:
                q = q_nope
            output = scaled_dot_product_attention(
                q, k, v, cache=None, scale=self.scale, mask=mask
            )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        if cache is not None and cache.caches:
            # Evaluate unread cache arrays, or long generations exhaust Metal buffers.
            state = [a for c in cache.caches for a in c.state]
            output = mx.depends(output, [a for a in state if isinstance(a, mx.array)])
        return self.o_proj(output), topk


class MoE(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.args = args
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.n_routed_experts
        )
        self.gate = nn.Linear(args.hidden_size, args.n_routed_experts, bias=False)
        self.shared_experts = DeepseekV3MLP(
            args, intermediate_size=args.moe_intermediate_size * args.n_shared_experts
        )
        self.sharding_group = None

    def __call__(self, x):
        args = self.args
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)
        inds, scores = mistral4_expert_select(
            self.gate(x.astype(mx.float32)),
            args.num_experts_per_tok,
            args.n_group,
            args.topk_group,
            args.routed_scaling_factor,
            True,
        )
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        y = y + self.shared_experts(x)
        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)
        return y


class DecoderLayer(nn.Module):
    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args, layer_idx)
        if args.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = MoE(args)
        else:
            self.mlp = DeepseekV3MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(self, x, mask, rope, log_pos, cache, shared_kv, prev_topk):
        r, topk = self.self_attn(
            self.input_layernorm(x), mask, rope, log_pos, cache, shared_kv, prev_topk
        )
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r, topk
