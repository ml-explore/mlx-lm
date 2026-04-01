# Copyright © 2025 Apple Inc.

"""Mistral Small 4 (119B MoE) with absorbed Multi-head Latent Attention.

Mistral Small 4 uses the same MLA architecture as DeepSeek V2/V3 but ships the
KV decompression weights as a single ``kv_b_proj`` linear layer rather than
per-head ``MultiLinear`` tensors.  During ``sanitize`` we decompose
``kv_b_proj`` into the absorbed ``embed_q`` (W_UK) and ``unembed_out`` (W_UV)
``MultiLinear`` weights, exactly as ``deepseek_v3.py`` does.

At inference the KV cache stores only the compressed latent ``c_kv`` and the
RoPE key component ``k_pe`` — a ~25× reduction compared to caching the full
decompressed keys and values.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .mla import MultiLinear
from .pipeline import PipelineMixin
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "mistral4"
    hidden_size: int = 4096
    num_hidden_layers: int = 36
    intermediate_size: int = 12288
    num_attention_heads: int = 32
    rms_norm_eps: float = 1e-6
    vocab_size: int = 131072
    head_dim: int = 128
    max_position_embeddings: int = 262144
    num_key_value_heads: Optional[int] = None
    rope_theta: float = 10000.0
    rope_parameters: Optional[Dict[str, Union[float, str]]] = None
    rope_scaling: Optional[Dict] = None
    tie_word_embeddings: bool = False

    # MoE
    n_routed_experts: int = 128
    num_experts_per_tok: int = 4
    moe_intermediate_size: int = 2048
    n_shared_experts: Optional[int] = 1
    first_k_dense_replace: int = 0
    routed_scaling_factor: float = 1.0
    norm_topk_prob: bool = True
    n_group: int = 1
    topk_group: int = 1

    # MLA
    kv_lora_rank: int = 256
    q_lora_rank: int = 1024
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 64
    qk_head_dim: int = 128
    v_head_dim: int = 128
    attention_bias: bool = False

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        # Unify rope config: prefer rope_parameters, fall back to rope_scaling
        if self.rope_parameters is not None and self.rope_scaling is None:
            self.rope_scaling = self.rope_parameters


# --------------------------------------------------------------------------- #
# Absorbed Multi-head Latent Attention
# --------------------------------------------------------------------------- #


class Mistral4Attention(nn.Module):
    """MLA with weight absorption — caches compressed (c_kv, k_pe) only.

    At load time ``sanitize`` decomposes the checkpoint's ``kv_b_proj`` into
    two ``MultiLinear`` modules:

    * ``embed_q``   — absorbs W_UK into the query path
    * ``unembed_out`` — absorbs W_UV into the output path

    The attention computation follows the same pattern as ``deepseek_v3.py``:
    RoPE scores are pre-computed and passed as an additive mask to the nope
    SDPA so that we never need to materialise full-rank keys.
    """

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.scale = self.q_head_dim ** -0.5

        # Query projections (LoRA-decomposed or direct)
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                self.hidden_size, self.num_heads * self.q_head_dim, bias=False
            )
        else:
            self.q_a_proj = nn.Linear(
                self.hidden_size, self.q_lora_rank, bias=config.attention_bias
            )
            self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
            )

        # KV compression (shared across heads, MQA-style)
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)

        # Absorbed projections — weights set by sanitize from kv_b_proj
        self.embed_q = MultiLinear(
            self.qk_nope_head_dim, self.kv_lora_rank, self.num_heads
        )
        self.unembed_out = MultiLinear(
            self.kv_lora_rank, self.v_head_dim, self.num_heads
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )

        # Attention scale with YaRN mscale correction
        if config.rope_scaling is not None:
            mscale_all_dim = config.rope_scaling.get("mscale_all_dim", 0)
            if mscale_all_dim:
                scaling_factor = config.rope_scaling["factor"]
                if scaling_factor > 1:
                    s = 0.1 * mscale_all_dim * math.log(scaling_factor) + 1.0
                    self.scale = self.scale * s * s

        # RoPE — Mistral 4 uses YaRN with interleaved (traditional) layout
        rope_scaling = config.rope_scaling or {}
        self.rope = initialize_rope(
            dims=self.qk_rope_head_dim,
            base=rope_scaling.get("rope_theta", config.rope_theta),
            traditional=True,
            max_position_embeddings=config.max_position_embeddings,
            scaling_config=config.rope_scaling,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        # --- Query path ---
        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))

        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        # --- KV compression path ---
        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = self.kv_a_layernorm(compressed_kv)

        # --- RoPE ---
        offset = cache.offset if cache is not None else 0
        q_pe = self.rope(q_pe, offset)
        k_pe = self.rope(k_pe, offset)

        # --- Expand latent for cache (1 "head") ---
        kv_latent = mx.expand_dims(kv_latent, axis=1)

        if cache is not None:
            kv_latent, k_pe = cache.update_and_fetch(kv_latent, k_pe)

        # --- Absorbed attention ---
        # Concatenate nope and rope components into unified Q/K so that
        # mx.fast.scaled_dot_product_attention uses its tiled flash-attention
        # kernel without materialising a (B, H, L, S) score tensor in RAM.
        # Correctness: dot(q_nope, k_nope) + dot(q_pe, k_pe)
        #            = dot(concat(q_nope, q_pe), concat(k_nope, k_pe))
        if L == 1:
            # Generation: absorb W_UK into q_nope; K = concat(c_kv, k_pe)
            q_nope = self.embed_q(q_nope)                        # (B, H, 1, kv_lora_rank)
            k = mx.concatenate([kv_latent, k_pe], axis=-1)       # (B, 1, S, kv_lora_rank + qk_rope_head_dim)
            q = mx.concatenate([q_nope, q_pe], axis=-1)          # (B, H, 1, kv_lora_rank + qk_rope_head_dim)
            v = kv_latent                                         # (B, 1, S, kv_lora_rank)
            output = scaled_dot_product_attention(
                q, k, v, cache=cache, scale=self.scale, mask=mask
            )
            output = self.unembed_out(output)                     # (B, H, 1, v_head_dim)
        else:
            # Prefill: expand latent to per-head K/V; broadcast k_pe to match
            k_nope = self.embed_q(kv_latent, transpose=False)     # (B, H, S, qk_nope_head_dim)
            k = mx.concatenate(
                [k_nope, mx.broadcast_to(k_pe, k_nope.shape)], axis=-1
            )                                                      # (B, H, S, q_head_dim)
            v = self.unembed_out(kv_latent)                       # (B, H, S, v_head_dim)
            q = mx.concatenate([q_nope, q_pe], axis=-1)           # (B, H, L, q_head_dim)
            output = scaled_dot_product_attention(
                q, k, v, cache=cache, scale=self.scale, mask=mask
            )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


# --------------------------------------------------------------------------- #
# MLP / MoE
# --------------------------------------------------------------------------- #


class Mistral4MLP(nn.Module):
    def __init__(
        self, config: ModelArgs, hidden_size: int = None, intermediate_size: int = None
    ):
        super().__init__()
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class MoEGate(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.norm_topk_prob = config.norm_topk_prob
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.weight = mx.zeros((self.n_routed_experts, config.hidden_size))

    def __call__(self, x):
        gates = x @ self.weight.T
        scores = mx.softmax(gates, axis=-1, precise=True)

        if self.n_group > 1:
            bsz, seq_len = x.shape[:2]
            scores = scores.reshape(bsz, seq_len, self.n_group, -1)
            group_scores = scores.max(axis=-1, keepdims=True)
            k = self.n_group - self.topk_group
            group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
            scores = mx.put_along_axis(
                scores, group_idx, mx.array(0.0, scores.dtype), axis=-2
            )
            scores = scores.reshape(bsz, seq_len, -1)

        k = self.top_k
        inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
        scores = mx.take_along_axis(scores, inds, axis=-1)
        if self.top_k > 1 and self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        scores = scores * self.routed_scaling_factor

        return inds, scores


class Mistral4MoE(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            config.moe_intermediate_size,
            config.n_routed_experts,
        )
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = Mistral4MLP(
                config=config, intermediate_size=intermediate_size
            )
        self.sharding_group = None

    def __call__(self, x):
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)

        inds, scores = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(x)

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)

        return y


# --------------------------------------------------------------------------- #
# Decoder layer / model
# --------------------------------------------------------------------------- #


class Mistral4DecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Mistral4Attention(config)
        self.mlp = (
            Mistral4MoE(config)
            if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
            )
            else Mistral4MLP(config)
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


class Mistral4Model(PipelineMixin, nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Mistral4DecoderLayer(config, idx)
            for idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(x)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * len(self.pipeline_layers)
        mask = create_attention_mask(h, cache[0], return_array=True)

        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        for l, c in zip(self.pipeline_layers, cache):
            h = l(h, mask, cache=c)

        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            if cache[-1] is not None:
                cache[-1].keys = mx.depends(cache[-1].keys, h)

        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = Mistral4Model(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ):
        out = self.model(inputs, cache, input_embeddings=input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights):
        # --- FP8 dequantization (Mistral ships FP8 weights) ---
        new_weights = {}
        for k, v in weights.items():
            if "weight_scale_inv" in k:
                scale_inv = v
                wk = k.replace("_scale_inv", "")
                weight = weights[wk]
                weight = _dequant_fp8(weight, scale_inv)
                new_weights[wk] = weight
            elif "activation_scale" in k:
                # Skip activation scales (not used in MLX)
                continue
            elif k not in new_weights:
                new_weights[k] = v
        weights = new_weights

        # --- Stack/split MoE expert weights ---
        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"

            # Pre-stacked fused format (original Mistral FP8 checkpoint):
            # experts.gate_up_proj  (n_experts, 2*moe_intermediate, hidden)
            # experts.down_proj     (n_experts, hidden, moe_intermediate)
            gate_up_key = f"{prefix}.mlp.experts.gate_up_proj"
            if gate_up_key in weights:
                gate_up = weights.pop(gate_up_key)
                down = weights.pop(f"{prefix}.mlp.experts.down_proj")

                # FP8 dequant per expert (these keys lack "weight_" so the
                # first FP8 loop above does not catch them)
                gate_up_scale_key = f"{gate_up_key}_scale_inv"
                if gate_up_scale_key in weights:
                    gu_s = weights.pop(gate_up_scale_key)
                    d_s = weights.pop(
                        f"{prefix}.mlp.experts.down_proj_scale_inv"
                    )
                    n = self.args.n_routed_experts
                    gate_up = mx.stack([
                        _dequant_fp8(
                            gate_up[e], gu_s[e] if gu_s.ndim > 0 else gu_s
                        )
                        for e in range(n)
                    ])
                    down = mx.stack([
                        _dequant_fp8(
                            down[e], d_s[e] if d_s.ndim > 0 else d_s
                        )
                        for e in range(n)
                    ])

                # Split fused gate_up → gate, up
                gate, up = mx.split(gate_up, 2, axis=1)
                weights[f"{prefix}.mlp.switch_mlp.gate_proj.weight"] = gate
                weights[f"{prefix}.mlp.switch_mlp.up_proj.weight"] = up
                weights[f"{prefix}.mlp.switch_mlp.down_proj.weight"] = down

            else:
                # Per-expert format (pre-quantized MLX checkpoints)
                for m in ("gate_proj", "down_proj", "up_proj"):
                    for k in ("weight", "scales", "biases"):
                        expert_key = f"{prefix}.mlp.experts.0.{m}.{k}"
                        if expert_key in weights:
                            to_join = [
                                weights.pop(
                                    f"{prefix}.mlp.experts.{e}.{m}.{k}"
                                )
                                for e in range(self.args.n_routed_experts)
                            ]
                            weights[
                                f"{prefix}.mlp.switch_mlp.{m}.{k}"
                            ] = mx.stack(to_join)

            # --- Decompose kv_b_proj → embed_q + unembed_out ---
            attn_prefix = f"{prefix}.self_attn"
            bproj_key = f"{attn_prefix}.kv_b_proj.weight"
            if bproj_key in weights:
                num_heads = self.args.num_attention_heads
                d_nope = self.args.qk_nope_head_dim
                d_v = self.args.v_head_dim
                kv_lora_rank = self.args.kv_lora_rank

                quantized = f"{attn_prefix}.kv_b_proj.scales" in weights
                v = weights.pop(bproj_key)

                if quantized:
                    scales = weights.pop(f"{attn_prefix}.kv_b_proj.scales")
                    biases = weights.pop(f"{attn_prefix}.kv_b_proj.biases")
                    bits = (v.shape[-1] * 32) // kv_lora_rank
                    group_size = kv_lora_rank // scales.shape[-1]
                    v = mx.dequantize(
                        v, scales, biases, bits=bits, group_size=group_size
                    )

                # v shape: (num_heads * (d_nope + d_v), kv_lora_rank)
                v = v.reshape(num_heads, d_nope + d_v, kv_lora_rank)

                # embed_q weight: (H, kv_lora_rank, d_nope) — W_UK transposed
                wk = mx.contiguous(v[:, :d_nope, :].swapaxes(-1, -2))
                # unembed_out weight: (H, d_v, kv_lora_rank) — W_UV
                wv = mx.contiguous(v[:, d_nope:, :])

                if quantized:
                    wk, wk_s, wk_b = mx.quantize(wk, bits=bits, group_size=group_size)
                    wv, wv_s, wv_b = mx.quantize(wv, bits=bits, group_size=group_size)
                    weights[f"{attn_prefix}.embed_q.scales"] = wk_s
                    weights[f"{attn_prefix}.embed_q.biases"] = wk_b
                    weights[f"{attn_prefix}.unembed_out.scales"] = wv_s
                    weights[f"{attn_prefix}.unembed_out.biases"] = wv_b

                weights[f"{attn_prefix}.embed_q.weight"] = wk
                weights[f"{attn_prefix}.unembed_out.weight"] = wv

        # Remove any unused keys
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
            # Shard attention
            if layer.self_attn.q_lora_rank is None:
                layer.self_attn.q_proj = shard_linear(
                    layer.self_attn.q_proj, "all-to-sharded", group=group
                )
            else:
                layer.self_attn.q_b_proj = shard_linear(
                    layer.self_attn.q_b_proj, "all-to-sharded", group=group
                )
            layer.self_attn.num_heads //= N
            num_heads = layer.self_attn.num_heads
            sh = rank * num_heads
            eh = sh + num_heads

            def shard_heads(w):
                return w[sh:eh]

            layer.self_attn.embed_q.apply(shard_heads)
            layer.self_attn.unembed_out.apply(shard_heads)

            layer.self_attn.o_proj = shard_linear(
                layer.self_attn.o_proj, "sharded-to-all", group=group
            )

            # Shard MLP / MoE
            if isinstance(layer.mlp, Mistral4MLP):
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
                layer.mlp.sharding_group = group
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

    @property
    def layers(self):
        return self.model.pipeline_layers


def _dequant_fp8(weight, scale_inv):
    """Dequantize FP8 block-scaled weights to bfloat16."""
    dtype = mx.bfloat16
    weight = mx.from_fp8(weight, dtype=dtype)
    if scale_inv.ndim == 0:
        return (weight * scale_inv).astype(dtype)
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
