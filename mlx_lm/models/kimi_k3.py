# Copyright © 2026 Apple Inc.
#
# Speculative preflight port for Kimi K3 (weights + tech report due
# 2026-07-27). Assembled from the public record: KDA + hybrid MLA plumbing
# from kimi_linear.py (Kimi Linear, arXiv 2510.26692), Block AttnRes from
# arXiv 2603.15031, latent MoE following the Nemotron-H `moe_latent_size`
# deployable variant (arXiv 2601.18089; routing stays in model space), and
# a sigmoid attention-output gate on the MLA layers (arXiv 2505.06708
# lineage). Every K3-specific guess is config-gated; revisit against the
# real config.json when weights land.

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import ArraysCache, KVCache
from .kimi_linear import (
    KimiDeltaAttention,
    KimiMLP,
    _group_expert_select,
)
from .mla import MultiLinear
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    head_dim: int
    rope_theta: float
    rms_norm_eps: float
    linear_attn_config: Dict[str, Any]
    model_max_length: int
    num_experts: int
    moe_intermediate_size: int
    kv_lora_rank: int
    rope_scaling: Optional[Dict[str, Any]] = None
    tie_word_embeddings: bool = False
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None
    mla_use_nope: bool = False
    num_experts_per_token: int = 1
    num_shared_experts: int = 0
    moe_router_activation_func: str = "sigmoid"
    moe_renormalize: bool = True
    routed_scaling_factor: float = 1.0
    first_k_dense_replace: int = 0
    moe_layer_freq: int = 1
    use_grouped_topk: bool = True
    num_expert_group: int = 1
    topk_group: int = 1
    # K3 additions (speculative until the tech report lands):
    # Stable LatentMoE — shared projections around experts in a latent space.
    moe_latent_size: Optional[int] = None
    # Gated MLA — sigmoid gate on the MLA attention output before o_proj.
    # "low_rank" mirrors KDA's g_a/g_b pair; "full" is a single projection.
    mla_gate: str = "low_rank"
    mla_gate_rank: Optional[int] = None
    # Block AttnRes — softmax attention over block summaries of the residual
    # stream. None or 0 disables.
    attnres_block_size: Optional[int] = 6


class KimiLatentSparseMoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        hidden = args.hidden_size
        experts = args.num_experts
        if experts is None:
            raise ValueError("num_experts must be specified for MoE layers")

        self.moe_latent_size = args.moe_latent_size
        expert_dim = self.moe_latent_size or hidden

        self.gate = nn.Linear(hidden, experts, bias=False)
        self.switch_mlp = SwitchGLU(expert_dim, args.moe_intermediate_size, experts)
        self.e_score_correction_bias = mx.zeros((experts,), dtype=mx.float32)

        if self.moe_latent_size is not None:
            self.fc1_latent_proj = nn.Linear(hidden, self.moe_latent_size, bias=False)
            self.fc2_latent_proj = nn.Linear(self.moe_latent_size, hidden, bias=False)

        if args.num_shared_experts:
            shared_hidden = args.moe_intermediate_size * args.num_shared_experts
            self.shared_experts = KimiMLP(args, intermediate_size=shared_hidden)
        else:
            self.shared_experts = None

    def __call__(self, x: mx.array) -> mx.array:
        scores = self.gate(x)
        inds, weights = _group_expert_select(
            scores,
            self.e_score_correction_bias,
            self.args.num_experts_per_token,
            self.args.num_expert_group,
            self.args.topk_group,
            self.args.routed_scaling_factor,
            self.args.moe_renormalize,
            self.args.moe_router_activation_func,
        )
        z = x if self.moe_latent_size is None else self.fc1_latent_proj(x)
        out = self.switch_mlp(z, inds)
        out = (out * weights[..., None]).sum(axis=-2)
        if self.moe_latent_size is not None:
            out = self.fc2_latent_proj(out)
        if self.shared_experts is not None:
            out = out + self.shared_experts(x)
        return out


class KimiGatedMLAAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.num_heads = args.num_attention_heads
        self.qk_nope_head_dim = args.qk_nope_head_dim or args.head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim or 0
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim or args.head_dim
        self.kv_lora_rank = args.kv_lora_rank
        self.scale = self.q_head_dim**-0.5

        hidden = args.hidden_size
        out_dim = self.num_heads * self.v_head_dim
        self.q_proj = nn.Linear(hidden, self.num_heads * self.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden,
            args.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = nn.RMSNorm(args.kv_lora_rank, eps=args.rms_norm_eps)
        self.embed_q = MultiLinear(
            self.qk_nope_head_dim, args.kv_lora_rank, self.num_heads
        )
        self.unembed_out = MultiLinear(
            args.kv_lora_rank, self.v_head_dim, self.num_heads
        )
        self.o_proj = nn.Linear(out_dim, hidden, bias=False)

        self.mla_gate = args.mla_gate
        if self.mla_gate == "low_rank":
            rank = args.mla_gate_rank or self.v_head_dim
            self.g_a_proj = nn.Linear(hidden, rank, bias=False)
            self.g_b_proj = nn.Linear(rank, out_dim, bias=False)
        elif self.mla_gate == "full":
            self.g_proj = nn.Linear(hidden, out_dim, bias=False)
        elif self.mla_gate != "none":
            raise ValueError(f"Unsupported mla_gate '{self.mla_gate}'")

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        q = self.q_proj(x).reshape(B, L, self.num_heads, self.q_head_dim)
        q = q.transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = self.kv_a_layernorm(compressed_kv)

        kv_latent = mx.expand_dims(kv_latent, axis=1)

        if cache is not None:
            kv_latent, k_pe = cache.update_and_fetch(kv_latent, k_pe)

        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(
                mask,
                pe_scores,
                mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
            )

        if L == 1:
            q_nope = self.embed_q(q_nope)
            k = v = kv_latent
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)

        output = scaled_dot_product_attention(
            q_nope, k, v, cache=cache, scale=self.scale, mask=pe_scores
        )

        if L == 1:
            output = self.unembed_out(output)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        if self.mla_gate == "low_rank":
            output = output * mx.sigmoid(self.g_b_proj(self.g_a_proj(x)))
        elif self.mla_gate == "full":
            output = output * mx.sigmoid(self.g_proj(x))

        return self.o_proj(output)


class BlockAttnRes(nn.Module):
    """Softmax attention over per-block residual-stream summaries.

    Keys and values are the token embedding plus each completed block's
    summed layer outputs; queries are learned per combine point and start
    at zero so the initial state is an equal-weight average (arXiv
    2603.15031). Keys are RMSNorm-ed, values are raw.
    """

    def __init__(self, dim: int, num_combines: int, eps: float):
        super().__init__()
        self.queries = mx.zeros((num_combines, dim))
        self.key_norm = nn.RMSNorm(dim, eps=eps)

    def __call__(self, idx: int, reps: List[mx.array]) -> mx.array:
        values = mx.stack(reps, axis=-2)
        keys = self.key_norm(values)
        scores = keys @ self.queries[idx]
        weights = mx.softmax(scores, axis=-1, precise=True)
        return (values * weights[..., None]).sum(axis=-2)


class KimiK3DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        kda_layers = args.linear_attn_config["kda_layers"]
        self.is_linear = (layer_idx + 1) in kda_layers

        if self.is_linear:
            self.self_attn = KimiDeltaAttention(args, layer_idx)
        else:
            self.self_attn = KimiGatedMLAAttention(args)

        if (
            args.num_experts > 0
            and layer_idx >= args.first_k_dense_replace
            and layer_idx % args.moe_layer_freq == 0
        ):
            self.mlp = KimiLatentSparseMoE(args)
        else:
            self.mlp = KimiMLP(args)

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
        y = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + y
        z = self.mlp(self.post_attention_layernorm(h))
        return h + z


class KimiK3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            KimiK3DecoderLayer(args, i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        kda_layers = args.linear_attn_config["kda_layers"]
        self.ssm_idx = kda_layers[0] - 1
        for i in range(len(self.layers)):
            if (i + 1) not in kda_layers:
                self.attn_idx = i
                break

        self.block_size = args.attnres_block_size or 0
        if self.block_size:
            num_combines = math.ceil(args.num_hidden_layers / self.block_size)
            self.attnres = BlockAttnRes(
                args.hidden_size, num_combines, args.rms_norm_eps
            )

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)

        ssm_mask = create_ssm_mask(h, cache[self.ssm_idx])
        attn_mask = create_attention_mask(h, cache[self.attn_idx], return_array=True)

        reps = [h]
        block_in = h
        combine_idx = 0

        for i, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
            if self.block_size and i > 0 and i % self.block_size == 0:
                reps.append(h - block_in)
                h = self.attnres(combine_idx, reps)
                combine_idx += 1
                block_in = h
            mask = ssm_mask if layer.is_linear else attn_mask
            h = layer(h, mask=mask, cache=layer_cache)

        if self.block_size:
            reps.append(h - block_in)
            h = self.attnres(combine_idx, reps)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = KimiK3Model(args)
        if args.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        out = self.model(inputs, cache)
        if self.lm_head is None:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches: List[Any] = []
        for layer in self.layers:
            if layer.is_linear:
                caches.append(ArraysCache(size=4))
            else:
                caches.append(KVCache())
        return caches

    def sanitize(self, weights: Dict[str, mx.array]) -> mx.array:
        weights = {k: v for k, v in weights.items() if not k.startswith("model.mtp")}

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        for layer_idx, layer in enumerate(self.layers):
            prefix = f"model.layers.{layer_idx}"

            if isinstance(layer.mlp, KimiLatentSparseMoE):
                # Expert weights may arrive under either the kimi_linear
                # ("block_sparse_moe", w1/w2/w3) or deepseek ("mlp.experts",
                # gate/up/down) naming; handle both until the real
                # checkpoint settles it.
                for src_prefix, names in (
                    (
                        f"{prefix}.block_sparse_moe",
                        [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")],
                    ),
                    (
                        f"{prefix}.mlp",
                        [
                            ("gate_proj", "gate_proj"),
                            ("down_proj", "down_proj"),
                            ("up_proj", "up_proj"),
                        ],
                    ),
                ):
                    dst_prefix = f"{prefix}.mlp"
                    for src, dst in names:
                        key = f"{src_prefix}.experts.0.{src}.weight"
                        if key in weights:
                            stacked = [
                                weights.pop(f"{src_prefix}.experts.{i}.{src}.weight")
                                for i in range(self.args.num_experts)
                            ]
                            weights[f"{dst_prefix}.switch_mlp.{dst}.weight"] = mx.stack(
                                stacked
                            )

                    for name in (
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    ):
                        src_key = f"{src_prefix}.shared_experts.{name}.weight"
                        if src_key in weights and src_prefix != dst_prefix:
                            weights[f"{dst_prefix}.shared_experts.{name}.weight"] = (
                                weights.pop(src_key)
                            )

                    for name in ("fc1_latent_proj", "fc2_latent_proj"):
                        src_key = f"{src_prefix}.{name}.weight"
                        if src_key in weights and src_prefix != dst_prefix:
                            weights[f"{dst_prefix}.{name}.weight"] = weights.pop(
                                src_key
                            )

                    gate_key = f"{src_prefix}.gate.weight"
                    if gate_key in weights and src_prefix != dst_prefix:
                        weights[f"{dst_prefix}.gate.weight"] = weights.pop(gate_key)

                    bias_key = f"{src_prefix}.gate.e_score_correction_bias"
                    if bias_key in weights:
                        weights[f"{dst_prefix}.e_score_correction_bias"] = weights.pop(
                            bias_key
                        )

            attn = getattr(layer, "self_attn", None)
            if isinstance(attn, KimiDeltaAttention):
                attn_prefix = f"{prefix}.self_attn"
                for src_name, dst_name in (
                    ("q_conv1d", "q_conv"),
                    ("k_conv1d", "k_conv"),
                    ("v_conv1d", "v_conv"),
                ):
                    src_key = f"{attn_prefix}.{src_name}.weight"
                    if src_key in weights:
                        w = weights.pop(src_key)
                        if w.ndim == 3:
                            w = w.moveaxis(2, 1)
                        weights[f"{attn_prefix}.{dst_name}.conv.weight"] = w
                dt_key = f"{attn_prefix}.dt_bias"
                if dt_key in weights:
                    if weights[dt_key].ndim > 1:
                        weights[dt_key] = mx.reshape(weights[dt_key], (-1,))

            attn_prefix = f"{prefix}.self_attn"
            kv_b_key = f"{attn_prefix}.kv_b_proj.weight"
            if kv_b_key in weights:
                qk_nope = self.args.qk_nope_head_dim or self.args.head_dim
                v_head = self.args.v_head_dim or self.args.head_dim
                head_dim = qk_nope + v_head
                num_heads = self.args.num_attention_heads

                quantized = f"{attn_prefix}.kv_b_proj.scales" in weights
                v = weights.pop(kv_b_key)

                if quantized:
                    dims = self.args.kv_lora_rank
                    scales = weights.pop(f"{attn_prefix}.kv_b_proj.scales")
                    biases = weights.pop(f"{attn_prefix}.kv_b_proj.biases")
                    bits = (v.shape[-1] * 32) // dims
                    group_size = dims // scales.shape[-1]
                    v = mx.dequantize(
                        v, scales, biases, bits=bits, group_size=group_size
                    )

                v = v.reshape(num_heads, head_dim, -1)
                wk = mx.contiguous(v[:, :qk_nope, :].swapaxes(-1, -2))
                wv = mx.contiguous(v[:, qk_nope:, :])

                if quantized:
                    wk, wk_s, wk_b = mx.quantize(wk, bits=bits, group_size=group_size)
                    wv, wv_s, wv_b = mx.quantize(wv, bits=bits, group_size=group_size)
                    weights[f"{attn_prefix}.embed_q.scales"] = wk_s
                    weights[f"{attn_prefix}.embed_q.biases"] = wk_b
                    weights[f"{attn_prefix}.unembed_out.scales"] = wv_s
                    weights[f"{attn_prefix}.unembed_out.biases"] = wv_b

                weights[f"{attn_prefix}.embed_q.weight"] = wk
                weights[f"{attn_prefix}.unembed_out.weight"] = wv

        return weights

    @property
    def cast_predicate(self):
        def predicate(path: str):
            if "e_score_correction_bias" in path:
                return False
            if path.endswith("A_log") or path.endswith("dt_bias"):
                return False
            return True

        return predicate

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("mlp.gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate
