# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .deepseek_v3 import (
    DeepseekV3MLP,
    DeepseekV3Model,
)
from .ministral3 import _get_llama_4_attn_scale
from .mla import MultiLinear
from .pipeline import PipelineMixin
from .rope_utils import apply_yarn_mscale, initialize_rope
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    n_shared_experts: int
    n_routed_experts: int
    routed_scaling_factor: float
    kv_lora_rank: int
    q_lora_rank: int
    norm_topk_prob: bool
    max_position_embeddings: int
    rms_norm_eps: float
    topk_group: int
    num_experts_per_tok: int
    first_k_dense_replace: int
    n_group: int
    qk_rope_head_dim: int
    qk_nope_head_dim: int
    v_head_dim: int
    head_dim: Optional[int] = None
    qk_head_dim: Optional[int] = None
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False
    rope_interleave: Optional[bool] = None
    attention_bias: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    rope_parameters: Optional[Dict] = None

    def __post_init__(self, **kwargs):
        if self.qk_head_dim is None:
            self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        if self.head_dim is None:
            self.head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        if self.rope_parameters is not None:
            self.rope_theta = self.rope_parameters.get("rope_theta", 100000.0)
            self.rope_scaling = self.rope_parameters


class Mistral4Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.max_position_embeddings = args.max_position_embeddings
        self.rope_theta = args.rope_theta
        self.q_lora_rank = args.q_lora_rank
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.kv_lora_rank = args.kv_lora_rank
        self.v_head_dim = args.v_head_dim
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.q_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim

        self.scale = apply_yarn_mscale(self.q_head_dim**-0.5, args.rope_parameters)

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                self.hidden_size, self.num_heads * self.q_head_dim, bias=False
            )
        else:
            self.q_a_proj = nn.Linear(
                self.hidden_size, self.q_lora_rank, bias=args.attention_bias
            )
            self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=1e-6)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=args.attention_bias,
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=1e-6)
        # kv_b_proj absorbed, so the cache holds the compressed latent.
        self.embed_q = MultiLinear(
            self.qk_nope_head_dim, self.kv_lora_rank, self.num_heads
        )
        self.unembed_out = MultiLinear(
            self.kv_lora_rank, self.v_head_dim, self.num_heads
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=args.attention_bias,
        )

        self.rope = initialize_rope(
            dims=self.qk_rope_head_dim,
            base=self.rope_theta,
            traditional=(
                args.rope_interleave if args.rope_interleave is not None else True
            ),
            max_position_embeddings=self.max_position_embeddings,
            scaling_config=self.args.rope_parameters,
        )

    def __call__(
        self,
        x: mx.array,
        attn_scale: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))

        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_rope = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed_kv = self.kv_a_proj_with_mqa(x)
        k_latent, k_rope = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_rope = k_rope.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = mx.expand_dims(self.kv_a_layernorm(k_latent), axis=1)

        offset = cache.offset if cache is not None else 0
        q_rope = self.rope(q_rope, offset)
        k_rope = self.rope(k_rope, offset)

        # The llama-4 scale applies to the whole query, so scale both halves.
        q_nope = q_nope * attn_scale
        q_rope = q_rope * attn_scale

        if cache is not None:
            kv_latent, k_rope = cache.update_and_fetch(kv_latent, k_rope)

        if L == 1:
            # Decode: attend to the latent directly. pe_scores is [B, H, 1, L].
            pe_scores = (q_rope * self.scale) @ k_rope.swapaxes(-1, -2)
            if mask is not None:
                pe_scores = mx.where(
                    mask,
                    pe_scores,
                    mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
                )
            output = scaled_dot_product_attention(
                self.embed_q(q_nope),
                kv_latent,
                kv_latent,
                cache=cache,
                scale=self.scale,
                mask=pe_scores,
            )
            output = self.unembed_out(output)
        else:
            k_nope = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)
            k_rope = mx.broadcast_to(
                k_rope, [B, self.num_heads, k_rope.shape[2], self.qk_rope_head_dim]
            )
            output = scaled_dot_product_attention(
                mx.concatenate([q_nope, q_rope], axis=-1),
                mx.concatenate([k_nope, k_rope], axis=-1),
                v,
                cache=cache,
                scale=self.scale,
                mask=mask,
            )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


@mx.compile
def mistral4_expert_select(
    gates,
    top_k,
    n_group,
    topk_group,
    routed_scaling_factor,
    norm_topk_prob,
):
    scores = mx.softmax(gates.astype(mx.float32), axis=-1)

    if n_group > 1:
        scores_grouped = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores_grouped, 2, axis=-1).sum(axis=-1, keepdims=True)
        # Zero out bottom (n_group - topk_group) groups
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        scores_grouped = mx.put_along_axis(
            scores_grouped, mx.stop_gradient(group_idx), mx.array(0.0), axis=-2
        )
        scores_for_choice = mx.flatten(scores_grouped, -2, -1)
    else:
        scores_for_choice = scores

    inds = mx.argpartition(-scores_for_choice, kth=top_k - 1, axis=-1)[..., :top_k]

    selected_scores = mx.take_along_axis(scores, inds, axis=-1)
    if norm_topk_prob:
        denominator = selected_scores.sum(axis=-1, keepdims=True) + 1e-20
        selected_scores = selected_scores / denominator
    selected_scores = selected_scores * routed_scaling_factor

    return inds, selected_scores


class Mistral4MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.num_experts_per_tok = args.num_experts_per_tok
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.n_routed_experts,
        )

        self.gate = nn.Linear(args.hidden_size, args.n_routed_experts, bias=False)
        if args.n_shared_experts is not None:
            intermediate_size = args.moe_intermediate_size * args.n_shared_experts
            self.shared_experts = DeepseekV3MLP(
                args, intermediate_size=intermediate_size
            )

        self.sharding_group = None

    def __call__(self, x):
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)

        inds, scores = mistral4_expert_select(
            self.gate(x),
            self.num_experts_per_tok,
            self.args.n_group,
            self.args.topk_group,
            self.args.routed_scaling_factor,
            self.args.norm_topk_prob,
        )
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        if self.args.n_shared_experts is not None:
            y = y + self.shared_experts(x)

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)

        return y


class Mistral4DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Mistral4Attention(args)
        self.mlp = (
            Mistral4MoE(args)
            if (
                args.n_routed_experts is not None
                and layer_idx >= args.first_k_dense_replace
            )
            else DeepseekV3MLP(args)
        )
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        attn_scale: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), attn_scale, mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class Mistral4Model(DeepseekV3Model, PipelineMixin, nn.Module):
    def __init__(self, args: ModelArgs):
        PipelineMixin.__init__(self)
        self.args = args
        self.vocab_size = args.vocab_size
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Mistral4DecoderLayer(args, idx) for idx in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

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

        offset = cache[0].offset if cache[0] is not None else 0
        # "causal" suits prefill; at L == 1 it is None, which decode wants.
        mask = create_attention_mask(h, cache[0])

        attn_scale = _get_llama_4_attn_scale(
            h.shape[1],
            offset,
            self.args.rope_parameters["llama_4_scaling_beta"],
            self.args.rope_parameters["original_max_position_embeddings"],
        ).astype(h.dtype)

        # Receive from the previous process in the pipeline
        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        for l, c in zip(self.pipeline_layers, cache):
            h = l(h, attn_scale, mask, cache=c)

        # Send to the next process in the pipeline
        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            if cache[-1] is not None:
                cache[-1].keys = mx.depends(cache[-1].keys, h)

        # Broadcast h while keeping it in the graph
        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Mistral4Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        out = self.model(inputs, cache=cache, input_embeddings=input_embeddings)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    @property
    def layers(self):
        return self.model.pipeline_layers

    def sanitize(self, weights):
        def broadcasts(scale_shape, weight_shape):
            if len(scale_shape) > len(weight_shape):
                return False
            pad = (1,) * (len(weight_shape) - len(scale_shape)) + tuple(scale_shape)
            return all(s in (1, w) for s, w in zip(pad, weight_shape))

        def dequant(weight, scale_inv):
            dtype = mx.bfloat16
            weight = mx.from_fp8(weight, dtype=dtype)
            # Per-tensor (rank 0) and per-expert ([E, 1, 1]) scales broadcast.
            if broadcasts(scale_inv.shape, weight.shape):
                return (weight * scale_inv).astype(dtype)
            bs = 128
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
            # Static activation scales have no consumer here.
            if k.endswith("activation_scale"):
                continue
            # Expert scales are named "experts.down_proj_scale_inv", so match
            # the suffix rather than a "weight_scale_inv" substring.
            if k.endswith("_scale_inv"):
                wk = k.replace("_scale_inv", "")
                new_weights[wk] = dequant(weights[wk], v)
            elif k not in new_weights:
                new_weights[k] = v
        weights = new_weights

        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"

            # Handle fused gate_up_proj format (Mistral4NaiveMoe)
            gup_key = f"{prefix}.mlp.experts.gate_up_proj"
            if gup_key in weights:
                gate_up = weights.pop(gup_key)
                gate, up = mx.split(gate_up, 2, axis=1)
                weights[f"{prefix}.mlp.switch_mlp.gate_proj.weight"] = gate
                weights[f"{prefix}.mlp.switch_mlp.up_proj.weight"] = up
            down_key = f"{prefix}.mlp.experts.down_proj"
            if down_key in weights:
                weights[f"{prefix}.mlp.switch_mlp.down_proj.weight"] = weights.pop(
                    down_key
                )

            for m in ["gate_proj", "down_proj", "up_proj"]:
                for k in ["weight", "scales", "biases"]:
                    if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                            for e in range(self.args.n_routed_experts)
                        ]
                        weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)

            # Absorb kv_b_proj into embed_q / unembed_out.
            # TODO: affine only; non-affine modes have no biases key.
            attn = f"{prefix}.self_attn"
            if f"{attn}.kv_b_proj.weight" in weights:
                quantized = f"{attn}.kv_b_proj.scales" in weights
                w = weights.pop(f"{attn}.kv_b_proj.weight")
                head_dim = self.args.qk_nope_head_dim + self.args.v_head_dim
                if quantized:
                    dims = self.args.kv_lora_rank
                    scales = weights.pop(f"{attn}.kv_b_proj.scales")
                    biases = weights.pop(f"{attn}.kv_b_proj.biases")
                    bits = (w.shape[-1] * 32) // dims
                    group_size = dims // scales.shape[-1]
                    w = mx.dequantize(
                        w, scales, biases, bits=bits, group_size=group_size
                    )
                w = w.reshape(self.args.num_attention_heads, head_dim, -1)
                wk = mx.contiguous(
                    w[:, : self.args.qk_nope_head_dim, :].swapaxes(-1, -2)
                )
                wv = mx.contiguous(w[:, self.args.qk_nope_head_dim :, :])
                if quantized:
                    wk, wk_scales, wk_biases = mx.quantize(
                        wk, bits=bits, group_size=group_size
                    )
                    wv, wv_scales, wv_biases = mx.quantize(
                        wv, bits=bits, group_size=group_size
                    )
                    weights[f"{attn}.embed_q.scales"] = wk_scales
                    weights[f"{attn}.embed_q.biases"] = wk_biases
                    weights[f"{attn}.unembed_out.scales"] = wv_scales
                    weights[f"{attn}.unembed_out.biases"] = wv_biases
                weights[f"{attn}.embed_q.weight"] = wk
                weights[f"{attn}.unembed_out.weight"] = wv

        return {k: v for k, v in weights.items() if "rotary_emb.inv_freq" not in k}

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()

        for layer in self.model.layers:
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
            sh = group.rank() * num_heads
            eh = sh + num_heads

            def shard_heads(w):
                return w[sh:eh]

            layer.self_attn.embed_q.apply(shard_heads)
            layer.self_attn.unembed_out.apply(shard_heads)

            layer.self_attn.o_proj = shard_linear(
                layer.self_attn.o_proj, "sharded-to-all", group=group
            )

            if isinstance(layer.mlp, DeepseekV3MLP):
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
                # Shard in place: the MoE aggregates the partial sums itself.
                layer.mlp.sharding_group = group
                if hasattr(layer.mlp, "shared_experts"):
                    shard_inplace(
                        layer.mlp.shared_experts.gate_proj,
                        "all-to-sharded",
                        group=group,
                    )
                    shard_inplace(
                        layer.mlp.shared_experts.down_proj,
                        "sharded-to-all",
                        group=group,
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
