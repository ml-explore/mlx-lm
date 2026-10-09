# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import (
    QuantizedAllToShardedLinear,
    QuantizedShardedToAllLinear,
    shard_inplace,
    shard_linear,
    sum_gradients,
)
from mlx.utils import tree_unflatten

from .activations import swiglu
from .base import (
    BaseModelArgs,
    create_attention_mask,
    gather_last_axis,
    scaled_dot_product_attention,
)
from .deepseek_v3 import (
    DeepseekV3MLP,
    DeepseekV3Model,
)
from .ministral3 import _get_llama_4_attn_scale
from .mla import MultiLinear, QuantizedMultiLinear
from .pipeline import PipelineMixin
from .rope_utils import apply_yarn_mscale, initialize_rope
from .switch_layers import (
    QuantizedSwitchLinear,
    SwitchGLU,
    SwitchLinear,
    _gather_sort,
    _scatter_unsort,
)


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
            compressed_kv = self.kv_a_proj_with_mqa(x)
        else:
            # sanitize fuses the two projections unless they are stored differently.
            if "qkv_a_proj" in self:
                q, compressed_kv = mx.split(
                    self.qkv_a_proj(x), [self.q_lora_rank], axis=-1
                )
            else:
                q, compressed_kv = self.q_a_proj(x), self.kv_a_proj_with_mqa(x)
            q = self.q_b_proj(self.q_a_layernorm(q))

        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_rope = mx.split(q, [self.qk_nope_head_dim], axis=-1)

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
    inds = mx.stop_gradient(inds)

    selected_scores = mx.take_along_axis(scores, inds, axis=-1)
    if norm_topk_prob:
        denominator = selected_scores.sum(axis=-1, keepdims=True) + 1e-20
        selected_scores = selected_scores / denominator
    selected_scores = selected_scores * routed_scaling_factor

    return inds, selected_scores


class Fp8Linear(nn.QuantizedLinear):
    """Linear on e4m3 weights, run as mxfp8 with unit group scales.

    The kernel has no slot for the per-tensor scale, so it scales the output.
    Fused layers have one scale per output row.
    """

    def __init__(self, input_dims: int, output_dims: int, per_row: bool = False):
        super().__init__(input_dims, output_dims, False, 32, 8, "mxfp8")
        self.output_scale = mx.ones((output_dims,)) if per_row else mx.array(1.0)
        self.freeze()

    def __call__(self, x):
        y = super().__call__(x)
        return (y * self.output_scale).astype(y.dtype)


class Fp8MultiLinear(QuantizedMultiLinear):
    """MultiLinear on e4m3 weights, see Fp8Linear."""

    def __init__(self, input_dims: int, output_dims: int, num_heads: int):
        super().__init__(input_dims, output_dims, num_heads, 32, 8, "mxfp8")
        self.output_scale = mx.array(1.0)
        self.freeze()

    def __call__(self, x, transpose=True):
        y = super().__call__(x, transpose)
        return (y * self.output_scale).astype(y.dtype)


class Fp8SwitchLinear(QuantizedSwitchLinear):
    """Experts on e4m3 weights with one scale per expert, see Fp8Linear."""

    def __init__(self, input_dims: int, output_dims: int, num_experts: int):
        super().__init__(input_dims, output_dims, num_experts, False, 32, 8, "mxfp8")
        self.output_scale = mx.ones((num_experts, 1, 1))
        self.freeze()

    def __call__(self, x, indices, sorted_indices=False):
        y = super().__call__(x, indices, sorted_indices)
        return (y * self.output_scale[indices]).astype(y.dtype)


class FusedSwitchGLU(nn.Module):
    """SwitchGLU with gate_proj and up_proj fused into one projection."""

    def __init__(self, input_dims: int, hidden_dims: int, num_experts: int):
        super().__init__()
        self.gate_up_proj = SwitchLinear(
            input_dims, 2 * hidden_dims, num_experts, bias=False
        )
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=False)

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        indices = mx.stop_gradient(indices)
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        gate, up = mx.split(self.gate_up_proj(x, idx, sorted_indices=do_sort), 2, -1)
        x = self.down_proj(swiglu(gate, up), idx, sorted_indices=do_sort)

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)


class FusedGLU(nn.Module):
    """Gated MLP with gate_proj and up_proj fused into one projection."""

    def __init__(self, input_dims: int, hidden_dims: int):
        super().__init__()
        self.gate_up_proj = nn.Linear(input_dims, 2 * hidden_dims, bias=False)
        self.down_proj = nn.Linear(hidden_dims, input_dims, bias=False)

    def __call__(self, x):
        gate, up = mx.split(self.gate_up_proj(x), 2, axis=-1)
        return self.down_proj(swiglu(gate, up))


def _fuse(weights, parts, fused):
    """Join the output rows of layers that read the same input.

    Returns False and changes nothing when the layers are stored differently,
    for example quantized with different bits.
    """
    keys = [{k[len(p) + 1 :] for k in weights if k.startswith(p + ".")} for p in parts]
    if "weight" not in keys[0] or any(k != keys[0] for k in keys):
        return False
    for k in keys[0]:
        vs = [weights[f"{p}.{k}"] for p in parts]
        if k == "output_scale" and vs[0].ndim > 0:
            return False
        if k not in ("output_scale", "bias") and any(
            v.dtype != vs[0].dtype or v.shape[-1] != vs[0].shape[-1] for v in vs
        ):
            return False

    rows = [weights[f"{p}.weight"].shape[-2] for p in parts]
    for k in keys[0]:
        vs = [weights.pop(f"{p}.{k}") for p in parts]
        if k == "output_scale":
            # Each fp8 layer has one scale. Fused, there is one per row.
            vs = [mx.full((n,), s, dtype=s.dtype) for n, s in zip(rows, vs)]
            weights[f"{fused}.{k}"] = mx.concatenate(vs)
        else:
            axis = -1 if k == "bias" else -2
            weights[f"{fused}.{k}"] = mx.concatenate(vs, axis=axis)
    return True


class Fp8AllToShardedLinear(QuantizedAllToShardedLinear):
    """Fp8Linear with the outputs sharded across the group."""

    def __call__(self, x):
        y = super().__call__(x)
        return (y * self.output_scale).astype(y.dtype)


class Fp8ShardedToAllLinear(QuantizedShardedToAllLinear):
    """Fp8Linear with the inputs sharded across the group."""

    def __call__(self, x):
        y = super().__call__(x)
        return (y * self.output_scale).astype(y.dtype)


# The fp8 output scale applies to each shard and to each partial sum, so every
# rank keeps all of it.
def _shard_linear(layer, sharding, group):
    if not isinstance(layer, Fp8Linear):
        return shard_linear(layer, sharding, group=group)
    if sharding == "all-to-sharded":
        cls = Fp8AllToShardedLinear
    else:
        cls = Fp8ShardedToAllLinear
    scale = layer.pop("output_scale")
    sharded = cls.from_quantized_linear(layer, group=group)
    sharded.output_scale = scale
    sharded.freeze()
    return sharded


def _shard_inplace(layer, sharding, group, segments=1):
    rows = sharding == "all-to-sharded"

    def predicate(path, w):
        # A scalar or per-expert fp8 scale is the same for every shard.
        if path == "output_scale" and w.ndim != 1:
            return None
        # A bias or a per-row scale follows the output rows.
        if path in ("bias", "output_scale"):
            return (-1, segments) if rows else None
        return (max(w.ndim - 2, 0), segments) if rows else (-1, segments)

    shard_inplace(layer, predicate, group=group)


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
        self.sharding_group = None

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
            if self.sharding_group is not None:
                out = gather_last_axis(out, self.sharding_group)
        return out

    @property
    def layers(self):
        return self.model.pipeline_layers

    def sanitize(self, weights):

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

        # fp8 layers keep the e4m3 bytes, and the scale moves to the layer.
        new_weights = {}
        for k, v in weights.items():
            # Static activation scales have no consumer here.
            if k.endswith("activation_scale"):
                continue
            new_weights[k.replace(".weight_scale_inv", ".output_scale")] = v
        weights = new_weights

        def fused(parts, name):
            return f"{name}.weight" in weights or _fuse(weights, parts, name)

        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"
            experts = f"{prefix}.mlp.experts"
            switch = f"{prefix}.mlp.switch_mlp"
            shared = f"{prefix}.mlp.shared_experts"
            attn = f"{prefix}.self_attn"

            # HF stores the experts with gate_up_proj fused (Mistral4NaiveMoe).
            for m in ["gate_up_proj", "down_proj"]:
                if f"{experts}.{m}" in weights:
                    weights[f"{switch}.{m}.weight"] = weights.pop(f"{experts}.{m}")
                if f"{experts}.{m}_scale_inv" in weights:
                    weights[f"{switch}.{m}.output_scale"] = weights.pop(
                        f"{experts}.{m}_scale_inv"
                    )

            for m in ["gate_proj", "down_proj", "up_proj"]:
                for k in ["weight", "scales", "biases"]:
                    if f"{experts}.0.{m}.{k}" in weights:
                        to_join = [
                            weights.pop(f"{experts}.{e}.{m}.{k}")
                            for e in range(self.args.n_routed_experts)
                        ]
                        weights[f"{switch}.{m}.{k}"] = mx.stack(to_join)

            # Absorb kv_b_proj into embed_q / unembed_out.
            # TODO: affine only; non-affine modes have no biases key.
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
                s = weights.pop(f"{attn}.kv_b_proj.output_scale", None)
                if s is not None:
                    weights[f"{attn}.embed_q.output_scale"] = s
                    weights[f"{attn}.unembed_out.output_scale"] = s
                weights[f"{attn}.embed_q.weight"] = wk
                weights[f"{attn}.unembed_out.weight"] = wv

            # Projections that read the same input run as one matmul.
            layer = self.model.layers[l]
            a = layer.self_attn
            if "q_a_proj" in a and fused(
                [f"{attn}.q_a_proj", f"{attn}.kv_a_proj_with_mqa"], f"{attn}.qkv_a_proj"
            ):
                rows = a.q_lora_rank + a.kv_lora_rank + a.qk_rope_head_dim
                a.qkv_a_proj = nn.Linear(
                    a.hidden_size, rows, bias=self.args.attention_bias
                )
                del a.q_a_proj, a.kv_a_proj_with_mqa
            mlp = layer.mlp
            if not isinstance(mlp, Mistral4MoE):
                continue
            if isinstance(mlp.switch_mlp, SwitchGLU) and fused(
                [f"{switch}.gate_proj", f"{switch}.up_proj"], f"{switch}.gate_up_proj"
            ):
                g = mlp.switch_mlp.gate_proj
                mlp.switch_mlp = FusedSwitchGLU(
                    g.input_dims, g.output_dims, g.num_experts
                )
            if isinstance(mlp.get("shared_experts"), DeepseekV3MLP) and fused(
                [f"{shared}.gate_proj", f"{shared}.up_proj"], f"{shared}.gate_up_proj"
            ):
                out_dims, in_dims = mlp.shared_experts.gate_proj.weight.shape
                mlp.shared_experts = FusedGLU(in_dims, out_dims)

        # e4m3 bytes read as uint32 are mxfp8 with unit (2^0) group scales.
        for k in [k for k in weights if k.endswith(".output_scale")]:
            p = k.removesuffix(".output_scale")
            w = weights[f"{p}.weight"]
            if w.dtype == mx.uint8:
                weights[f"{p}.weight"] = w.view(mx.uint32)
                weights[f"{p}.scales"] = mx.full(
                    (*w.shape[:-1], w.shape[-1] // 32), 127, mx.uint8
                )

        fp8_modules = []
        for p, m in self.named_modules():
            if f"{p}.output_scale" not in weights:
                continue
            if isinstance(m, nn.Linear):
                out_dims, in_dims = m.weight.shape
                per_row = weights[f"{p}.output_scale"].ndim == 1
                fp8_modules.append((p, Fp8Linear(in_dims, out_dims, per_row)))
            elif isinstance(m, MultiLinear):
                h, o, i = m.weight.shape
                fp8_modules.append((p, Fp8MultiLinear(i, o, h)))
            elif isinstance(m, SwitchLinear):
                fp8_modules.append(
                    (p, Fp8SwitchLinear(m.input_dims, m.output_dims, m.num_experts))
                )
        if fp8_modules:
            self.update_modules(tree_unflatten(fp8_modules))

        return {k: v for k, v in weights.items() if "rotary_emb.inv_freq" not in k}

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()

        # Each rank computes a slice of the vocabulary
        if not self.args.tie_word_embeddings:
            vocab = self.args.vocab_size
            assert vocab % N == 0, f"group size {N} must divide vocab_size {vocab}"
            self.lm_head = shard_linear(self.lm_head, "all-to-sharded", group=group)
            self.sharding_group = group

        for layer in self.model.layers:
            attn = layer.self_attn
            if attn.q_lora_rank is None:
                attn.q_proj = _shard_linear(attn.q_proj, "all-to-sharded", group)
            else:
                attn.q_b_proj = _shard_linear(attn.q_b_proj, "all-to-sharded", group)

            attn.num_heads //= N
            num_heads = attn.num_heads
            sh = group.rank() * num_heads
            eh = sh + num_heads

            def shard_heads(w):
                # The fp8 output scale is one scalar for all heads.
                return w[sh:eh] if w.ndim > 0 else w

            attn.embed_q.apply(shard_heads)
            attn.unembed_out.apply(shard_heads)

            attn.o_proj = _shard_linear(attn.o_proj, "sharded-to-all", group)

            mlp = layer.mlp
            if isinstance(mlp, DeepseekV3MLP):
                mlp.gate_proj = _shard_linear(mlp.gate_proj, "all-to-sharded", group)
                mlp.down_proj = _shard_linear(mlp.down_proj, "sharded-to-all", group)
                mlp.up_proj = _shard_linear(mlp.up_proj, "all-to-sharded", group)

            else:
                # Shard in place: the MoE aggregates the partial sums itself.
                mlp.sharding_group = group
                if hasattr(mlp, "shared_experts"):
                    shared = mlp.shared_experts
                    if isinstance(shared, FusedGLU):
                        _shard_inplace(
                            shared.gate_up_proj, "all-to-sharded", group, segments=2
                        )
                    else:
                        _shard_inplace(shared.gate_proj, "all-to-sharded", group)
                        _shard_inplace(shared.up_proj, "all-to-sharded", group)
                    _shard_inplace(shared.down_proj, "sharded-to-all", group)
                switch = mlp.switch_mlp
                if isinstance(switch, FusedSwitchGLU):
                    # Each rank takes its part of both halves of gate_up_proj.
                    _shard_inplace(
                        switch.gate_up_proj, "all-to-sharded", group, segments=2
                    )
                else:
                    _shard_inplace(switch.gate_proj, "all-to-sharded", group)
                    _shard_inplace(switch.up_proj, "all-to-sharded", group)
                _shard_inplace(switch.down_proj, "sharded-to-all", group)
