# Copyright © 2026 Apple Inc.

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import ArraysCache, KVCache
from .mla import MultiLinear
from .pipeline import PipelineMixin
from .rope_utils import apply_yarn_mscale, initialize_rope
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    num_experts: int
    num_experts_per_tok: int
    num_shared_experts: int
    n_group: int
    topk_group: int
    first_k_dense_replace: int
    layer_group_size: int
    group_norm_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    routed_scaling_factor: float
    head_dim: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    qk_nope_head_dim: int
    v_head_dim: int
    q_lora_rank: Optional[int] = None
    moe_shared_expert_intermediate_size: Optional[int] = None
    rope_interleave: bool = True
    partial_rotary_factor: float = 0.5
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    use_qkv_bias: bool = False
    use_bias: bool = False
    use_qk_norm: bool = True
    score_function: str = "sigmoid"
    norm_topk_prob: bool = True
    moe_router_enable_expert_bias: bool = True
    tie_word_embeddings: bool = False
    num_nextn_predict_layers: int = 0


@mx.compile
def _recurrent_gla_step(
    q_t: mx.array,
    k_t: mx.array,
    v_t: mx.array,
    h: mx.array,
    exp_g: mx.array,
) -> Tuple[mx.array, mx.array]:
    h = h * exp_g + k_t.transpose(0, 1, 3, 2) @ v_t
    return (q_t @ h).astype(q_t.dtype), h


def recurrent_gla(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    *,
    scale: float,
    h: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
    chunk: int = 128,
) -> Tuple[mx.array, mx.array]:
    """
    Lightning-attention recurrence ``h_t = h_{t-1} exp(g) + k_t^T v_t``,
    ``y_t = q_t h_t``, with ``g`` a per-head decay constant over time.

    Past one step it runs chunked. With ``r_t`` the number of valid positions up
    to ``t``, unrolling gives

        y_t = sum_{s <= t} exp(g (r_t - r_s)) (q_t . k_s) v_s + exp(g r_t) q_t h_0

    so each block of ``C`` steps is one decay-weighted causal attention plus one
    state carry, both matmuls.

    Args:
        mask (mx.array, optional): A ``[B, L]`` validity mask. Padded positions
          contribute no ``k v^T`` and do not decay the state.
    """
    B, H, L, K = q.shape
    V = v.shape[-1]
    out_dtype = q.dtype
    if h is None:
        # Hold the state in float32, as the reference does.
        h = mx.zeros((B, H, K, V), dtype=mx.float32)

    if L == 1:
        exp_g = mx.exp(g)[:, None, None].astype(h.dtype)
        y, h_t = _recurrent_gla_step(q * scale, k, v, h, exp_g)
        if mask is None:
            return y, h_t
        keep = mask[:, 0][:, None, None, None]
        return mx.where(keep, y, 0), mx.where(keep, h_t, h)

    # exp(|g| C / 2) must stay in float32 and |g| < 1, so 176 is the ceiling.
    C = min(chunk, L, 128)
    n_chunks = (L + C - 1) // C
    pad = n_chunks * C - L

    # Rank counts valid positions, so padding neither contributes nor decays.
    if mask is None:
        m = (mx.arange(n_chunks * C) < L).astype(mx.float32)[None]
    else:
        m = mx.pad(mask.astype(mx.float32), [(0, 0), (0, pad)])
    if pad:
        pad_width = [(0, 0), (0, 0), (0, pad), (0, 0)]
        q = mx.pad(q, pad_width)
        k = mx.pad(k, pad_width)
        v = mx.pad(v, pad_width)

    q = mx.unflatten(q, 2, (n_chunks, C))
    k = mx.unflatten(k, 2, (n_chunks, C))
    v = mx.unflatten(v, 2, (n_chunks, C)).astype(mx.float32)

    m = mx.unflatten(m, 1, (n_chunks, C))[:, None]
    rank = mx.cumsum(m, axis=-1)
    total = rank[..., -1:]
    gh = g.reshape(1, H, 1, 1)

    # Centring the rank splits exp(g (r_i - r_j)) over i and j within float32.
    centre = 0.5 * C
    e = gh * (rank - centre)
    eq = mx.exp(e) * scale * m
    ek = mx.exp(-e) * m
    q = q * eq[..., None]
    k = k * ek[..., None]

    # The strict upper triangle overflows to +-inf; `tril` selects, so it drops
    # those instead of multiplying them into NaN.
    scores = mx.tril(q @ k.swapaxes(-1, -2))
    intra = scores @ v
    carry = k.swapaxes(-1, -2) @ v
    decay = mx.exp(gh * (total - centre))[..., None]
    state_decay = mx.exp(gh * total)[..., None]

    cross = []
    for i in range(n_chunks):
        cross.append(q[:, :, i] @ h)
        h = state_decay[:, :, i] * h + carry[:, :, i] * decay[:, :, i]
    cross = mx.stack(cross, axis=2)

    cross_scale = mx.exp(g * centre).reshape(1, H, 1, 1, 1)
    y = (intra + cross * cross_scale).astype(out_dtype)
    return mx.flatten(y, 2, 3)[:, :, :L], h


class GroupRMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5, groups: int = 1):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.groups = groups
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.unflatten(x, axis=-1, shape=(self.groups, -1))
        x = mx.fast.rms_norm(x, weight=None, eps=self.eps)
        return self.weight * mx.flatten(x, -2)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs, intermediate_size: Optional[int] = None):
        super().__init__()
        dim = (
            intermediate_size
            if intermediate_size is not None
            else args.intermediate_size
        )
        self.gate_proj = nn.Linear(args.hidden_size, dim, bias=args.use_bias)
        self.up_proj = nn.Linear(args.hidden_size, dim, bias=args.use_bias)
        self.down_proj = nn.Linear(dim, args.hidden_size, bias=args.use_bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class MultiLatentAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.q_lora_rank = args.q_lora_rank
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.kv_lora_rank = args.kv_lora_rank
        self.v_head_dim = args.v_head_dim
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim

        self.scale = self.qk_head_dim**-0.5

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                args.hidden_size, self.num_heads * self.qk_head_dim, bias=False
            )
        else:
            self.q_a_proj = nn.Linear(
                args.hidden_size, self.q_lora_rank, bias=args.use_qkv_bias
            )
            self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=args.rms_norm_eps)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            args.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=args.use_qkv_bias,
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=args.rms_norm_eps)

        self.embed_q = MultiLinear(
            self.qk_nope_head_dim, self.kv_lora_rank, self.num_heads
        )
        self.unembed_out = MultiLinear(
            self.kv_lora_rank, self.v_head_dim, self.num_heads
        )

        self.dense = nn.Linear(
            self.num_heads * self.v_head_dim,
            args.hidden_size,
            bias=args.use_qkv_bias,
        )

        self.scale = apply_yarn_mscale(self.scale, args.rope_scaling)

        self.rope = initialize_rope(
            dims=self.qk_rope_head_dim,
            base=args.rope_theta,
            traditional=args.rope_interleave,
            max_position_embeddings=args.max_position_embeddings,
            scaling_config=args.rope_scaling,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))

        q = q.reshape(B, L, self.num_heads, self.qk_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = self.kv_a_layernorm(compressed_kv)

        offset = cache.offset if cache is not None else 0
        q_pe = self.rope(q_pe, offset)
        k_pe = self.rope(k_pe, offset)

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
        return self.dense(output)


class LinearAttention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_qk_norm = args.use_qk_norm
        self.num_hidden_layers = args.num_hidden_layers
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_attention_heads
        self.head_dim = args.head_dim or args.hidden_size // self.num_attention_heads
        self.scale = self.head_dim**-0.5

        self.query_key_value = nn.Linear(
            args.hidden_size,
            (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=args.use_qkv_bias,
        )

        self.dense = nn.Linear(
            self.num_attention_heads * self.head_dim,
            args.hidden_size,
            bias=args.use_bias,
        )

        self.g_proj = nn.Linear(
            args.hidden_size, self.num_attention_heads * self.head_dim, bias=False
        )
        self.g_norm = GroupRMSNorm(
            self.num_attention_heads * self.head_dim,
            eps=args.rms_norm_eps,
            groups=args.group_norm_size,
        )

        if args.use_qk_norm:
            self.key_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
            self.query_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        self.rope = initialize_rope(
            int(self.head_dim * args.partial_rotary_factor),
            args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )
        self._slope = self._get_slopes()

    def _get_slopes(self) -> mx.array:
        n = self.num_attention_heads

        def power_of_2_slopes(n):
            return [2 ** (-(2 ** -(math.log2(n) - 3)) * (i + 1)) for i in range(n)]

        if math.log2(n).is_integer():
            slopes = power_of_2_slopes(n)
        else:
            p = 2 ** math.floor(math.log2(n))
            slopes = power_of_2_slopes(p) + power_of_2_slopes(2 * p)[::2][: n - p]

        slopes = mx.array(slopes, dtype=mx.float32)
        denom = max(1, self.num_hidden_layers - 1)
        layer_factor = 1 - self.layer_idx / denom + 1e-5
        return -slopes * layer_factor

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        offset: int = 0,
    ) -> mx.array:
        B, L, _ = x.shape

        qkv = self.query_key_value(x).reshape(
            B,
            L,
            self.num_attention_heads + 2 * self.num_key_value_heads,
            self.head_dim,
        )
        q, k, v = mx.split(
            qkv,
            [
                self.num_attention_heads,
                self.num_attention_heads + self.num_key_value_heads,
            ],
            axis=2,
        )

        queries = q.transpose(0, 2, 1, 3)
        keys = k.transpose(0, 2, 1, 3)
        values = v.transpose(0, 2, 1, 3)

        if self.use_qk_norm:
            queries = self.query_layernorm(queries)
            keys = self.key_layernorm(keys)

        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)

        if cache is None:
            cache = [None]
        output, cache[0] = recurrent_gla(
            q=queries,
            k=keys,
            v=values,
            g=self._slope,
            scale=self.scale,
            h=cache[0],
            mask=mask,
        )
        if hasattr(cache, "advance"):
            # Keeps ``lengths`` aligned with the next chunk of a padded prompt.
            cache.advance(L)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        output = self.g_norm(output) * mx.sigmoid(self.g_proj(x))
        return self.dense(output)


class Gate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.n_group = args.n_group
        self.topk_group = args.topk_group
        self.norm_topk_prob = args.norm_topk_prob
        self.routed_scaling_factor = args.routed_scaling_factor
        self.score_function = args.score_function

        self.gate_proj = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.expert_bias = (
            mx.zeros((args.num_experts,))
            if args.moe_router_enable_expert_bias
            else None
        )

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        gates = self.gate_proj(x)
        in_type = gates.dtype
        gates = gates.astype(mx.float32)
        if self.score_function == "sigmoid":
            scores = mx.sigmoid(gates)
        else:
            scores = mx.softmax(gates, axis=-1)

        # The bias steers the selection only, the weights use the raw scores.
        orig_scores = scores
        if self.expert_bias is not None:
            scores = scores + self.expert_bias

        n_drop = self.n_group - self.topk_group
        if n_drop > 0:
            scores = mx.unflatten(scores, axis=-1, shape=(self.n_group, -1))
            group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
            group_idx = mx.argpartition(group_scores, kth=n_drop - 1, axis=-2)[
                ..., :n_drop, :
            ]
            # -inf, not 0, because expert_bias can make a kept score negative.
            scores = mx.put_along_axis(
                scores,
                mx.stop_gradient(group_idx),
                mx.array(-mx.inf, scores.dtype),
                axis=-2,
            )
            scores = mx.flatten(scores, -2, -1)

        inds = mx.argpartition(-scores, kth=self.top_k - 1, axis=-1)[..., : self.top_k]
        inds = mx.stop_gradient(inds)
        scores = mx.take_along_axis(orig_scores, inds, axis=-1)
        if self.top_k > 1 and self.norm_topk_prob:
            scores = scores / (scores.sum(axis=-1, keepdims=True) + 1e-20)
        scores = scores * self.routed_scaling_factor
        return inds, scores.astype(in_type)


class SparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts_per_tok = args.num_experts_per_tok
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.num_experts,
            bias=args.use_bias,
        )
        self.gate = Gate(args)
        shared_size = (
            args.moe_shared_expert_intermediate_size
            or args.moe_intermediate_size * args.num_shared_experts
        )
        self.shared_experts = (
            MLP(args, intermediate_size=shared_size)
            if args.num_shared_experts > 0
            else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        topk_idx, topk_weight = self.gate(x)
        out = self.switch_mlp(x, topk_idx)
        out = (out * topk_weight[..., None]).sum(axis=-2)
        if self.shared_experts is not None:
            out = out + self.shared_experts(x)
        return out


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        n_layers = args.num_hidden_layers
        group = args.layer_group_size
        self.is_global = (layer_idx + 1) % group == 0 or layer_idx >= (
            n_layers // group
        ) * group

        if self.is_global:
            self.attention = MultiLatentAttention(args)
        else:
            self.attention = LinearAttention(args, layer_idx=layer_idx)

        if args.num_experts is not None and layer_idx >= args.first_k_dense_replace:
            self.mlp = SparseMoeBlock(args)
        else:
            self.mlp = MLP(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        offset: int = 0,
    ) -> mx.array:
        if self.is_global:
            r = self.attention(self.input_layernorm(x), mask, cache)
        else:
            r = self.attention(self.input_layernorm(x), mask, cache, offset=offset)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class LanguageModel(PipelineMixin, nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.layer_group_size = args.layer_group_size
        self._set_mask_indices()

    def _set_mask_indices(self):
        # One layer of each kind, to size the two masks and read the offset.
        layers = self.pipeline_layers
        self._attn_idx = next((i for i, l in enumerate(layers) if l.is_global), None)
        self._gla_idx = next((i for i, l in enumerate(layers) if not l.is_global), None)

    def pipeline(self, group, split=None):
        super().pipeline(group, split=split)
        self._set_mask_indices()
        if self._attn_idx is None:
            raise ValueError(
                "Every pipeline rank needs one MLA layer, which carries the "
                "position offset the linear layers rope with. Pass a split with "
                f"at least {self.layer_group_size} layers per rank."
            )

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.word_embeddings(inputs)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * len(self.pipeline_layers)

        attn_cache = cache[self._attn_idx] if self._attn_idx is not None else None
        attn_mask = create_attention_mask(h, attn_cache, return_array=True)
        gla_mask = (
            create_ssm_mask(h, cache[self._gla_idx])
            if self._gla_idx is not None
            else None
        )
        offset = attn_cache.offset if attn_cache is not None else 0

        # Receive from the previous process in the pipeline
        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        for layer, c in zip(self.pipeline_layers, cache):
            mask = attn_mask if layer.is_global else gla_mask
            h = layer(h, mask, c, offset=offset)

        # Send to the next process in the pipeline
        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            if cache[-1] is not None:
                if hasattr(cache[-1], "keys"):
                    cache[-1].keys = mx.depends(cache[-1].keys, h)
                else:
                    cache[-1][0] = mx.depends(cache[-1][0], h)

        # Broadcast h while keeping it in the graph
        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LanguageModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            return self.model.word_embeddings.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights):
        n_layers = self.args.num_hidden_layers

        # Drop MTP and any extra non-base layers (Ling 2.6 has 1 MTP layer
        # appended after num_hidden_layers; deletes weights for those).
        weights = {
            k: v
            for k, v in weights.items()
            if not (
                k.startswith("model.layers.")
                and k.split(".")[2].isdigit()
                and int(k.split(".")[2]) >= n_layers
            )
        }

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        for l in range(n_layers):
            prefix = f"model.layers.{l}"

            # MoE expert stacking + gate remap
            if l >= self.args.first_k_dense_replace:
                for m in ["gate_proj", "down_proj", "up_proj"]:
                    for k in ["weight", "scales", "biases"]:
                        if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                            stacked = [
                                weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                                for e in range(self.args.num_experts)
                            ]
                            weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(
                                stacked
                            )

                if f"{prefix}.mlp.gate.weight" in weights:
                    weights[f"{prefix}.mlp.gate.gate_proj.weight"] = weights.pop(
                        f"{prefix}.mlp.gate.weight"
                    )
                if f"{prefix}.mlp.gate.bias" in weights:
                    weights[f"{prefix}.mlp.gate.gate_proj.bias"] = weights.pop(
                        f"{prefix}.mlp.gate.bias"
                    )

            # MLA kv_b_proj split for global attention layers.
            kv_b_key = f"{prefix}.attention.kv_b_proj.weight"
            if kv_b_key in weights:
                v = weights.pop(kv_b_key)
                head_dim = self.args.qk_nope_head_dim + self.args.v_head_dim
                num_heads = self.args.num_attention_heads
                v = v.reshape(num_heads, head_dim, -1)
                wk = mx.contiguous(
                    v[:, : self.args.qk_nope_head_dim, :].swapaxes(-1, -2)
                )
                wv = mx.contiguous(v[:, self.args.qk_nope_head_dim :, :])
                weights[f"{prefix}.attention.embed_q.weight"] = wk
                weights[f"{prefix}.attention.unembed_out.weight"] = wv

        return weights

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("mlp.gate.gate_proj"):
                return {"group_size": 64, "bits": 8}
            # The absorbed MLA matrices stay unquantized, as in bailing_moe_v3.
            if path.endswith(("attention.embed_q", "attention.unembed_out")):
                return False
            return True

        return predicate

    @property
    def cast_predicate(self):
        def predicate(k):
            return "expert_bias" not in k

        return predicate

    @property
    def layers(self):
        return self.model.pipeline_layers

    def make_cache(self):
        caches = []
        for l in self.layers:
            if l.is_global:
                caches.append(KVCache())
            else:
                caches.append(ArraysCache(size=1))
        return caches
