# Copyright © 2026 Apple Inc.

"""Muse Spark.

https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B-NVFP4
"""

import re
from dataclasses import dataclass, field
from typing import Any, Callable

import mlx.core as mx
from mlx import nn
from mlx.nn.layers.distributed import shard_linear, sum_gradients

from ..kernels import (
    expert_routing,
    norm_add_norm,
    post_expert_norm_combine,
    pre_norm,
    qk_norm,
    qk_norm_rope,
    rms_norm_sigmoid_gate,
)
from .activations import precise_swiglu
from .base import (
    BaseModelArgs,
    create_attention_mask,
    gather_last_axis,
    scaled_dot_product_attention,
)
from .cache import KVCache, RotatingKVCache
from .pipeline import PipelineMixin
from .switch_layers import SwitchGLU

# For checkpoints without them. Stop on <|eot|>, not on <|eom|>.
TOKENIZER_DEFAULTS = {
    "chat_template_type": "muse_spark",
    "tool_parser_type": "muse_spark",
    "eos_token_ids": [200001, 200008],
}


@dataclass
class TextArgs(BaseModelArgs):
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_hidden_layers: int
    vocab_size: int
    rms_norm_eps: float
    post_norm_eps: float
    rope_parameters: dict
    # A 0 marks a layer without RoPE.
    layer_rope_theta: list[float]
    layer_types: list[str]
    sliding_window: int
    final_logit_softcapping: float
    output_multiplier: float
    num_local_experts: int
    num_experts_per_tok: int
    moe_hidden_size: int
    moe_intermediate_size: int
    qk_scale_factor: float = 5.4730064863875
    routed_scaling_factor: float = 1.0
    num_shared_experts: int = 0
    mlp_layer_types: list[str] | None = None

    def __post_init__(self) -> None:
        if self.num_shared_experts or any(
            t != "sparse" for t in self.mlp_layer_types or []
        ):
            raise NotImplementedError("Dense feed-forward layers are not supported.")
        self.rope_theta = self.rope_parameters["rope_theta"]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict
    vision_config: dict | None = None
    # Image rows go at <|patch|>: the processor expands each <|patch|> of the
    # prompt to the rows of one image.
    image_token_id: int = 200092
    out_hidden_size: int | None = None
    projector_hidden_size: int | None = None

    def __post_init__(self) -> None:
        self.text = TextArgs.from_dict(self.text_config)
        self.vision = None
        if self.vision_config:
            self.vision = VisionArgs.from_dict(self.vision_config)
            if self.out_hidden_size is None:
                self.out_hidden_size = (
                    self.vision.hidden_size * self.vision.merge_size**2
                )
            if self.projector_hidden_size is None:
                self.projector_hidden_size = self.out_hidden_size


class InputScaleRMSNorm(nn.Module):

    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(self.weight * x, None, self.eps)


def residual_gate_coefficients(
    gate: mx.array, *, temperature: float = 0.3, eps: float = 1e-3
) -> tuple[mx.array, mx.array]:
    """Return the weights of the stream and of the branch for a gate."""
    w = gate.astype(mx.float32) / temperature
    beta = mx.sigmoid(w)
    alpha = mx.sqrt(mx.clip(mx.sigmoid(-w) * (1.0 + beta), eps, None))
    return alpha, beta


def norm_gain(weight: mx.array) -> mx.array:
    return (1.0 + weight.astype(mx.float32)).astype(mx.bfloat16)


def interleave_rope_rows(w: mx.array, head_dim: int) -> mx.array:
    """Hugging Face orders the q and k rows of a head for RoPE pairs (i, i +
    head_dim / 2). The model pairs neighbor rows (2i, 2i + 1)."""
    shape = w.shape
    w = w.reshape(-1, 2, head_dim // 2, *shape[1:])
    return mx.swapaxes(w, 1, 2).reshape(shape)


POOLED_TP_MAX_TOKENS = 8

# Added to the sum of the top-k router scores.
_ROUTER_SCORE_EPS = 1e-15


def row_parallel_linear(
    linear: nn.Linear, x: mx.array, group: mx.distributed.Group
) -> mx.array:
    assert "bias" not in linear, "Linear layers with a bias are not supported."
    rows = linear.weight.shape[0] // group.size()
    r = group.rank()
    part = x @ linear.weight[r * rows : (r + 1) * rows].T
    return gather_last_axis(part, group)


_UNUSED_QUANT_TENSORS = (
    "mlp.experts.gate_up_proj_input_scale",
    "mlp.experts.down_proj_input_scale",
)

# Norm weights the checkpoint stores centered at zero; sanitize stores the gain.
_ZERO_CENTERED_NORMS = (
    "input_layernorm.weight",
    "pre_feedforward_layernorm.weight",
    "post_feedforward_layernorm.weight",
    "mlp.pre_expert_norm.weight",
    "mlp.experts.post_expert_norm.weight",
)

_NVFP4_E2M1_MAX = 6.0
_NVFP4_E4M3_MAX = 448.0
_NVFP4_SCALE_2_TO_GLOBAL = _NVFP4_E2M1_MAX * _NVFP4_E4M3_MAX


# TODO: mx.contiguous of a slice fails on arrays this large; copy in chunks.
_MAX_COPY_ELEMENTS = 1 << 31


def shard_experts_inplace(
    module: nn.Module, sharding: str, group: mx.distributed.Group
) -> None:
    N, r = group.size(), group.rank()
    updates = {}
    for name, w in module.parameters().items():
        if not isinstance(w, mx.array):
            continue
        if w.ndim < 3:
            # we don't shard scales for the nvfp4 weights
            continue
        axis = w.ndim - 2 if sharding == "all-to-sharded" else w.ndim - 1
        n = w.shape[axis] // N
        lo, hi = r * n, (r + 1) * n
        per_expert = w.size // w.shape[0]
        step = max(1, _MAX_COPY_ELEMENTS // (2 * per_expert))
        parts = []
        for i in range(0, w.shape[0], step):
            chunk = w[i : i + step]
            chunk = chunk[:, lo:hi] if axis == 1 else chunk[..., lo:hi]
            chunk = mx.contiguous(chunk)
            mx.eval(chunk)
            parts.append(chunk)
        updates[name] = mx.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
    module.update(updates)


class Attention(nn.Module):
    def __init__(self, args: TextArgs, *, use_rope: bool):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim

        self.scale = head_dim**-0.5
        # Queries carry a tuned constant scale of their own, on top of the
        # usual 1/sqrt(head_dim) applied inside the attention call
        self.query_scale = args.qk_scale_factor
        self.eps = args.rms_norm_eps

        self.qkvg_proj = nn.Linear(
            dim, 2 * (n_heads + n_kv_heads) * head_dim, bias=False
        )
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.use_rope = use_rope
        if use_rope:
            self.rope = nn.RoPE(head_dim, traditional=True, base=args.rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        B, L, _ = x.shape

        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim
        qkvg = self.qkvg_proj(x)
        norm_args = {
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "eps": self.eps,
            "query_scale": self.query_scale,
        }
        if self.use_rope:
            # Also rotates q and k, at the positions after the cache.
            offset = cache.offset if cache is not None else 0
            queries, keys = qk_norm_rope(
                qkvg, **norm_args, rope_base=self.rope.base, offset=offset
            )
        else:
            queries, keys = qk_norm(qkvg, **norm_args)
        values = qkvg[..., q_dim + kv_dim : q_dim + 2 * kv_dim]
        gate = qkvg[..., q_dim + 2 * kv_dim :]

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        return self.o_proj(rms_norm_sigmoid_gate(output, gate, eps=self.eps))


class PreciseSwiGLU(nn.Module):
    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        return precise_swiglu(x, gate, x)


class MoEGate(nn.Module):
    """The router, (num_experts, dim) for a gemv, and its score bias."""

    def __init__(self, dim: int, num_experts: int):
        super().__init__()
        self.weight = mx.zeros((num_experts, dim), dtype=mx.float32)
        self.e_score_correction_bias = mx.zeros((num_experts,), dtype=mx.float32)


class MoE(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()

        dim = args.hidden_size
        compressed = args.moe_hidden_size
        num_experts = args.num_local_experts

        self.top_k = args.num_experts_per_tok
        self.route_scale = args.routed_scaling_factor
        self.score_eps = _ROUTER_SCORE_EPS
        self.router_multiplier = args.output_multiplier

        self.gate = MoEGate(dim, num_experts)

        # The experts run at a compressed width, between the two projections.
        self.pre_expert_proj = nn.Linear(dim, compressed, bias=False)
        self.pre_expert_norm = nn.RMSNorm(compressed, eps=args.rms_norm_eps)
        self.experts = SwitchGLU(
            compressed,
            args.moe_intermediate_size,
            num_experts,
            activation=PreciseSwiGLU(),
        )
        # The checkpoint stores this norm under the experts.
        self.experts.post_expert_norm = InputScaleRMSNorm(
            compressed, eps=args.post_norm_eps
        )
        self.post_expert_proj = nn.Linear(compressed, dim, bias=False)

        self.sharding_group = None

    def __call__(self, x: mx.array) -> mx.array:
        group = self.sharding_group
        if group is not None:
            x = sum_gradients(group)(x)
        # Quantized expert projections stay whole.
        split = (
            group is not None
            and group.size() > 1
            and x.size // x.shape[-1] <= POOLED_TP_MAX_TOKENS
            and isinstance(self.pre_expert_proj, nn.Linear)
            and isinstance(self.post_expert_proj, nn.Linear)
        )

        logits = x.astype(mx.float32) @ self.gate.weight.T
        inds, weights = expert_routing(
            logits,
            self.gate.e_score_correction_bias,
            top_k=self.top_k,
            multiplier=self.router_multiplier,
            score_eps=self.score_eps,
            route_scale=self.route_scale,
        )

        h = (
            row_parallel_linear(self.pre_expert_proj, x, group)
            if split
            else self.pre_expert_proj(x)
        )
        h = self.experts(self.pre_expert_norm(h), inds)
        if group is not None:
            # Sum the expert outputs before the post-expert norm.
            h = mx.distributed.all_sum(h, group=group)
        norm = self.experts.post_expert_norm
        h = post_expert_norm_combine(h, weights, norm_weight=norm.weight, eps=norm.eps)
        return (
            row_parallel_linear(self.post_expert_proj, h, group)
            if split
            else self.post_expert_proj(h)
        )


class TransformerBlock(nn.Module):
    def __init__(self, args: TextArgs, *, use_rope: bool, use_sliding: bool):
        super().__init__()

        dim = args.hidden_size
        self.use_sliding = use_sliding
        self.post_norm_eps = args.post_norm_eps

        self.input_layernorm = nn.RMSNorm(dim, eps=args.rms_norm_eps)
        self.self_attn = Attention(args, use_rope=use_rope)
        self.pre_feedforward_layernorm = nn.RMSNorm(dim, eps=args.rms_norm_eps)
        self.mlp = MoE(args)
        self.post_feedforward_layernorm = InputScaleRMSNorm(dim, eps=args.post_norm_eps)
        # Sanitize computes these from the fixed gates. The defaults are the
        # values for a zero gate.
        zero_gate = mx.zeros((dim,))
        self.add_attn_alpha, self.add_attn_beta = residual_gate_coefficients(zero_gate)
        self.add_ffn_alpha, self.add_ffn_beta = residual_gate_coefficients(zero_gate)

    def __call__(
        self,
        x: mx.array,
        *,
        mask: mx.array | None = None,
        cache: Any | None = None,
        attn_input: mx.array | None = None,
        next_norm: nn.Module | None = None,
    ) -> tuple[mx.array, mx.array | None]:
        if attn_input is None:
            attn_input = self.input_layernorm(x).astype(mx.bfloat16)
        r = self.self_attn(attn_input, mask, cache)
        h, ffn_input = norm_add_norm(
            r,
            x,
            alpha=self.add_attn_alpha,
            beta=self.add_attn_beta,
            post_weight=None,
            post_eps=self.post_norm_eps,
            next_norm=self.pre_feedforward_layernorm,
        )
        r = self.mlp(ffn_input)
        return norm_add_norm(
            r,
            h,
            alpha=self.add_ffn_alpha,
            beta=self.add_ffn_beta,
            post_weight=self.post_feedforward_layernorm.weight,
            post_eps=self.post_feedforward_layernorm.eps,
            next_norm=next_norm,
        )


class WeightlessRMSNorm(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, None, self.eps)


class PreciseLayerNorm(nn.LayerNorm):
    """As PyTorch: LayerNorm in fp32, rounded once."""

    def __call__(self, x: mx.array) -> mx.array:
        f = mx.float32
        w, b = self.weight.astype(f), self.bias.astype(f)
        return mx.fast.layer_norm(x.astype(f), w, b, self.eps).astype(x.dtype)


def _precise_gelu(x: mx.array) -> mx.array:
    """As PyTorch: GELU in fp32, rounded once."""
    return nn.gelu(x.astype(mx.float32)).astype(x.dtype)


def _splice_vision_embeddings(
    residual: mx.array,
    input_ids: mx.array,
    *,
    vision_embeds: mx.array,
    patch_token_id: int,
) -> mx.array:
    """Put the rows of vision_embeds, in order, at the patch tokens."""
    # A shape check only, so no host sync.
    if vision_embeds.shape[0] == 0:
        return residual
    mask = input_ids == patch_token_id
    # The row of each masked token: the number of masked tokens before it.
    idx = mx.cumsum(mask.flatten()).reshape(mask.shape) - 1
    idx = mx.clip(idx, 0, vision_embeds.shape[0] - 1)
    rows = vision_embeds[idx].astype(residual.dtype)
    return mx.where(mask[..., None], rows, residual)


def _pixel_shuffle_downsample(
    x: mx.array, *, grid_h: int, grid_w: int, factor: int = 2
) -> mx.array:
    """Merge each factor x factor block of patches into one token."""
    d = x.shape[-1]
    h_blocks = grid_h // factor
    w_blocks = grid_w // factor
    x = x.reshape(h_blocks, factor, w_blocks, factor, d)
    # The features of a token: channel, then row and column in the block.
    x = x.transpose(0, 2, 4, 1, 3)
    return x.reshape(h_blocks * w_blocks, d * factor * factor)


def _linspace(start: float, end: float, steps: int, /, *, dtype: mx.Dtype) -> mx.array:
    """torch.linspace on CUDA: start and end in dtype, the steps in fp32."""
    start = mx.array(start, dtype=dtype).astype(mx.float32)
    end = mx.array(end, dtype=dtype).astype(mx.float32)
    step = (end - start) / (steps - 1)
    i = mx.arange(steps)
    v = mx.where(i < steps // 2, start + step * i, end - step * (steps - 1 - i))
    return v.astype(dtype)


def _position_embedding(
    table: mx.array, *, size: int, grid_h: int, grid_w: int, ij: bool
) -> mx.array:
    """The size x size position table at a grid_h x grid_w grid, sampled
    bilinearly (align_corners=False, zero padding)."""
    dtype = table.dtype
    pos_h = _linspace(-1 + 1 / grid_h, 1 - 1 / grid_h, grid_h, dtype=dtype)
    pos_w = _linspace(-1 + 1 / grid_w, 1 - 1 / grid_w, grid_w, dtype=dtype)
    t = mx.arange(grid_h * grid_w)
    if ij:
        x, y = pos_w[t % grid_w], pos_h[t // grid_w]
    else:
        # The legacy grid: transposed unless the grid is square.
        x, y = pos_h[t % grid_h], pos_w[t // grid_h]
    ix = ((x.astype(mx.float32) + 1) * size - 1) / 2
    iy = ((y.astype(mx.float32) + 1) * size - 1) / 2
    x0, y0 = mx.floor(ix), mx.floor(iy)
    corners = (
        (x0, y0, (x0 + 1 - ix) * (y0 + 1 - iy)),
        (x0 + 1, y0, (ix - x0) * (y0 + 1 - iy)),
        (x0, y0 + 1, (x0 + 1 - ix) * (iy - y0)),
        (x0 + 1, y0 + 1, (ix - x0) * (iy - y0)),
    )
    table = table.astype(mx.float32)
    out = mx.zeros((grid_h * grid_w, table.shape[-1]), dtype=mx.float32)
    for cx, cy, w in corners:
        inside = (cx >= 0) & (cx < size) & (cy >= 0) & (cy < size)
        idx = mx.clip(cy, 0, size - 1) * size + mx.clip(cx, 0, size - 1)
        out = out + mx.where(inside, w, 0)[:, None] * table[idx.astype(mx.int32)]
    return out.astype(dtype)


def _attention_tiles(
    *, grid_h: int, grid_w: int, tile: int
) -> tuple[list[mx.array], mx.array] | None:
    """The token indices of each tile x tile block, grouped by block shape, and
    the order that puts the grouped tokens back. None for a single block."""
    if grid_h <= tile and grid_w <= tile:
        return None
    pos = mx.arange(grid_h * grid_w).reshape(grid_h, grid_w)
    groups = {}
    for r in range(0, grid_h, tile):
        for c in range(0, grid_w, tile):
            block = pos[r : r + tile, c : c + tile]
            groups.setdefault(block.shape, []).append(block.flatten())
    groups = [mx.stack(g) for g in groups.values()]
    order = mx.concatenate([g.flatten() for g in groups])
    return groups, mx.argsort(order)


@dataclass
class VisionArgs(BaseModelArgs):
    hidden_size: int = 1536
    intermediate_size: int = 8960
    num_hidden_layers: int = 50
    num_attention_heads: int = 16
    num_channels: int = 3
    patch_size: int = 14
    patch_temporal: int = 2
    merge_size: int = 2
    # The position table; images can be larger or smaller.
    pos_emb_height: int = 32
    pos_emb_width: int = 32
    layer_norm_eps: float = 1e-5
    rope_parameters: dict = field(default_factory=dict)
    # window_attention attends inside pos_emb blocks of patches.
    layer_types: list[str] | None = None
    use_ij_pos_encoding: bool = False

    def __post_init__(self) -> None:
        if self.pos_emb_height != self.pos_emb_width:
            raise NotImplementedError("Only square position tables are supported.")
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.grid_size = self.pos_emb_height
        if self.layer_types is None:
            # As transformers: full attention in every 4th and the last layer.
            self.layer_types = [
                (
                    "full_attention"
                    if (i + 1) % 4 == 0 or i == self.num_hidden_layers - 1
                    else "window_attention"
                )
                for i in range(self.num_hidden_layers)
            ]


class VisionRoPE2D(nn.Module):
    """2D RoPE: head_dim // 4 frequencies per axis, positions start at 1,
    width then height, interleaved pairs."""

    def __init__(self, *, head_dim: int, theta: float = 10000.0):
        super().__init__()
        dim_quarter = head_dim // 4
        self._inv_freq = 1.0 / (
            theta ** (mx.arange(0, dim_quarter, dtype=mx.float32) * 4.0 / head_dim)
        )

    def __call__(self, *, grid_h: int, grid_w: int) -> tuple[mx.array, mx.array]:
        """The cos and sin of the patches of a grid_h x grid_w image, row by row,
        as (1, N, 1, head_dim // 2) to match q and k as (1, N, n_heads, head_dim)."""
        shape = (grid_h, grid_w, self._inv_freq.shape[0])
        fw = mx.arange(1, grid_w + 1, dtype=mx.float32)[:, None] * self._inv_freq
        fh = mx.arange(1, grid_h + 1, dtype=mx.float32)[:, None] * self._inv_freq
        fw, fh = mx.broadcast_to(fw[None], shape), mx.broadcast_to(fh[:, None], shape)
        freqs = mx.concatenate([fw, fh], axis=-1).reshape(grid_h * grid_w, -1)
        return mx.cos(freqs)[None, :, None], mx.sin(freqs)[None, :, None]


def _apply_vision_rope(x: mx.array, *, cos: mx.array, sin: mx.array) -> mx.array:
    """Rotate x of shape (..., N, heads, head_dim)."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    # Rotate in fp32, then cast back.
    return mx.stack([o1, o2], axis=-1).reshape(x.shape).astype(x.dtype)


class VisionAttention(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def _attention(self, q: mx.array, k: mx.array, v: mx.array) -> mx.array:
        """q, k, v and the output: (batch, N, n_heads, head_dim)."""
        q, k, v = (a.transpose(0, 2, 1, 3) for a in (q, k, v))
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=None)
        return out.transpose(0, 2, 1, 3)

    def __call__(
        self,
        x: mx.array,
        *,
        cos: mx.array,
        sin: mx.array,
        tiles: tuple[list[mx.array], mx.array] | None = None,
    ) -> mx.array:
        # One image, no batch dimension.
        N = x.shape[0]
        q, k, v = mx.split(self.qkv_proj(x), 3, axis=-1)
        shape = (1, N, self.n_heads, self.head_dim)
        q = _apply_vision_rope(q.reshape(shape), cos=cos, sin=sin)
        k = _apply_vision_rope(k.reshape(shape), cos=cos, sin=sin)
        v = v.reshape(shape)

        if tiles is None:
            out = self._attention(q, k, v)
        else:
            # Blocks of the same shape, which attend only to themselves.
            groups, order = tiles
            outs = [self._attention(q[0][i], k[0][i], v[0][i]) for i in groups]
            out = mx.concatenate([o.reshape(-1, *shape[2:]) for o in outs])[order]
        return self.proj(out.reshape(N, -1))


class VisionMLP(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        self.fc1 = nn.Linear(args.hidden_size, args.intermediate_size, bias=True)
        self.fc2 = nn.Linear(args.intermediate_size, args.hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(_precise_gelu(self.fc1(x)))


class VisionBlock(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        self.norm1 = PreciseLayerNorm(args.hidden_size, eps=args.layer_norm_eps)
        self.attn = VisionAttention(args)
        self.norm2 = PreciseLayerNorm(args.hidden_size, eps=args.layer_norm_eps)
        self.mlp = VisionMLP(args)

    def __call__(
        self,
        x: mx.array,
        *,
        cos: mx.array,
        sin: mx.array,
        tiles: tuple[list[mx.array], mx.array] | None = None,
    ) -> mx.array:
        x = x + self.attn(self.norm1(x), cos=cos, sin=sin, tiles=tiles)
        x = x + self.mlp(self.norm2(x))
        return x


class VisionAdapter(nn.Module):
    def __init__(self, input_dims: int, dims: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dims, dims, bias=False)
        self.fc2 = nn.Linear(dims, dims, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        x = _precise_gelu(self.fc1(x))
        x = _precise_gelu(self.fc2(x))
        return x


class VisionPatchEmbedder(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        dim = args.hidden_size
        patch_dim = args.num_channels * args.patch_size**2 * args.patch_temporal
        self.patch_embedding = nn.Linear(patch_dim, dim, bias=False)
        self.position_embedding_table = nn.Embedding(args.grid_size**2, dim)


class VisionTower(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        self.args = args
        dim = args.hidden_size
        self.patch_embedder = VisionPatchEmbedder(args)
        self.ln_pre = PreciseLayerNorm(dim, eps=args.layer_norm_eps)
        theta = args.rope_parameters.get("rope_theta", 10000.0)
        self.rope = VisionRoPE2D(head_dim=args.head_dim, theta=theta)
        self.layers = [VisionBlock(args) for _ in range(args.num_hidden_layers)]
        self.ln_post = PreciseLayerNorm(dim, eps=args.layer_norm_eps)

    def __call__(self, pixel_values: mx.array) -> mx.array:
        """pixel_values: (num_channels, H, W), with H and W multiples of
        patch_size * merge_size."""
        args = self.args
        C, H, W = pixel_values.shape
        p, t, f = args.patch_size, args.patch_temporal, args.merge_size
        if H % (p * f) or W % (p * f):
            raise ValueError(f"Image size {H}x{W} is not a multiple of {p * f}.")
        gh, gw = H // p, W // p

        img = mx.broadcast_to(pixel_values[None], (t, C, H, W))
        patches = img.reshape(t, C, gh, p, gw, p)
        patches = patches.transpose(2, 4, 0, 1, 3, 5)
        patches = patches.reshape(gh * gw, t * C * p * p)

        h = self.patch_embedder.patch_embedding(patches)
        h = h + _position_embedding(
            self.patch_embedder.position_embedding_table.weight,
            size=args.grid_size,
            grid_h=gh,
            grid_w=gw,
            ij=args.use_ij_pos_encoding,
        )
        h = self.ln_pre(h)
        cos, sin = self.rope(grid_h=gh, grid_w=gw)
        tiles = _attention_tiles(grid_h=gh, grid_w=gw, tile=args.grid_size)
        for block, layer_type in zip(self.layers, args.layer_types):
            full = layer_type == "full_attention"
            h = block(h, cos=cos, sin=sin, tiles=None if full else tiles)
        h = self.ln_post(h)

        return _pixel_shuffle_downsample(h, grid_h=gh, grid_w=gw, factor=f)


class MuseSparkTextModel(PipelineMixin, nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.args = args
        self.sliding_window = args.sliding_window
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(
                args,
                use_rope=bool(rope_theta),
                use_sliding=layer_type == "sliding_attention",
            )
            for rope_theta, layer_type in zip(args.layer_rope_theta, args.layer_types)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    @property
    def fa_idx(self) -> int:
        layers = self.pipeline_layers
        return next((i for i, l in enumerate(layers) if not l.use_sliding), 0)

    @property
    def swa_idx(self) -> int | None:
        layers = self.pipeline_layers
        return next((i for i, l in enumerate(layers) if l.use_sliding), None)

    def embed(self, inputs: mx.array) -> mx.array:
        """The normalized token embeddings."""
        return mx.fast.rms_norm(self.embed_tokens(inputs), None, self.args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        *,
        cache: list[Any] | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        # input_embeddings are already normalized.
        h = self.embed(inputs) if input_embeddings is None else input_embeddings
        h = h.astype(mx.float32)

        layers = self.pipeline_layers
        if cache is None:
            cache = [None] * len(layers)

        fa_mask = create_attention_mask(h, cache[self.fa_idx])
        if self.swa_idx is not None:
            swa_mask = create_attention_mask(
                h, cache[self.swa_idx], window_size=self.sliding_window
            )

        rank, size = self.pipeline_rank, self.pipeline_size
        attn_input = None
        if rank < size - 1:
            h = mx.distributed.recv_like(h, rank + 1)
            # Equal to the fused norm of the previous stage, not to input_layernorm.
            norm = layers[0].input_layernorm
            attn_input = pre_norm(h, norm, dtype=norm.weight.dtype)

        # Each block also makes the next block's attention input, so that the
        # norm is fused with its add.
        for i, (layer, c) in enumerate(zip(layers, cache)):
            mask = swa_mask if layer.use_sliding else fa_mask
            nxt = layers[i + 1].input_layernorm if i + 1 < len(layers) else None
            h, attn_input = layer(
                h, mask=mask, cache=c, attn_input=attn_input, next_norm=nxt
            )

        # Send to the next process in the pipeline
        if rank != 0:
            h = mx.distributed.send(h, (rank - 1) % size)
            if cache[-1] is not None:
                cache[-1].keys = mx.depends(cache[-1].keys, h)

        # Broadcast h while keeping it in the graph
        if size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        # The norm reads the fp32 stream.
        return self.norm(h).astype(mx.bfloat16)


class MuseSparkModel(nn.Module):
    """The language model and the vision tower, named as in the checkpoint."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.language_model = MuseSparkTextModel(args.text)
        # The vision tower is not sharded.
        if args.vision is not None:
            self.vision_tower = VisionTower(args.vision)
            self.vision_adapter = VisionAdapter(
                args.out_hidden_size, args.projector_hidden_size
            )
            self.vision_projection = nn.Linear(
                args.projector_hidden_size, args.text.hidden_size, bias=False
            )

    # mlx_lm.utils pipelines model.model with these.
    @property
    def layers(self) -> list[TransformerBlock | None]:
        return self.language_model.layers

    @property
    def start_idx(self) -> int:
        return self.language_model.start_idx

    @property
    def end_idx(self) -> int | None:
        return self.language_model.end_idx

    def pipeline(self, group, split=None) -> None:
        self.language_model.pipeline(group, split)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = MuseSparkModel(args)
        text = args.text
        self.lm_head = nn.Linear(text.hidden_size, text.vocab_size, bias=False)
        self.sharding_group = None
        if args.vision is not None:
            self.perception_emb_norm = WeightlessRMSNorm(eps=text.rms_norm_eps)

    def encode_image(self, pixel_values: mx.array | list[mx.array]) -> mx.array:
        """pixel_values: one image (num_channels, H, W), a list of images of
        any size, or images of one size (num_images, num_channels, H, W)."""
        if isinstance(pixel_values, mx.array) and pixel_values.ndim == 3:
            pixel_values = [pixel_values]
        assert len(pixel_values), "No images."
        assert all(p.ndim == 3 for p in pixel_values), "Images must be (C, H, W)."
        tower, adapter = self.model.vision_tower, self.model.vision_adapter
        dtype = tower.patch_embedder.patch_embedding.weight.dtype
        # TODO: consider packing the images into one sequence, with attention
        # per image (MLX has no varlen attention).
        h = mx.concatenate([adapter(tower(p.astype(dtype))) for p in pixel_values])
        h = self.model.vision_projection(h)
        return self.perception_emb_norm(h)

    def embed_inputs(
        self,
        inputs: mx.array,
        *,
        pixel_values: mx.array | list[mx.array] | None = None,
    ) -> mx.array:
        """The normalized input embeddings, with the vision rows at the patch tokens."""
        h = self.model.language_model.embed(inputs)
        # The vision rows take the dtype of the text embeddings.
        if pixel_values is not None:
            h = _splice_vision_embeddings(
                h,
                inputs,
                vision_embeds=self.encode_image(pixel_values),
                patch_token_id=self.args.image_token_id,
            )
        return h

    def __call__(
        self,
        inputs: mx.array,
        *,
        cache: list[Any] | None = None,
        input_embeddings: mx.array | None = None,
        pixel_values: mx.array | list[mx.array] | None = None,
    ) -> mx.array:
        if pixel_values is not None:
            assert input_embeddings is None, "Pass pixel_values or input_embeddings."
            input_embeddings = self.embed_inputs(inputs, pixel_values=pixel_values)
        out = self.model.language_model(
            inputs, cache=cache, input_embeddings=input_embeddings
        )
        out = self.lm_head(out.astype(self.lm_head.weight.dtype))
        if self.sharding_group is not None:
            out = gather_last_axis(out, self.sharding_group)
        out = out.astype(mx.float32) * self.args.text.output_multiplier
        cap = self.args.text.final_logit_softcapping
        return mx.tanh(out / cap) * cap

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        # Muse Spark's audio tower is not supported.
        drop_prefixes = ("model.audio_tower.",)
        if self.args.vision is None:
            drop_prefixes += (
                "model.vision_tower.",
                "model.vision_adapter.",
                "model.vision_projection.",
            )
        # The gates are fixed, so compute their weights once.
        gates = {
            "post_attention_residual_gate.gate": "add_attn",
            "post_feedforward_residual_gate.gate": "add_ffn",
        }
        layer_re = re.compile(r"^(model\.language_model\.layers\.\d+\.)(.+)$")
        # Weights from mlx_lm.convert already store the gate weights and the
        # norm gains.
        converted = any(k.endswith("_alpha") for k in weights)

        sanitized = {}
        for k, v in weights.items():
            if k.startswith(drop_prefixes):
                continue
            m = layer_re.match(k)
            if m is None or converted:
                sanitized[k] = v
                continue

            prefix, suffix = m.groups()
            experts = prefix + "mlp.experts."
            if suffix in gates:
                alpha, beta = residual_gate_coefficients(v)
                sanitized[prefix + gates[suffix] + "_alpha"] = alpha
                sanitized[prefix + gates[suffix] + "_beta"] = beta
            elif suffix in _ZERO_CENTERED_NORMS:
                sanitized[k] = norm_gain(v)
            elif suffix == "mlp.experts.gate_up_proj":
                gate_proj, up_proj = mx.split(v, 2, axis=1)
                if v.dtype == mx.uint8:
                    # In mlx quantized weights are packed into uint32
                    gate_proj = gate_proj.view(mx.uint32)
                    up_proj = up_proj.view(mx.uint32)
                sanitized[experts + "gate_proj.weight"] = gate_proj
                sanitized[experts + "up_proj.weight"] = up_proj
            elif suffix == "mlp.experts.gate_up_proj_weight_scale":
                # One e4m3 block scale per 16 inputs
                gate_scales, up_scales = mx.split(v, 2, axis=1)
                sanitized[experts + "gate_proj.scales"] = gate_scales
                sanitized[experts + "up_proj.scales"] = up_scales
            elif suffix == "mlp.experts.gate_up_proj_weight_scale_2":
                # Column 0 is the gate half, column 1 the up half.
                gate_gs, up_gs = v[:, 0], v[:, 1]
                sanitized[experts + "gate_proj.global_scale"] = (
                    gate_gs * _NVFP4_SCALE_2_TO_GLOBAL
                )
                sanitized[experts + "up_proj.global_scale"] = (
                    up_gs * _NVFP4_SCALE_2_TO_GLOBAL
                )
            elif suffix == "mlp.experts.down_proj":
                sanitized[experts + "down_proj.weight"] = (
                    v.view(mx.uint32) if v.dtype == mx.uint8 else v
                )
            elif suffix == "mlp.experts.down_proj_weight_scale":
                sanitized[experts + "down_proj.scales"] = v
            elif suffix == "mlp.experts.down_proj_weight_scale_2":
                sanitized[experts + "down_proj.global_scale"] = (
                    v * _NVFP4_SCALE_2_TO_GLOBAL
                )
            elif suffix in _UNUSED_QUANT_TENSORS:
                # we don't quantize activations
                continue
            else:
                sanitized[k] = v

        # The model runs q, k, v and the output gate as one matmul.
        head_dim = self.args.text.head_dim
        projections = ("q_proj", "k_proj", "v_proj", "gate_proj")
        for k in [k for k in sanitized if k.endswith("self_attn.q_proj.weight")]:
            attn = k[: -len("q_proj.weight")]
            for param in ("weight", "scales", "biases"):
                keys = [f"{attn}{p}.{param}" for p in projections]
                if all(key in sanitized for key in keys):
                    q, k_, v, g = (sanitized.pop(key) for key in keys)
                    q = interleave_rope_rows(q, head_dim)
                    k_ = interleave_rope_rows(k_, head_dim)
                    sanitized[f"{attn}qkvg_proj.{param}"] = mx.concatenate(
                        [q, k_, v, g]
                    )

        vision_q = [
            k
            for k in sanitized
            if k.startswith("model.vision_tower.") and k.endswith(".attn.q_proj.weight")
        ]
        for k in vision_q:
            attn = k[: -len("q_proj.weight")]
            for param in ("weight", "bias"):
                keys = [f"{attn}{p}_proj.{param}" for p in "qkv"]
                q, k_, v = (sanitized.pop(key) for key in keys)
                q = interleave_rope_rows(q, self.args.vision.head_dim)
                k_ = interleave_rope_rows(k_, self.args.vision.head_dim)
                sanitized[f"{attn}qkv_proj.{param}"] = mx.concatenate([q, k_, v])
        return sanitized

    def shard(self, group: mx.distributed.Group | None = None) -> None:
        group = group or mx.distributed.init()
        N = group.size()
        n_kv_heads = self.args.text.num_key_value_heads
        assert (
            n_kv_heads % N == 0
        ), f"group size {N} must divide num_key_value_heads {n_kv_heads}"
        vocab = self.args.text.vocab_size
        assert vocab % N == 0, f"group size {N} must divide vocab_size {vocab}"
        self.lm_head = shard_linear(self.lm_head, "all-to-sharded", group=group)
        self.sharding_group = group
        for layer in self.layers:
            attn = layer.self_attn
            q_dim = attn.n_heads * attn.head_dim
            kv_dim = attn.n_kv_heads * attn.head_dim
            attn.qkvg_proj = shard_linear(
                attn.qkvg_proj,
                "all-to-sharded",
                segments=[q_dim, q_dim + kv_dim, q_dim + 2 * kv_dim],
                group=group,
            )
            attn.o_proj = shard_linear(attn.o_proj, "sharded-to-all", group=group)
            attn.n_heads //= N
            attn.n_kv_heads //= N

            layer.mlp.sharding_group = group
            experts = layer.mlp.experts
            shard_experts_inplace(experts.gate_proj, "all-to-sharded", group)
            shard_experts_inplace(experts.down_proj, "sharded-to-all", group)
            shard_experts_inplace(experts.up_proj, "all-to-sharded", group)

    @property
    def quant_predicate(self) -> Callable[[str, nn.Module], bool]:
        """Quantize only MoE layers except the first one and the last one."""
        experts_re = re.compile(
            r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(gate|up|down)_proj$"
        )
        full_precision = {0, self.args.text.num_hidden_layers - 1}

        def predicate(path: str, module: nn.Module) -> bool:
            m = experts_re.match(path)
            return m is not None and int(m.group(1)) not in full_precision

        return predicate

    @property
    def cast_predicate(self) -> Callable[[str], bool]:
        """Keep the gate weights in fp32 when the model is converted."""

        def predicate(k: str) -> bool:
            return not k.endswith(("_alpha", "_beta"))

        return predicate

    @property
    def layers(self) -> list[TransformerBlock]:
        return self.model.language_model.pipeline_layers

    def make_cache(self) -> list[KVCache | RotatingKVCache]:
        return [
            (
                RotatingKVCache(max_size=self.model.language_model.sliding_window)
                if layer.use_sliding
                else KVCache()
            )
            for layer in self.layers
        ]
