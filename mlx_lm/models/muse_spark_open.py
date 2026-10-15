# Copyright © 2026 Apple Inc.

"""Muse Spark Open.

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
    "chat_template_type": "muse_spark_open",
    "tool_parser_type": "muse_spark_open",
    "eos_token_ids": [200001, 200008],
}


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_hidden_layers: int
    vocab_size: int
    rms_norm_eps: float
    rms_post_norm_eps: float
    rope_theta: float
    every_n_layers_nope: int
    global_attn_cfg: list[int]
    output_soft_cap_temp: float
    moe: dict = field(default_factory=dict)
    metap: dict = field(default_factory=dict)
    use_fp32_for_output: bool = False
    numerics_args: dict = field(default_factory=dict)
    ffn_dim_multiplier: float = 2.0
    # Image rows go at <|patch|>: the processor expands each <|image|> of the
    # prompt to these. The config does not store this id.
    patch_token_id: int = 200092
    vision_config: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        spec = self.numerics_args.get("out_dtype_per_module_overrides") or ""
        parts = (e.strip().split(":") for e in spec.split(","))
        overrides = {p[0]: p[2].lower() for p in parts if len(p) == 3}
        self.use_fp32_for_output = (
            self.use_fp32_for_output
            or self.numerics_args.get("use_fp32_for_output")
            or overrides.get("output") == "fp32"
        )

        if (
            not self.moe.get("is_enabled", False)
            or self.moe.get("n_dense_layers_before_moe", 0) != 0
            or self.moe.get("interleave_moe_layer_step", 1) != 1
        ):
            raise NotImplementedError("Dense feed-forward layers are not supported.")
        if not self.moe.get("router_in_fp32", False):
            raise NotImplementedError("Only the fp32 router is supported.")

        self.use_rope_layers, self.layer_types, self.sliding_window = layer_schedule(
            num_hidden_layers=self.num_hidden_layers,
            every_n_layers_nope=self.every_n_layers_nope,
            global_attn_cfg=self.global_attn_cfg,
        )

        use_metap = bool(self.metap.get("use_metap", False))
        if use_metap and self.metap.get("router_has_one_inf_dim", False):
            width_mult = self.hidden_size / self.metap.get("base_width", 256)
            mup = self.metap.get("metap_mode", "sp") == "mup"
            self.metap_multiplier = width_mult**-1 if mup else width_mult**-0.5
        else:
            self.metap_multiplier = 1.0
        self.m_emb = self.metap.get("m_emb", 1.0) if use_metap else 1.0

        self.expert_hidden_size = self.hidden_size // self.moe.get(
            "compression_factor", 1
        )

        self.vision = None
        if self.vision_config:
            navit = dict(self.vision_config.get("navit_config", self.vision_config))
            if self.vision_config.get("projector_input_dim") is not None:
                navit["projector_input_dim"] = self.vision_config["projector_input_dim"]
            self.vision = VisionArgs.from_dict(navit)


def layer_schedule(
    *, num_hidden_layers: int, every_n_layers_nope: int, global_attn_cfg: list[int]
) -> tuple[list[bool], list[str], int | None]:

    offset = every_n_layers_nope - (num_hidden_layers % every_n_layers_nope)
    use_rope = []
    layer_types = []
    windows = set()
    for i in range(num_hidden_layers):
        use_rope.append((num_hidden_layers - i - 1) % every_n_layers_nope != 0)
        window = global_attn_cfg[(i + offset) % len(global_attn_cfg)]
        if isinstance(window, int) and window > 0:
            layer_types.append("sliding_attention")
            windows.add(window)
        else:
            layer_types.append("full_attention")
    if len(windows) > 1:
        raise NotImplementedError(
            f"global_attn_cfg mixes window sizes {sorted(windows)}; "
            "only one window size per model is supported"
        )
    return use_rope, layer_types, windows.pop() if windows else None


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


POOLED_TP_MAX_TOKENS = 8


def row_parallel_linear(
    linear: nn.Linear, x: mx.array, group: mx.distributed.Group
) -> mx.array:
    assert "bias" not in linear, "Linear layers with a bias are not supported."
    rows = linear.weight.shape[0] // group.size()
    r = group.rank()
    part = x @ linear.weight[r * rows : (r + 1) * rows].T
    return gather_last_axis(part, group)


_UNUSED_QUANT_TENSORS = (
    "feed_forward.experts.w13_input_scale",
    "feed_forward.experts.w2_input_scale",
    "self_attn.q_scale",
    "self_attn.k_scale",
    "self_attn.v_scale",
)

# Norm weights the checkpoint stores centered at zero; sanitize stores the gain.
_ZERO_CENTERED_NORMS = (
    "pre_attn_norm.weight",
    "pre_ffn_norm.weight",
    "post_ffn_norm.weight",
    "feed_forward.pre_expert_norm.weight",
    "feed_forward.post_expert_norm.weight",
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
    def __init__(self, args: ModelArgs, *, use_rope: bool):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim

        self.scale = head_dim**-0.5
        # Queries carry a tuned constant scale of their own, on top of the
        # usual 1/sqrt(head_dim) applied inside the attention call
        self.query_scale = 43.7840518911 / (head_dim**0.5)
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


class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        compressed = args.expert_hidden_size
        num_experts = args.moe.get("num_experts", 0)

        self.top_k = args.moe.get("topk", 1)
        self.route_scale = args.moe.get("route_scale", 1.0)
        self.score_eps = args.moe.get("router_score_division_eps", 0.0)
        self.router_multiplier = args.metap_multiplier

        self.router = mx.zeros((num_experts, dim), dtype=mx.float32)
        self.expert_bias = mx.zeros((num_experts,), dtype=mx.float32)

        # The experts run at a compressed width; pooler and unpooler bracket them.
        self.pooler = nn.Linear(dim, compressed, bias=False)
        self.pre_expert_norm = nn.RMSNorm(compressed, eps=args.rms_norm_eps)
        hidden_dims = args.moe.get("override_routed_hidden_dim") or int(
            dim * args.ffn_dim_multiplier
        )
        self.switch_mlp = SwitchGLU(
            compressed, hidden_dims, num_experts, activation=PreciseSwiGLU()
        )
        self.post_expert_norm = InputScaleRMSNorm(
            compressed, eps=args.rms_post_norm_eps
        )
        self.unpooler = nn.Linear(compressed, dim, bias=False)

        self.sharding_group = None

    def __call__(self, x: mx.array) -> mx.array:
        group = self.sharding_group
        if group is not None:
            x = sum_gradients(group)(x)
        # Quantized poolers stay whole.
        split = (
            group is not None
            and group.size() > 1
            and x.size // x.shape[-1] <= POOLED_TP_MAX_TOKENS
            and isinstance(self.pooler, nn.Linear)
            and isinstance(self.unpooler, nn.Linear)
        )

        logits = x.astype(mx.float32) @ self.router.T
        inds, weights = expert_routing(
            logits,
            self.expert_bias,
            top_k=self.top_k,
            multiplier=self.router_multiplier,
            score_eps=self.score_eps,
            route_scale=self.route_scale,
        )

        h = row_parallel_linear(self.pooler, x, group) if split else self.pooler(x)
        h = self.switch_mlp(self.pre_expert_norm(h), inds)
        if group is not None:
            # Sum the expert outputs before the post-expert norm.
            h = mx.distributed.all_sum(h, group=group)
        h = post_expert_norm_combine(
            h,
            weights,
            norm_weight=self.post_expert_norm.weight,
            eps=self.post_expert_norm.eps,
        )
        return (
            row_parallel_linear(self.unpooler, h, group) if split else self.unpooler(h)
        )


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, *, use_rope: bool, use_sliding: bool):
        super().__init__()

        dim = args.hidden_size
        self.use_sliding = use_sliding
        self.post_norm_eps = args.rms_post_norm_eps

        self.pre_attn_norm = nn.RMSNorm(dim, eps=args.rms_norm_eps)
        self.self_attn = Attention(args, use_rope=use_rope)
        self.pre_ffn_norm = nn.RMSNorm(dim, eps=args.rms_norm_eps)
        self.feed_forward = MoE(args)
        self.post_ffn_norm = InputScaleRMSNorm(dim, eps=args.rms_post_norm_eps)
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
            attn_input = self.pre_attn_norm(x).astype(mx.bfloat16)
        r = self.self_attn(attn_input, mask, cache)
        h, ffn_input = norm_add_norm(
            r,
            x,
            alpha=self.add_attn_alpha,
            beta=self.add_attn_beta,
            post_weight=None,
            post_eps=self.post_norm_eps,
            next_norm=self.pre_ffn_norm,
        )
        r = self.feed_forward(ffn_input)
        return norm_add_norm(
            r,
            h,
            alpha=self.add_ffn_alpha,
            beta=self.add_ffn_beta,
            post_weight=self.post_ffn_norm.weight,
            post_eps=self.post_ffn_norm.eps,
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
    vision_output_dim: int = 6144
    num_hidden_layers: int = 50
    num_attention_heads: int = 16
    intermediate_size: int = 8960
    num_channels: int = 3
    image_size: int = 448
    patch_size: int = 14
    patch_temporal: int = 2
    downsample_factor: int = 2
    sparse_attention_factor: int = 4
    use_ij_pos_encoding: bool = False
    projector_input_dim: int | None = None

    def __post_init__(self) -> None:
        # The width comes from vision_output_dim, not hidden_size.
        self.hidden_size = self.vision_output_dim // self.downsample_factor**2
        self.head_dim = self.hidden_size // self.num_attention_heads
        # The size of the position table and of the sparse attention blocks;
        # images can be larger or smaller.
        self.grid_size = self.image_size // self.patch_size
        if self.projector_input_dim is None:
            self.projector_input_dim = self.vision_output_dim


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
        self.out_proj = nn.Linear(dim, dim, bias=True)

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
        return self.out_proj(out.reshape(N, -1))


class VisionMLP(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        self.c_fc = nn.Linear(args.hidden_size, args.intermediate_size, bias=True)
        self.c_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.c_proj(_precise_gelu(self.c_fc(x)))


class VisionBlock(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        self.ln_1 = PreciseLayerNorm(args.hidden_size, bias=True)
        self.attn = VisionAttention(args)
        self.ln_2 = PreciseLayerNorm(args.hidden_size, bias=True)
        self.mlp = VisionMLP(args)

    def __call__(
        self,
        x: mx.array,
        *,
        cos: mx.array,
        sin: mx.array,
        tiles: tuple[list[mx.array], mx.array] | None = None,
    ) -> mx.array:
        x = x + self.attn(self.ln_1(x), cos=cos, sin=sin, tiles=tiles)
        x = x + self.mlp(self.ln_2(x))
        return x


class VisionAdapter(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        dim = args.projector_input_dim
        self.c_fc = nn.Linear(args.vision_output_dim, dim, bias=False)
        self.c_proj = nn.Linear(dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        x = _precise_gelu(self.c_fc(x))
        x = _precise_gelu(self.c_proj(x))
        return x


class VisionTower(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        self.args = args
        dim = args.hidden_size
        patch_dim = args.num_channels * args.patch_size**2 * args.patch_temporal
        self.conv1_linear = nn.Linear(patch_dim, dim, bias=False)
        self.positional_embedding_vlm = mx.zeros((args.grid_size**2, dim))
        self.ln_pre = PreciseLayerNorm(dim, bias=True)
        self.rope = VisionRoPE2D(head_dim=args.head_dim)
        self.encoder = [VisionBlock(args) for _ in range(args.num_hidden_layers)]
        self.ln_post = PreciseLayerNorm(dim, bias=True)
        self.adapter = VisionAdapter(args)

    def __call__(self, pixel_values: mx.array) -> mx.array:
        """pixel_values: (num_channels, H, W), with H and W multiples of
        patch_size * downsample_factor."""
        args = self.args
        C, H, W = pixel_values.shape
        p, t, f = args.patch_size, args.patch_temporal, args.downsample_factor
        if H % (p * f) or W % (p * f):
            raise ValueError(f"Image size {H}x{W} is not a multiple of {p * f}.")
        gh, gw = H // p, W // p

        img = mx.broadcast_to(pixel_values[None], (t, C, H, W))
        patches = img.reshape(t, C, gh, p, gw, p)
        patches = patches.transpose(2, 4, 0, 1, 3, 5)
        patches = patches.reshape(gh * gw, t * C * p * p)

        h = self.conv1_linear(patches)
        h = h + _position_embedding(
            self.positional_embedding_vlm,
            size=args.grid_size,
            grid_h=gh,
            grid_w=gw,
            ij=args.use_ij_pos_encoding,
        )
        h = self.ln_pre(h)
        cos, sin = self.rope(grid_h=gh, grid_w=gw)
        # Attention inside grid_size blocks, except in every
        # sparse_attention_factor-th layer and the last layer.
        tiles = _attention_tiles(grid_h=gh, grid_w=gw, tile=args.grid_size)
        n, k = len(self.encoder), args.sparse_attention_factor
        for i, block in enumerate(self.encoder):
            full = k <= 1 or i == n - 1 or (i + 1) % k == 0
            h = block(h, cos=cos, sin=sin, tiles=None if full else tiles)
        h = self.ln_post(h)

        h = _pixel_shuffle_downsample(h, grid_h=gh, grid_w=gw, factor=f)
        return self.adapter(h)


class MuseSparkOpenModel(PipelineMixin, nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.sliding_window = args.sliding_window
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(
                args,
                use_rope=use_rope,
                use_sliding=layer_type == "sliding_attention",
            )
            for use_rope, layer_type in zip(args.use_rope_layers, args.layer_types)
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
        h = mx.fast.rms_norm(self.embed_tokens(inputs), None, self.args.rms_norm_eps)
        if self.args.m_emb != 1.0:
            h = h * self.args.m_emb
        return h

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
            # Equal to the fused norm of the previous stage, not to pre_attn_norm.
            norm = layers[0].pre_attn_norm
            attn_input = pre_norm(h, norm, dtype=norm.weight.dtype)

        # Each block also makes the next block's attention input, so that the
        # norm is fused with its add.
        for i, (layer, c) in enumerate(zip(layers, cache)):
            mask = swa_mask if layer.use_sliding else fa_mask
            nxt = layers[i + 1].pre_attn_norm if i + 1 < len(layers) else None
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


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = MuseSparkOpenModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.sharding_group = None

        # The vision tower is not sharded.
        if args.vision is not None:
            self.vision_model = VisionTower(args.vision)
            self.vision_projection = nn.Linear(
                args.vision.projector_input_dim, args.hidden_size, bias=False
            )
            self.perception_emb_norm = WeightlessRMSNorm(eps=1e-5)

    def encode_image(self, pixel_values: mx.array | list[mx.array]) -> mx.array:
        """pixel_values: one image (num_channels, H, W), a list of images of
        any size, or images of one size (num_images, num_channels, H, W)."""
        if isinstance(pixel_values, mx.array) and pixel_values.ndim == 3:
            pixel_values = [pixel_values]
        assert len(pixel_values), "No images."
        assert all(p.ndim == 3 for p in pixel_values), "Images must be (C, H, W)."
        dtype = self.vision_model.conv1_linear.weight.dtype
        # TODO: consider packing the images into one sequence, with attention
        # per image (MLX has no varlen attention).
        h = mx.concatenate([self.vision_model(p.astype(dtype)) for p in pixel_values])
        h = self.vision_projection(h)
        return self.perception_emb_norm(h)

    def embed_inputs(
        self,
        inputs: mx.array,
        *,
        pixel_values: mx.array | list[mx.array] | None = None,
    ) -> mx.array:
        """The normalized input embeddings, with the vision rows at the patch tokens."""
        h = self.model.embed(inputs)
        # The vision rows take the dtype of the text embeddings.
        if pixel_values is not None:
            h = _splice_vision_embeddings(
                h,
                inputs,
                vision_embeds=self.encode_image(pixel_values),
                patch_token_id=self.args.patch_token_id,
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
        out = self.model(inputs, cache=cache, input_embeddings=input_embeddings)
        # The head weight is fp32 with use_fp32_for_output (see sanitize).
        out = self.lm_head(out.astype(self.lm_head.weight.dtype))
        if self.sharding_group is not None:
            out = gather_last_axis(out, self.sharding_group)
        out = out.astype(mx.float32) * self.args.metap_multiplier
        cap = self.args.output_soft_cap_temp
        return mx.tanh(out / cap) * cap

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        drop_prefixes = (
            "audio_",
            "input_drop.",
            # Partial copies of the vision_model.* tensors.
            "modalities.avocado_perception.encoder.vision_model.",
            "modalities.avocado_perception.encoder.vision_module.",
            "modalities.vision_perception.encoder.vision_model.",
            "modalities.vision_perception.encoder.vision_module.",
            # Muse Spark's audio tower is not supported.
            "modalities.soft_audio_perception.",
        )
        if self.args.vision is None:
            drop_prefixes += ("vision_", "modalities.", "pre_vision_projection_norm")
        renames = {
            "self_attn.pre_qkv_proj_norm.weight": "pre_attn_norm.weight",
        }
        # The gates are fixed, so compute their weights once. The underscore
        # names are from MLX checkpoints that store the gate.
        gates = {
            "add_attn.gate": "add_attn",
            "add_ffn.gate": "add_ffn",
            "add_attn_gate": "add_attn",
            "add_ffn_gate": "add_ffn",
        }
        layer_re = re.compile(r"^(model\.layers\.\d+\.)(.+)$")
        # Weights from mlx_lm.convert already store the norm gains.
        converted = any(k.endswith(".feed_forward.router") for k in weights)

        sanitized = {}
        for k, v in weights.items():
            if k.startswith(drop_prefixes):
                continue
            # The projection is stored only under the modalities prefix.
            if k in (
                "modalities.avocado_perception.encoder.vision_projection.weight",
                "modalities.vision_perception.encoder.vision_projection.weight",
            ):
                sanitized["vision_projection.weight"] = v
                continue
            if k.startswith("vision_"):
                sanitized[k.replace(".encoder.layers.", ".encoder.")] = v
                continue
            # embed_norm and the attention post-norm are weightless here
            if k.endswith(("embed_norm.weight", "num_tokens_per_expert")):
                continue

            m = layer_re.match(k)
            if m is None:
                sanitized[k] = v
                continue

            prefix, suffix = m.groups()
            suffix = renames.get(suffix, suffix)
            switch_mlp = prefix + "feed_forward.switch_mlp."
            if suffix in gates:
                alpha, beta = residual_gate_coefficients(v)
                sanitized[prefix + gates[suffix] + "_alpha"] = alpha
                sanitized[prefix + gates[suffix] + "_beta"] = beta
            elif suffix in _ZERO_CENTERED_NORMS and not converted:
                sanitized[prefix + suffix] = norm_gain(v)
            elif suffix == "feed_forward.router.weight":
                # to use gemv instead of gemv_t
                sanitized[prefix + "feed_forward.router"] = mx.contiguous(v.T)
            elif suffix == "feed_forward.experts.w13_weight":
                gate_proj, up_proj = mx.split(v, 2, axis=1)
                if v.dtype == mx.uint8:
                    # In mlx quantized weights are packed into uint32
                    gate_proj = gate_proj.view(mx.uint32)
                    up_proj = up_proj.view(mx.uint32)
                sanitized[switch_mlp + "gate_proj.weight"] = gate_proj
                sanitized[switch_mlp + "up_proj.weight"] = up_proj
            elif suffix == "feed_forward.experts.w13_weight_scale":
                # One e4m3 block scale per 16 inputs
                gate_scales, up_scales = mx.split(v, 2, axis=1)
                sanitized[switch_mlp + "gate_proj.scales"] = gate_scales
                sanitized[switch_mlp + "up_proj.scales"] = up_scales
            elif suffix == "feed_forward.experts.w13_weight_scale_2":
                # Column 0 is the gate half, column 1 the up half.
                gate_gs, up_gs = v[:, 0], v[:, 1]
                sanitized[switch_mlp + "gate_proj.global_scale"] = (
                    gate_gs * _NVFP4_SCALE_2_TO_GLOBAL
                )
                sanitized[switch_mlp + "up_proj.global_scale"] = (
                    up_gs * _NVFP4_SCALE_2_TO_GLOBAL
                )
            elif suffix == "feed_forward.experts.w2_weight":
                sanitized[switch_mlp + "down_proj.weight"] = (
                    v.view(mx.uint32) if v.dtype == mx.uint8 else v
                )
            elif suffix == "feed_forward.experts.w2_weight_scale":
                sanitized[switch_mlp + "down_proj.scales"] = v
            elif suffix == "feed_forward.experts.w2_weight_scale_2":
                sanitized[switch_mlp + "down_proj.global_scale"] = (
                    v * _NVFP4_SCALE_2_TO_GLOBAL
                )
            elif suffix in _UNUSED_QUANT_TENSORS:
                # we don't quantize activations
                continue
            else:
                sanitized[prefix + suffix] = v

        if self.args.use_fp32_for_output:
            head = sanitized["lm_head.weight"]
            sanitized["lm_head.weight"] = head.astype(mx.float32)

        projections = ("q_proj", "k_proj", "v_proj", "o_gate_proj")
        for k in [k for k in sanitized if k.endswith("self_attn.q_proj.weight")]:
            attn = k[: -len("q_proj.weight")]
            for param in ("weight", "scales", "biases"):
                keys = [f"{attn}{p}.{param}" for p in projections]
                if all(key in sanitized for key in keys):
                    sanitized[f"{attn}qkvg_proj.{param}"] = mx.concatenate(
                        [sanitized.pop(key) for key in keys]
                    )

        vision_q = [
            k
            for k in sanitized
            if k.startswith("vision_model.") and k.endswith(".attn.q_proj.weight")
        ]
        for k in vision_q:
            attn = k[: -len("q_proj.weight")]
            for param in ("weight", "bias"):
                keys = [f"{attn}{p}_proj.{param}" for p in "qkv"]
                sanitized[f"{attn}qkv_proj.{param}"] = mx.concatenate(
                    [sanitized.pop(key) for key in keys]
                )
        return sanitized

    def shard(self, group: mx.distributed.Group | None = None) -> None:
        group = group or mx.distributed.init()
        N = group.size()
        n_kv_heads = self.args.num_key_value_heads
        assert (
            n_kv_heads % N == 0
        ), f"group size {N} must divide num_key_value_heads {n_kv_heads}"
        vocab = self.args.vocab_size
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

            layer.feed_forward.sharding_group = group
            switch_mlp = layer.feed_forward.switch_mlp
            shard_experts_inplace(switch_mlp.gate_proj, "all-to-sharded", group)
            shard_experts_inplace(switch_mlp.down_proj, "sharded-to-all", group)
            shard_experts_inplace(switch_mlp.up_proj, "all-to-sharded", group)

    @property
    def quant_predicate(self) -> Callable[[str, nn.Module], bool]:
        """Quantize only MoE layers except the first one and the last one."""
        experts_re = re.compile(
            r"^model\.layers\.(\d+)\.feed_forward\.switch_mlp\.\w+$"
        )
        full_precision = {0, self.args.num_hidden_layers - 1}

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
        return self.model.pipeline_layers

    def make_cache(self) -> list[KVCache | RotatingKVCache]:
        return [
            (
                RotatingKVCache(max_size=self.model.sliding_window)
                if layer.use_sliding
                else KVCache()
            )
            for layer in self.layers
        ]
