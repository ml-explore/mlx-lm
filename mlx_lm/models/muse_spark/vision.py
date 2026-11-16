# Copyright © 2026 Apple Inc.

import mlx.core as mx
from mlx import nn

from mlx_lm.models.muse_spark.config import VisionArgs
from mlx_lm.models.muse_spark.layers import PreciseLayerNorm

__all__ = [
    "VisionAdapter",
    "VisionAttention",
    "VisionBlock",
    "VisionMLP",
    "VisionPatchEmbedder",
    "VisionRoPE2D",
    "VisionTower",
]


def _precise_gelu(x: mx.array) -> mx.array:
    """As PyTorch: GELU in fp32, rounded once."""
    return nn.gelu(x.astype(mx.float32)).astype(x.dtype)


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
