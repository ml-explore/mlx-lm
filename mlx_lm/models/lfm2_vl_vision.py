# Copyright © 2026 Apple Inc.

# LFM2-VL image encoder: a SigLIP2 NaFlex vision tower (variable resolution,
# learned position grid resized per image) and the pixel-unshuffle MLP
# projector into the language model. Written after HF transformers'
# modeling_siglip2.py and modeling_lfm2_vl.py.

from dataclasses import dataclass
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass
class VisionConfig:
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_channels: int = 3
    patch_size: int = 16
    num_patches: int = 256
    layer_norm_eps: float = 1e-6

    @classmethod
    def from_dict(cls, params: dict) -> "VisionConfig":
        names = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in params.items() if k in names})


@lru_cache(maxsize=64)
def _resize_matrix(n_in: int, n_out: int) -> np.ndarray:
    """[n_out, n_in] weights of a bilinear resize with antialias, as torch's
    F.interpolate(mode="bilinear", align_corners=False, antialias=True)."""
    scale = n_in / n_out
    support = scale if scale >= 1 else 1.0
    inv = 1 / scale if scale >= 1 else 1.0
    m = np.zeros((n_out, n_in), dtype=np.float64)
    for i in range(n_out):
        center = scale * (i + 0.5)
        lo = max(int(center - support + 0.5), 0)
        hi = min(int(center + support + 0.5), n_in)
        j = np.arange(lo, hi)
        w = np.clip(1 - np.abs((j - center + 0.5) * inv), 0, None)
        m[i, lo:hi] = w / w.sum()
    return m.astype(np.float32)


class PositionTable(nn.Module):
    """The learned position grid. Not an nn.Embedding, so quantization
    leaves it in float: it is resized, not looked up."""

    def __init__(self, num_patches: int, dims: int):
        super().__init__()
        self.weight = mx.zeros((num_patches, dims))


class Embeddings(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        p = config.patch_size
        self.patch_embedding = nn.Linear(config.num_channels * p * p, config.hidden_size)
        self.grid = int(config.num_patches**0.5)
        self.position_embedding = PositionTable(config.num_patches, config.hidden_size)

    def positions(self, h: int, w: int) -> mx.array:
        """The position grid resized to h x w patches, [h * w, D]."""
        g = self.grid
        table = self.position_embedding.weight.astype(mx.float32).reshape(g, g, -1)
        rh = mx.array(_resize_matrix(g, h))
        rw = mx.array(_resize_matrix(g, w))
        out = mx.einsum("hi,ijd,wj->hwd", rh, table, rw)
        return out.reshape(h * w, -1)

    def __call__(self, patches: mx.array, h: int, w: int) -> mx.array:
        dtype = self.position_embedding.weight.dtype
        x = self.patch_embedding(patches.astype(dtype))
        return x + self.positions(h, w).astype(x.dtype)


class Attention(nn.Module):
    def __init__(self, dims: int, heads: int):
        super().__init__()
        self.heads = heads
        self.scale = (dims // heads) ** -0.5
        self.q_proj = nn.Linear(dims, dims)
        self.k_proj = nn.Linear(dims, dims)
        self.v_proj = nn.Linear(dims, dims)
        self.out_proj = nn.Linear(dims, dims)

    def __call__(self, x: mx.array) -> mx.array:
        B, L, D = x.shape
        q, k, v = (
            p(x).reshape(B, L, self.heads, -1).transpose(0, 2, 1, 3)
            for p in (self.q_proj, self.k_proj, self.v_proj)
        )
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.out_proj(o.transpose(0, 2, 1, 3).reshape(B, L, D))


class MLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu_approx(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        eps = config.layer_norm_eps
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=eps)
        self.self_attn = Attention(config.hidden_size, config.num_attention_heads)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=eps)
        self.mlp = MLP(config)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.self_attn(self.layer_norm1(x))
        return x + self.mlp(self.layer_norm2(x))


class Encoder(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.layers = [EncoderLayer(config) for _ in range(config.num_hidden_layers)]


class VisionModel(nn.Module):
    """SigLIP2 vision tower without the pooling head."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, patches: mx.array, h: int, w: int) -> mx.array:
        """patches [B, h * w, C * p * p] of B crops of the same size."""
        x = self.embeddings(patches, h, w)
        for layer in self.encoder.layers:
            x = layer(x)
        return self.post_layernorm(x)


class Projector(nn.Module):
    """Pixel unshuffle (factor x factor patches into one token), then a
    two-layer MLP into the language model's width."""

    def __init__(self, vision_dims, text_dims, hidden, factor, bias, act, layer_norm):
        super().__init__()
        self.factor = factor
        dims = vision_dims * factor * factor
        if layer_norm:
            self.layer_norm = nn.LayerNorm(dims)
        self.linear_1 = nn.Linear(dims, hidden, bias=bias)
        self.linear_2 = nn.Linear(hidden, text_dims, bias=bias)
        self.act = nn.gelu if act == "gelu" else nn.gelu_approx

    def unshuffle(self, x: mx.array) -> mx.array:
        # HF's pixel_unshuffle: [B, h, w, C] -> [B, h/f, w/f, C*f*f].
        B, h, w, C = x.shape
        f = self.factor
        x = x.reshape(B, h, w // f, C * f).transpose(0, 2, 1, 3)
        x = x.reshape(B, w // f, h // f, C * f * f).transpose(0, 2, 1, 3)
        return x

    def __call__(self, features: mx.array, h: int, w: int) -> mx.array:
        """features [B, h * w, C] -> image tokens [B, (h/f) * (w/f), D]."""
        B = features.shape[0]
        x = self.unshuffle(features.reshape(B, h, w, -1))
        x = x.reshape(B, -1, x.shape[-1])
        if "layer_norm" in self:
            x = self.layer_norm(x)
        return self.linear_2(self.act(self.linear_1(x)))
