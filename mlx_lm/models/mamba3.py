# Copyright © 2026 Apple Inc.

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_ssm_mask
from .cache import ArraysCache


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "mamba3"
    vocab_size: int = 50257
    d_model: int = 256
    n_layer: int = 2
    d_state: int = 64
    headdim: int = 64
    mimo_rank: int = 2
    chunk_size: int = 16
    ssm_cfg: dict | None = None
    d_intermediate: int = 0
    rms_norm: bool = True
    norm_epsilon: float = 1e-5
    residual_in_fp32: bool = True
    tie_embeddings: bool = True
    pad_vocab_size_multiple: int = 16
    attn_layer_idx: list | None = None
    expand: int = 2
    ngroups: int = 1
    rope_fraction: float = 0.5
    A_floor: float = 1e-4
    is_mimo: bool = True
    is_outproj_norm: bool = False

    def __post_init__(self):
        if self.ssm_cfg is not None:
            for key in (
                "d_state",
                "expand",
                "headdim",
                "ngroups",
                "rope_fraction",
                "A_floor",
                "is_mimo",
                "mimo_rank",
                "is_outproj_norm",
                "chunk_size",
            ):
                if key in self.ssm_cfg:
                    setattr(self, key, self.ssm_cfg[key])
            self.is_mimo = self.ssm_cfg.get("is_mimo", False)
            if self.ssm_cfg.get("layer", "Mamba3") != "Mamba3":
                raise ValueError("Expected a Mamba3 mixer.")
        if not self.is_mimo:
            self.mimo_rank = 1
        if self.attn_layer_idx:
            raise ValueError("Mamba3 attention layers are not supported.")
        if min(self.d_model, self.n_layer, self.headdim, self.mimo_rank) <= 0:
            raise ValueError("Model dimensions and MIMO rank must be positive.")
        if self.expand * self.d_model % self.headdim:
            raise ValueError("headdim must divide expand * d_model.")
        if self.ngroups != 1:
            raise ValueError("Mamba3 currently requires ngroups=1.")
        if self.rope_fraction not in (0.5, 1.0):
            raise ValueError("rope_fraction must be 0.5 or 1.0.")
        if self.d_state < 4 or self.d_state % 4:
            raise ValueError("d_state must be a positive multiple of four.")


def rotate_bc(x, angle, is_mimo=True):
    n, half = angle.shape[-1], x.shape[-1] // 2
    cos, sin = mx.cos(angle)[:, :, None], mx.sin(angle)[:, :, None]
    if is_mimo:
        a, b = x[..., :n], x[..., half : half + n]
        return mx.concatenate(
            [a * cos - b * sin, x[..., n:half], a * sin + b * cos, x[..., half + n :]],
            -1,
        )
    pairs = x[..., : 2 * n].reshape(*x.shape[:-1], n, 2)
    a, b = pairs[..., 0], pairs[..., 1]
    rotated = mx.stack([a * cos - b * sin, a * sin + b * cos], -1)
    return mx.concatenate([rotated.flatten(-2), x[..., 2 * n :]], -1)


class Mamba3(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.d_inner = args.expand * args.d_model
        self.nheads = self.d_inner // args.headdim
        self.headdim = args.headdim
        self.d_state = args.d_state
        self.rank = args.mimo_rank
        self.is_mimo = args.is_mimo
        self.num_angles = int(self.d_state * args.rope_fraction) // 2
        self.A_floor = args.A_floor
        self.is_outproj_norm = args.is_outproj_norm
        self.in_proj = nn.Linear(
            args.d_model,
            2 * self.d_inner
            + 2 * self.rank * self.d_state
            + 3 * self.nheads
            + self.num_angles,
            bias=False,
        )
        self.B_norm = nn.RMSNorm(self.d_state, eps=1e-5)
        self.C_norm = nn.RMSNorm(self.d_state, eps=1e-5)
        self.B_bias = mx.ones((self.nheads, self.rank, self.d_state))
        self.C_bias = mx.ones((self.nheads, self.rank, self.d_state))
        self.dt_bias = mx.zeros(self.nheads)
        self.D = mx.ones(self.nheads)
        if self.is_mimo:
            self.mimo_x = mx.ones((self.nheads, self.rank, self.headdim)) / self.rank
            self.mimo_z = mx.ones((self.nheads, self.rank, self.headdim))
            self.mimo_o = mx.ones((self.nheads, self.rank, self.headdim)) / self.rank
        if self.is_outproj_norm:
            self.norm = nn.RMSNorm(self.d_inner, eps=1e-5)
        self.out_proj = nn.Linear(self.d_inner, args.d_model, bias=False)

    def __call__(self, u, mask=None, cache=None):
        batch, length, _ = u.shape
        sizes = [
            self.d_inner,
            self.d_inner,
            self.rank * self.d_state,
            self.rank * self.d_state,
            self.nheads,
            self.nheads,
            self.nheads,
        ]
        offsets, total = [], 0
        for size in sizes:
            total += size
            offsets.append(total)
        projection_dtype = (
            self.in_proj.scales.dtype
            if isinstance(self.in_proj, nn.QuantizedLinear)
            else self.in_proj.weight.dtype
        )
        projected = self.in_proj(u.astype(projection_dtype))
        z, x, B, C, dt, A, trap, angles = mx.split(projected, offsets, axis=-1)
        dtype = x.dtype
        x = x.reshape(batch, length, self.nheads, self.headdim).astype(mx.float32)
        z = z.reshape(batch, length, self.nheads, self.headdim).astype(mx.float32)
        B = self.B_norm(B.reshape(batch, length, self.rank, self.d_state))
        C = self.C_norm(C.reshape(batch, length, self.rank, self.d_state))
        dt = nn.softplus(dt.astype(mx.float32) + self.dt_bias)
        A = A.astype(mx.float32)
        A = -mx.maximum(mx.maximum(A, 0) + 1 / (1 - mx.minimum(A, 0)), self.A_floor)
        decay = mx.exp(A * dt)
        trap = mx.sigmoid(trap.astype(mx.float32))
        angle_steps = (
            mx.tanh(angles.astype(mx.float32))[:, :, None] * dt[..., None] * mx.pi
        )

        if cache is not None and cache[0] is not None:
            state, previous, angle = cache[0], cache[1], cache[2]
        else:
            state = mx.zeros((batch, self.nheads, self.headdim, self.d_state))
            previous = mx.zeros(state.shape)
            angle = mx.zeros((batch, self.nheads, self.num_angles))
        outputs = []
        for t in range(length):
            next_angle = angle + angle_steps[:, t]
            b = rotate_bc(
                B[:, t, None].astype(mx.float32) + self.B_bias, next_angle, self.is_mimo
            )
            c = rotate_bc(
                C[:, t, None].astype(mx.float32) + self.C_bias, next_angle, self.is_mimo
            )
            values = x[:, t, :, None] * (self.mimo_x if self.is_mimo else 1)
            bx = mx.einsum("bhrp,bhrn->bhpn", values, b)
            alpha = decay[:, t, :, None, None]
            gamma = (trap[:, t] * dt[:, t])[:, :, None, None]
            beta = ((1 - trap[:, t]) * dt[:, t])[:, :, None, None] * alpha
            next_state = alpha * state + beta * previous + gamma * bx
            if mask is not None:
                valid = mask[:, t, None, None, None]
                state = mx.where(valid, next_state, state)
                previous = mx.where(valid, bx, previous)
                angle = mx.where(valid[..., 0], next_angle, angle)
            else:
                state, previous, angle = next_state, bx, next_angle
            y = mx.einsum("bhpn,bhrn->bhrp", state, c)
            y = y + values * self.D[None, :, None, None]
            if self.is_outproj_norm:
                y = mx.fast.rms_norm(y, None, 1e-5) * self.norm.weight.reshape(
                    self.nheads, 1, self.headdim
                )
            y = y * nn.silu(z[:, t, :, None] * (self.mimo_z if self.is_mimo else 1))
            outputs.append(
                mx.sum(y * (self.mimo_o if self.is_mimo else 1), axis=2).astype(dtype)
            )
        if cache is not None:
            cache[0], cache[1], cache[2] = state, previous, angle
            cache.advance(length)
        y = mx.stack(outputs, axis=1).reshape(batch, length, self.d_inner)
        return self.out_proj(y)


class GatedMLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        hidden = (args.d_intermediate + 127) // 128 * 128
        self.fc1 = nn.Linear(args.d_model, 2 * hidden, bias=False)
        self.fc2 = nn.Linear(hidden, args.d_model, bias=False)

    def __call__(self, x):
        y, gate = mx.split(self.fc1(x), 2, axis=-1)
        return self.fc2(y * nn.silu(gate))


class ResidualBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        norm = nn.RMSNorm if args.rms_norm else nn.LayerNorm
        self.norm = norm(args.d_model, eps=args.norm_epsilon)
        self.mixer = Mamba3(args)
        self.residual_in_fp32 = args.residual_in_fp32
        if args.d_intermediate:
            self.norm2 = norm(args.d_model, eps=args.norm_epsilon)
            self.mlp = GatedMLP(args)

    def __call__(self, x, mask, cache):
        residual = x.astype(mx.float32) if self.residual_in_fp32 else x
        x = residual + self.mixer(
            self.norm(x.astype(self.norm.weight.dtype)), mask, cache
        )
        if hasattr(self, "mlp"):
            x = x + self.mlp(self.norm2(x.astype(self.norm2.weight.dtype)))
        return x


class Backbone(nn.Module):
    def __init__(self, args):
        super().__init__()
        official = args.ssm_cfg is not None
        vocab_size = args.vocab_size
        if official:
            multiple = args.pad_vocab_size_multiple
            vocab_size = (vocab_size + multiple - 1) // multiple * multiple
        self.embedding = nn.Embedding(vocab_size, args.d_model)
        block = ResidualBlock if official else Mamba3
        self.layers = [block(args) for _ in range(args.n_layer)]
        norm = nn.RMSNorm if official and args.rms_norm else nn.LayerNorm
        self.norm_f = norm(args.d_model, eps=args.norm_epsilon)

    def __call__(self, inputs, cache):
        hidden = self.embedding(inputs)
        mask = create_ssm_mask(hidden, cache[0])
        for layer, c in zip(self.layers, cache):
            hidden = layer(hidden, mask, c)
        return self.norm_f(hidden.astype(self.norm_f.weight.dtype))


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.backbone = Backbone(args)
        self.tied = args.ssm_cfg is not None and args.tie_embeddings
        if not self.tied:
            self.lm_head = nn.Linear(
                args.d_model, self.backbone.embedding.weight.shape[0], bias=False
            )

    def __call__(self, inputs, cache=None):
        if cache is None:
            cache = [None] * len(self.layers)
        hidden = self.backbone(inputs, cache)
        if self.tied:
            return self.backbone.embedding.as_linear(hidden)
        output_dtype = (
            self.lm_head.scales.dtype
            if isinstance(self.lm_head, nn.QuantizedLinear)
            else self.lm_head.weight.dtype
        )
        return self.lm_head(hidden.astype(output_dtype))

    def make_cache(self):
        return [ArraysCache(size=3) for _ in self.layers]

    @property
    def layers(self):
        return self.backbone.layers

    def sanitize(self, weights):
        if "embedding.weight" in weights:
            weights = {
                k if k.startswith("lm_head.") else "backbone." + k: v
                for k, v in weights.items()
            }
        if self.tied:
            weights = {k: v for k, v in weights.items() if not k.startswith("lm_head.")}
        return weights

    @property
    def cast_predicate(self):
        return lambda key: key.endswith(".weight")
