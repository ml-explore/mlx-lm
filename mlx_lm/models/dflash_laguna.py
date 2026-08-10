# Copyright © 2026 Apple Inc.
#
# DFlash speculator for Laguna-XS-2.1 (poolside/Laguna-XS-2.1-DFlash) — MLX.
#
# EAGLE-3 style BLOCK speculator. It has NO embedding / lm_head of its own — it
# reuses the TARGET Laguna's embed_tokens and lm_head.
#
# Mechanism (per block): build [anchor, MASK*(block-1)] token block, embed it via
# the TARGET embedding; each draft layer's attention takes q from the block and
# INJECTS the target's fused aux hidden states as extra K/V context
# (k/v = concat(proj(target_hidden), proj(block))). One parallel forward predicts
# the whole block; block position k predicts the token AT anchor+k (so [0]
# reproduces the anchor, [1:] are the speculative tokens). Laguna adds per-head
# softplus output gating (g_proj); within-block attention is causal.
#
# Config: fuse = hidden_norm(fc(concat(aux_norm_j(aux_j)))); per-head gate ON;
# causal within block.
from dataclasses import dataclass, field
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

MASK_TOKEN_ID = 12
NH, NKV, HD, THETA = 64, 8, 128, 500000.0


def _rms(x, w, eps=1e-6):
    x = x.astype(mx.float32)
    return (x * mx.rsqrt(mx.mean(x * x, -1, keepdims=True) + eps)) * w.astype(
        mx.float32
    )


def _rope(x, offset):
    return mx.fast.rope(x, HD, traditional=False, base=THETA, scale=1.0, offset=offset)


@dataclass
class ModelArgs:
    model_type: str = "dflash_laguna"
    hidden_size: int = 2048
    num_hidden_layers: int = 5
    num_aux_hidden_states: int = 5
    block_size: int = 16
    mask_token_id: int = MASK_TOKEN_ID
    rms_norm_eps: float = 1e-6
    aux_hidden_state_layer_ids: List[int] = field(
        default_factory=lambda: [1, 13, 25, 33, 39]
    )

    @classmethod
    def from_dict(cls, d):
        dc = d.get("dflash_config", {})
        return cls(
            hidden_size=d.get("hidden_size", 2048),
            num_hidden_layers=d.get("num_hidden_layers", 5),
            num_aux_hidden_states=len(dc.get("target_layer_ids", [1, 13, 25, 33, 39])),
            block_size=dc.get("block_size", 16),
            mask_token_id=dc.get("mask_token_id", MASK_TOKEN_ID),
            rms_norm_eps=d.get("rms_norm_eps", 1e-6),
            aux_hidden_state_layer_ids=dc.get("target_layer_ids", [1, 13, 25, 33, 39]),
        )


class DFlashAttention(nn.Module):
    """q from the mask block; K/V = concat(proj(target_hidden), proj(block))."""

    def __init__(self, eps):
        super().__init__()
        H = 2048
        self.q_proj = nn.Linear(H, NH * HD, bias=False)
        self.k_proj = nn.Linear(H, NKV * HD, bias=False)
        self.v_proj = nn.Linear(H, NKV * HD, bias=False)
        self.o_proj = nn.Linear(NH * HD, H, bias=False)
        self.g_proj = nn.Linear(H, NH, bias=False)  # per-head gate
        self.q_norm = nn.RMSNorm(HD, eps=eps)
        self.k_norm = nn.RMSNorm(HD, eps=eps)

    def __call__(self, h, target_hidden, blk_off, block_mask):
        C, B = target_hidden.shape[1], h.shape[1]
        q = _rms((h @ self.q_proj.weight.T).reshape(1, B, NH, HD), self.q_norm.weight)
        kc = (target_hidden @ self.k_proj.weight.T).reshape(1, C, NKV, HD)
        kn = (h @ self.k_proj.weight.T).reshape(1, B, NKV, HD)
        k = _rms(mx.concatenate([kc, kn], axis=1), self.k_norm.weight)
        v = mx.concatenate(
            [
                (target_hidden @ self.v_proj.weight.T).reshape(1, C, NKV, HD),
                (h @ self.v_proj.weight.T).reshape(1, B, NKV, HD),
            ],
            axis=1,
        )
        q, k, v = (t.transpose(0, 2, 1, 3) for t in (q, k, v))
        k = mx.concatenate([_rope(k[:, :, :C], 0), _rope(k[:, :, C:], blk_off)], axis=2)
        q = _rope(q, blk_off)
        k = mx.repeat(k, NH // NKV, axis=1)
        v = mx.repeat(v, NH // NKV, axis=1)
        s = (q @ k.transpose(0, 1, 3, 2)) * (HD**-0.5) + block_mask
        o = (mx.softmax(s, axis=-1) @ v).transpose(0, 2, 1, 3).reshape(1, B, NH * HD)
        g = nn.softplus(h @ self.g_proj.weight.T)  # per-head gate
        o = (o.reshape(1, B, NH, HD) * g[..., None]).reshape(1, B, NH * HD)
        return o @ self.o_proj.weight.T


class DFlashLayer(nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.self_attn = DFlashAttention(eps)
        self.mlp = _MLP()
        self.input_layernorm = nn.RMSNorm(2048, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(2048, eps=eps)

    def __call__(self, x, target_hidden, blk_off, block_mask):
        x = x + self.self_attn(
            _rms(x, self.input_layernorm.weight), target_hidden, blk_off, block_mask
        )
        return x + self.mlp(_rms(x, self.post_attention_layernorm.weight))


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(2048, 8192, bias=False)
        self.up_proj = nn.Linear(2048, 8192, bias=False)
        self.down_proj = nn.Linear(8192, 2048, bias=False)

    def __call__(self, x):
        return (
            nn.silu(x @ self.gate_proj.weight.T) * (x @ self.up_proj.weight.T)
        ) @ self.down_proj.weight.T


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        eps = args.rms_norm_eps
        self.layers = [DFlashLayer(eps) for _ in range(args.num_hidden_layers)]
        self.aux_hidden_norms = [
            nn.RMSNorm(2048, eps=eps) for _ in range(args.num_aux_hidden_states)
        ]
        self.fc = nn.Linear(args.num_aux_hidden_states * 2048, 2048, bias=False)
        self.hidden_norm = nn.RMSNorm(2048, eps=eps)
        self.norm = nn.RMSNorm(2048, eps=eps)

    def fuse(self, aux: List[mx.array]) -> mx.array:
        parts = [_rms(aux[j], self.aux_hidden_norms[j].weight) for j in range(len(aux))]
        return _rms(
            mx.concatenate(parts, axis=-1) @ self.fc.weight.T, self.hidden_norm.weight
        )

    def draft_block(self, target_embed, target_hidden, anchor_pos):
        """target_embed: [1,B,H] embedding of [anchor, MASK*(B-1)] (from target).
        target_hidden: [1,C,H] fused aux context. Returns block hidden [1,B,H]."""
        B = target_embed.shape[1]
        bm = mx.where(mx.arange(B)[:, None] >= mx.arange(B)[None, :], 0.0, -1e9)
        mask = mx.concatenate([mx.zeros((B, target_hidden.shape[1])), bm], axis=1)[
            None, None
        ]
        x = target_embed.astype(mx.float32)
        for layer in self.layers:
            x = layer(x, target_hidden, anchor_pos, mask)
        return _rms(x, self.norm.weight)

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            "dflash_laguna is a target-coupled DFlash speculator, not a "
            "standalone causal language model. Drive it through fuse() and "
            "draft_block() with the target Laguna model's embedding, aux "
            "hidden states, and LM head."
        )

    def sanitize(self, weights):
        out = {}
        for k, v in weights.items():
            if k.endswith(".self_attn.qkv_proj.weight"):
                base = k[: -len("qkv_proj.weight")]
                out[f"{base}q_proj.weight"] = v[: NH * HD]
                out[f"{base}k_proj.weight"] = v[NH * HD : NH * HD + NKV * HD]
                out[f"{base}v_proj.weight"] = v[NH * HD + NKV * HD :]
            else:
                out[k] = v
        return out
