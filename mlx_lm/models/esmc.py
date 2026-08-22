"""MLX implementation of ESM-C (EvolutionaryScale Cambrian) protein language model.

ESM-C is an encoder-only (bidirectional) transformer with RoPE, QK-LayerNorm,
SwiGLU feed-forward networks, and ESM3-style residual scaling. This port mirrors
the PyTorch reference in the `transformers` ESMC implementation; weight keys are
remapped in `sanitize()` (the reference fuses LayerNorm+QKV via Transformer
Engine module names).

The model is an encoder, so it is used for representations / masked-LM logits
rather than autoregressive generation. `Model.__call__` returns MLM logits;
`Model.encode` returns the final hidden state plus every intermediate hidden
state (matching the reference `output_hidden_states` stack).
"""

import math
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "esmc"
    vocab_size: int = 64
    d_model: int = 960
    n_heads: int = 30
    n_layers: int = 30
    pad_token_id: int = 1
    mask_token_id: int = 32
    # ESM-C FFN: SwiGLU with hidden = round_to_256(expansion_ratio * d_model).
    expansion_ratio: float = 8.0 / 3.0
    # ESM3 residue-scaling reference depth.
    residue_scaling_base: float = 36.0
    layer_norm_eps: float = 1e-5
    rope_base: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def ffn_hidden(self) -> int:
        return int(((self.expansion_ratio * self.d_model) + 255) // 256 * 256)

    @property
    def residue_scaling_factor(self) -> float:
        return math.sqrt(self.n_layers / self.residue_scaling_base)


class Attention(nn.Module):
    """Multi-head self-attention with QK-LayerNorm (over full d_model) and RoPE."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        d = args.d_model
        # Fused pre-norm + QKV projection in the reference (no bias on the proj).
        self.ln_qkv = nn.LayerNorm(d, eps=args.layer_norm_eps)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        # QK-LayerNorm is applied over the full model dim before the head split.
        self.q_ln = nn.LayerNorm(d, eps=args.layer_norm_eps, bias=False)
        self.k_ln = nn.LayerNorm(d, eps=args.layer_norm_eps, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=args.rope_base)

    def __call__(self, x: mx.array, mask: Optional[mx.array]) -> mx.array:
        B, L, _ = x.shape
        qkv = self.qkv(self.ln_qkv(x))
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = self.q_ln(q)
        k = self.k_ln(k)

        # (B, L, D) -> (B, n_heads, L, head_dim)
        def heads(t):
            return t.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        q, k, v = heads(q), heads(k), heads(v)
        q = self.rope(q)
        k = self.rope(k)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(out)


class FFN(nn.Module):
    """Pre-norm SwiGLU feed-forward network (bias-free)."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        d, h = args.d_model, args.ffn_hidden
        self.ln = nn.LayerNorm(d, eps=args.layer_norm_eps)
        self.fc1 = nn.Linear(d, 2 * h, bias=False)
        self.fc2 = nn.Linear(h, d, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(self.ln(x))
        x1, x2 = mx.split(x, 2, axis=-1)
        return self.fc2((x1 * mx.sigmoid(x1)) * x2)


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attn = Attention(args)
        self.ffn = FFN(args)
        self.scaling_factor = args.residue_scaling_factor

    def __call__(self, x: mx.array, mask: Optional[mx.array]) -> mx.array:
        x = x + self.attn(x, mask) / self.scaling_factor
        x = x + self.ffn(x) / self.scaling_factor
        return x


class TransformerStack(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.blocks = [TransformerBlock(args) for _ in range(args.n_layers)]
        self.norm = nn.LayerNorm(args.d_model, eps=args.layer_norm_eps, bias=False)

    def __call__(self, x: mx.array, mask: Optional[mx.array], collect: bool):
        hidden = []
        for block in self.blocks:
            if collect:
                hidden.append(x)  # residual stream entering this block
            x = block(x, mask)
        normed = self.norm(x)
        if collect:
            hidden.append(normed)  # final post-norm output
        return normed, hidden


class ESMCModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed = nn.Embedding(args.vocab_size, args.d_model)
        self.transformer = TransformerStack(args)

    def __call__(self, input_ids, mask, collect):
        x = self.embed(input_ids)
        return self.transformer(x, mask, collect)


def _attention_mask(attention_mask: Optional[mx.array], dtype) -> Optional[mx.array]:
    """Build the additive chain-aware mask matching the reference.

    The reference derives ``sequence_id = attention_mask - 1`` (padding -> -1,
    real -> 0) and attends where ``sequence_id_i == sequence_id_j``. With no
    padding this is full attention (mask is None / all-zero).
    """
    if attention_mask is None:
        return None
    if bool(mx.all(attention_mask != 0).item()):
        return None  # all real tokens -> full attention, no mask needed
    sid = attention_mask.astype(mx.int32) - 1  # (B, L)
    allowed = sid[:, :, None] == sid[:, None, :]  # (B, L, L)
    neg = mx.array(-1e9, dtype=dtype)  # large additive penalty (fp32-safe)
    add = mx.where(allowed, mx.array(0.0, dtype=dtype), neg)
    return add[:, None, :, :]  # (B, 1, L, L)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.esmc = ESMCModel(args)
        # MLM head: Linear -> GELU -> LayerNorm -> Linear (indices match HF keys).
        self.lm_head = [
            nn.Linear(args.d_model, args.d_model),
            nn.GELU(),
            nn.LayerNorm(args.d_model, eps=args.layer_norm_eps),
            nn.Linear(args.d_model, args.vocab_size),
        ]

    def encode(self, input_ids, attention_mask=None, output_hidden_states=False):
        """Return (last_hidden_state, hidden_states) like the reference encoder.

        ``hidden_states`` is a stacked array of shape (n_layers+1, B, L, d_model)
        when ``output_hidden_states`` else None.
        """
        mask = _attention_mask(attention_mask, self.esmc.embed.weight.dtype)
        last, hidden = self.esmc(input_ids, mask, output_hidden_states)
        hs = mx.stack(hidden, axis=0) if output_hidden_states else None
        return last, hs

    def _lm_head(self, x):
        for layer in self.lm_head:
            x = layer(x)
        return x

    def __call__(self, input_ids, attention_mask=None):
        last, _ = self.encode(input_ids, attention_mask)
        return self._lm_head(last)

    def sanitize(self, weights):
        out = {}
        for k, v in weights.items():
            if k.endswith("._extra_state"):
                continue
            k = k.replace(
                ".attn.layernorm_qkv.layer_norm_weight", ".attn.ln_qkv.weight"
            )
            k = k.replace(".attn.layernorm_qkv.layer_norm_bias", ".attn.ln_qkv.bias")
            k = k.replace(".attn.layernorm_qkv.weight", ".attn.qkv.weight")
            k = k.replace(".ffn.layer_norm_weight", ".ffn.ln.weight")
            k = k.replace(".ffn.layer_norm_bias", ".ffn.ln.bias")
            k = k.replace(".ffn.fc1_weight", ".ffn.fc1.weight")
            k = k.replace(".ffn.fc2_weight", ".ffn.fc2.weight")
            out[k] = v
        return out

    @property
    def layers(self):
        return self.esmc.transformer.blocks
