# Copyright © 2023-2024 Apple Inc.
#
# OLMo Hybrid 7B — first MLX / mlx-lm implementation
#
# Architecture: 32 layers in a 3:1 GatedDeltaNet / full-Attention pattern (24 GDN + 8 Attn)
# Paper:  Yang et al. "GatedDeltaNet" (arxiv 2412.06464), AllenAI OLMo Hybrid 2025
# HF hub: allenai/OLMo-Hybrid-7B  (Apache 2.0)
#
# Key design notes
# ─────────────────
# • GatedDeltaNet (GDN) is a sequential recurrence — state S ∈ (B,H,Dv,Dk).
#   Memory cost is O(1) in context length (vs O(L) for standard KV cache).
# • ShortConv uses mx.conv1d with groups=C for native depthwise support.
#   HF checkpoint stores weights as (C,1,K); sanitize() transposes to (C,K,1).
# • Chunked eval (CHUNK=32) breaks MLX's lazy graph chain during the T-step
#   recurrence to cap peak memory to ~CHUNK S-matrices rather than T.
# • GDNCache / KVCache are heterogeneous — make_cache() returns the right type
#   per layer. The generation loop assigns caches by position, so this is transparent.
#
# mlx-lm usage (after placing this file in mlx_lm/models/):
#   mlx_lm.generate --model allenai/OLMo-Hybrid-7B --prompt "Once upon a time"

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope


# ─────────────────────────────────────────────────────────────────────────────
# Config  (field names match HF config.json — BaseModelArgs.from_dict filters extras)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "olmo_hybrid"

    # Standard transformer dims
    vocab_size: int = 100352
    hidden_size: int = 3840
    num_hidden_layers: int = 32
    num_attention_heads: int = 30
    num_key_value_heads: Optional[int] = None
    intermediate_size: int = 11008
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 65536
    tie_word_embeddings: bool = False

    # RoPE  (OLMo2-style large-context theta)
    rope_theta: float = 500000.0
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Any]] = None

    # Per-layer type schedule ("linear_attention" | "full_attention")
    # Present in config.json as a list of strings — drives the 3:1 hybrid pattern.
    layer_types: List[str] = field(default_factory=list)

    # GatedDeltaNet-specific
    linear_num_key_heads:   int  = 30
    linear_num_value_heads: int  = 30
    linear_key_head_dim:    int  = 96    # D_k per head
    linear_value_head_dim:  int  = 192   # D_v per head
    linear_conv_kernel_dim: int  = 4     # ShortConv kernel size K
    linear_allow_neg_eigval: bool = True

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if not self.layer_types:
            # Fallback: derive 3:1 pattern if config.json omits layer_types
            self.layer_types = [
                "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
                for i in range(self.num_hidden_layers)
            ]

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def n_kv_heads(self) -> int:
        return self.num_key_value_heads


# ─────────────────────────────────────────────────────────────────────────────
# Cache objects
# ─────────────────────────────────────────────────────────────────────────────

class GDNCache:
    """
    Per-layer recurrent cache for GatedDeltaNet.

    State tuple:
      S:        mx.array (B, H, D_v, D_k) — associative memory matrix
      conv_ctx: (q_ctx, k_ctx, v_ctx)     — last K-1 raw projections for ShortConv
                each mx.array (B, K-1, H*dim)

    Conforms to the mlx-lm cache protocol (has .offset property).
    """

    def __init__(self):
        self._state: Optional[Tuple] = None

    @property
    def state(self) -> Optional[Tuple]:
        return self._state

    @state.setter
    def state(self, v: Optional[Tuple]):
        self._state = v

    @property
    def offset(self) -> int:
        # Position tracking is handled by the recurrence itself; always report 0
        # so attention layers don't pick up the wrong offset from a GDN cache.
        return 0


class KVCache:
    """
    Growing KV cache for full-attention layers. Allocates in steps to amortise
    re-allocation overhead. Compatible with mlx-lm's cache protocol.
    """

    def __init__(self, head_dim: int, n_heads: int):
        self.head_dim = head_dim
        self.n_heads  = n_heads
        self.keys:   Optional[mx.array] = None
        self.values: Optional[mx.array] = None
        self._offset = 0

    @property
    def offset(self) -> int:
        return self._offset

    @property
    def state(self):
        return [self.keys, self.values] if self.keys is not None else []

    def update_and_fetch(
        self,
        keys:   mx.array,   # (B, H, T, D)
        values: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        if self.keys is None:
            self.keys   = keys
            self.values = values
        else:
            self.keys   = mx.concatenate([self.keys,   keys],   axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)
        self._offset += keys.shape[2]
        return self.keys, self.values


# ─────────────────────────────────────────────────────────────────────────────
# ShortConv  (causal depthwise conv1d, used inside GDN)
# ─────────────────────────────────────────────────────────────────────────────

class ShortConv(nn.Module):
    """
    Causal depthwise conv1d + SiLU applied to q, k, v before the delta-rule recurrence.

    MLX conv1d weight shape: (out_ch, kW, in_ch // groups) = (C, K, 1)
    HF checkpoint stores:    (C, 1, K) — sanitize() transposes to (C, K, 1).

    The `prefix` arg passes real prior-token context during decode, avoiding the
    zero-pad corruption that occurs after the first generated token.
    """

    def __init__(self, channels: int, kernel_size: int = 4):
        super().__init__()
        self.channels    = channels
        self.kernel_size = kernel_size
        self.weight = mx.zeros((channels, kernel_size, 1))   # (C, K, 1)

    def __call__(
        self,
        x:      mx.array,
        prefix: Optional[mx.array] = None,
    ) -> mx.array:
        """
        x:      (B, T, C)
        prefix: (B, K-1, C) real prior context, or None → zero-pad (prefill only)
        Returns (B, T, C)
        """
        B, T, C = x.shape
        K   = self.kernel_size
        pad = prefix if prefix is not None else mx.zeros((B, K - 1, C))
        x_pad = mx.concatenate([pad, x], axis=1)                     # (B, K-1+T, C)
        out   = mx.conv1d(x_pad, self.weight, padding=0, groups=C)   # (B, T, C)
        return nn.silu(out)


# ─────────────────────────────────────────────────────────────────────────────
# GatedDeltaNet  (linear_attention layers)
# ─────────────────────────────────────────────────────────────────────────────

class GatedDeltaNet(nn.Module):
    """
    Gated DeltaNet recurrent mixer.

    Per-token update for each head h:
        dt    = softplus(a_proj(x) + dt_bias)         — continuous time step, always > 0
        alpha = exp(dt · (−exp(A_log)))  ∈ (0, 1]     — per-head decay (forget gate)
        beta  = sigmoid(b_proj(x)) · 2  ∈ (0, 2]      — write scale
                (×2 from allow_neg_eigval, enabling eigenvalues of write-op (I − β k kᵀ) < 0)

        k  = L2_norm(ShortConv(k_proj(x)))
        q  = L2_norm(ShortConv(q_proj(x))) / √D_k
        v  = ShortConv(v_proj(x))

        # Delta-rule state update
        S ← alpha · S + beta · (v − S @ k) ⊗ k
        # Gated readout
        y ← o_norm(S @ q) ⊙ silu(g_proj(x))
        out ← o_proj(reshape(y))

    Note on chunked eval:
        The T-step Python loop accumulates lazy MLX tensors. Without periodic
        mx.eval(), MLX holds all intermediate S matrices on the graph until
        the whole sequence is processed (~13 GB at T=256, 24 GDN layers).
        Calling mx.eval(S) and mx.eval(chunk) every CHUNK=32 tokens caps
        in-flight memory to CHUNK S-matrices, independent of T.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        H   = args.linear_num_key_heads
        D_k = args.linear_key_head_dim
        D_v = args.linear_value_head_dim
        hid = args.hidden_size

        self.num_heads        = H
        self.key_dim          = D_k
        self.val_dim          = D_v
        self.allow_neg_eigval = args.linear_allow_neg_eigval
        self.q_scale          = 1.0 / math.sqrt(D_k)

        self.q_proj = nn.Linear(hid, H * D_k, bias=False)
        self.k_proj = nn.Linear(hid, H * D_k, bias=False)
        self.v_proj = nn.Linear(hid, H * D_v, bias=False)
        self.b_proj = nn.Linear(hid, H,       bias=False)   # beta
        self.g_proj = nn.Linear(hid, H * D_v, bias=False)   # output gate
        self.a_proj = nn.Linear(hid, H,       bias=False)   # dt input

        # Mamba-style decay parameterisation
        self.dt_bias = mx.zeros((H,))   # learned bias added to dt
        self.A_log   = mx.zeros((H,))   # decay: alpha = exp(dt * −exp(A_log))

        K = args.linear_conv_kernel_dim
        self.q_conv1d = ShortConv(H * D_k, kernel_size=K)
        self.k_conv1d = ShortConv(H * D_k, kernel_size=K)
        self.v_conv1d = ShortConv(H * D_v, kernel_size=K)

        self.o_norm = nn.RMSNorm(D_v, eps=args.rms_norm_eps)
        self.o_proj = nn.Linear(H * D_v, hid, bias=False)

    @staticmethod
    def _l2_norm(x: mx.array, eps: float = 1e-8) -> mx.array:
        return x / (mx.linalg.norm(x, axis=-1, keepdims=True) + eps)

    def __call__(
        self,
        x:     mx.array,
        cache: Optional[GDNCache] = None,
    ) -> mx.array:
        """
        x:     (B, T, hidden_size)
        cache: GDNCache updated in-place (pass None for training / stateless use)
        Returns (B, T, hidden_size)
        """
        B, T, _ = x.shape
        H, D_k, D_v = self.num_heads, self.key_dim, self.val_dim
        pad_len = self.q_conv1d.kernel_size - 1   # K-1

        # ── Unpack recurrent state ────────────────────────────────────────────
        if cache is None or cache.state is None:
            S        = mx.zeros((B, H, D_v, D_k))
            q_ctx = k_ctx = v_ctx = None
        else:
            S, (q_ctx, k_ctx, v_ctx) = cache.state

        # ── Linear projections ────────────────────────────────────────────────
        q_raw = self.q_proj(x)   # (B, T, H*D_k)
        k_raw = self.k_proj(x)
        v_raw = self.v_proj(x)   # (B, T, H*D_v)

        beta = mx.sigmoid(self.b_proj(x))                     # (B, T, H)
        if self.allow_neg_eigval:
            beta = beta * 2.0

        gate = nn.silu(self.g_proj(x))                        # (B, T, H*D_v)

        dt   = nn.softplus(self.a_proj(x) + self.dt_bias)     # (B, T, H)
        A    = -mx.exp(self.A_log)                            # (H,)
        alpha = mx.exp(dt * A)                                # (B, T, H) ∈ (0,1]

        # ── ShortConv with real prior context ─────────────────────────────────
        # During prefill (q_ctx=None): zero-pad applied automatically.
        # During decode:  q/k/v_ctx = last K-1 raw tokens from prior call — no bug.
        q = self.q_conv1d(q_raw, prefix=q_ctx)   # (B, T, H*D_k)
        k = self.k_conv1d(k_raw, prefix=k_ctx)
        v = self.v_conv1d(v_raw, prefix=v_ctx)   # (B, T, H*D_v)

        # Save new conv context: last pad_len raw (pre-conv) tokens
        if T >= pad_len:
            new_conv_ctx = (q_raw[:, -pad_len:], k_raw[:, -pad_len:], v_raw[:, -pad_len:])
        else:
            def _slide(old, new, C):
                base = old if old is not None else mx.zeros((B, pad_len, C))
                return mx.concatenate([base, new], axis=1)[:, -pad_len:]
            new_conv_ctx = (
                _slide(q_ctx, q_raw, q_raw.shape[-1]),
                _slide(k_ctx, k_raw, k_raw.shape[-1]),
                _slide(v_ctx, v_raw, v_raw.shape[-1]),
            )

        # ── Per-head reshape + normalise ──────────────────────────────────────
        q = self._l2_norm(q.reshape(B, T, H, D_k)) * self.q_scale   # (B, T, H, D_k)
        k = self._l2_norm(k.reshape(B, T, H, D_k))
        v = v.reshape(B, T, H, D_v)

        alpha = alpha.reshape(B, T, H, 1, 1)   # broadcast over (D_v, D_k)
        beta  = beta.reshape(B, T, H, 1, 1)
        gate  = gate.reshape(B, T, H, D_v)

        # ── Chunked sequential delta-rule recurrence ──────────────────────────
        # See class docstring for why chunked eval is necessary.
        CHUNK = 32
        materialized_chunks: list = []
        outputs_chunk:       list = []

        for t in range(T):
            q_t = q[:, t]    # (B, H, D_k)
            k_t = k[:, t]
            v_t = v[:, t]    # (B, H, D_v)
            a_t = alpha[:, t]
            b_t = beta[:, t]

            Sk    = mx.einsum("bhvd,bhd->bhv", S, k_t)
            outer = mx.einsum("bhv,bhd->bhvd", v_t - Sk, k_t)
            S     = a_t * S + b_t * outer
            y_t   = mx.einsum("bhvd,bhd->bhv", S, q_t)
            outputs_chunk.append(y_t)

            if (t + 1) % CHUNK == 0:
                mx.eval(S)                                     # break S dependency chain
                chunk = mx.stack(outputs_chunk, axis=1)        # (B, CHUNK, H, D_v)
                mx.eval(chunk)                                 # release S_t references
                materialized_chunks.append(chunk)
                outputs_chunk = []

        if outputs_chunk:
            chunk = mx.stack(outputs_chunk, axis=1)
            mx.eval(chunk)
            materialized_chunks.append(chunk)

        # ── Readout: RMSNorm → gate → project ────────────────────────────────
        y   = mx.concatenate(materialized_chunks, axis=1)         # (B, T, H, D_v)
        y   = self.o_norm(y.reshape(B * T * H, D_v)).reshape(B, T, H, D_v)
        y   = (y * gate).reshape(B, T, H * D_v)
        out = self.o_proj(y)

        if cache is not None:
            cache.state = (S, new_conv_ctx)

        # Materialise before returning — frees intermediate graph nodes.
        # With stop_gradient applied by the caller (HybridLayer), the concrete
        # tensor is treated as a constant in the backward pass.
        mx.eval(out)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Full Attention  (full_attention layers — OLMo2 post-norm + QK-norm style)
# ─────────────────────────────────────────────────────────────────────────────

class Attention(nn.Module):
    """
    Causal MHA with RoPE, GQA support, and OLMo2-style QK-norm.

    QK-norm is applied to the full (H * D) projection *before* reshape into heads,
    which matches the HF reference. The norm weights in the checkpoint are (H*D,),
    not (D,) per head — a silent correctness bug if you apply it after reshape.

    Post-norm: RMSNorm is applied to the mixer output in HybridLayer, not here.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        H   = args.num_attention_heads
        Hkv = args.num_key_value_heads
        D   = args.head_dim
        hid = args.hidden_size

        self.num_heads    = H
        self.num_kv_heads = Hkv
        self.head_dim     = D
        self.scale        = D ** -0.5

        self.q_proj = nn.Linear(hid, H * D,   bias=False)
        self.k_proj = nn.Linear(hid, Hkv * D, bias=False)
        self.v_proj = nn.Linear(hid, Hkv * D, bias=False)
        self.o_proj = nn.Linear(H * D, hid,   bias=False)

        # QK-norm: applied on full (H*D) projection — NOT per-head (D,).
        # See note above about weight shape.
        self.q_norm = nn.RMSNorm(H * D,   eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(Hkv * D, eps=args.rms_norm_eps)

        self.rope = initialize_rope(
            D,
            args.rope_theta,
            args.rope_traditional,
            args.rope_scaling,
            args.max_position_embeddings,
        )

    def __call__(
        self,
        x:     mx.array,
        mask:  Optional[mx.array],
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        B, T, _ = x.shape
        H, Hkv, D = self.num_heads, self.num_kv_heads, self.head_dim

        offset = cache.offset if cache is not None else 0

        q = self.q_norm(self.q_proj(x)).reshape(B, T, H,   D).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x)).reshape(B, T, Hkv, D).transpose(0, 2, 1, 3)
        v = self.v_proj(x)             .reshape(B, T, Hkv, D).transpose(0, 2, 1, 3)

        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        out = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, T, H * D)
        return self.o_proj(out)


# ─────────────────────────────────────────────────────────────────────────────
# SwiGLU FFN
# ─────────────────────────────────────────────────────────────────────────────

class SwiGLU(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj   = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid Decoder Layer
# ─────────────────────────────────────────────────────────────────────────────

class HybridLayer(nn.Module):
    """
    One decoder layer. Norm structure differs by layer type:

    linear_attention (GDN) — pre-norm:
        h = x + GDN(input_layernorm(x))
        x = h + FFN(post_attention_layernorm(h))

    full_attention — post-norm (OLMo2 style):
        h = x + post_attention_layernorm(Attn(x))
        x = h + post_feedforward_layernorm(FFN(h))

    For GDN layers: mx.stop_gradient + mx.checkpoint applied at the mixer call.
    - stop_gradient: no backprop through the sequential recurrence
      (LoRA adapters on attention layers are sufficient for instruction tuning)
    - checkpoint: discards GDN's T intermediate S-matrices from the autograd tape,
      recomputes them if needed. Combined with stop_gradient, recompute never
      triggers — we get the memory savings for free.
    """

    def __init__(self, args: ModelArgs, layer_type: str):
        super().__init__()
        assert layer_type in ("linear_attention", "full_attention"), \
            f"Unknown layer_type: {layer_type!r}"
        self.layer_type = layer_type

        if layer_type == "linear_attention":
            self.input_layernorm          = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.mixer = GatedDeltaNet(args)
        else:
            self.post_attention_layernorm   = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.post_feedforward_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.mixer = Attention(args)

        self.mlp = SwiGLU(args.hidden_size, args.intermediate_size)

    def __call__(
        self,
        x:    mx.array,
        mask: Optional[mx.array],
        cache=None,   # GDNCache | KVCache | None
    ) -> mx.array:
        if self.layer_type == "linear_attention":
            normed = self.input_layernorm(x)
            def _gdn_fwd(n):
                return self.mixer(n, cache=cache)
            x = x + mx.stop_gradient(mx.checkpoint(_gdn_fwd)(normed))
            x = x + self.mlp(self.post_attention_layernorm(x))
        else:
            x = x + self.post_attention_layernorm(
                self.mixer(x, mask=mask, cache=cache)
            )
            x = x + self.post_feedforward_layernorm(self.mlp(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Model  (mlx-lm entry point)
# ─────────────────────────────────────────────────────────────────────────────

class OlmoHybridModel(nn.Module):
    """Inner model (token embeddings + decoder layers + final norm)."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            HybridLayer(args, args.layer_types[i])
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        mask:   Optional[mx.array] = None,
        cache:  Optional[List]     = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask=mask, cache=c)
        return self.norm(h)


class Model(nn.Module):
    """
    OLMo Hybrid 7B — mlx-lm top-level model class.

    Interface:
        logits = model(input_ids, cache=cache)   # (B, T, vocab_size)
        cache  = model.make_cache()              # mixed GDN / KV cache list
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model = OlmoHybridModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache:  Optional[List] = None,
    ) -> mx.array:
        B, T = inputs.shape

        # Determine RoPE offset from first attention-layer cache
        attn_offset = 0
        if cache is not None:
            for c in cache:
                if isinstance(c, KVCache):
                    attn_offset = c.offset
                    break

        # Causal mask (prefill only; decode is T=1 so no mask needed).
        # create_attention_mask only reads .shape[1], so passing inputs directly
        # avoids a redundant embed_tokens call before the model forward pass.
        mask = None
        if T > 1:
            mask = create_attention_mask(inputs, cache[0] if cache else None)

        h = self.model(inputs, mask=mask, cache=cache)

        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(h)
        else:
            return self.lm_head(h)

    def make_cache(self) -> List:
        """
        Create per-layer cache objects. Called by mlx-lm's make_prompt_cache().
        Returns a heterogeneous list: GDNCache for linear_attention layers,
        KVCache for full_attention layers.
        """
        caches = []
        for lt in self.args.layer_types:
            if lt == "linear_attention":
                caches.append(GDNCache())
            else:
                caches.append(KVCache(
                    head_dim=self.args.head_dim,
                    n_heads=self.args.num_key_value_heads,
                ))
        return caches

    def sanitize(self, weights: dict) -> dict:
        """
        Map HuggingFace safetensors weight names → MLX attribute paths.
        Called by mlx-lm's load() after reading the checkpoint shards.

        Key remappings
        ──────────────
        model.embed_tokens.*          → model.embed_tokens.*
        model.norm.*                  → model.norm.*
        lm_head.*                     → lm_head.*

        GDN layers  (HF prefix: model.layers.{i}.linear_attn.*):
          → model.layers.{i}.mixer.*

        Attention layers (HF prefix: model.layers.{i}.self_attn.*):
          → model.layers.{i}.mixer.*

        ShortConv weight reshape:
          HF stores (C, 1, K)  [PyTorch: out_ch, in_ch//groups, kW]
          MLX needs  (C, K, 1)  [MLX conv1d: out_ch, kW, in_ch//groups]
          Transform: w.transpose(0, 2, 1)
        """
        out = {}
        for hf_name, w in weights.items():
            mlx_name = self._hf_to_mlx(hf_name)

            # Fallback: handle weights already in old MLX format — i.e. converted
            # by a standalone model.py that lacked the OlmoHybridModel wrapper.
            # Those keys look like "layers.{i}.mixer.*" / "embed_tokens.weight" /
            # "norm.weight" — just prepend "model." to match the new structure.
            if mlx_name is None:
                if hf_name.startswith("layers.") or hf_name in ("embed_tokens.weight", "norm.weight"):
                    mlx_name = "model." + hf_name
                elif hf_name == "lm_head.weight":
                    mlx_name = "lm_head.weight"

            if mlx_name is None:
                continue

            # Normalise depthwise conv weights to (C, K, 1) — MLX conv1d format.
            # HF safetensors stores (C, 1, K) → transpose to (C, K, 1)
            # Old MLX .npz stores  (C, K)    → unsqueeze last dim to (C, K, 1)
            if "conv1d.weight" in mlx_name:
                if w.ndim == 3 and w.shape[1] == 1:
                    w = w.transpose(0, 2, 1)   # (C, 1, K) → (C, K, 1)
                elif w.ndim == 2:
                    w = w[:, :, None]           # (C, K)    → (C, K, 1)

            out[mlx_name] = w
        return out

    @staticmethod
    def _hf_to_mlx(hf_name: str) -> Optional[str]:
        """Return the MLX attribute path for an HF weight name, or None to skip."""
        # Passthrough: key is already in MLX format (e.g. loading from a previously
        # converted safetensors rather than raw HF weights).
        # HF format uses ".linear_attn." / ".self_attn." — MLX uses ".mixer.".
        # Any key containing ".mixer." has already been sanitized.
        if ".mixer." in hf_name:
            return hf_name

        # Strip leading "model." for most weights
        name = hf_name.removeprefix("model.")

        # Top-level weights
        if hf_name == "lm_head.weight":         return "lm_head.weight"
        if name == "embed_tokens.weight":        return "model.embed_tokens.weight"
        if name == "norm.weight":                return "model.norm.weight"

        # Per-layer weights: model.layers.{i}.*
        m = re.match(r"^layers\.(\d+)\.(.+)$", name)
        if not m:
            return None

        i, rest = m.group(1), m.group(2)

        # Norms and FFN — same key in both layer types
        if rest in (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "post_feedforward_layernorm.weight",
        ):
            return f"model.layers.{i}.{rest}"

        if rest.startswith("mlp."):
            return f"model.layers.{i}.{rest}"

        # GDN layers — HF uses "linear_attn." prefix
        if rest.startswith("linear_attn."):
            sub = rest[len("linear_attn."):]
            return f"model.layers.{i}.mixer.{sub}"

        # Attention layers — HF uses "self_attn." prefix
        if rest.startswith("self_attn."):
            sub = rest[len("self_attn."):]
            return f"model.layers.{i}.mixer.{sub}"

        return None  # unknown key — skip (e.g. rotary_emb.inv_freq)

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self) -> int:
        return self.args.head_dim

    @property
    def n_kv_heads(self) -> int:
        return self.args.num_key_value_heads
