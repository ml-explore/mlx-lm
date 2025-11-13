# ABOUTME: Provides array-only decode helpers specialized for LLaMA blocks.
# ABOUTME: Projects transformer weights without relying on model cache objects.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx


@dataclass
class _LlamaOutputs:
    q: mx.array
    k: mx.array
    v: mx.array


class LlamaArrayGraph:
    """Thin utility that exposes the building blocks of LLaMA layers."""

    def __init__(self, model) -> None:
        core = getattr(model, "model", model)
        args = getattr(model, "args", getattr(core, "args", None))
        if not hasattr(core, "layers") or not hasattr(core, "embed_tokens"):
            raise ValueError("LlamaArrayGraph expects a LLaMA-style model.")
        self._outer = model
        self._core = core
        self._args = args
        self.layers = list(core.layers)
        self.embed_tokens = core.embed_tokens
        self.norm = core.norm
        tied = True
        if args is not None:
            tied = bool(getattr(args, "tie_word_embeddings", True))
        if tied:
            self._logits_fn = core.embed_tokens.as_linear
        else:
            head = getattr(model, "lm_head", None)
            if head is None:
                raise ValueError("Model missing lm_head for untied embeddings.")
            self._logits_fn = head

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def embed(self, token_ids: mx.array) -> mx.array:
        if token_ids.ndim == 1:
            token_ids = mx.expand_dims(token_ids, axis=1)
        return self.embed_tokens(token_ids)

    def project_qkv(self, attn, x_norm: mx.array) -> _LlamaOutputs:
        q = attn.q_proj(x_norm)
        k = attn.k_proj(x_norm)
        v = attn.v_proj(x_norm)
        B, L, _ = q.shape
        q = q.reshape(B, L, attn.n_heads, -1).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
        return _LlamaOutputs(q=q, k=k, v=v)

    def apply_rope(self, attn, tensor: mx.array, offsets: mx.array) -> mx.array:
        if offsets.ndim == 2:
            offsets = mx.reshape(offsets, (offsets.shape[0],))
        return attn.rope(tensor, offset=offsets)

    def attention_output(self, attn, attn_ctx: mx.array) -> mx.array:
        """Convert attention context [B,H,L,D] to [B,L,hidden]."""
        B, _, L, D = attn_ctx.shape
        context = attn_ctx.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return attn.o_proj(context)

    def mlp(self, layer, hidden: mx.array) -> mx.array:
        return layer.mlp(layer.post_attention_layernorm(hidden))

    def logits(self, hidden: mx.array) -> mx.array:
        return self._logits_fn(self.norm(hidden))


__all__ = ["LlamaArrayGraph"]
