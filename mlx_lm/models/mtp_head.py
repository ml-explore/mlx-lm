# Copyright © 2026 Apple Inc.

"""Multi-Token Prediction (MTP) speculative-decoding head.

Some Qwen3-Next / Qwen3.5-MoE checkpoints ship an extra "MTP" block under
the ``mtp.*`` prefix in the safetensors files (e.g. the upstream
``Qwen/Qwen3-Next-80B-A3B`` and ``Qwen/Qwen3.6-35B-A3B`` releases). The
block is one decoder layer that consumes (last-hidden, next-token-embedding)
pairs and produces a hidden state for the *following* token. Combined with
a verify pass through the main model, this gives bit-exact greedy
speculative decoding at roughly 1.15-1.25x throughput.

This module provides:

* :class:`MTPHead` -- the architecture (single decoder layer + dual
  RMSNorms + concat-and-project ``fc``), structurally a Qwen3-Next decoder
  layer hard-wired to the full-attention + MoE branch.
* :func:`load_mtp_head` -- load + quantize an ``MTPHead`` from a sidecar
  ``model-mtp.safetensors`` file.

The MTP head does *not* own ``embed_tokens`` or ``lm_head`` -- those are
shared with the main model and passed in at forward time. See
``mlx_lm.mtp_generate`` for the matching speculative-decode generator.

Reference: companion impl in ``mlx_fast/mtp/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .qwen3_5 import TextModelArgs
from .qwen3_next import Qwen3NextAttention, Qwen3NextSparseMoeBlock


class MTPLayer(nn.Module):
    """A single MTP transformer block.

    Mirrors ``Qwen3NextDecoderLayer`` hard-wired to ``full-attention + MoE``
    (no GatedDeltaNet branch, no dense-MLP fallback).
    """

    def __init__(self, args: TextModelArgs):
        super().__init__()
        # Marker used by SnapKV / other cache utilities; MTP layer is the
        # full-attention branch only.
        self.is_linear = False
        self.self_attn = Qwen3NextAttention(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.mlp = Qwen3NextSparseMoeBlock(args)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class MTPHead(nn.Module):
    """Full MTP head: dual-RMSNorm + concat-and-project + one decoder block
    + final RMSNorm.

    Does NOT include ``embed_tokens`` or ``lm_head``; those are shared with
    the main model and passed in at forward time.

    Expected weight layout (under the ``language_model.mtp.*`` prefix in the
    upstream safetensors)::

        language_model.mtp.pre_fc_norm_embedding.weight
        language_model.mtp.pre_fc_norm_hidden.weight
        language_model.mtp.fc.weight
        language_model.mtp.layers.0.input_layernorm.weight
        language_model.mtp.layers.0.post_attention_layernorm.weight
        language_model.mtp.layers.0.self_attn.{q,k,v,o}_proj.{weight,scales,biases}
        language_model.mtp.layers.0.self_attn.{q,k}_norm.weight
        language_model.mtp.layers.0.mlp.gate.weight
        language_model.mtp.layers.0.mlp.shared_expert.{gate,up,down}_proj.{weight,scales,biases}
        language_model.mtp.layers.0.mlp.shared_expert_gate.weight
        language_model.mtp.layers.0.mlp.switch_mlp.{gate,up,down}_proj.{weight,scales,biases}
        language_model.mtp.norm.weight
    """

    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.pre_fc_norm_embedding = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.pre_fc_norm_hidden = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        # fc takes concatenated [hidden_normed; embedding_normed] of width 2H,
        # projects back to H.
        self.fc = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
        self.layers = [MTPLayer(args)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        hidden_state: mx.array,
        token_embedding: mx.array,
        cache: Optional[Any] = None,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass. Returns post-norm hidden of shape ``(B, L, H)``.

        Caller applies ``lm_head`` afterward to obtain draft-token logits.
        """
        h_normed = self.pre_fc_norm_hidden(hidden_state)
        e_normed = self.pre_fc_norm_embedding(token_embedding)
        x = self.fc(mx.concatenate([e_normed, h_normed], axis=-1))
        for layer in self.layers:
            x = layer(x, mask=mask, cache=cache)
        return self.norm(x)


# --- Quantization predicate -------------------------------------------------

# Tensors stored as bf16 (NOT quantized), full path under the MTPHead root.
_MTP_UNQUANTIZED_PATHS = {
    "fc",  # 4096 -> 2048 projection kept dense in standard extracts
    "layers.0.mlp.gate",  # MoE router
    "layers.0.mlp.shared_expert_gate",  # shared-expert router
}


def mtp_quant_predicate(path: str, module: nn.Module):
    """Quantization predicate suitable for ``mlx_lm.utils.quantize_model``.

    The path is relative to the ``MTPHead`` root.
    """
    if not hasattr(module, "to_quantized"):
        return False
    if path in _MTP_UNQUANTIZED_PATHS:
        return False
    return True


# --- Loader -----------------------------------------------------------------


def _strip_mtp_prefix(weights: dict) -> dict:
    """Drop the ``language_model.mtp.`` or ``mtp.`` prefix from every key.

    Accepts both layouts: the upstream HF shards use
    ``language_model.mtp.*``, while standalone MTP-only sidecars may use
    plain ``mtp.*``.
    """
    out = {}
    for k, v in weights.items():
        if k.startswith("language_model.mtp."):
            out[k[len("language_model.mtp.") :]] = v
        elif k.startswith("mtp."):
            out[k[len("mtp.") :]] = v
    return out


def _build_args_from_config(text_config: dict) -> TextModelArgs:
    """Construct ``TextModelArgs`` from a model's ``text_config`` dict."""
    field_names = {f.name for f in TextModelArgs.__dataclass_fields__.values()}
    init_kwargs = {k: v for k, v in text_config.items() if k in field_names}
    init_kwargs.setdefault(
        "model_type", text_config.get("model_type", "qwen3_5_moe_text")
    )
    init_kwargs.setdefault(
        "intermediate_size",
        text_config.get(
            "intermediate_size", text_config.get("moe_intermediate_size", 512)
        ),
    )
    return TextModelArgs(**init_kwargs)


def load_mtp_head(
    weights_path,
    config: dict,
    *,
    group_size: int = 64,
    bits: int = 4,
    mode: str = "affine",
) -> Tuple[MTPHead, TextModelArgs]:
    """Load an :class:`MTPHead` from a ``model-mtp.safetensors`` file.

    The default quantization (4-bit affine, group_size=64) matches the
    ``mlx-community`` Qwen3-Next/Qwen3.6 base releases.

    Args:
        weights_path: path to ``model-mtp.safetensors``.
        config: the ``config.json`` dict for the main model. Must contain a
            ``text_config`` sub-dict (the standard Qwen3-Next layout) -- if
            absent, ``config`` itself is treated as the text config.
        group_size: quantization group size for matrix weights.
        bits: bits per weight.
        mode: quantization mode (``affine`` by default).

    Returns:
        ``(head, args)`` -- the loaded ``MTPHead`` and the
        ``TextModelArgs`` used to build it.
    """
    # Local import to avoid pulling utils at module import time.
    from ..utils import quantize_model

    text_config = config.get("text_config", config)
    args = _build_args_from_config(text_config)

    head = MTPHead(args)

    head, _ = quantize_model(
        head,
        config={},
        group_size=group_size,
        bits=bits,
        mode=mode,
        quant_predicate=mtp_quant_predicate,
    )

    raw = mx.load(str(weights_path))
    weights = _strip_mtp_prefix(raw)
    if not weights:
        raise ValueError(
            "No tensors with 'language_model.mtp.' prefix found in "
            f"{weights_path}. Got example keys: {list(raw.keys())[:5]}"
        )

    head.load_weights(list(weights.items()))
    mx.eval(head.parameters())
    return head, args
