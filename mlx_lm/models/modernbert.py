# Copyright © 2026
#
# ModernBERT support for MLX-LM
#
# This module provides:
# - A ModernBERT encoder backbone (ModernBertModel)
# - Task heads for:
#     - sequence classification
#     - regression
#     - masked language modeling
# - Load-time head selection via args.task (model_config overlay in mlx_lm.load)
#
# The implementation follows the Hugging Face Transformers ModernBERT architecture:
# - Embeddings: tok_embeddings + LayerNorm + Dropout
# - Encoder layer: (attn_norm -> attn -> residual) + (mlp_norm -> mlp -> residual)
# - Attention: packed Wqkv + Wo with RoPE applied to q/k
# - Attention schedule:
#     - global/full attention when layer_id % global_attn_every_n_layers == 0
#     - sliding/local attention otherwise with window size local_attention
# - MLP: Wi -> split(2) -> act(input) * gate -> dropout -> Wo
# - final_norm: LayerNorm
#
# Checkpoint note:
# - The reference MLX checkpoint stores backbone weights under keys prefixed with `model.*`
#   (e.g. model.layers.0.attn.Wqkv.weight). The top-level `Model` wrapper exposes the
#   backbone at `self.model` to match that layout.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple



import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, scaled_dot_product_attention





# -----------------------------
# Utilities
# -----------------------------


def _masked_mean_pool(last_hidden_state: mx.array, attention_mask: mx.array) -> mx.array:
    """
    Mean pool over sequence length using attention_mask.

    last_hidden_state: (B, T, H)
    attention_mask: (B, T) with 1 for tokens to include.
    returns: (B, H)
    """
    m = attention_mask.astype(last_hidden_state.dtype)
    denom = mx.maximum(m.sum(axis=1, keepdims=True), mx.array(1.0, dtype=m.dtype))
    summed = (last_hidden_state * m[:, :, None]).sum(axis=1)
    return summed / denom


def _make_full_attention_bias(attention_mask: mx.array, *, dtype: mx.Dtype) -> mx.array:
    """
    Match HF `_prepare_4d_attention_mask` / `AttentionMaskConverter._expand_mask` behavior.

    HF does:
      expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
      inverted_mask = 1.0 - expanded_mask
      return inverted_mask.masked_fill(inverted_mask.to(bool), finfo(dtype).min)

    We implement the same semantics for the common case tgt_len == src_len == T.

    attention_mask: (B, T) with 1 keep, 0 pad
    returns: (B, 1, T, T) additive bias in `dtype` with:
      - 0.0 where keys are allowed
      - finfo(dtype).min where keys are masked
    """
    bsz, src_len = attention_mask.shape
    tgt_len = src_len

    expanded = attention_mask[:, None, None, :].astype(dtype)
    expanded = mx.broadcast_to(expanded, (bsz, 1, tgt_len, src_len))

    inverted = mx.array(1.0, dtype=dtype) - expanded
    finfo_min = mx.array(mx.finfo(dtype).min, dtype=dtype)

    # masked_fill(inverted_mask.to(bool), finfo.min)
    return mx.where(inverted.astype(mx.bool_), finfo_min, inverted)


def _make_sliding_window_bias(
    attention_mask: mx.array,
    window: int,
    *,
    global_attention_mask: mx.array,
    dtype: mx.Dtype,
) -> mx.array:
    """
    Match HF ModernBERT sliding_window_mask construction:

      global_attention_mask = _prepare_4d_attention_mask(attention_mask, self.dtype)
      rows = torch.arange(global_attention_mask.shape[2]).unsqueeze(0)
      distance = torch.abs(rows - rows.T)
      window_mask = (distance <= local_attention // 2).unsqueeze(0).unsqueeze(0)
      sliding_window_mask = global_attention_mask.masked_fill(window_mask.logical_not(), finfo(dtype).min)

    In other words:
    - start from the global 4D additive mask (already contains pad masking via finfo.min)
    - additionally mask out keys outside the sliding window by setting them to finfo.min

    attention_mask: (B, T) with 1 keep, 0 pad
    global_attention_mask: (B, 1, T, T) additive mask in `dtype`
    returns: (B, 1, T, T) additive mask in `dtype`
    """
    _, _, T, _ = global_attention_mask.shape

    half = int(window) // 2
    rows = mx.arange(T)[None, :]  # (1, T)
    distance = mx.abs(rows - rows.T)  # (T, T)

    window_mask = (distance <= half)  # (T, T)
    window_mask = window_mask[None, None, :, :]  # (1, 1, T, T)

    finfo_min = mx.array(mx.finfo(dtype).min, dtype=dtype)
    # masked_fill(window_mask.logical_not(), finfo.min)
    return mx.where(window_mask, global_attention_mask, finfo_min)


def _rotate_half(x: mx.array) -> mx.array:
    """
    Match HF `rotate_half` exactly:
      x1 = x[..., : x.shape[-1] // 2]
      x2 = x[..., x.shape[-1] // 2 :]
      return cat((-x2, x1), dim=-1)
    """
    h = x.shape[-1] // 2
    x1 = x[..., :h]
    x2 = x[..., h:]
    return mx.concatenate([-x2, x1], axis=-1)


def _apply_rotary_pos_emb(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array,
    *,
    unsqueeze_dim: int = 1,
) -> Tuple[mx.array, mx.array]:
    """
    Match HF `apply_rotary_pos_emb` broadcasting semantics exactly.

    HF does:
      cos = cos.unsqueeze(unsqueeze_dim)
      sin = sin.unsqueeze(unsqueeze_dim)
      q_embed = (q * cos) + (rotate_half(q) * sin)
      k_embed = (k * cos) + (rotate_half(k) * sin)

    For our ModernBERT tensors:
      q,k are shaped (B, nh, T, hd) so unsqueeze_dim=1 is correct.
      cos,sin are shaped (B, T, hd).
    """
    if unsqueeze_dim == 1:
        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]
    elif unsqueeze_dim == 2:
        cos = cos[:, :, None, :]
        sin = sin[:, :, None, :]
    else:
        raise ValueError(f"Unsupported unsqueeze_dim={unsqueeze_dim} for rotary pos emb")

    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _act(name: str):
    name = (name or "").lower()
    # HF uses ACT2FN; for ModernBERT checkpoint, hidden_activation is "gelu".
    if name in ("gelu", "gelu_new", "gelu_fast", "gelu_approx"):
        return nn.gelu_approx
    if name in ("relu",):
        return nn.relu
    if name in ("silu", "swish"):
        return nn.silu
    return nn.gelu_approx


# -----------------------------
# Config / Args
# -----------------------------


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str

    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int

    # Positions / tokens
    max_position_embeddings: int = 8192
    position_embedding_type: str = "absolute"
    pad_token_id: int = 0

    # Normalization / dropout
    norm_eps: float = 1e-5
    norm_bias: bool = False

    embedding_dropout: float = 0.0
    attention_dropout: float = 0.0
    mlp_dropout: float = 0.0

    attention_bias: bool = False
    mlp_bias: bool = False

    # ModernBERT attention schedule + RoPE bases
    local_attention: int = 128
    global_attn_every_n_layers: int = 3
    local_rope_theta: float = 10000.0
    global_rope_theta: float = 160000.0

    # tied embeddings flag (not enforced here; checkpoint may contain both weights)
    tie_word_embeddings: bool = True

    # Task selection via load-time model_config overlay
    task: Optional[str] = None  # backbone|sequence_classification|regression|masked_lm
    num_labels: int = 2

    # classification head config
    classifier_pooling: str = "mean"  # mean|cls
    classifier_dropout: float = 0.0
    classifier_bias: bool = False
    classifier_activation: str = "gelu"

    # allow config alias
    layer_norm_eps: Optional[float] = None

    def __post_init__(self):
        if self.layer_norm_eps is not None:
            self.norm_eps = self.layer_norm_eps
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads "
                f"(got hidden_size={self.hidden_size}, num_attention_heads={self.num_attention_heads})"
            )


# -----------------------------
# Layers
# -----------------------------


class LayerNorm(nn.LayerNorm):
    """
    Use MLX's built-in LayerNorm.

    This keeps the same checkpoint key layout:
      - <prefix>.weight
      - <prefix>.bias (only if bias=True / norm_bias=True)
    """

    def __init__(self, dims: int, eps: float = 1e-5, bias: bool = False):
        super().__init__(dims, eps=eps, bias=bias)


class ModernBertEmbeddings(nn.Module):
    """
    HF ModernBertEmbeddings equivalent (as far as weights indicate).
    The reference checkpoint does not include position embeddings, so we do not
    create position embedding parameters here.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.norm = LayerNorm(args.hidden_size, eps=args.norm_eps, bias=args.norm_bias)
        self.drop = (
            nn.Dropout(p=args.embedding_dropout)
            if args.embedding_dropout and args.embedding_dropout > 0.0
            else (lambda x: x)
        )

    def __call__(self, input_ids: mx.array) -> mx.array:
        x = self.tok_embeddings(input_ids)
        x = self.norm(x)
        return self.drop(x)


class ModernBertRotaryEmbedding:
    """
    Stateless RoPE helper (NOT an nn.Module) to avoid introducing extra parameters
    that are not present in the MLX safetensors checkpoint.

    Matches the HF "default" RoPE math:
      inv_freq = 1 / (theta ** (arange(0, dim, 2) / dim))
      freqs = inv_freq @ position_ids
      emb = concat(freqs, freqs)
      cos/sin = cos(emb), sin(emb)
    """

    def __init__(self, dim: int, base: float):
        self.dim = int(dim)
        self.base = float(base)

    def inv_freq(self) -> mx.array:
        return 1.0 / (
            self.base ** (mx.arange(0, self.dim, 2).astype(mx.float32) / float(self.dim))
        )

    def cos_sin(
        self,
        *,
        x: mx.array,
        position_ids: mx.array,
        inv_freq: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        inv = inv_freq[None, :, None].astype(mx.float32)
        pos = position_ids[:, None, :].astype(mx.float32)
        freqs = (inv @ pos).transpose(0, 2, 1)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb)
        sin = mx.sin(emb)
        return cos.astype(x.dtype), sin.astype(x.dtype)

    def __call__(self, x: mx.array, position_ids: mx.array) -> Tuple[mx.array, mx.array]:
        return self.cos_sin(x=x, position_ids=position_ids, inv_freq=self.inv_freq())


class ModernBertAttention(nn.Module):
    """
    HF ModernBertAttention (sdpa/eager behavior) implemented in MLX:
    - packed Wqkv + Wo
    - RoPE on q/k
    - attention mask: full/global or sliding/local based on layer index schedule

    NOTE:
    Cos/sin generation is delegated to ModernBertModel via a shared RoPE cache keyed by
    attention type ("full_attention" vs "sliding_attention") to better match HF.
    """

    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.config = args
        self.layer_id = int(layer_id)

        self.num_heads = args.num_attention_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.scale = self.head_dim**-0.5

        # Keep checkpoint naming: attn.Wqkv and attn.Wo
        self.Wqkv = nn.Linear(
            args.hidden_size, 3 * self.all_head_size, bias=args.attention_bias
        )
        self.Wo = nn.Linear(args.hidden_size, args.hidden_size, bias=args.attention_bias)

        # Dropout on output projection
        self.out_drop = (
            nn.Dropout(p=args.attention_dropout)
            if args.attention_dropout and args.attention_dropout > 0.0
            else (lambda x: x)
        )

        # Attention schedule (HF semantics):
        # - global/full attention when layer_id % global_attn_every_n_layers == 0
        # - sliding/local attention otherwise
        every = int(args.global_attn_every_n_layers) if args.global_attn_every_n_layers else 0
        self.is_global = (every > 0) and (self.layer_id % every == 0)
        self.local_window = int(args.local_attention)

        # Identify layer type for RoPE caching
        self.layer_type = "full_attention" if self.is_global else "sliding_attention"

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        attention_mask: mx.array,
        position_ids: mx.array,
        rope_cos_sin: Tuple[mx.array, mx.array],
    ) -> mx.array:
        B, T, _ = hidden_states.shape

        qkv = self.Wqkv(hidden_states)  # (B, T, 3*H)
        q, k, v = mx.split(qkv, 3, axis=-1)

        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        cos, sin = rope_cos_sin  # (B, T, hd)
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        # Match HF mask mechanics for sdpa:
        # 1) global_attention_mask = _prepare_4d_attention_mask(attention_mask, dtype)
        # 2) sliding_window_mask = global_attention_mask.masked_fill(~window_mask, finfo.min)
        mask_dtype = q.dtype

        global_attention_mask = _make_full_attention_bias(attention_mask, dtype=mask_dtype)  # (B,1,T,T)

        if self.is_global:
            attn_mask = global_attention_mask
        else:
            attn_mask = _make_sliding_window_bias(
                attention_mask,
                self.local_window,
                global_attention_mask=global_attention_mask,
                dtype=mask_dtype,
            )  # (B,1,T,T)

        out = scaled_dot_product_attention(
            q,
            k,
            v,
            cache=None,
            scale=self.scale,
            mask=attn_mask,
        )  # (B, nh, T, hd)

        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.all_head_size)

        out = self.Wo(out)
        out = self.out_drop(out)

        return out


class ModernBertMLP(nn.Module):
    """
    HF ModernBertMLP:
      input, gate = Wi(hidden_states).chunk(2, dim=-1)
      out = Wo(drop(act(input) * gate))
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.Wi = nn.Linear(
            args.hidden_size, int(args.intermediate_size) * 2, bias=args.mlp_bias
        )
        self.act = _act(getattr(args, "hidden_activation", "gelu"))
        self.drop = (
            nn.Dropout(p=args.mlp_dropout)
            if args.mlp_dropout and args.mlp_dropout > 0.0
            else (lambda x: x)
        )
        self.Wo = nn.Linear(args.intermediate_size, args.hidden_size, bias=args.mlp_bias)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        # HF ModernBertMLP does:
        #   input, gate = Wi(hidden_states).chunk(2, dim=-1)
        #   out = Wo(drop(act(input) * gate))
        input, gate = mx.split(self.Wi(hidden_states), 2, axis=-1)
        return self.Wo(self.drop(self.act(input) * gate))


class ModernBertEncoderLayer(nn.Module):
    """
    HF ModernBertEncoderLayer:
      attn_norm: Identity for layer 0 else LayerNorm
      attn: ModernBertAttention
      mlp_norm: LayerNorm
      residuals
    """

    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.layer_id = int(layer_id)
        self.attn_norm = (
            (lambda x: x)
            if self.layer_id == 0
            else LayerNorm(args.hidden_size, eps=args.norm_eps, bias=args.norm_bias)
        )
        self.attn = ModernBertAttention(args, layer_id=self.layer_id)
        self.mlp_norm = LayerNorm(args.hidden_size, eps=args.norm_eps, bias=args.norm_bias)
        self.mlp = ModernBertMLP(args)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        attention_mask: mx.array,
        position_ids: mx.array,
        rope_cos_sin: Tuple[mx.array, mx.array],
    ) -> mx.array:
        attn_out = self.attn(
            self.attn_norm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            rope_cos_sin=rope_cos_sin,
        )
        hidden_states = hidden_states + attn_out
        mlp_out = self.mlp(self.mlp_norm(hidden_states))
        hidden_states = hidden_states + mlp_out
        return hidden_states


class ModernBertModel(nn.Module):
    """
    Backbone model.

    IMPORTANT for checkpoint compatibility:
    - This class is intended to live at attribute path `model` under the top-level `Model` wrapper.
    - Therefore checkpoint keys like `model.layers.0.attn.Wqkv.weight` map to:
        wrapper.model.layers[0].attn.Wqkv.weight

    HF alignment:
    - HF uses a shared ModernBertRotaryEmbedding module with per-layer-type parameters.
      We emulate this by caching inv_freq vectors for each layer_type and generating
      cos/sin from that cache.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.embeddings = ModernBertEmbeddings(args)
        self.layers = [
            ModernBertEncoderLayer(args, layer_id=i) for i in range(args.num_hidden_layers)
        ]
        self.final_norm = LayerNorm(args.hidden_size, eps=args.norm_eps, bias=args.norm_bias)

        # Shared RoPE cache keyed by layer_type.
        # This is NOT loaded from weights; it's computed from config.
        self._rope = {
            "full_attention": ModernBertRotaryEmbedding(
                dim=args.hidden_size // args.num_attention_heads,
                base=float(args.global_rope_theta),
            ),
            "sliding_attention": ModernBertRotaryEmbedding(
                dim=args.hidden_size // args.num_attention_heads,
                base=float(args.local_rope_theta),
            ),
        }
        self._inv_freq = {
            "full_attention": self._rope["full_attention"].inv_freq(),
            "sliding_attention": self._rope["sliding_attention"].inv_freq(),
        }

    def _rope_cos_sin(
        self,
        *,
        x: mx.array,
        position_ids: mx.array,
        layer_type: str,
    ) -> Tuple[mx.array, mx.array]:
        rope = self._rope[layer_type]
        inv = self._inv_freq[layer_type]

        if position_ids.ndim == 2 and position_ids.shape[0] == 1:
            position_ids = mx.broadcast_to(position_ids, (x.shape[0], position_ids.shape[1]))

        return rope.cos_sin(x=x, position_ids=position_ids, inv_freq=inv)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> Dict[str, mx.array]:
        if attention_mask is None:
            attention_mask = mx.ones(input_ids.shape, dtype=mx.int32)
        attention_mask = attention_mask.astype(mx.int32)

        B, T = input_ids.shape
        # Match HF ModernBertModel.forward default:
        #   position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        # i.e. shape (1, T), not (B, T).
        position_ids = mx.arange(T)[None, :].astype(mx.int32)

        hidden_states = self.embeddings(input_ids)

        # Precompute cos/sin for both layer types once per forward (HF does this in a shared module).
        # HF computes rotary embeddings using position_ids with shape (1, T), and expands internally.
        cos_sin_full = self._rope_cos_sin(x=hidden_states, position_ids=position_ids, layer_type="full_attention")
        cos_sin_slide = self._rope_cos_sin(x=hidden_states, position_ids=position_ids, layer_type="sliding_attention")

        for layer in self.layers:
            layer_type = layer.attn.layer_type
            rope_cos_sin = cos_sin_full if layer_type == "full_attention" else cos_sin_slide
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                rope_cos_sin=rope_cos_sin,
            )

        hidden_states = self.final_norm(hidden_states)

        pooling = (self.args.classifier_pooling or "mean").lower()
        pooled = hidden_states[:, 0, :] if pooling == "cls" else _masked_mean_pool(hidden_states, attention_mask)

        return {"last_hidden_state": hidden_states, "pooled_output": pooled}


# -----------------------------
# Task heads
# -----------------------------


class ModernBertSequenceHead(nn.Module):
    def __init__(self, args: ModelArgs, out_dim: int):
        super().__init__()
        self.dropout = nn.Dropout(p=args.classifier_dropout) if args.classifier_dropout > 0.0 else (lambda x: x)
        self.proj = nn.Linear(args.hidden_size, out_dim, bias=args.classifier_bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(self.dropout(x))


class ModernBertMaskedLMHead(nn.Module):
    """
    Minimal MLM head compatible with checkpoint keys:
      head.dense.weight
      head.norm.weight
      decoder.weight (+ decoder.bias)

    Note: HF uses LayerNorm in head.norm for this model.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dense = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.norm = LayerNorm(args.hidden_size, eps=args.norm_eps, bias=args.norm_bias)
        self.decoder = nn.Linear(args.hidden_size, args.vocab_size, bias=True)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        h = self.dense(hidden_states)
        h = nn.gelu_approx(h)
        h = self.norm(h)
        return self.decoder(h)


# -----------------------------
# Loader entrypoint for MLX-LM
# -----------------------------


class Model(nn.Module):
    """
    Top-level wrapper expected by mlx_lm.utils._get_classes()

    This wrapper is responsible for:
    - exposing the backbone at `self.model` to match checkpoint key prefix `model.*`
    - adding optional task heads selected at load time via args.task
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type

        # Backbone always exists and must be at `self.model` for weight loading.
        self.model = ModernBertModel(args)

        task = (args.task or "").lower() if args.task is not None else ""
        if task in ("sequence_classification", "classification", "cls"):
            self.classifier = ModernBertSequenceHead(args, out_dim=args.num_labels)
        elif task in ("regression", "sequence_regression", "reg"):
            self.regressor = ModernBertSequenceHead(args, out_dim=1)
        elif task in ("masked_lm", "mlm"):
            # For MLM, we need `head.*` and `decoder.*` weights. Provide a head module.
            self.head = ModernBertMaskedLMHead(args)
        elif task in ("", "backbone", "encoder"):
            # Backbone-only, no extra modules needed.
            pass
        else:
            raise ValueError(
                f"Unsupported ModernBERT task '{args.task}'. "
                "Supported: sequence_classification, regression, masked_lm, backbone"
            )

    def __call__(self, input_ids: mx.array, attention_mask: Optional[mx.array] = None):
        task = (self.args.task or "").lower() if self.args.task is not None else ""

        if task in ("sequence_classification", "classification", "cls"):
            out = self.model(input_ids, attention_mask=attention_mask)
            pooled = out["pooled_output"]
            return self.classifier(pooled)

        if task in ("regression", "sequence_regression", "reg"):
            out = self.model(input_ids, attention_mask=attention_mask)
            pooled = out["pooled_output"]
            return self.regressor(pooled).squeeze(-1)  # (B,)

        if task in ("masked_lm", "mlm"):
            out = self.model(input_ids, attention_mask=attention_mask)
            return self.head(out["last_hidden_state"])

        # backbone
        return self.model(input_ids, attention_mask=attention_mask)

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """
        Adjust checkpoint weights for the selected task.

        - For sequence classification/regression/backbone, ignore MLM head weights
          present in the reference checkpoint (head.*, decoder.*).
        - For MLM, keep them.
        """
        task = (self.args.task or "").lower() if self.args.task is not None else ""
        if task in ("masked_lm", "mlm"):
            return weights
        drop_prefixes = ("head.", "decoder.")
        return {k: v for k, v in weights.items() if not k.startswith(drop_prefixes)}

    @property
    def layers(self):
        # Expose backbone layers for potential adapter injection utilities.
        return getattr(self.model, "layers", None)
