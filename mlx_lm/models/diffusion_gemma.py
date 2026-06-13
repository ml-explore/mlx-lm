# Copyright © 2026 Apple Inc.
#
# DiffusionGemma (google/diffusiongemma-26B-A4B-it) — text-only port for mlx-lm.
#
# DiffusionGemma is an encoder-decoder BLOCK-DIFFUSION ("multi-canvas") MoE built
# on the Gemma 4 stack. Unlike an autoregressive LM, it denoises fixed-size token
# CANVASES instead of emitting one token at a time:
#
#   encoder (Gemma4 blocks, no LM head)  →  prefills the prompt into a KV cache
#   decoder (bidirectional, is_causal=False)  →  denoises a 256-token canvas,
#       seeing prompt context by CONCATENATING the encoder cache K/V with the
#       decoder's own K/V; self-conditioning feeds the previous step's logits
#       through a small MLP back into the canvas embeddings.
#   block-autoregressive commits  →  a finalized canvas is appended to the cache,
#       then the next canvas is denoised, until EOS/stop.
#
# Parity references (do NOT reinvent — mirror these, credit @Blaizzy):
#   - mlx-vlm DiffusionGemma4Backbone (Pedro Cuenca / Blaizzy) — the working MLX
#     impl (vision+text). We build the text-only path.
#   - HF transformers modular_diffusion_gemma.py / generation_diffusion_gemma.py —
#     canonical architecture + sampler semantics.
#   - mlx_lm.models.gemma4_text — the base whose blocks we extend (it already has
#     SwitchGLU MoE, per-layer-type RoPE, global heads, logit softcap).
# Full recon: ~/Projects/mlx-diffusion-gemma/RECON.md. Upstream lane: ml-explore/
# mlx-lm issue #1391 (our shape comment posted; building on it while silent).
#
# ── BUILD SEQUENCE (this file, top-down; each step gated by tests/test_models.py
#    conventions + logits parity vs transformers on a tiny random-init config) ──
#   [x] 1. ModelArgs                — config (mirrors DiffusionGemmaTextConfig).
#   [ ] 2. building blocks          — RMSNorm, Attention (generic mask =
#                                     bidirectional injectable), summed MLP+MoE
#                                     (SwitchGLU) w/ per-expert router scale,
#                                     layer_scalar, v_norm, self-conditioning MLP.
#   [ ] 3. Encoder / Decoder        — encoder prefills cache; decoder does the
#                                     cache-concat bidirectional canvas attention.
#   [ ] 4. Model (+ sanitize)       — tie encoder↔decoder; sanitize the existing
#                                     mlx-community conversions (MoE key splits,
#                                     drop vision tower); softcapped logits.
#   [ ] 5. diffusion_generate       — new top-level sibling to stream_generate:
#                                     EntropyBoundSampler + temp schedule (0.8→0.4)
#                                     + adaptive stop; per-canvas streaming.

import weakref
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache
from .gemma4_text import RMSNormNoScale
from .rope_utils import initialize_rope
from .switch_layers import SwitchLinear, _gather_sort, _scatter_unsort


@dataclass
class ModelArgs(BaseModelArgs):
    """Mirrors HF `DiffusionGemmaTextConfig`. Note this is NOT gemma4_text's config:
    DiffusionGemma drops the per-layer-input embeddings and KV-sharing, and adds the
    diffusion knobs (`use_bidirectional_attention`, `canvas_length`)."""

    model_type: str = "diffusion_gemma_text"
    vocab_size: int = 262_144
    hidden_size: int = 2304
    intermediate_size: int = 9216
    num_hidden_layers: int = 30
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 256
    # Global (full-attention) layers use wider heads + their own KV-head count.
    global_head_dim: int = 512
    num_global_key_value_heads: Optional[int] = None
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 131_072
    sliding_window: int = 512
    # 5:1 sliding:full by default (sliding_window_pattern == 6 in HF terms); the
    # LAST layer is forced to full_attention.
    sliding_window_pattern: int = 6
    layer_types: Optional[List[str]] = None
    final_logit_softcapping: float = 30.0
    tie_word_embeddings: bool = True
    pad_token_id: int = 0
    # ── MoE (summed with the dense MLP; SwitchGLU experts + a per-expert router
    #    scale) — None on dense checkpoints. ──
    num_experts: Optional[int] = None
    top_k_experts: Optional[int] = None
    moe_intermediate_size: Optional[int] = None
    # ── diffusion ──
    # "all" → every token bidirectional (decoder canvas); "vision" → only image
    # tokens bidirectional (multimodal, unused in the text-only port); None → causal.
    use_bidirectional_attention: Optional[str] = None
    canvas_length: int = 256
    # Per-layer-type RoPE: sliding = default θ=10k full-rotary; full = proportional
    # partial_rotary_factor=0.25 θ=1e6 (the Gemma 4 scheme).
    rope_parameters: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.rope_parameters is None:
            self.rope_parameters = {
                "full_attention": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1_000_000.0,
                    "rope_type": "proportional",
                },
                "sliding_attention": {
                    "partial_rotary_factor": 1.0,
                    "rope_theta": 10_000.0,
                    "rope_type": "default",
                },
            }
        if self.layer_types is None:
            # (pattern-1) sliding then 1 full, tiled to depth.
            pattern = ["sliding_attention"] * (self.sliding_window_pattern - 1) + [
                "full_attention"
            ]
            self.layer_types = (pattern * (self.num_hidden_layers // len(pattern) + 1))[
                : self.num_hidden_layers
            ]
        # The last layer must be full_attention (HF enforces this).
        if self.layer_types and self.layer_types[-1] != "full_attention":
            self.layer_types[-1] = "full_attention"
        if self.num_global_key_value_heads is None:
            self.num_global_key_value_heads = self.num_key_value_heads


# ── Slice 1: building blocks ──────────────────────────────────────────────────
# Mirrors mlx-vlm's DiffusionGemma4Backbone (Blaizzy / Pedro Cuenca) so the wire
# weights load 1:1. The MoE here differs from gemma4_text's plain SwitchGLU: it
# carries a per-token router `scale` AND a `per_expert_scale`, and the gates use
# GeGLU. A layer's feed-forward sums the dense MLP path with this MoE path.


@partial(mx.compile, shapeless=True)
def geglu(gate, x):
    return nn.gelu_approx(gate) * x


class MLP(nn.Module):
    """The dense feed-forward path (GeGLU)."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(geglu(self.gate_proj(x), self.up_proj(x)))


class Router(nn.Module):
    """Top-k expert router with the DiffusionGemma scales: the RMS-normed hidden
    is multiplied by a learned per-channel `scale` (× hidden_size**-0.5) before the
    gate projection, and the softmax weights by a learned `per_expert_scale`."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.eps = config.rms_norm_eps
        self.proj = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.scale = mx.ones((config.hidden_size,))
        self.per_expert_scale = mx.ones((config.num_experts,))
        self._root_size = config.hidden_size**-0.5

    def __call__(self, x):
        x = mx.fast.rms_norm(x, None, self.eps)
        x = x * self.scale * self._root_size
        scores = self.proj(x)
        top_k = self.config.top_k_experts
        indices = mx.argpartition(scores, kth=-top_k, axis=-1)[..., -top_k:]
        weights = mx.take_along_axis(scores, indices, axis=-1)
        weights = mx.softmax(weights, axis=-1, precise=True)
        weights = weights * self.per_expert_scale[indices]
        return indices, weights


class Experts(nn.Module):
    """Routed experts (SwitchLinear gate_up + down, GeGLU), with gather-sort for
    larger token counts. Returns the weighted sum over the top-k experts."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.hidden_dims = config.moe_intermediate_size
        self.gate_up_proj = SwitchLinear(
            input_dims=config.hidden_size,
            output_dims=2 * config.moe_intermediate_size,
            num_experts=config.num_experts,
            bias=False,
        )
        self.down_proj = SwitchLinear(
            input_dims=config.moe_intermediate_size,
            output_dims=config.hidden_size,
            num_experts=config.num_experts,
            bias=False,
        )

    def __call__(self, x, top_k_indices, top_k_weights):
        x = mx.expand_dims(x, (-2, -3))
        do_sort = top_k_indices.size >= 64
        indices = top_k_indices
        inv_order = None
        if do_sort:
            x, indices, inv_order = _gather_sort(x, top_k_indices)

        gate_up = self.gate_up_proj(x, indices, sorted_indices=do_sort)
        gate = gate_up[..., : self.hidden_dims]
        up = gate_up[..., self.hidden_dims :]
        y = self.down_proj(geglu(gate, up), indices, sorted_indices=do_sort)

        if do_sort:
            y = _scatter_unsort(y, inv_order, top_k_indices.shape)

        y = y.squeeze(-2)
        return (y * top_k_weights[..., None]).sum(axis=-2)


# ── cache helpers (duck-typed; the encoder-decoder cache machinery is wired in a
#    later slice — these read whatever prefix cache is present without importing it).
def _cache_offset(cache) -> int:
    if cache is None or getattr(cache, "keys", None) is None:
        return 0
    offset = getattr(cache, "offset", 0)
    if isinstance(offset, mx.array):
        return int(mx.max(offset).item())
    return int(offset)


def _cache_state(cache):
    if cache is None or getattr(cache, "keys", None) is None:
        return None
    if hasattr(cache, "decoder_state"):
        return cache.decoder_state
    if hasattr(cache, "_temporal_order"):
        return cache._temporal_order(cache.keys), cache._temporal_order(cache.values)
    return cache.state


class Attention(nn.Module):
    """Gemma4 attention + diffusion deltas: full (global) layers use wider heads
    (`global_head_dim`) and SHARE keys as values (no v_proj, values = keys); sliding
    layers have a v_proj + a no-scale `v_norm`. Per-layer-type RoPE. In `decoder`
    mode the canvas concatenates the encoder cache K/V so it attends to the prompt;
    sliding layers slice the encoder cache to the window. (Mirror of mlx-vlm.)"""

    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.is_sliding = self.layer_type == "sliding_attention"

        self.head_dim = (
            config.global_head_dim
            if not self.is_sliding and config.global_head_dim
            else config.head_dim
        )
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = (
            config.num_global_key_value_heads
            if not self.is_sliding and config.num_global_key_value_heads is not None
            else config.num_key_value_heads
        )
        self.scale = 1.0  # mlx-vlm passes 1.0 (no SDPA rescale); mirror for parity.

        self.q_proj = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        # Full (global) layers reuse keys as values; only sliding layers carry v_proj.
        self.v_proj = (
            nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
            if self.is_sliding
            else None
        )
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = RMSNormNoScale(self.head_dim, eps=config.rms_norm_eps)

        rope_params = config.rope_parameters.get(self.layer_type, {})
        self.rope = initialize_rope(
            dims=self.head_dim,
            traditional=False,
            base=rope_params.get("rope_theta", 10000.0),
            scaling_config=rope_params,
            max_position_embeddings=config.max_position_embeddings,
        )

    def __call__(self, x, mask=None, cache=None, *, decoder=False, offset=None):
        B, L, _ = x.shape
        if offset is None:
            offset = _cache_offset(cache)

        queries = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        queries = self.rope(queries, offset=offset)

        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
        values = (
            self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
            if self.v_proj is not None
            else keys
        )
        keys = self.k_norm(keys).transpose(0, 2, 1, 3)
        keys = self.rope(keys, offset=offset)
        values = self.v_norm(values).transpose(0, 2, 1, 3)

        if decoder:
            state = _cache_state(cache)
            if state is not None:
                encoder_keys, encoder_values = state
                if self.is_sliding:
                    window = max(self.config.sliding_window - 1, 0)
                    encoder_len = encoder_keys.shape[2]
                    if window and encoder_len > window and offset >= encoder_len:
                        encoder_keys = encoder_keys[:, :, -window:, :]
                        encoder_values = encoder_values[:, :, -window:, :]
                        if mask is not None and not isinstance(mask, str):
                            mask = mask[..., -(window + L):]
                keys = mx.concatenate([encoder_keys, keys], axis=2)
                values = mx.concatenate([encoder_values, values], axis=2)
            attn_cache = None
        else:
            if cache is not None:
                keys, values = cache.update_and_fetch(keys, values)
            attn_cache = cache

        output = scaled_dot_product_attention(
            queries, keys, values, cache=attn_cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class SelfConditioning(nn.Module):
    """Folds the previous denoise step's signal back into the canvas embeddings:
    pre_norm → GeGLU MLP → add to embeds → no-scale post_norm."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.pre_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_norm = RMSNormNoScale(config.hidden_size, eps=config.rms_norm_eps)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, inputs_embeds, self_conditioning_signal):
        normed = self.pre_norm(self_conditioning_signal)
        signal = self.down_proj(geglu(self.gate_proj(normed), self.up_proj(normed)))
        return self.post_norm(inputs_embeds + signal)


class DecoderLayer(nn.Module):
    """A Gemma4 sandwich-norm block whose feed-forward SUMS a dense MLP path and a
    routed MoE path, each through its own pre/post feed-forward norms; the block
    output is scaled by a per-layer `layer_scalar`."""

    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config)
        self.router = Router(config)
        self.experts = Experts(config)
        eps = config.rms_norm_eps
        h = config.hidden_size
        self.input_layernorm = nn.RMSNorm(h, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(h, eps=eps)
        self.pre_feedforward_layernorm = nn.RMSNorm(h, eps=eps)
        self.post_feedforward_layernorm = nn.RMSNorm(h, eps=eps)
        self.post_feedforward_layernorm_1 = nn.RMSNorm(h, eps=eps)
        self.pre_feedforward_layernorm_2 = nn.RMSNorm(h, eps=eps)
        self.post_feedforward_layernorm_2 = nn.RMSNorm(h, eps=eps)
        self.layer_scalar = mx.ones((1,))

    def __call__(self, x, mask=None, cache=None, *, decoder=False, offset=None, layer_scalar=None):
        residual = x
        h = self.input_layernorm(x)
        h = self.self_attn(h, mask, cache, decoder=decoder, offset=offset)
        h = self.post_attention_layernorm(h)
        h = residual + h

        residual = h
        h1 = self.pre_feedforward_layernorm(h)
        h1 = self.mlp(h1)
        h1 = self.post_feedforward_layernorm_1(h1)

        flat = residual.reshape(-1, residual.shape[-1])
        top_k_indices, top_k_weights = self.router(flat)
        h2 = self.pre_feedforward_layernorm_2(flat)
        h2 = self.experts(h2, top_k_indices, top_k_weights)
        h2 = h2.reshape(residual.shape)
        h2 = self.post_feedforward_layernorm_2(h2)

        h = self.post_feedforward_layernorm(h1 + h2)
        h = residual + h
        return h * (self.layer_scalar if layer_scalar is None else layer_scalar)


# ── Slice 2: the decoder model (canvas denoiser) ──────────────────────────────
# The decoder embeds a token canvas (× sqrt(hidden)), folds in the previous step's
# self-conditioning signal, builds per-layer-type masks over [encoder-cache | canvas]
# (full layers see the whole valid prefix; sliding layers see a window), and runs
# the DecoderLayers in `decoder=True` mode so each canvas attends to the prompt's
# encoder cache. With no cache (cache=None) it degrades to pure canvas self-attention.


class DecoderModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_scale = config.hidden_size**0.5
        self.layers = [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_conditioning = SelfConditioning(config)

    def _embed_canvas(self, canvas_ids, self_conditioning_logits=None,
                      self_conditioning_embeddings=None):
        inputs_embeds = self.embed_tokens(canvas_ids) * self.embed_scale
        if self_conditioning_logits is not None and self_conditioning_embeddings is not None:
            raise ValueError(
                "Only one of self_conditioning_logits or self_conditioning_embeddings can be set."
            )
        if self_conditioning_embeddings is not None:
            soft_embeddings = self_conditioning_embeddings.astype(inputs_embeds.dtype)
        elif self_conditioning_logits is None:
            soft_embeddings = mx.zeros_like(inputs_embeds)
        else:
            probs = mx.softmax(self_conditioning_logits, axis=-1, precise=True)
            if isinstance(self.embed_tokens, nn.QuantizedEmbedding):
                soft_embeddings = mx.quantized_matmul(
                    probs.astype(inputs_embeds.dtype),
                    self.embed_tokens.weight, self.embed_tokens.scales, self.embed_tokens.biases,
                    transpose=False, group_size=self.embed_tokens.group_size,
                    bits=self.embed_tokens.bits, mode=getattr(self.embed_tokens, "mode", "affine"),
                )
            else:
                soft_embeddings = probs @ self.embed_tokens.weight
            soft_embeddings = soft_embeddings.astype(inputs_embeds.dtype) * self.embed_scale
        return self.self_conditioning(inputs_embeds, soft_embeddings)

    def _make_decoder_masks(self, h, caches, decoder_attention_mask=None):
        if isinstance(decoder_attention_mask, dict):
            return decoder_attention_mask
        B, canvas_length, _ = h.shape
        masks = {}
        for layer_type in set(self.config.layer_types):
            cache = next(
                (c for c, layer in zip(caches or [], self.layers)
                 if layer.layer_type == layer_type),
                None,
            )
            state = _cache_state(cache)
            encoder_len = state[0].shape[2] if state is not None else 0
            valid_encoder_len = min(_cache_offset(cache), encoder_len)
            key_len = encoder_len + canvas_length

            if layer_type == "full_attention":
                if decoder_attention_mask is None:
                    if encoder_len == valid_encoder_len:
                        masks[layer_type] = None
                    else:
                        row = mx.concatenate(
                            [mx.arange(encoder_len) < valid_encoder_len,
                             mx.ones((canvas_length,), dtype=mx.bool_)], axis=0,
                        )
                        masks[layer_type] = mx.broadcast_to(
                            row[None, None, None, :], (B, 1, canvas_length, key_len))
                else:
                    full = decoder_attention_mask.astype(mx.bool_)
                    if full.shape[-1] != key_len:
                        full = full[..., -key_len:]
                    masks[layer_type] = mx.broadcast_to(
                        full[:, None, None, :], (B, 1, canvas_length, key_len))
                continue

            # sliding_attention
            window_prefix = max(self.config.sliding_window - 1, 0)
            if decoder_attention_mask is None:
                if encoder_len == valid_encoder_len and encoder_len <= window_prefix:
                    masks[layer_type] = None
                    continue
                start = max(0, valid_encoder_len - window_prefix)
                positions = mx.arange(encoder_len)
                encoder_mask = (positions >= start) & (positions < valid_encoder_len)
                row = mx.concatenate(
                    [encoder_mask, mx.ones((canvas_length,), dtype=mx.bool_)], axis=0)
                masks[layer_type] = mx.broadcast_to(
                    row[None, None, None, :], (B, 1, canvas_length, key_len))
            else:
                full = decoder_attention_mask.astype(mx.bool_)
                if full.shape[-1] != key_len:
                    full = full[..., -key_len:]
                start = max(0, valid_encoder_len - window_prefix)
                positions = mx.arange(encoder_len)
                keep = mx.concatenate(
                    [(positions >= start) & (positions < valid_encoder_len),
                     mx.ones((canvas_length,), dtype=mx.bool_)], axis=0)
                row = full[:, None, None, :] & keep[None, None, None, :]
                masks[layer_type] = mx.broadcast_to(row, (B, 1, canvas_length, key_len))
        return masks

    def __call__(self, canvas_ids, cache=None, self_conditioning_logits=None,
                 self_conditioning_embeddings=None, decoder_attention_mask=None):
        h = self._embed_canvas(canvas_ids, self_conditioning_logits, self_conditioning_embeddings)
        cache = cache or [None] * len(self.layers)
        masks = self._make_decoder_masks(h, cache, decoder_attention_mask)
        offset = _cache_offset(cache[0]) if cache else 0
        for layer, c in zip(self.layers, cache):
            h = layer(h, masks.get(layer.layer_type), c, decoder=True, offset=offset)
        return self.norm(h)


# ── Slice 3: encoder (prompt prefill), backbone, top-level Model ───────────────


class _EncoderLayerScalar(nn.Module):
    """The only encoder weight NOT tied to the decoder — its per-layer scalar."""

    def __init__(self):
        super().__init__()
        self.layer_scalar = mx.ones((1,))


class EncoderLanguageModel(nn.Module):
    """The encoder's view of the decoder: the SAME layers (tied weights) but its own
    per-layer scalars. A weakref keeps the decoder out of this module's tree so its
    weights aren't double-counted."""

    def __init__(self, decoder: "DecoderModel"):
        super().__init__()
        self._decoder_ref = weakref.ref(decoder)
        self.layers = [_EncoderLayerScalar() for _ in decoder.layers]

    @property
    def decoder(self):
        return self._decoder_ref()


class EncoderModel(nn.Module):
    """Prefills the prompt into a KV cache by running the DECODER's layers in encoder
    mode (decoder=False) with the encoder's own per-layer scalars. Text-only (no
    vision tower). Cache = standard mlx-lm KVCache (full) / RotatingKVCache (sliding),
    so the decoder's cache-concat reads their `.state` via `_cache_state`."""

    def __init__(self, config: ModelArgs, decoder: "DecoderModel"):
        super().__init__()
        self.config = config
        self.language_model = EncoderLanguageModel(decoder)
        self._decoder_ref = weakref.ref(decoder)

    @property
    def decoder(self):
        return self._decoder_ref()

    def make_cache(self):
        caches = []
        for layer_type in self.config.layer_types:
            if layer_type == "full_attention":
                caches.append(KVCache())
            else:
                caches.append(RotatingKVCache(max_size=self.config.sliding_window))
        return caches

    def _make_encoder_masks(self, h, cache):
        # PARITY ITEM: use_bidirectional_attention="all" makes the encoder fully
        # bidirectional. We mirror mlx-vlm's create_attention_mask path here; verify
        # the "all" bidirectional case against transformers in the parity gate.
        return [
            create_attention_mask(h, c) for c in cache
        ]

    def __call__(self, input_ids, attention_mask=None, cache=None):
        h = self.decoder.embed_tokens(input_ids) * self.decoder.embed_scale
        if cache is None:
            cache = self.make_cache()
        masks = self._make_encoder_masks(h, cache)
        for i, (layer, c, mask) in enumerate(zip(self.decoder.layers, cache, masks)):
            h = layer(h, mask, c, decoder=False,
                      layer_scalar=self.language_model.layers[i].layer_scalar)
        return self.decoder.norm(h), cache


class DiffusionGemma4Backbone(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.decoder = DecoderModel(config)
        self.encoder = EncoderModel(config, self.decoder)

    def __call__(self, input_ids=None, attention_mask=None, cache=None, canvas_ids=None,
                 self_conditioning_logits=None, self_conditioning_embeddings=None,
                 decoder_attention_mask=None):
        if input_ids is not None:
            _, cache = self.encoder(input_ids, attention_mask=attention_mask, cache=cache)
        elif cache is None:
            raise ValueError("Either input_ids or cache must be provided.")
        if canvas_ids is None:
            batch_size = input_ids.shape[0]
            canvas_ids = mx.random.randint(
                0, self.config.vocab_size, (batch_size, self.config.canvas_length))
        hidden_states = self.decoder(
            canvas_ids, cache=cache,
            self_conditioning_logits=self_conditioning_logits,
            self_conditioning_embeddings=self_conditioning_embeddings,
            decoder_attention_mask=decoder_attention_mask)
        return hidden_states, cache


@partial(mx.compile, shapeless=True)
def _softcap(softcap, x):
    return mx.tanh(x.astype(mx.float32) / softcap) * softcap


class Model(nn.Module):
    """Top-level mlx-lm model. NOTE: this is an encoder-decoder DIFFUSION model —
    `__call__(input_ids=...)` prefills the prompt, denoises ONE random canvas, and
    returns its softcapped logits (B, canvas_length, vocab). It is NOT an
    autoregressive next-token step; real generation goes through `diffusion_generate`
    (slice 5), which the standard mlx-lm generate loop will dispatch to."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = DiffusionGemma4Backbone(config)
        self.final_logit_softcapping = config.final_logit_softcapping

    def __call__(self, input_ids=None, cache=None, canvas_ids=None, **kwargs):
        hidden_states, _ = self.model(
            input_ids=input_ids, cache=cache, canvas_ids=canvas_ids,
            self_conditioning_logits=kwargs.get("self_conditioning_logits"),
            self_conditioning_embeddings=kwargs.get("self_conditioning_embeddings"),
            decoder_attention_mask=kwargs.get("decoder_attention_mask"))
        logits = self.model.decoder.embed_tokens.as_linear(hidden_states)
        return _softcap(float(self.final_logit_softcapping), logits)

    @property
    def layers(self):
        return self.model.decoder.layers

    def make_cache(self):
        return self.model.encoder.make_cache()

    def sanitize(self, weights):
        sanitized = {}
        for key, value in weights.items():
            if "rotary_emb" in key or key == "lm_head.weight":
                continue
            # Encoder text weights are tied to the decoder; the checkpoint only
            # carries the encoder's separate layer scalars.
            if key.startswith("model.encoder.language_model."):
                if key.endswith(".layer_scalar"):
                    sanitized[key] = value
                continue
            # Drop the vision tower (this is the text-only port).
            if key.startswith("model.encoder.embed_vision.") or key.startswith(
                "model.encoder.vision_tower."
            ):
                continue
            # MoE expert tensors → SwitchLinear's `.weight` name.
            if key.endswith(".experts.down_proj"):
                sanitized[key.replace(".experts.down_proj", ".experts.down_proj.weight")] = value
                continue
            if key.endswith(".experts.gate_up_proj"):
                sanitized[key.replace(".experts.gate_up_proj", ".experts.gate_up_proj.weight")] = value
                continue
            sanitized[key] = value
        return sanitized

    @property
    def quant_predicate(self):
        def predicate(path, m):
            if not hasattr(m, "to_quantized"):
                return False
            if "router" in path or path.endswith(
                ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
            ):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate
