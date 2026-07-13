# Copyright © 2026 Apple Inc.

# Nemotron-Labs-Diffusion-3B: a TRI-MODE Ministral/Llama-style transformer that
# supports both standard autoregressive (AR) decoding AND masked block
# diffusion (dLM). The backbone (``NemotronLabsDiffusionModel.encoder`` in the
# HF reference) is a Ministral-3 decoder — GQA attention, SwiGLU MLP, RMSNorm,
# YaRN RoPE — with two non-standard wrinkles carried over from the reference:
#
#   1. A Llama-4-style per-position query scale
#      ``1 + beta * log1p(floor(pos / orig_max_pos))`` applied to the queries
#      after RoPE (``beta = rope_parameters.llama_4_scaling_beta``). For every
#      position below ``original_max_position_embeddings`` this is exactly 1.0,
#      so it only bends the very-long-context tail.
#   2. A separate ``diffusion_head`` (untied Linear) is the LM head — the HF
#      module is named ``diffusion_head`` rather than ``lm_head``.
#
# The HF weight tree prefixes the transformer with ``encoder.`` and names the
# head ``diffusion_head.``; ``sanitize`` remaps those to the standard mlx-lm
# ``model.`` / ``lm_head.`` layout so the 4bit affine quant loads and quantizes
# cleanly. There is also a ``linear_spec_lora`` self-speculation adapter shipped
# alongside the base weights — it lives in a separate ``adapter_model.safetensors``
# and is loaded on demand by ``load_linear_spec_lora`` (below); ``sanitize`` still
# drops any adapter tensors that leak into the base weight tree.
#
# Three generation paths are exposed:
#   * AR:        the standard causal ``Model.__call__(inputs, cache)`` so
#                ``mlx_lm.generate`` works out of the box.
#   * Diffusion: the opt-in ``diffusion_generate`` module-level function (and the
#                ``Model.diffusion_generate`` convenience method) — LLaDA-style
#                semi-autoregressive block denoising with bidirectional attention
#                within each block and confidence-based unmasking.
#   * Linear self-spec: ``linear_spec_generate`` — the SAME model drafts the next
#                block bidirectionally (with the ``linear_spec_lora`` adapter
#                toggled ON) and then AR-verifies it causally (adapter OFF),
#                accepting the longest draft prefix that matches the AR argmax plus
#                one bonus token. Lossless: every emitted token comes from the
#                causal (adapter-OFF) verify, so the stream is bit-identical to
#                plain AR greedy; the LoRA only speeds the draft.

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, trim_prompt_cache
from .rope_utils import initialize_rope

# Defaults from the published config.json.
DEFAULT_MASK_TOKEN_ID = 100
DEFAULT_EOS_TOKEN_ID = 11


class SpecLoRALinear(nn.Module):
    """A base (possibly quantized) ``nn.Linear`` carrying the ``linear_spec_lora``
    PEFT delta, gated by a runtime ``active`` flag.

    Mirrors ``mlx_lm.tuner.lora.LoRALinear`` (a wrapped ``self.linear`` + a
    low-rank ``scale * B @ A`` correction) but with two differences dictated by
    the shipped adapter:

    * **Runtime toggle.** Linear self-spec needs the SAME weights to run with the
      adapter ON (bidirectional draft) and OFF (causal verify). When
      ``active`` is ``False`` the call is byte-identical to the wrapped base
      linear, so the AR / verify path is unchanged.
    * **PEFT layout.** The adapter stores ``lora_A`` as ``(r, in)`` and ``lora_B``
      as ``(out, r)`` (HF/PEFT convention), so the delta is
      ``scale * (x @ A.T) @ B.T`` with ``scale = lora_alpha / r``. (mlx-lm's own
      trainer uses the transposed ``(in, r)`` / ``(r, out)`` layout.) Keeping the
      PEFT layout lets the ``adapter_model.safetensors`` load with no transpose.
    """

    @staticmethod
    def from_base(linear: nn.Module, r: int, scale: float) -> "SpecLoRALinear":
        # Recover the *unpacked* input width (quantized weights are bit-packed).
        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            input_dims = input_dims * 32 // linear.bits
        wrap = SpecLoRALinear(input_dims, output_dims, r, scale)
        wrap.linear = linear
        return wrap

    def __init__(self, input_dims: int, output_dims: int, r: int, scale: float):
        super().__init__()
        self.linear = nn.Linear(input_dims, output_dims, bias=False)
        self.r = r
        self.scale = scale
        # Toggled by ``Model.set_linear_spec_lora`` — draft ON, verify OFF.
        self.active = False
        # PEFT layout: A is (r, in), B is (out, r). Zero-init == identity delta.
        self.lora_a = mx.zeros((r, input_dims))
        self.lora_b = mx.zeros((output_dims, r))

    def __call__(self, x: mx.array) -> mx.array:
        y = self.linear(x)
        if not self.active:
            return y
        z = (x @ self.lora_a.T) @ self.lora_b.T
        return y + (self.scale * z).astype(y.dtype)


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "nemotron_labs_diffusion"
    hidden_size: int = 3072
    num_hidden_layers: int = 26
    intermediate_size: int = 9216
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-5
    vocab_size: int = 131072
    max_position_embeddings: int = 262144
    rope_theta: float = 1000000.0
    rope_traditional: bool = False
    rope_parameters: Optional[Dict[str, Union[float, str]]] = None
    attention_bias: bool = False
    mlp_bias: bool = False
    tie_word_embeddings: bool = False
    sliding_window: Optional[int] = None
    # Diffusion-specific.
    mask_token_id: int = DEFAULT_MASK_TOKEN_ID
    block_size: int = 32
    dlm_paradigm: str = "bidirectional"
    ar_loss_weight: float = 1.0
    eos_token_id: int = DEFAULT_EOS_TOKEN_ID
    bos_token_id: int = 1

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        # ``rope_parameters`` (transformers v5 name) carries both the scaling
        # config and the base theta; mirror theta up so ``initialize_rope`` gets
        # a consistent (base, scaling_config) pair.
        if self.rope_parameters is not None:
            self.rope_theta = float(
                self.rope_parameters.get("rope_theta", self.rope_theta)
            )

    @property
    def rope_scaling(self) -> Optional[dict]:
        """``initialize_rope`` reads the scaling config from here; the reference
        stores it under ``rope_parameters`` (aliased ``rope_scaling`` in HF)."""
        return self.rope_parameters


def _llama4_query_scale(
    offset: int, length: int, beta: Optional[float], orig_max_pos: Optional[int]
) -> Optional[mx.array]:
    """Llama-4-style per-position query scale, matching the reference's
    ``_get_llama_4_attn_scale``: ``1 + beta * log(1 + floor(pos / orig_max))``.

    Returns ``None`` when disabled (no ``beta``) or when the whole window is
    below ``orig_max`` (scale == 1 everywhere), so the caller can skip the
    multiply on the common short-context path.
    """
    if not beta or not orig_max_pos:
        return None
    # Only positions >= orig_max_pos change the scale; skip otherwise.
    if offset + length <= orig_max_pos:
        return None
    pos = mx.arange(offset, offset + length, dtype=mx.float32)
    scale = 1.0 + beta * mx.log1p(mx.floor(pos / orig_max_pos))
    return scale.reshape(1, 1, length, 1)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim or dim // self.n_heads
        self.scale = head_dim**-0.5

        bias = args.attention_bias
        self.q_proj = nn.Linear(dim, self.n_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(self.n_heads * head_dim, dim, bias=bias)

        self.rope = initialize_rope(
            head_dim,
            args.rope_theta,
            args.rope_traditional,
            args.rope_scaling,
            args.max_position_embeddings,
        )

        rp = args.rope_parameters or {}
        self._llama4_beta = rp.get("llama_4_scaling_beta")
        self._llama4_orig_max = rp.get("original_max_position_embeddings")

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        prefix_cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        # ``prefix_cache`` selects the read-only bidirectional DRAFT path used by
        # linear self-spec: the block queries attend to the already-committed
        # causal prefix KV (read from ``prefix_cache``, never mutated) plus the
        # block itself with FULL (non-causal) attention. The standard AR / verify
        # path (``prefix_cache is None``) is untouched.
        if prefix_cache is not None:
            offset = prefix_cache.offset
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)
            qscale = _llama4_query_scale(
                offset, L, self._llama4_beta, self._llama4_orig_max
            )
            if qscale is not None:
                queries = queries * qscale.astype(queries.dtype)
            if offset > 0:
                pk = prefix_cache.keys[..., :offset, :]
                pv = prefix_cache.values[..., :offset, :]
                keys = mx.concatenate([pk, keys], axis=2)
                values = mx.concatenate([pv, values], axis=2)
            output = scaled_dot_product_attention(
                queries, keys, values, cache=None, scale=self.scale, mask=None
            )
            output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
            return self.o_proj(output)

        offset = cache.offset if cache is not None else 0
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)

        # Llama-4-style per-position query scaling (no-op below orig_max_pos).
        qscale = _llama4_query_scale(
            offset, L, self._llama4_beta, self._llama4_orig_max
        )
        if qscale is not None:
            queries = queries * qscale.astype(queries.dtype)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        hidden_dim = args.intermediate_size
        bias = args.mlp_bias
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

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


class NemotronLabsDiffusionTransformer(nn.Module):
    """The Ministral-3 backbone (``encoder`` in the HF reference)."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        mask: Optional[mx.array] = None,
        bidirectional: bool = False,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        # AR path builds a causal mask; the diffusion path is bidirectional
        # (mask stays None so every position attends to every other).
        if mask is None and not bidirectional:
            mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = NemotronLabsDiffusionTransformer(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def _head(self, out: mx.array) -> mx.array:
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        mask: Optional[mx.array] = None,
        bidirectional: bool = False,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        """Autoregressive (causal) forward by default; pass ``bidirectional=True``
        for the diffusion denoiser (full non-causal attention, no cache)."""
        out = self.model(
            inputs,
            cache=cache,
            mask=mask,
            bidirectional=bidirectional,
            input_embeddings=input_embeddings,
        )
        return self._head(out)

    def diffusion_generate(self, prompt: mx.array, **kwargs):
        """Opt-in masked block-diffusion generation (see ``diffusion_generate``)."""
        return diffusion_generate(self, prompt, **kwargs)

    def linear_spec_generate(self, prompt: mx.array, **kwargs):
        """Opt-in linear self-speculative decoding (see ``linear_spec_generate``)."""
        return linear_spec_generate(self, prompt, **kwargs)

    def load_linear_spec_lora(self, adapter_path: str) -> "Model":
        """Attach the ``linear_spec_lora`` self-speculation adapter (see the
        module-level ``load_linear_spec_lora``)."""
        return load_linear_spec_lora(self, adapter_path)

    def set_linear_spec_lora(self, active: bool):
        """Toggle every attached ``SpecLoRALinear`` on (draft) or off (verify).

        A no-op when no adapter has been loaded — the draft then runs on the base
        weights, exactly like the reference's ``_toggle_adapters`` with no PEFT
        modules attached."""
        for layer in self.model.layers:
            attn = layer.self_attn
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                module = getattr(attn, name, None)
                if isinstance(module, SpecLoRALinear):
                    module.active = active

    def sanitize(self, weights):
        """Remap the HF weight tree to the mlx-lm layout.

        The reference names the backbone ``encoder.*`` and the LM head
        ``diffusion_head.*``; we host them as ``model.*`` and ``lm_head.*``.
        Precomputed rotary buffers and any bundled self-speculation LoRA
        (``linear_spec_lora`` / PEFT ``lora_*``) tensors are dropped — the LoRA
        adapter is not part of this port.
        """
        sanitized = {}
        for key, value in weights.items():
            if "rotary_emb.inv_freq" in key:
                continue
            # Drop bundled self-speculation LoRA / PEFT adapter tensors.
            if "lora_" in key or ".base_layer." in key or "linear_spec" in key:
                continue

            new_key = key
            if key.startswith("encoder."):
                new_key = "model." + key[len("encoder.") :]
            elif key.startswith("model.encoder."):
                new_key = "model." + key[len("model.encoder.") :]
            elif key.startswith("diffusion_head."):
                new_key = "lm_head." + key[len("diffusion_head.") :]

            sanitized[new_key] = value

        if self.args.tie_word_embeddings:
            sanitized.pop("lm_head.weight", None)
        return sanitized

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [KVCache() for _ in self.layers]


# ---------------------------------------------------------------------------
# Masked block-diffusion sampler (LLaDA-style, pure MLX).
#
# Semi-autoregressive: the generation window is split into blocks of
# ``block_length``. Within a block, attention is bidirectional and masked
# positions are revealed a few at a time by confidence, re-running a full
# forward each denoising step. Blocks are decoded left-to-right so later blocks
# condition on the finalized earlier ones. No KV cache (bidirectional attention
# depends on the still-masked tail); the orchestrator can layer a Fast-dLLM-style
# block cache later if throughput demands it.
# ---------------------------------------------------------------------------


def add_gumbel_noise(logits: mx.array, temperature: float) -> mx.array:
    """Log-space Gumbel-max perturbation; a no-op at ``temperature == 0``."""
    if temperature == 0.0:
        return logits
    logits = logits.astype(mx.float32)
    eps = 1e-7
    noise = mx.clip(mx.random.uniform(shape=logits.shape), eps, 1.0 - eps)
    return logits - temperature * mx.log(-mx.log(noise))


def get_num_transfer_tokens(mask_index: mx.array, steps: int) -> mx.array:
    """Per-row reveal schedule: ``mask_count`` split evenly over ``steps`` with
    the remainder front-loaded. Returns an int array ``[B, steps]``."""
    mask_count = mask_index.sum(axis=1, keepdims=True)
    base = mask_count // steps
    remainder = mask_count % steps
    num_transfer = mx.repeat(base, steps, axis=1)
    step_idx = mx.arange(steps).reshape(1, steps)
    num_transfer = num_transfer + (step_idx < remainder).astype(num_transfer.dtype)
    return num_transfer


def _select_topk(confidence: mx.array, k_per_row: mx.array) -> mx.array:
    """Boolean mask selecting the top-``k`` positions per row (exact rank via
    argsort so ties never over-reveal)."""
    B, L = confidence.shape
    order = mx.argsort(-confidence, axis=1)
    ranks = mx.zeros((B, L), dtype=mx.int32)
    col_positions = mx.arange(L).reshape(1, L)
    ranks = mx.put_along_axis(
        ranks,
        order,
        mx.broadcast_to(col_positions.astype(mx.int32), (B, L)),
        axis=1,
    )
    return ranks < k_per_row.astype(mx.int32)


def diffusion_generate(
    model: Model,
    prompt: mx.array,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 32,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: Optional[int] = None,
    tokenizer=None,
):
    """Masked block-diffusion decoding for Nemotron-Labs-Diffusion.

    Args:
        model: a built ``Model``.
        prompt: token ids, shape ``[1, prompt_len]`` (or ``[prompt_len]``).
        steps: total denoising steps, split evenly across the blocks.
        gen_length: number of tokens to generate (must divide by
            ``block_length``, and ``steps`` must divide by the block count).
        block_length: semi-autoregressive block size (defaults to the config's
            ``block_size``).
        temperature: Gumbel sampling temperature (0 == greedy/argmax).
        remasking: ``"low_confidence"`` (default) or ``"random"``.
        mask_id: mask token id (defaults to the config's ``mask_token_id``).
        tokenizer: optional; if given the decoded text is returned too.

    Returns:
        Generated ids ``[1, gen_length]`` (plus decoded text if a tokenizer was
        supplied).
    """
    if mask_id is None:
        mask_id = getattr(model.args, "mask_token_id", DEFAULT_MASK_TOKEN_ID)
    if block_length is None:
        block_length = getattr(model.args, "block_size", 32)

    if prompt.ndim == 1:
        prompt = prompt[None, :]
    prompt_len = prompt.shape[1]

    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if gen_length <= 0:
        raise ValueError(f"gen_length must be positive, got {gen_length}")
    if block_length <= 0:
        raise ValueError(f"block_length must be positive, got {block_length}")
    if gen_length % block_length != 0:
        raise ValueError(
            f"gen_length ({gen_length}) must be divisible by "
            f"block_length ({block_length})"
        )
    num_blocks = gen_length // block_length
    if steps % num_blocks != 0:
        raise ValueError(
            f"steps ({steps}) must be divisible by num_blocks ({num_blocks})"
        )
    steps_per_block = steps // num_blocks

    total_len = prompt_len + gen_length
    x = mx.full((1, total_len), mask_id, dtype=prompt.dtype)
    x[:, :prompt_len] = prompt

    neg_inf = mx.array(-float("inf"), dtype=mx.float32)
    col_index = mx.arange(total_len).reshape(1, total_len)

    for b in range(num_blocks):
        block_start = prompt_len + b * block_length
        block_end = prompt_len + (b + 1) * block_length

        block_mask_index = x[:, block_start:block_end] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(
            block_mask_index, steps_per_block
        )

        for i in range(steps_per_block):
            mask_index = x == mask_id
            # Bidirectional full forward over the whole sequence (no KV cache).
            logits = model(x, bidirectional=True)

            noised = add_gumbel_noise(logits, temperature)
            x0 = mx.argmax(noised, axis=-1)

            if remasking == "low_confidence":
                p = mx.softmax(logits.astype(mx.float32), axis=-1)
                x0_p = mx.take_along_axis(p, x0[..., None], axis=-1).squeeze(-1)
            elif remasking == "random":
                x0_p = mx.random.uniform(shape=x0.shape)
            else:
                raise ValueError(f"Unknown remasking strategy: {remasking}")

            # Never reveal past the current block boundary.
            beyond_block = col_index >= block_end
            x0_p = mx.where(beyond_block, neg_inf, x0_p.astype(mx.float32))

            x0 = mx.where(mask_index, x0, x)
            confidence = mx.where(mask_index, x0_p, neg_inf)

            transfer_index = _select_topk(
                confidence, num_transfer_tokens[:, i : i + 1]
            )
            x = mx.where(transfer_index, x0, x)
            mx.eval(x)

    out = x[:, prompt_len:]
    mx.eval(out)
    if tokenizer is not None:
        return out, tokenizer.decode(out[0].tolist())
    return out


# ---------------------------------------------------------------------------
# Linear self-speculative decoding: diffusion draft + AR verify.
#
# Mirrors the reference ``NemotronLabsDiffusionModel.linear_spec_generate``. Each
# outer step:
#   1. Draft the next block bidirectionally with the ``linear_spec_lora`` adapter
#      ON (attending to the committed causal prefix KV + the block itself, no
#      cache mutation). At ``threshold == 0`` the whole block is filled in one
#      forward; at ``threshold > 0`` masked positions are revealed by confidence
#      over several forwards (always forcing >=1 reveal so it can't stall).
#   2. Verify the drafted block causally with the adapter OFF (KV-cached). The AR
#      argmax gives the ground-truth continuation.
#   3. Accept the longest prefix where draft[i+1] == AR_argmax[i], plus one bonus
#      token (the AR argmax always contributes its own next token). Crop the KV
#      cache back to the accepted length and re-seed with the last accepted AR
#      token.
#
# Every emitted token is an AR (adapter-OFF, causal) argmax, so the stream is
# bit-identical to plain greedy AR — this is a *lossless* self-spec; the LoRA
# only makes the draft accept more per verify.
# ---------------------------------------------------------------------------


def load_linear_spec_lora(model: "Model", adapter_path: str) -> "Model":
    """Load the ``linear_spec_lora`` PEFT adapter and wrap the target linears.

    Reads ``adapter_config.json`` (rank ``r``, ``lora_alpha``, ``target_modules``)
    and ``adapter_model.safetensors`` from ``adapter_path``, then replaces each
    targeted attention projection with a :class:`SpecLoRALinear` carrying the
    adapter's ``lora_A`` / ``lora_B`` weights. The wrappers start inactive; call
    ``model.set_linear_spec_lora(True/False)`` (done automatically inside
    ``linear_spec_generate``) to gate the delta.
    """
    with open(os.path.join(adapter_path, "adapter_config.json")) as fh:
        cfg = json.load(fh)
    r = int(cfg["r"])
    alpha = float(cfg["lora_alpha"])
    scale = alpha / r
    targets = cfg.get("target_modules", ["o_proj"])
    if isinstance(targets, str):
        targets = [targets]

    weights = mx.load(os.path.join(adapter_path, "adapter_model.safetensors"))

    def _find(layer_idx: int, proj: str, ab: str):
        # The adapter keys are ``base_model.model.encoder.layers.N.self_attn.
        # <proj>.lora_<A|B>.weight``; match on the stable suffix to be robust to
        # a differing base prefix.
        suffix = f"layers.{layer_idx}.self_attn.{proj}.lora_{ab}.weight"
        for key, value in weights.items():
            if key.endswith(suffix):
                return value
        return None

    n_wrapped = 0
    for li, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        for proj in targets:
            base = getattr(attn, proj, None)
            if base is None:
                continue
            wa = _find(li, proj, "A")
            wb = _find(li, proj, "B")
            if wa is None or wb is None:
                continue
            wrap = SpecLoRALinear.from_base(base, r, scale)
            wrap.lora_a = wa.astype(mx.float32)
            wrap.lora_b = wb.astype(mx.float32)
            setattr(attn, proj, wrap)
            n_wrapped += 1

    if n_wrapped == 0:
        raise ValueError(
            f"No linear_spec_lora tensors matched target_modules={targets} "
            f"under {adapter_path}"
        )
    model._spec_lora_layers = n_wrapped
    return model


def _spec_sample(logits: mx.array, temperature: float) -> mx.array:
    """Greedy (``temperature == 0``) or temperature-sampled token ids.

    ``logits`` is ``[B, L, V]`` → returns ``[B, L]`` int ids."""
    if temperature and temperature > 0.0:
        B, L, V = logits.shape
        flat = (logits.astype(mx.float32) / temperature).reshape(-1, V)
        return mx.random.categorical(flat).reshape(B, L)
    return mx.argmax(logits, axis=-1)


def _spec_draft_logits(model: "Model", block: mx.array, caches) -> mx.array:
    """One read-only bidirectional draft forward over ``block`` (the adapter, if
    loaded and toggled on, is active). Attends to the committed prefix KV in
    ``caches`` without mutating it. Returns logits ``[1, block_len, V]``."""
    h = model.model.embed_tokens(block)
    for layer, c in zip(model.model.layers, caches):
        r = layer.self_attn(layer.input_layernorm(h), mask=None, prefix_cache=c)
        h = h + r
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    h = model.model.norm(h)
    return model._head(h)


def linear_spec_generate(
    model: "Model",
    prompt: mx.array,
    max_new_tokens: int = 128,
    block_length: Optional[int] = None,
    temperature: float = 0.0,
    threshold: float = 0.0,
    mask_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    tokenizer=None,
):
    """Linear self-speculative decoding (diffusion draft + AR verify).

    Args:
        model: a built ``Model`` (optionally with ``linear_spec_lora`` loaded via
            ``load_linear_spec_lora``).
        prompt: token ids, shape ``[1, prompt_len]`` (or ``[prompt_len]``).
        max_new_tokens: number of tokens to generate.
        block_length: draft/verify block size (defaults to the config's
            ``block_size``).
        temperature: 0 == greedy (the lossless, verified path).
        threshold: draft unmask confidence gate; 0 fills the whole block in one
            draft forward, >0 reveals by confidence over several forwards.
        mask_id / eos_token_id: default to the config values.
        tokenizer: optional; if given the decoded text is returned too.

    Returns:
        Generated ids ``[1, total_gen]`` (plus decoded text if a tokenizer was
        supplied). Acceptance statistics are stashed on ``model.spec_stats``:
        ``{"nfe", "steps", "accepted_total", "generated", "avg_accepted",
        "accept_counts"}``.
    """
    if mask_id is None:
        mask_id = getattr(model.args, "mask_token_id", DEFAULT_MASK_TOKEN_ID)
    if block_length is None:
        block_length = getattr(model.args, "block_size", 32)
    if eos_token_id is None:
        eos_token_id = getattr(model.args, "eos_token_id", None)

    if prompt.ndim == 1:
        prompt = prompt[None, :]
    if prompt.shape[0] != 1:
        raise ValueError("linear_spec_generate requires batch_size == 1")
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")
    if block_length <= 0:
        raise ValueError(f"block_length must be positive, got {block_length}")

    caches = model.make_cache()

    # ---- Prefill (causal, adapter OFF). ----
    model.set_linear_spec_lora(False)
    prefill_logits = model(prompt, cache=caches)
    nfe = 1
    next_token = _spec_sample(prefill_logits[:, -1:, :], temperature)  # [1, 1]

    generated = [next_token]
    total_gen = 1
    accept_counts = []

    if eos_token_id is not None and next_token[0, 0].item() == eos_token_id:
        total_gen = 1
    else:
        while total_gen < max_new_tokens:
            cache_len = caches[0].offset

            block = mx.full((1, block_length), mask_id, dtype=prompt.dtype)
            block[:, 0:1] = next_token.astype(block.dtype)

            # ---- Draft (bidirectional, adapter ON). ----
            model.set_linear_spec_lora(True)
            while True:
                is_mask = block == mask_id
                if not is_mask.any().item():
                    break

                draft_logits = _spec_draft_logits(model, block, caches)
                nfe += 1
                # LLaDA layout: logit[i] predicts position i (no shift).
                draft_tokens = mx.argmax(draft_logits, axis=-1)  # [1, bl]

                if threshold > 0:
                    probs = mx.softmax(draft_logits.astype(mx.float32), axis=-1)
                    conf = mx.take_along_axis(
                        probs, draft_tokens[..., None], axis=-1
                    ).squeeze(-1)  # [1, bl]
                    neg = mx.array(-float("inf"), dtype=mx.float32)
                    conf = mx.where(is_mask, conf, neg)
                    unmask = conf >= threshold
                    if not unmask.any().item():
                        best = mx.argmax(conf.reshape(-1)).item()
                        cols = mx.arange(block_length).reshape(1, block_length)
                        unmask = cols == best
                    block = mx.where(unmask, draft_tokens, block)
                else:
                    block = mx.where(is_mask, draft_tokens, block)
                    break

            # ---- Verify (causal, adapter OFF, KV-cached). ----
            model.set_linear_spec_lora(False)
            verify_logits = model(block, cache=caches)
            nfe += 1
            ar_tokens = _spec_sample(verify_logits, temperature)  # [1, bl]

            # Accept the longest matching prefix + one bonus AR token.
            block_l = block[0].tolist()
            ar_l = ar_tokens[0].tolist()
            accepted = 0
            for i in range(block_length - 1):
                if ar_l[i] == block_l[i + 1]:
                    accepted += 1
                else:
                    break
            accepted += 1

            accepted_toks = ar_tokens[:, :accepted]
            generated.append(accepted_toks)
            total_gen += accepted
            accept_counts.append(accepted)

            # Verify appended the whole block; crop back to the accepted length.
            trim_prompt_cache(caches, block_length - accepted)
            next_token = ar_tokens[:, accepted - 1 : accepted]

            # EOS: truncate the just-appended chunk at the first EOS and stop.
            if eos_token_id is not None:
                acc_l = ar_l[:accepted]
                if eos_token_id in acc_l:
                    first_eos = acc_l.index(eos_token_id)
                    generated[-1] = accepted_toks[:, : first_eos + 1]
                    total_gen = total_gen - accepted + first_eos + 1
                    break

            if total_gen >= max_new_tokens:
                break

    ids = mx.concatenate(generated, axis=1)[:, :max_new_tokens]
    mx.eval(ids)
    total_gen = ids.shape[1]

    model.spec_stats = {
        "nfe": nfe,
        "steps": len(accept_counts),
        "accepted_total": total_gen,
        "generated": total_gen,
        "avg_accepted": (
            sum(accept_counts) / len(accept_counts) if accept_counts else float(total_gen)
        ),
        "accept_counts": accept_counts,
    }

    if tokenizer is not None:
        return ids, tokenizer.decode(ids[0].tolist())
    return ids
