# Copyright © 2025-2026 Apple Inc.

# LLaDA2-MoE: a Mixture-of-Experts masked-diffusion language model (the MoE
# version of LLaDA). Architecturally it fuses two well-trodden pieces of the
# mlx-lm zoo:
#
#   * The MoE backbone is DeepSeek-V3 / Hunyuan-shaped: sigmoid-scored router
#     with a per-expert bias term, grouped-topk (``n_group`` / ``topk_group``)
#     routing, ``norm_topk_prob`` renormalization + ``routed_scaling_factor``,
#     one always-on shared expert, and a ``first_k_dense_replace`` split where
#     the first layer(s) use a plain dense MLP. We reuse the exact
#     ``group_expert_select`` routing from ``deepseek_v3`` and the ``SwitchGLU``
#     stacked-expert kernel.
#
#   * Generation is diffusion (masked bidirectional denoising, NO KV cache),
#     block-wise semi-autoregressive with a block-causal attention mask, mirrored
#     from the reference ``modeling_llada2_moe.py`` and the ``llada`` port.
#
# Two attention details differ from a vanilla causal LM:
#   1. Attention is BIDIRECTIONAL (no causal mask in the plain forward). During
#      diffusion generation a block-causal mask is supplied so a block can attend
#      to itself and all previous blocks but not future ones.
#   2. RoPE is PARTIAL (``partial_rotary_factor 0.5``): only the first
#      ``head_dim * 0.5`` dims of each head are rotated; the rest pass through.
#      Per-head query/key RMSNorm (over the full head_dim) is applied before RoPE.
#
# HF tensor names are matched directly (``model.word_embeddings``,
# ``...attention.query_key_value``, ``...attention.dense``,
# ``...mlp.gate.expert_bias``, ...) so the 4-bit affine quant loads with only the
# per-expert weight-stacking remap done in ``sanitize``.

from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import BaseModelArgs
from .switch_layers import SwitchGLU

# LLaDA2-mini special token ids (from the reference generate defaults).
DEFAULT_MASK_TOKEN_ID = 156895
DEFAULT_EOS_TOKEN_ID = 156892


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "llada2_moe"
    vocab_size: int = 157184
    hidden_size: int = 2048
    intermediate_size: int = 5120
    moe_intermediate_size: int = 512
    num_hidden_layers: int = 20
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 128
    # MoE
    num_experts: int = 256
    num_shared_experts: int = 1
    num_experts_per_tok: int = 8
    n_group: int = 8
    topk_group: int = 4
    routed_scaling_factor: float = 2.5
    norm_topk_prob: bool = True
    moe_router_enable_expert_bias: bool = True
    first_k_dense_replace: int = 1
    # Norm / rope / diffusion
    rms_norm_eps: float = 1e-6
    rope_theta: float = 600000.0
    partial_rotary_factor: float = 0.5
    max_position_embeddings: int = 8192
    use_qkv_bias: bool = False
    use_bias: bool = False
    tie_word_embeddings: bool = False
    # Special tokens (diffusion sampler defaults; not always in config.json)
    mask_token_id: int = DEFAULT_MASK_TOKEN_ID
    eos_token_id: int = DEFAULT_EOS_TOKEN_ID
    pad_token_id: int = 156892


# ---------------------------------------------------------------------------
# Attention: fused QKV projection, per-head QK RMSNorm, partial RoPE, GQA,
# bidirectional (mask supplied by the diffusion sampler).
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.num_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        # Single fused projection: [q (nh) | k (nkv) | v (nkv)] * head_dim.
        op_size = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
        self.query_key_value = nn.Linear(dim, op_size, bias=args.use_qkv_bias)
        # Per-head RMSNorm over the full head_dim, applied before RoPE.
        self.query_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.key_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.dense = nn.Linear(self.num_heads * self.head_dim, dim, bias=args.use_bias)

        # Partial RoPE: rotate only the first ``rotary_dim`` head dims; the rest
        # pass through. nn.RoPE with dims < head_dim does exactly this. The
        # reference uses the GPT-NeoX rotate_half convention -> traditional=False.
        rotary_dim = int(self.head_dim * args.partial_rotary_factor)
        self.rope = nn.RoPE(rotary_dim, traditional=False, base=args.rope_theta)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        B, L, _ = x.shape
        qkv = self.query_key_value(x)
        qkv = qkv.reshape(B, L, self.num_heads + 2 * self.num_kv_heads, self.head_dim)
        q, k, v = mx.split(
            qkv,
            [self.num_heads, self.num_heads + self.num_kv_heads],
            axis=2,
        )
        # QK-norm over head_dim (order-invariant vs the transpose below).
        q = self.query_layernorm(q).transpose(0, 2, 1, 3)
        k = self.key_layernorm(k).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        q = self.rope(q)
        k = self.rope(k)

        # GQA is handled by the SDPA kernel (num_kv_heads < num_heads). Mask is
        # None (bidirectional) in the plain forward, block-causal in diffusion.
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.dense(out)


# ---------------------------------------------------------------------------
# Dense MLP (used for the shared expert and the first_k_dense_replace layer).
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, args: ModelArgs, intermediate_size: int):
        super().__init__()
        dim = args.hidden_size
        self.gate_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.up_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


# ---------------------------------------------------------------------------
# Grouped-topk router with sigmoid scoring + per-expert bias. This is the exact
# DeepSeek-V3 / Hunyuan ("noaux_tc") routing: the bias is added only for the
# selection, while the combine weights come from the un-biased sigmoid scores.
# ---------------------------------------------------------------------------


@mx.compile
def group_expert_select(
    gates,
    e_score_correction_bias,
    top_k,
    n_group,
    topk_group,
    routed_scaling_factor,
    norm_topk_prob,
):
    scores = mx.sigmoid(gates.astype(mx.float32))
    orig_scores = scores
    scores = scores + e_score_correction_bias
    if n_group > 1:
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        scores = mx.put_along_axis(
            scores, mx.stop_gradient(group_idx), mx.array(0.0), axis=-2
        )
        scores = mx.flatten(scores, -2, -1)

    k = top_k
    inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)
    if top_k > 1 and norm_topk_prob:
        denominator = scores.sum(axis=-1, keepdims=True) + 1e-20
        scores = scores / denominator
    scores = scores * routed_scaling_factor

    return inds, scores


class MoEGate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.n_routed_experts = args.num_experts
        self.routed_scaling_factor = args.routed_scaling_factor
        self.n_group = args.n_group
        self.topk_group = args.topk_group
        self.weight = mx.zeros((args.num_experts, args.hidden_size))
        # HF name: ``expert_bias`` (registered buffer). Kept unquantized/uncast.
        self.expert_bias = mx.zeros((args.num_experts,))

    def __call__(self, x):
        return group_expert_select(
            x @ self.weight.T,
            self.expert_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )


class LLaDA2MoeSparseMoeBlock(nn.Module):
    """Routed experts (stacked via SwitchGLU) plus one always-on shared expert."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts_per_tok = args.num_experts_per_tok
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.num_experts,
        )
        self.gate = MoEGate(args)
        if args.num_shared_experts is not None and args.num_shared_experts > 0:
            self.shared_experts = MLP(
                args,
                intermediate_size=args.moe_intermediate_size * args.num_shared_experts,
            )

    def __call__(self, x: mx.array) -> mx.array:
        inds, scores = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        if "shared_experts" in self:
            y = y + self.shared_experts(x)
        return y


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.attention = Attention(args)
        # first_k_dense_replace leading layers are dense; the rest are MoE.
        if args.num_experts is not None and layer_idx >= args.first_k_dense_replace:
            self.mlp = LLaDA2MoeSparseMoeBlock(args)
        else:
            self.mlp = MLP(args, intermediate_size=args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        h = x + self.attention(self.input_layernorm(x), mask)
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class LLaDA2MoeModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        h = self.word_embeddings(inputs)
        for layer in self.layers:
            h = layer(h, mask)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LLaDA2MoeModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        # ``cache`` is accepted for API-compatibility with the standard generate
        # loop but ignored: LLaDA2 is a diffusion model with no KV cache. The
        # real decoding path is ``diffusion_generate`` / module-level ``generate``.
        out = self.model(inputs, mask)
        return self.lm_head(out)

    def sanitize(self, weights):
        """Stack per-expert HF weights into the SwitchGLU layout.

        The HF checkpoint stores each expert separately
        (``...mlp.experts.{e}.{gate,up,down}_proj.{weight,scales,biases}``); the
        stacked ``switch_mlp`` expects ``[num_experts, out, in]``. All other
        tensor names already match the mlx-lm layout used here (word_embeddings,
        attention.query_key_value/dense, mlp.gate.weight/expert_bias,
        shared_experts, norm, lm_head), so no other remap is needed. Works for
        both fp/bf16 and pre-quantized (4-bit affine) checkpoints — the ``scales``
        / ``biases`` tensors are stacked the same way as ``weight`` when present.
        """
        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}.mlp"
            if f"{prefix}.experts.0.gate_proj.weight" not in weights:
                continue
            for name in ("gate_proj", "up_proj", "down_proj"):
                for k in ("weight", "scales", "biases"):
                    if f"{prefix}.experts.0.{name}.{k}" not in weights:
                        continue
                    to_join = [
                        weights.pop(f"{prefix}.experts.{e}.{name}.{k}")
                        for e in range(self.args.num_experts)
                    ]
                    weights[f"{prefix}.switch_mlp.{name}.{k}"] = mx.stack(to_join)
        return weights

    @property
    def layers(self):
        return self.model.layers

    @property
    def cast_predicate(self):
        # Keep the router bias in full precision (matches DeepSeek-V3).
        def predicate(k):
            return "expert_bias" not in k

        return predicate


# ---------------------------------------------------------------------------
# Diffusion sampler (block-wise semi-autoregressive masked denoising).
#
# Ported from the reference ``LLaDA2MoeModelLM.generate``: the full target
# length is filled with ``mask_id``, then processed block-by-block. A
# block-causal attention mask lets a block attend to itself and all previous
# blocks. Within a block, ``steps`` denoising iterations progressively unmask
# tokens: each step samples every masked position and commits either every
# position whose confidence clears ``threshold`` or, failing that, the top-k
# most-confident ones (k from a per-step reveal schedule). No KV cache.
# ---------------------------------------------------------------------------


def _num_transfer_tokens(block_length: int, steps: int) -> mx.array:
    """Per-step reveal schedule summing to ``block_length`` (int array [steps])."""
    base = block_length // steps
    remainder = block_length % steps
    sched = mx.full((steps,), base, dtype=mx.int32)
    if remainder > 0:
        sched = sched + (mx.arange(steps) < remainder).astype(mx.int32)
    return sched


def _select_topk(confidence: mx.array, k: int) -> mx.array:
    """Boolean mask [1, L] selecting the ``k`` highest-confidence positions.

    Exact rank via argsort so ties never reveal more than ``k``.
    """
    L = confidence.shape[-1]
    order = mx.argsort(-confidence, axis=-1)  # best first
    ranks = mx.zeros((1, L), dtype=mx.int32)
    cols = mx.arange(L).reshape(1, L).astype(mx.int32)
    ranks = mx.put_along_axis(ranks, order, cols, axis=-1)
    return ranks < k


def diffusion_generate(
    model: Model,
    prompt: mx.array,
    steps: int = 32,
    gen_length: int = 256,
    block_length: int = 32,
    temperature: float = 0.0,
    threshold: float = 0.95,
    mask_id: int = DEFAULT_MASK_TOKEN_ID,
    eos_id: int = DEFAULT_EOS_TOKEN_ID,
    eos_early_stop: bool = False,
    tokenizer=None,
):
    """Block-wise diffusion generation for LLaDA2-MoE.

    Args:
        model: a built ``Model``.
        prompt: token ids ``[1, prompt_len]`` (or ``[prompt_len]``).
        steps: denoising iterations per block.
        gen_length: number of tokens to generate.
        block_length: semi-autoregressive block size.
        temperature: 0 == greedy/argmax. >0 adds Gumbel noise before argmax.
        threshold: confidence bar; positions above it are committed immediately.
        mask_id / eos_id: special token ids.
        eos_early_stop: stop once a completed EOS is produced.
        tokenizer: optional; if given the decoded text is returned too.

    Returns:
        Generated ids ``[1, n]`` (trailing decoded text if a tokenizer is given).
    """
    if prompt.ndim == 1:
        prompt = prompt[None, :]
    if steps <= 0 or gen_length <= 0 or block_length <= 0:
        raise ValueError("steps, gen_length and block_length must be positive")

    prompt_length = prompt.shape[1]
    num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length

    # Block-causal additive mask [1, 1, T, T]: 0 where a query block may attend
    # to a key block (self + earlier), -inf otherwise.
    # Additive mask must match the model's compute dtype — MLX SDPA rejects a
    # float32 mask against bf16 activations (won't downcast). Infer from the
    # final RMSNorm weight (non-quantized).
    mdt = model.model.norm.weight.dtype
    block_mask = mx.tril(mx.ones((num_blocks, num_blocks)))
    full = mx.repeat(mx.repeat(block_mask, block_length, axis=0), block_length, axis=1)
    neg_inf = mx.array(-1e9, dtype=mdt)
    attn_mask = mx.where(full > 0, mx.array(0.0, dtype=mdt), neg_inf)
    attn_mask = attn_mask[None, None]

    x = mx.full((1, total_length), mask_id, dtype=prompt.dtype)
    x[:, :prompt_length] = prompt

    prefill_blocks = prompt_length // block_length
    schedule = _num_transfer_tokens(block_length, steps)

    for nb in range(prefill_blocks, num_blocks):
        win = (nb + 1) * block_length
        cur_mask = attn_mask[:, :, :win, :win]
        blk_lo = win - block_length

        for step in range(steps):
            active_is_mask = x[:, blk_lo:win] == mask_id
            if int(active_is_mask.sum()) == 0:
                break

            logits = model(x[:, :win], mask=cur_mask)
            active_logits = logits[:, blk_lo:win, :].astype(mx.float32)

            if temperature > 0:
                eps = 1e-7
                noise = mx.random.uniform(shape=active_logits.shape).astype(mx.float32)
                noise = mx.clip(noise, eps, 1.0 - eps)
                noised = active_logits - temperature * mx.log(-mx.log(noise))
            else:
                noised = active_logits
            x0 = mx.argmax(noised, axis=-1)  # [1, block_length]

            probs = mx.softmax(active_logits, axis=-1)
            x0_p = mx.take_along_axis(probs, x0[..., None], axis=-1).squeeze(-1)

            confidence = mx.where(active_is_mask, x0_p, neg_inf)
            num_to = int(schedule[step])
            n_active = int(active_is_mask.sum())
            high_conf = confidence > threshold
            n_high = int(high_conf.sum())

            if n_high >= num_to:
                transfer = high_conf
            else:
                transfer = _select_topk(confidence, min(num_to, n_active))

            active = x[:, blk_lo:win]
            x[:, blk_lo:win] = mx.where(transfer, x0, active)
            mx.eval(x)

        if eos_id is not None and int((x[0, prompt_length:win] == eos_id).sum()) > 0:
            if eos_early_stop:
                break

    out = x[:, prompt_length : prompt_length + gen_length]
    mx.eval(out)
    if tokenizer is not None:
        return out, tokenizer.decode(out[0].tolist())
    return out


# Module-level alias mirroring ``llada.generate`` so callers can use either.
generate = diffusion_generate
