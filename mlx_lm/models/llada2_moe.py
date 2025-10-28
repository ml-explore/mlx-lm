# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    intermediate_size: int
    max_position_embeddings: int
    moe_intermediate_size: int
    num_experts: int
    num_shared_experts: int
    norm_topk_prob: bool
    num_attention_heads: int
    num_experts_per_tok: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    vocab_size: int
    first_k_dense_replace: int
    head_dim: Optional[int] = None
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    use_bias: bool = False
    use_qkv_bias: bool = False
    norm_head: bool = False
    norm_softmax: bool = False
    tie_word_embeddings: bool = False
    partial_rotary_factor: float = 1.0
    rotary_dim: Optional[int] = None
    moe_router_enable_expert_bias: bool = False
    routed_scaling_factor: float = 1.0
    score_function: str = "sigmoid"
    n_group: int = 1
    topk_group: int = 4
    mask_token_id: int = 156895
    eos_token_id: int = 156892


@partial(mx.compile, shapeless=True)
def swiglu(gate, up):
    return nn.silu(gate) * up


def is_eos_token(tokens: mx.array, eos_token_ids: set) -> mx.array:
    """Check if tokens match any EOS token ID."""
    return (tokens[:, None] == mx.array(list(eos_token_ids))).any(axis=-1)


class LLaDA2MoeMLP(nn.Module):
    def __init__(self, args: ModelArgs, intermediate_size: Optional[int] = None):
        super().__init__()
        self.intermediate_size = (
            intermediate_size
            if intermediate_size is not None
            else args.intermediate_size
        )

        self.gate_proj = nn.Linear(
            args.hidden_size, self.intermediate_size, bias=args.use_bias
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, args.hidden_size, bias=args.use_bias
        )
        self.up_proj = nn.Linear(
            args.hidden_size, self.intermediate_size, bias=args.use_bias
        )

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class LLaDA2MoeAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.head_dim or (args.hidden_size // self.num_attention_heads)
        self.scale = self.head_dim**-0.5

        self.query_key_value = nn.Linear(
            args.hidden_size,
            (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=args.use_qkv_bias,
        )
        self.dense = nn.Linear(
            self.num_attention_heads * self.head_dim,
            args.hidden_size,
            bias=args.use_bias,
        )
        self.query_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.key_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        rope_dim = args.rotary_dim or int(self.head_dim * args.partial_rotary_factor)
        self.rope = initialize_rope(
            rope_dim,
            args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ):
        B, L, _ = x.shape

        qkv = self.query_key_value(x)
        q_size = self.num_attention_heads * self.head_dim
        kv_size = self.num_key_value_heads * self.head_dim
        q, k, v = mx.split(qkv, [q_size, q_size + kv_size], axis=-1)

        queries = q.reshape(B, L, self.num_attention_heads, -1).transpose(0, 2, 1, 3)
        keys = k.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        values = v.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)

        queries = self.query_layernorm(queries)
        keys = self.key_layernorm(keys)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.dense(output)


def group_expert_select(
    gates,
    expert_bias,
    top_k,
    n_group,
    topk_group,
    routed_scaling_factor,
    norm_topk_prob,
    score_function,
):
    in_type = gates.dtype

    scores = (
        mx.sigmoid(gates.astype(mx.float32))
        if score_function == "sigmoid"
        else mx.softmax(gates.astype(mx.float32), axis=-1)
    )
    orig_scores = scores

    if expert_bias is not None:
        scores = scores + expert_bias

    if n_group > 1:
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        scores = mx.put_along_axis(
            scores, mx.stop_gradient(group_idx), mx.array(0.0, scores.dtype), axis=-2
        )
        scores = mx.flatten(scores, -2, -1)

    inds = mx.argpartition(scores, kth=-top_k, axis=-1)[..., -top_k:]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)

    if top_k > 1 and norm_topk_prob:
        scores = scores / (scores.sum(axis=-1, keepdims=True) + 1e-20)

    scores = scores * routed_scaling_factor
    return inds, scores.astype(in_type)


class LLaDA2MoeGate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.norm_topk_prob = args.norm_topk_prob
        self.top_k = args.num_experts_per_tok
        self.n_group = args.n_group
        self.topk_group = args.topk_group
        self.routed_scaling_factor = args.routed_scaling_factor
        self.score_function = args.score_function
        self.weight = mx.zeros((args.num_experts, args.hidden_size))
        self.expert_bias = (
            mx.zeros((args.num_experts,))
            if args.moe_router_enable_expert_bias
            else None
        )

    def __call__(self, x):
        orig_shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        gates = mx.matmul(x.astype(mx.float32), self.weight.T)

        indices, scores = group_expert_select(
            gates,
            self.expert_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
            self.score_function,
        )

        return indices.reshape(*orig_shape[:-1], -1), scores.reshape(
            *orig_shape[:-1], -1
        )


class LLaDA2MoeSparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts_per_tok = args.num_experts_per_tok
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.num_experts,
            bias=args.use_bias,
        )
        self.gate = LLaDA2MoeGate(args)
        self.shared_experts = (
            LLaDA2MoeMLP(
                args=args,
                intermediate_size=args.moe_intermediate_size * args.num_shared_experts,
            )
            if args.num_shared_experts > 0
            else None
        )

    def __call__(self, x):
        inds, scores = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        if self.shared_experts is not None:
            y = y + self.shared_experts(x)
        return y


class LLaDA2MoeDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.attention = LLaDA2MoeAttention(args)
        self.mlp = (
            LLaDA2MoeSparseMoeBlock(args)
            if layer_idx >= args.first_k_dense_replace
            else LLaDA2MoeMLP(args)
        )
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache=None):
        r = self.attention(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class LLaDA2MoeModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            LLaDA2MoeDecoderLayer(args, i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs, cache=None, inputs_embeds=None, mask=None):
        h = inputs_embeds if inputs_embeds is not None else self.word_embeddings(inputs)

        if cache is None:
            cache = [None] * len(self.layers)
        if mask is None:
            mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    """LLaDA2 MoE model with diffusion-based generation."""

    EXTRA_EOS_TOKENS = ["<|role_end|>"]

    # As per original paper, LLaDA does not support kv caching
    supports_prompt_cache = False

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LLaDA2MoeModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None, inputs_embeds=None, mask=None):
        out = self.model(inputs, cache, inputs_embeds, mask)
        if self.args.tie_word_embeddings:
            return self.model.word_embeddings.as_linear(out)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return (
            self.args.head_dim or self.args.hidden_size // self.args.num_attention_heads
        )

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads

    def _create_block_diagonal_mask(self, num_blocks: int, block_length: int):
        """Create block-diagonal attention mask for diffusion generation."""
        mask = mx.tril(mx.ones((num_blocks, num_blocks)))
        mask = mx.repeat(mx.repeat(mask, block_length, axis=0), block_length, axis=1)
        mask = mask[None, None, :, :]
        return mx.where(mask, 0.0, float("-inf")).astype(mx.bfloat16)

    def _select_tokens_to_update(
        self, confidence: mx.array, mask: mx.array, num_tokens: int, threshold: float
    ):
        """Select which tokens to update based on confidence scores."""
        conf = mx.where(mask, confidence, float("-inf"))[0]

        high_conf = conf > threshold
        if high_conf.sum().item() >= num_tokens:
            return high_conf

        k = min(num_tokens, mask.sum().item())
        idx = mx.argpartition(-conf, kth=k - 1)[:k]
        positions = mx.arange(len(conf))
        return (positions[:, None] == idx[None, :]).any(axis=1)

    def _find_stop_position(self, tokens: mx.array, mask_id: int, eos_ids: set):
        """Find first mask or EOS position in token sequence."""
        is_mask = tokens == mask_id
        is_eos = is_eos_token(tokens, eos_ids)
        stop_mask = is_mask | is_eos

        if not stop_mask.any():
            return len(tokens), False

        stop_idx = mx.argmax(stop_mask.astype(mx.int32)).item()
        return stop_idx, is_eos[stop_idx].item() if stop_idx < len(is_eos) else False

    def generate_step(
        self,
        inputs: mx.array,
        max_tokens: int = 2048,
        sampler: Optional[callable] = None,
        block_length: int = 32,
        steps: int = 32,
        minimal_topk: int = 1,
        threshold: float = 0.95,
        eos_token_ids: Optional[Union[int, list, set]] = None,
        mask_id: Optional[int] = None,
    ):
        """
        Diffusion-based text generation using block-wise iterative denoising.

        Args:
            inputs: Input token IDs (prompt).
            max_tokens: Maximum tokens to generate.
            sampler: Sampling function from make_sampler().
            block_length: Size of each generation block.
            steps: Number of denoising iterations per block.
            minimal_topk: Minimum tokens to keep (caps effective steps).
            threshold: Confidence threshold for token acceptance.
            eos_token_ids: EOS token ID(s) (int, list, or set).
            mask_id: Mask token ID for ungenerated positions.

        Yields:
            (tokens, logprobs): Generated tokens and empty logprobs array.
        """
        sampler = sampler or (lambda x: mx.argmax(x, axis=-1))

        if eos_token_ids is None:
            eos_token_ids = {self.args.eos_token_id}
        elif not isinstance(eos_token_ids, set):
            eos_token_ids = (
                {eos_token_ids}
                if isinstance(eos_token_ids, int)
                else set(eos_token_ids)
            )

        mask_id = mask_id or self.args.mask_token_id
        steps = min(steps, max_tokens // minimal_topk)

        batch_size, prompt_length = inputs.shape
        if batch_size != 1:
            raise ValueError(
                f"Diffusion generation only supports batch_size=1, got {batch_size}"
            )

        num_blocks = (prompt_length + max_tokens + block_length - 1) // block_length
        total_length = num_blocks * block_length

        mask = self._create_block_diagonal_mask(num_blocks, block_length)
        transfer_schedule = self._get_num_transfer_tokens(block_length, steps)

        x = mx.full((1, total_length), mask_id, dtype=mx.int32)
        x[:, :prompt_length] = inputs

        last_yield_pos = prompt_length
        prefill_blocks = prompt_length // block_length

        for block_idx in range(prefill_blocks, num_blocks):
            window_end = (block_idx + 1) * block_length
            cur_x = x[:, :window_end]
            cur_mask = mask[:, :, :window_end, :window_end]
            block_start = block_idx * block_length

            for step in range(steps):
                active_mask = cur_x[:, -block_length:] == mask_id
                if not active_mask.any():
                    break

                logits = self(cur_x, cache=None, mask=cur_mask)
                tokens, confidence = self._sample_with_sampler(
                    logits[:, -block_length:, :], sampler
                )

                num_transfer = int(transfer_schedule[step])
                update_mask = self._select_tokens_to_update(
                    confidence, active_mask, num_transfer, threshold
                )

                if not update_mask.any():
                    continue

                new_block = mx.where(update_mask, tokens[0], cur_x[0, -block_length:])
                cur_x = mx.concatenate(
                    [cur_x[:, :-block_length], new_block[None, :]], axis=1
                )
                x[:, :window_end] = cur_x

                start = max(last_yield_pos - block_start, 0)
                if start >= block_length:
                    continue

                remaining = cur_x[0, -block_length:][start:]
                stop_idx, hit_eos = self._find_stop_position(
                    remaining, mask_id, eos_token_ids
                )

                if stop_idx > 0:
                    end_idx = stop_idx + 1 if hit_eos else stop_idx
                    yield (remaining[:end_idx], mx.array([]))
                    last_yield_pos = block_start + start + end_idx

                if hit_eos:
                    return

            gen_end = min(window_end, prompt_length + max_tokens)
            if gen_end > last_yield_pos:
                remaining = x[0, last_yield_pos:gen_end]
                stop_idx, hit_eos = self._find_stop_position(
                    remaining, mask_id, eos_token_ids
                )

                if stop_idx > 0:
                    end_idx = stop_idx + 1 if hit_eos else stop_idx
                    yield (remaining[:end_idx], mx.array([]))
                    last_yield_pos += end_idx

                if hit_eos:
                    return

    @staticmethod
    def _get_num_transfer_tokens(block_length: int, steps: int):
        """Calculate token transfer schedule for denoising steps."""
        if steps == 0:
            return mx.array([], dtype=mx.int32)

        base = block_length // steps
        remainder = block_length % steps
        schedule = mx.full((steps,), base, dtype=mx.int32)
        schedule[:remainder] += 1
        return schedule

    def _sample_with_sampler(self, logits: mx.array, sampler: callable):
        """Sample tokens and return confidence scores."""
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        tokens = sampler(logprobs)
        probs = mx.exp(logprobs)
        confidence = mx.take_along_axis(probs, tokens[..., None], axis=-1).squeeze(-1)
        return tokens, confidence

    def sanitize(self, weights):
        """Convert HuggingFace weights to MLX format by stacking MoE expert weights."""
        for l in range(self.args.first_k_dense_replace, self.args.num_hidden_layers):
            prefix = f"model.layers.{l}.mlp"
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                if f"{prefix}.experts.0.{proj}.weight" in weights:
                    stacked = mx.stack(
                        [
                            weights.pop(f"{prefix}.experts.{e}.{proj}.weight")
                            for e in range(self.args.num_experts)
                        ]
                    )
                    weights[f"{prefix}.switch_mlp.{proj}.weight"] = stacked
        return weights
