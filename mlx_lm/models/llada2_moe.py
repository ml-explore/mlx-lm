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
    """LLaDA2 MoE model configuration arguments."""

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
    # Diffusion-specific parameters
    mask_token_id: int = 156895
    eos_token_id: int = 156892


@partial(mx.compile, shapeless=True)
def swiglu(gate, up):
    """SwiGLU activation function."""
    return nn.silu(gate) * up


@partial(mx.compile, shapeless=True)
def aggregate_expert_outputs(expert_outputs, scores):
    """Aggregate expert outputs weighted by routing scores."""
    return (
        (expert_outputs * scores[..., None]).sum(axis=-2).astype(expert_outputs.dtype)
    )


class LLaDA2MoeMLP(nn.Module):
    """Standard MLP for dense layers."""

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
    """Multi-head attention with fused QKV and QK normalization."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.head_dim or (args.hidden_size // self.num_attention_heads)
        self.scale = self.head_dim**-0.5

        # Fused QKV projection
        self.query_key_value = nn.Linear(
            args.hidden_size,
            (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=args.use_qkv_bias,
        )

        # Output projection
        self.dense = nn.Linear(
            self.num_attention_heads * self.head_dim,
            args.hidden_size,
            bias=args.use_bias,
        )

        # QK normalization
        self.query_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.key_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        # RoPE with partial rotary support
        if (rope_dim := args.rotary_dim) is None:
            rope_dim = int(self.head_dim * args.partial_rotary_factor)
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
    ) -> mx.array:
        B, L, D = x.shape

        # Fused QKV projection
        qkv = self.query_key_value(x)

        # Split into Q, K, V
        q_size = self.num_attention_heads * self.head_dim
        kv_size = self.num_key_value_heads * self.head_dim
        q, k, v = mx.split(qkv, [q_size, q_size + kv_size], axis=-1)

        # Reshape for multi-head attention
        queries = q.reshape(B, L, self.num_attention_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        keys = k.reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        values = v.reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        # Apply QK normalization
        queries = self.query_layernorm(queries)
        keys = self.key_layernorm(keys)

        # Apply RoPE
        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # Scaled dot product attention
        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        # Reshape and project output
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.dense(output)


@mx.compile
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
    """
    Group-based expert selection with sigmoid or softmax scoring.

    Implements the group-limited top-k routing algorithm from LLaDA2.
    """
    in_type = gates.dtype

    # Apply scoring function (sigmoid for LLaDA2)
    if score_function == "sigmoid":
        scores = mx.sigmoid(gates.astype(mx.float32))
    else:
        scores = mx.softmax(gates.astype(mx.float32), axis=-1)

    orig_scores = scores

    # Add expert bias if enabled
    if expert_bias is not None:
        scores = scores + expert_bias

    # Group-based routing
    if n_group > 1:
        # Reshape into groups
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        # Sum top-2 scores in each group
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        # Select top-k groups
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        # Zero out non-selected groups
        scores = mx.put_along_axis(
            scores, mx.stop_gradient(group_idx), mx.array(0.0, scores.dtype), axis=-2
        )
        scores = mx.flatten(scores, -2, -1)

    # Select top-k experts
    k = top_k
    inds = mx.argpartition(scores, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)

    # Normalize top-k probabilities
    if top_k > 1 and norm_topk_prob:
        denominator = scores.sum(axis=-1, keepdims=True) + 1e-20
        scores = scores / denominator

    # Apply routed scaling factor
    scores = scores * routed_scaling_factor

    return inds, scores.astype(in_type)


class LLaDA2MoeGate(nn.Module):
    """MoE router with group-limited top-k selection."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.norm_topk_prob = args.norm_topk_prob
        self.top_k = args.num_experts_per_tok
        self.n_group = args.n_group
        self.topk_group = args.topk_group
        self.routed_scaling_factor = args.routed_scaling_factor
        self.score_function = args.score_function

        # Gate projection (raw weight, not nn.Linear wrapper)
        self.weight = mx.random.normal((args.num_experts, args.hidden_size)) * 0.02

        # Expert bias for load balancing
        self.expert_bias = (
            mx.zeros((args.num_experts,))
            if args.moe_router_enable_expert_bias
            else None
        )

    def __call__(self, x):
        # Flatten batch and sequence dimensions
        orig_shape = x.shape
        x = x.reshape(-1, x.shape[-1])

        # Manual linear transformation
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

        # Reshape back to original batch/sequence dimensions
        indices = indices.reshape(*orig_shape[:-1], -1)
        scores = scores.reshape(*orig_shape[:-1], -1)

        return indices, scores


class LLaDA2MoeSparseMoeBlock(nn.Module):
    """Sparse MoE block with routed and shared experts."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts_per_tok = args.num_experts_per_tok

        # Routed experts
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.num_experts,
            bias=args.use_bias,
        )

        # Router gate
        self.gate = LLaDA2MoeGate(args)

        # Shared experts
        self.shared_experts = (
            LLaDA2MoeMLP(
                args=args,
                intermediate_size=args.moe_intermediate_size * args.num_shared_experts,
            )
            if args.num_shared_experts > 0
            else None
        )

    def __call__(self, x):
        # Route to experts
        topk_idx, topk_weight = self.gate(x)
        out = self.switch_mlp(x, topk_idx)
        out = aggregate_expert_outputs(out, topk_weight)

        # Add shared expert output
        if self.shared_experts is not None:
            out = out + self.shared_experts(x)

        return out


class LLaDA2MoeDecoderLayer(nn.Module):
    """Transformer decoder layer with attention and MoE/dense MLP."""

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.attention = LLaDA2MoeAttention(args)

        # Use MoE for layers >= first_k_dense_replace, dense for earlier layers
        self.mlp = (
            LLaDA2MoeSparseMoeBlock(args)
            if layer_idx >= args.first_k_dense_replace
            else LLaDA2MoeMLP(args)
        )

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
        # Pre-norm attention
        r = self.attention(self.input_layernorm(x), mask, cache)
        h = x + r

        # Pre-norm MLP
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class LLaDA2MoeModel(nn.Module):
    """LLaDA2 MoE base model."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers

        # Embeddings
        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)

        # Transformer layers
        self.layers = [
            LLaDA2MoeDecoderLayer(args, layer_idx)
            for layer_idx in range(args.num_hidden_layers)
        ]

        # Final normalization
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
    ):
        # Get embeddings
        if inputs_embeds is None:
            h = self.word_embeddings(inputs)
        else:
            h = inputs_embeds

        # Create attention mask if not provided
        if mask is None:
            mask = create_attention_mask(h, cache)

        # Process through transformer layers
        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        # Final normalization
        return self.norm(h)


class Model(nn.Module):
    """
    LLaDA2 MoE model with language modeling head and diffusion generation.

    This is a Diffusion Language Model (DLLM) that uses iterative block-wise
    denoising instead of standard autoregressive generation.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LLaDA2MoeModel(args)

        # Language modeling head
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

        # Mark as diffusion model for generation pipeline
        self.is_diffusion_model = True

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
    ):
        out = self.model(inputs, cache, inputs_embeds, mask)

        if self.args.tie_word_embeddings:
            out = self.model.word_embeddings.as_linear(out)
        else:
            out = self.lm_head(out)

        return out

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

    def stream_generate(
        self,
        inputs: mx.array,
        temperature: float = 0.0,
        block_length: int = 32,
        steps: int = 32,
        gen_length: int = 2048,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        eos_early_stop: bool = False,
        minimal_topk: int = 1,
        threshold: float = 0.95,
        eos_id: Optional[int] = None,
        mask_id: Optional[int] = None,
    ):
        """
        Stream diffusion generation block-by-block.

        This method yields completed blocks as they finish denoising, allowing
        for progressive output display.

        Args:
            inputs: Input token IDs (prompt)
            temperature: Sampling temperature (0 = greedy)
            block_length: Size of each generation block
            steps: Number of denoising iterations per block
            gen_length: Maximum tokens to generate
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling parameter
            eos_early_stop: Stop on first valid EOS token
            minimal_topk: Minimum tokens to keep (caps effective steps)
            threshold: Confidence threshold for token acceptance
            eos_id: End-of-sequence token ID
            mask_id: Mask token ID for ungenerated positions

        Yields:
            Tuple[mx.array, int, int]: (new_tokens, block_start, block_end)
                - new_tokens: The newly completed tokens for this block
                - block_start: Starting position (relative to generation start)
                - block_end: Ending position (relative to generation start)
        """
        # Use config defaults if not provided
        if eos_id is None:
            eos_id = self.args.eos_token_id
        if mask_id is None:
            mask_id = self.args.mask_token_id

        # Cap steps based on minimal_topk
        steps = min(steps, gen_length // minimal_topk)

        # Setup
        prompt_length = inputs.shape[1] if len(inputs.shape) > 1 else inputs.shape[0]
        if len(inputs.shape) == 1:
            inputs = mx.expand_dims(inputs, 0)

        num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
        total_length = num_blocks * block_length

        # Create block-diagonal attention mask
        block_mask = mx.tril(mx.ones((num_blocks, num_blocks)))
        block_diffusion_attention_mask = mx.repeat(
            mx.repeat(block_mask, block_length, axis=0), block_length, axis=1
        )
        block_diffusion_attention_mask = mx.expand_dims(
            mx.expand_dims(block_diffusion_attention_mask, 0), 0
        )
        block_diffusion_attention_mask = mx.where(
            block_diffusion_attention_mask.astype(mx.bool_),
            mx.array(0.0, dtype=mx.bfloat16),
            mx.array(float("-inf"), dtype=mx.bfloat16),
        )

        # Initialize sequence with mask tokens
        x = mx.full((1, total_length), mask_id, dtype=mx.int32)
        x[:, :prompt_length] = inputs

        # Determine how many blocks are already filled (prompt)
        prefill_blocks = prompt_length // block_length

        # Calculate token transfer schedule
        num_transfer_tokens_schedule = self._get_num_transfer_tokens(
            block_length, steps
        )

        # Track what we've yielded
        last_yielded_pos = prompt_length

        # Process each block
        for num_block in range(prefill_blocks, num_blocks):
            current_window_end = (num_block + 1) * block_length
            cur_x = x[:, :current_window_end]
            cur_attn_mask = block_diffusion_attention_mask[
                :, :, :current_window_end, :current_window_end
            ]

            # Iterative denoising for this block
            for step in range(steps):
                # Check if any tokens left to denoise
                active_block_mask = cur_x[:, -block_length:] == mask_id
                if active_block_mask.sum() == 0:
                    break

                # Forward pass with block-diagonal mask
                logits = self(cur_x, cache=None, mask=cur_attn_mask)

                # Focus on current block
                active_logits = logits[:, -block_length:, :]

                # Sample tokens with confidence
                x0, x0_p = self._sample_with_temperature_topk_topp(
                    active_logits, temperature=temperature, top_k=top_k, top_p=top_p
                )

                # Determine which tokens to transfer
                num_to_transfer = int(num_transfer_tokens_schedule[step])
                confidence = mx.where(active_block_mask, x0_p, float("-inf"))

                # Select high-confidence tokens
                high_conf_mask = confidence[0] > threshold
                num_high_confidence = high_conf_mask.sum().item()

                # Determine which positions to update
                if num_high_confidence >= num_to_transfer:
                    # Use high confidence mask
                    update_mask = high_conf_mask
                else:
                    # Take top-k by confidence
                    k = min(num_to_transfer, active_block_mask.sum().item())
                    idx = mx.argpartition(-confidence[0], kth=k - 1)[:k]
                    # Create update mask using broadcasting
                    positions = mx.arange(block_length)
                    update_mask = mx.any(
                        mx.expand_dims(positions, 0) == mx.expand_dims(idx, 1), axis=0
                    )

                # Update tokens in the current block
                if update_mask.any():
                    # Update only the last block_length positions
                    new_block = mx.where(update_mask, x0[0], cur_x[0, -block_length:])
                    cur_x = mx.concatenate(
                        [cur_x[:, :-block_length], mx.expand_dims(new_block, 0)], axis=1
                    )

                # Early stop on EOS
                if eos_early_stop:
                    # Check if any newly updated tokens are EOS
                    newly_updated = mx.where(
                        update_mask, x0[0], mx.array(-1, dtype=x0.dtype)
                    )
                    if (newly_updated == eos_id).any():
                        eos_mask = cur_x[0, prompt_length:] == eos_id
                        if eos_mask.any():
                            # Find first EOS position by converting to list
                            eos_list = eos_mask.tolist()
                            try:
                                eos_idx = eos_list.index(True)
                                eos_pos = eos_idx + prompt_length
                                if (cur_x[0, prompt_length:eos_pos] != mask_id).all():
                                    # Yield final tokens and return
                                    final_end = eos_pos + 1
                                    if final_end > last_yielded_pos:
                                        new_tokens = cur_x[
                                            0, last_yielded_pos:final_end
                                        ]
                                        block_start = last_yielded_pos - prompt_length
                                        block_end = final_end - prompt_length
                                        yield (new_tokens, block_start, block_end)
                                    return
                            except ValueError:
                                pass  # No True found

            # Update global sequence
            x[:, :current_window_end] = cur_x

            # Yield completed block (excluding mask tokens and stopping at EOS)
            gen_end = min(current_window_end, prompt_length + gen_length)
            if gen_end > last_yielded_pos:
                new_tokens = x[0, last_yielded_pos:gen_end]

                # Don't yield if block is all mask tokens
                if not (new_tokens == mask_id).all():
                    # Check for EOS tokens first
                    eos_positions = (new_tokens == eos_id).tolist()
                    if any(eos_positions):
                        # Truncate at first EOS (inclusive)
                        first_eos_idx = eos_positions.index(True)
                        new_tokens = new_tokens[: first_eos_idx + 1]
                        actual_end = last_yielded_pos + first_eos_idx + 1
                        block_start = last_yielded_pos - prompt_length
                        block_end = actual_end - prompt_length
                        yield (new_tokens, block_start, block_end)
                        # Stop after yielding block with EOS
                        return

                    # Check for mask tokens
                    mask_positions = (new_tokens == mask_id).tolist()
                    if any(mask_positions):
                        # Find first mask position
                        first_mask_idx = mask_positions.index(True)
                        if first_mask_idx > 0:
                            # Yield only up to first mask
                            new_tokens = new_tokens[:first_mask_idx]
                            actual_end = last_yielded_pos + first_mask_idx
                            block_start = last_yielded_pos - prompt_length
                            block_end = actual_end - prompt_length
                            yield (new_tokens, block_start, block_end)
                            last_yielded_pos = actual_end
                            # Stop yielding - rest is incomplete
                            break
                    else:
                        # No masks or EOS, yield entire block
                        block_start = last_yielded_pos - prompt_length
                        block_end = gen_end - prompt_length
                        yield (new_tokens, block_start, block_end)
                        last_yielded_pos = gen_end

            # Stop if EOS found
            if (x[0, prompt_length:current_window_end] == eos_id).any():
                break

    def generate(
        self,
        inputs: mx.array,
        temperature: float = 0.0,
        block_length: int = 32,
        steps: int = 32,
        gen_length: int = 2048,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        eos_early_stop: bool = False,
        minimal_topk: int = 1,
        threshold: float = 0.95,
        eos_id: Optional[int] = None,
        mask_id: Optional[int] = None,
    ) -> mx.array:
        """
        Diffusion-based generation using block-wise iterative denoising.

        This method operates differently from standard autoregressive generation.
        It creates a template filled with mask_id tokens and iteratively "denoises"
        them into actual tokens over multiple refinement steps per block.

        Args:
            inputs: Input token IDs (prompt)
            temperature: Sampling temperature (0 = greedy)
            block_length: Size of each generation block
            steps: Number of denoising iterations per block
            gen_length: Maximum tokens to generate
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling parameter
            eos_early_stop: Stop on first valid EOS token
            minimal_topk: Minimum tokens to keep (caps effective steps)
            threshold: Confidence threshold for token acceptance
            eos_id: End-of-sequence token ID
            mask_id: Mask token ID for ungenerated positions

        Returns:
            Generated token IDs
        """
        # Use config defaults if not provided
        if eos_id is None:
            eos_id = self.args.eos_token_id
        if mask_id is None:
            mask_id = self.args.mask_token_id

        # Cap steps based on minimal_topk
        steps = min(steps, gen_length // minimal_topk)

        # Setup
        prompt_length = inputs.shape[1] if len(inputs.shape) > 1 else inputs.shape[0]
        if len(inputs.shape) == 1:
            inputs = mx.expand_dims(inputs, 0)

        num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
        total_length = num_blocks * block_length

        # Create block-diagonal attention mask
        block_mask = mx.tril(mx.ones((num_blocks, num_blocks)))
        # Expand each block into block_length x block_length
        block_diffusion_attention_mask = mx.repeat(
            mx.repeat(block_mask, block_length, axis=0), block_length, axis=1
        )
        # Add batch and head dimensions
        block_diffusion_attention_mask = mx.expand_dims(
            mx.expand_dims(block_diffusion_attention_mask, 0), 0
        )
        # Convert to additive mask (0 for attend, -inf for masked)
        block_diffusion_attention_mask = mx.where(
            block_diffusion_attention_mask.astype(mx.bool_),
            mx.array(0.0, dtype=mx.bfloat16),
            mx.array(float("-inf"), dtype=mx.bfloat16),
        )

        # Initialize sequence with mask tokens
        x = mx.full((1, total_length), mask_id, dtype=mx.int32)
        x[:, :prompt_length] = inputs

        # Determine how many blocks are already filled (prompt)
        prefill_blocks = prompt_length // block_length

        # Calculate token transfer schedule
        num_transfer_tokens_schedule = self._get_num_transfer_tokens(
            block_length, steps
        )

        # Process each block
        for num_block in range(prefill_blocks, num_blocks):
            current_window_end = (num_block + 1) * block_length
            cur_x = x[:, :current_window_end]
            cur_attn_mask = block_diffusion_attention_mask[
                :, :, :current_window_end, :current_window_end
            ]

            # Iterative denoising for this block
            for step in range(steps):
                # Check if any tokens left to denoise
                active_block_mask = cur_x[:, -block_length:] == mask_id
                if active_block_mask.sum() == 0:
                    break

                # Forward pass with block-diagonal mask
                logits = self(cur_x, cache=None, mask=cur_attn_mask)

                # Focus on current block
                active_logits = logits[:, -block_length:, :]

                # Sample tokens with confidence
                x0, x0_p = self._sample_with_temperature_topk_topp(
                    active_logits, temperature=temperature, top_k=top_k, top_p=top_p
                )

                # Determine which tokens to transfer
                num_to_transfer = int(num_transfer_tokens_schedule[step])
                confidence = mx.where(active_block_mask, x0_p, float("-inf"))

                # Select high-confidence tokens
                high_conf_mask = confidence[0] > threshold
                num_high_confidence = high_conf_mask.sum().item()

                # Determine which positions to update
                if num_high_confidence >= num_to_transfer:
                    # Use high confidence mask
                    update_mask = high_conf_mask
                else:
                    # Take top-k by confidence
                    k = min(num_to_transfer, active_block_mask.sum().item())
                    idx = mx.argpartition(-confidence[0], kth=k - 1)[:k]
                    # Create update mask using broadcasting
                    # Compare each position against all selected indices
                    positions = mx.arange(block_length)
                    update_mask = mx.any(
                        mx.expand_dims(positions, 0) == mx.expand_dims(idx, 1), axis=0
                    )

                # Update tokens in the current block
                if update_mask.any():
                    # Update only the last block_length positions
                    new_block = mx.where(update_mask, x0[0], cur_x[0, -block_length:])
                    cur_x = mx.concatenate(
                        [cur_x[:, :-block_length], mx.expand_dims(new_block, 0)], axis=1
                    )

                # Early stop on EOS
                if eos_early_stop:
                    # Check if any newly updated tokens are EOS
                    newly_updated = mx.where(
                        update_mask, x0[0], mx.array(-1, dtype=x0.dtype)
                    )
                    if (newly_updated == eos_id).any():
                        eos_mask = cur_x[0, prompt_length:] == eos_id
                        if eos_mask.any():
                            # Find first EOS position by converting to list
                            eos_list = eos_mask.tolist()
                            try:
                                eos_idx = eos_list.index(True)
                                eos_pos = eos_idx + prompt_length
                                if (cur_x[0, prompt_length:eos_pos] != mask_id).all():
                                    return cur_x[:, : eos_pos + 1]
                            except ValueError:
                                pass  # No True found

            # Update global sequence
            x[:, :current_window_end] = cur_x

            # Stop if EOS found
            if (x[0, prompt_length:current_window_end] == eos_id).any():
                break

        # Extract final generation
        generated_answer = x[:, : prompt_length + gen_length]

        # Find first mask token and truncate (incomplete generation)
        generation_part = generated_answer[0, prompt_length:]
        mask_positions = (generation_part == mask_id).tolist()
        if any(mask_positions):
            first_mask_idx = mask_positions.index(True)
            if first_mask_idx > 0:
                # Truncate at first mask
                generated_answer = generated_answer[:, : prompt_length + first_mask_idx]

        # Find first EOS and truncate
        eos_mask = generated_answer[0, prompt_length:] == eos_id
        if eos_mask.any():
            # Find first EOS position by converting to list
            eos_list = eos_mask.tolist()
            try:
                first_eos_position = eos_list.index(True)
                return generated_answer[
                    :, prompt_length : prompt_length + first_eos_position + 1
                ]
            except ValueError:
                pass  # No True found

        return generated_answer[:, prompt_length:]

    @staticmethod
    def _get_num_transfer_tokens(block_length: int, steps: int) -> mx.array:
        """Calculate token transfer schedule for denoising steps."""
        if steps == 0:
            return mx.array([], dtype=mx.int32)

        base = block_length // steps
        remainder = block_length % steps
        num_transfer_tokens = mx.full((steps,), base, dtype=mx.int32)
        num_transfer_tokens[:remainder] += 1

        return num_transfer_tokens

    @staticmethod
    def _top_k_logits(logits: mx.array, k: Optional[int]) -> mx.array:
        """Apply top-k filtering to logits."""
        if k is None or k <= 0:
            return logits

        values = mx.topk(logits, k)
        min_values = mx.expand_dims(values[..., -1], -1)
        return mx.where(logits < min_values, float("-inf"), logits)

    @staticmethod
    def _top_p_logits(logits: mx.array, p: Optional[float]) -> mx.array:
        """Apply nucleus (top-p) filtering to logits."""
        if p is None or p >= 1.0:
            return logits

        sorted_logits = mx.sort(logits, axis=-1)[:, :, ::-1]
        sorted_indices = mx.argsort(logits, axis=-1)[:, :, ::-1]
        cumulative_probs = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)

        sorted_mask = cumulative_probs > p
        sorted_mask = mx.concatenate(
            [mx.zeros_like(sorted_mask[:, :, :1]), sorted_mask[:, :, :-1]], axis=-1
        )

        # Scatter mask back to original order
        mask = mx.take_along_axis(
            sorted_mask, mx.argsort(sorted_indices, axis=-1), axis=-1
        )
        return mx.where(mask, float("-inf"), logits)

    def _sample_with_temperature_topk_topp(
        self,
        logits: mx.array,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> tuple[mx.array, mx.array]:
        """Sample tokens with temperature, top-k, and top-p filtering."""
        # Apply temperature
        if temperature > 0 and temperature != 1.0:
            logits = logits / temperature

        # Apply top-k filtering
        logits = self._top_k_logits(logits, top_k)

        # Apply top-p filtering
        logits = self._top_p_logits(logits, top_p)

        # Sample
        probs = mx.softmax(logits, axis=-1)
        token = mx.random.categorical(mx.log(probs + 1e-10), axis=-1)
        token_prob = mx.take_along_axis(probs, mx.expand_dims(token, -1), axis=-1)

        return token, token_prob.squeeze(-1)

    def sanitize(self, weights):
        """
        Convert HuggingFace weights to MLX format.

        Key transformations:
        - Rename 'model.' prefix to match MLX structure
        - Combine individual expert weights into batched tensors
        - Handle gate.weight (raw matrix, not gate_proj)
        """
        import re
        from collections import defaultdict

        # Group expert weights by layer and projection type
        expert_weights = defaultdict(list)
        new_weights = {}

        for k, v in weights.items():
            # Keep track of original prefix
            has_model_prefix = k.startswith("model.")
            k_without_prefix = k[6:] if has_model_prefix else k

            # Check if this is an expert weight
            expert_match = re.match(
                r"layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight",
                k_without_prefix,
            )

            if expert_match:
                layer_idx, expert_idx, proj_type = expert_match.groups()
                # Keep the model prefix for the transformed key
                expert_key = (
                    f"model.layers.{layer_idx}.mlp.switch_mlp.{proj_type}.weight"
                )
                expert_weights[expert_key].append((int(expert_idx), v))
            else:
                # Keep the weight as-is (with model. prefix if it had one)
                new_weights[k] = v

        # Stack expert weights
        for expert_key, expert_list in expert_weights.items():
            # Sort by expert index
            expert_list.sort(key=lambda x: x[0])
            # Stack into single tensor
            stacked = mx.stack([w for _, w in expert_list])
            new_weights[expert_key] = stacked

        return new_weights
