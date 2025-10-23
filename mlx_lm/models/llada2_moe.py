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


def is_eos_token(tokens: mx.array, eos_token_ids) -> mx.array:
    """
    Check if tokens match any EOS token ID using broadcasting.

    Args:
        tokens: 1D array of token IDs to check
        eos_token_ids: Set, list, or array of EOS token IDs

    Returns:
        Boolean mask with same shape as tokens
    """
    eos_array = mx.array(list(eos_token_ids))
    return (tokens[:, None] == eos_array).any(axis=-1)


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
        B, L, _ = x.shape

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
        self.weight = mx.zeros((args.num_experts, args.hidden_size))

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

    def _create_block_diagonal_mask(
        self, num_blocks: int, block_length: int
    ) -> mx.array:
        """Create block-diagonal attention mask for diffusion generation."""
        block_mask = mx.tril(mx.ones((num_blocks, num_blocks)))
        block_diffusion_attention_mask = mx.repeat(
            mx.repeat(block_mask, block_length, axis=0), block_length, axis=1
        )
        block_diffusion_attention_mask = mx.expand_dims(
            mx.expand_dims(block_diffusion_attention_mask, 0), 0
        )
        return mx.where(
            block_diffusion_attention_mask.astype(mx.bool_),
            mx.array(0.0, dtype=mx.bfloat16),
            mx.array(float("-inf"), dtype=mx.bfloat16),
        )

    def _select_tokens_to_update(
        self,
        x0_p: mx.array,
        active_block_mask: mx.array,
        num_to_transfer: int,
        threshold: float,
        positions_array: mx.array,
    ) -> mx.array:
        """Determine which token positions should be updated based on confidence."""
        confidence = mx.where(active_block_mask, x0_p, float("-inf"))
        high_conf_mask = confidence[0] > threshold
        num_high_confidence = high_conf_mask.sum().item()

        if num_high_confidence >= num_to_transfer:
            return high_conf_mask

        # Take top-k by confidence
        k = min(num_to_transfer, active_block_mask.sum().item())
        idx = mx.argpartition(-confidence[0], kth=k - 1)[:k]
        return mx.any(
            mx.expand_dims(positions_array, 0) == mx.expand_dims(idx, 1), axis=0
        )

    def _check_early_stop_eos(
        self,
        update_mask: mx.array,
        x0: mx.array,
        cur_x: mx.array,
        prompt_length: int,
        eos_token_ids: set,
        mask_id: int,
        last_yielded_pos: int,
    ) -> tuple[bool, Optional[tuple[mx.array, int, int, str]]]:
        """
        Check if we should early stop due to EOS token.

        Returns:
            (should_stop, yield_data): If should_stop is True, yield_data contains
                                       (new_tokens, block_start, block_end, finish_reason) or None
        """
        # Check if any newly updated tokens are EOS
        newly_updated = mx.where(update_mask, x0[0], mx.array(-1, dtype=x0.dtype))
        has_eos_in_update = is_eos_token(newly_updated, eos_token_ids).any()
        if not has_eos_in_update:
            return False, None

        # Check for any EOS token in the generated sequence
        combined_eos_mask = is_eos_token(cur_x[0, prompt_length:], eos_token_ids)
        if not combined_eos_mask.any():
            return False, None

        # Find first EOS position
        # Note: .tolist() is faster than mx.argmax for short sequences (< 2048 tokens)
        # which is typical for block-by-block generation
        eos_idx = combined_eos_mask.tolist().index(True)
        eos_pos = eos_idx + prompt_length

        if (cur_x[0, prompt_length:eos_pos] != mask_id).all():
            # Valid EOS found - prepare yield data (exclude EOS token)
            final_end = eos_pos
            if final_end > last_yielded_pos:
                new_tokens = cur_x[0, last_yielded_pos:final_end]
                block_start = last_yielded_pos - prompt_length
                block_end = final_end - prompt_length
                return True, (new_tokens, block_start, block_end, "stop")
            return True, None

        return False, None

    def _process_completed_block(
        self,
        x: mx.array,
        last_yielded_pos: int,
        gen_end: int,
        prompt_length: int,
        mask_id: int,
        eos_token_ids: set,
    ) -> tuple[Optional[tuple[mx.array, int, int]], int, bool, Optional[str]]:
        """
        Process completed block and determine what to yield.

        Returns:
            (yield_data, new_last_yielded_pos, should_break, finish_reason):
                - yield_data: (new_tokens, block_start, block_end) or None
                - new_last_yielded_pos: Updated yielded position
                - should_break: Whether to break from block loop
                - finish_reason: "stop" if EOS found, None otherwise
        """
        if gen_end <= last_yielded_pos:
            return None, last_yielded_pos, False, None

        new_tokens = x[0, last_yielded_pos:gen_end]

        # Don't yield if block is all mask tokens
        if (new_tokens == mask_id).all():
            return None, last_yielded_pos, False, None

        # Check for EOS tokens first
        # Note: .tolist().index() is faster than mx.argmax for short sequences
        # (typical block_length is 32-128 tokens)
        eos_mask = is_eos_token(new_tokens, eos_token_ids)
        if eos_mask.any():
            # Truncate at first EOS (exclusive - don't include EOS in output)
            first_eos_idx = eos_mask.tolist().index(True)
            new_tokens = new_tokens[:first_eos_idx]
            actual_end = last_yielded_pos + first_eos_idx
            block_start = last_yielded_pos - prompt_length
            block_end = actual_end - prompt_length
            return (new_tokens, block_start, block_end), actual_end, True, "stop"

        # Check for mask tokens
        mask_mask = new_tokens == mask_id
        if mask_mask.any():
            # Find first mask position
            first_mask_idx = mask_mask.tolist().index(True)
            if first_mask_idx > 0:
                # Yield only up to first mask
                new_tokens = new_tokens[:first_mask_idx]
                actual_end = last_yielded_pos + first_mask_idx
                block_start = last_yielded_pos - prompt_length
                block_end = actual_end - prompt_length
                return (new_tokens, block_start, block_end), actual_end, True, "stop"
            return None, last_yielded_pos, True, "stop"

        # No masks or EOS, yield entire block
        block_start = last_yielded_pos - prompt_length
        block_end = gen_end - prompt_length
        return (new_tokens, block_start, block_end), gen_end, False, None

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
        eos_token_ids: Optional[Union[int, list, set]] = None,
        mask_id: Optional[int] = None,
    ):
        """
        Stream diffusion generation with token-by-token yielding.

        This method yields tokens as they become confident during the denoising
        process, providing responsive incremental output.

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
            eos_token_ids: End-of-sequence token ID(s). Can be a single int, list, or set.
            mask_id: Mask token ID for ungenerated positions

        Yields:
            Tuple[mx.array, int, int, Optional[str]]: (new_tokens, block_start, block_end, finish_reason)
                - new_tokens: The newly completed tokens
                - block_start: Starting position (relative to generation start)
                - block_end: Ending position (relative to generation start)
                - finish_reason: "stop" if EOS found, None otherwise
        """
        # Convert to set for consistent handling
        if eos_token_ids is None:
            eos_token_ids = {self.args.eos_token_id}
        elif isinstance(eos_token_ids, int):
            eos_token_ids = {eos_token_ids}
        elif isinstance(eos_token_ids, list):
            eos_token_ids = set(eos_token_ids)
        else:
            eos_token_ids = set(eos_token_ids)

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
        block_diffusion_attention_mask = self._create_block_diagonal_mask(
            num_blocks, block_length
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

        # Pre-compute positions array for update mask creation (optimization)
        positions_array = mx.arange(block_length)

        # Process each block
        for num_block in range(prefill_blocks, num_blocks):
            current_window_end = (num_block + 1) * block_length
            cur_x = x[:, :current_window_end]
            cur_attn_mask = block_diffusion_attention_mask[
                :, :, :current_window_end, :current_window_end
            ]

            # Track which positions in current block have been yielded
            block_start_abs = num_block * block_length
            last_yielded_in_block = (
                last_yielded_pos - block_start_abs
                if last_yielded_pos > block_start_abs
                else 0
            )

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
                update_mask = self._select_tokens_to_update(
                    x0_p, active_block_mask, num_to_transfer, threshold, positions_array
                )

                # Update tokens in the current block
                if update_mask.any():
                    # Update only the last block_length positions
                    new_block = mx.where(update_mask, x0[0], cur_x[0, -block_length:])
                    cur_x = mx.concatenate(
                        [cur_x[:, :-block_length], mx.expand_dims(new_block, 0)],
                        axis=1,
                    )

                    # Yield newly confident tokens within this block
                    # Update global sequence
                    x[:, :current_window_end] = cur_x

                    # Extract tokens from current block
                    block_tokens = cur_x[0, -block_length:]

                    # Determine valid yield range using array operations
                    start_idx = max(
                        last_yielded_in_block, prompt_length - block_start_abs
                    )
                    if start_idx >= block_length:
                        continue  # Nothing to yield yet

                    # Create boolean masks for stop conditions
                    remaining_tokens = block_tokens[start_idx:]
                    is_mask = remaining_tokens == mask_id
                    is_eos = is_eos_token(remaining_tokens, eos_token_ids)

                    # Find first stop position (mask or EOS)
                    stop_mask = is_mask | is_eos
                    if stop_mask.any():
                        # Find first True using argmax (works because True=1, False=0)
                        stop_offset = mx.argmax(stop_mask.astype(mx.int32)).item()
                        end_idx = start_idx + stop_offset

                        # Check if we hit EOS
                        hit_eos = (
                            is_eos[stop_offset].item()
                            if stop_offset < len(is_eos)
                            else False
                        )
                    else:
                        # No stop conditions found
                        end_idx = block_length
                        hit_eos = False

                    # Yield tokens if we have any
                    if end_idx > start_idx:
                        tokens_slice = block_tokens[start_idx:end_idx]
                        gen_start = (block_start_abs + start_idx) - prompt_length
                        gen_end = gen_start + (end_idx - start_idx)
                        yield (tokens_slice, gen_start, gen_end, None)

                        last_yielded_pos = block_start_abs + end_idx
                        last_yielded_in_block = end_idx

                    # If we hit EOS, stop iterating (block completion will handle final yield)
                    if hit_eos:
                        break

                # Early stop on EOS
                if eos_early_stop:
                    should_stop, yield_data = self._check_early_stop_eos(
                        update_mask,
                        x0,
                        cur_x,
                        prompt_length,
                        eos_token_ids,
                        mask_id,
                        last_yielded_pos,
                    )
                    if should_stop:
                        if yield_data is not None:
                            yield yield_data
                        return

            # Yield completed block (excluding mask tokens and stopping at EOS)
            gen_end = min(current_window_end, prompt_length + gen_length)
            yield_data, last_yielded_pos, should_break, finish_reason = (
                self._process_completed_block(
                    x, last_yielded_pos, gen_end, prompt_length, mask_id, eos_token_ids
                )
            )

            if yield_data is not None:
                yield (*yield_data, finish_reason)

            if should_break:
                break

            # Stop if EOS found
            eos_found = is_eos_token(
                x[0, prompt_length:current_window_end], eos_token_ids
            ).any()
            if eos_found:
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
        eos_token_ids: Optional[Union[int, list, set]] = None,
        mask_id: Optional[int] = None,
    ) -> mx.array:
        """
        Diffusion-based generation using block-wise iterative denoising.

        This is a convenience wrapper around stream_generate() that collects
        all generated tokens and returns them at once.

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
            eos_token_ids: End-of-sequence token ID(s). Can be a single int, list, or set.
            mask_id: Mask token ID for ungenerated positions

        Returns:
            Generated token IDs (excluding prompt)
        """
        # Collect all blocks from streaming generator
        all_tokens = []
        for new_tokens, _, _, _ in self.stream_generate(
            inputs=inputs,
            temperature=temperature,
            block_length=block_length,
            steps=steps,
            gen_length=gen_length,
            top_p=top_p,
            top_k=top_k,
            eos_early_stop=eos_early_stop,
            minimal_topk=minimal_topk,
            threshold=threshold,
            eos_token_ids=eos_token_ids,
            mask_id=mask_id,
        ):
            all_tokens.append(new_tokens)

        # Concatenate all generated tokens
        if all_tokens:
            return mx.concatenate(all_tokens, axis=0)
        else:
            # No tokens generated
            return mx.array([], dtype=mx.int32)

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

        # Use argpartition with negative kth to avoid creating -logits copy
        mask_idx = mx.argpartition(logits, kth=-k, axis=-1)[..., :-k]
        return mx.put_along_axis(
            logits, mask_idx, mx.array(-float("inf"), logits.dtype), axis=-1
        )

    @staticmethod
    def _top_p_logits(logits: mx.array, p: Optional[float]) -> mx.array:
        """Apply nucleus (top-p) filtering to logits."""
        if p is None or p >= 1.0:
            return logits

        # Sort descending by negating
        sorted_indices = mx.argsort(-logits, axis=-1)
        sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)

        # Get cumulative probabilities in sorted (descending) order
        sorted_probs = mx.softmax(sorted_logits, axis=-1)
        cumsum_probs = mx.cumsum(sorted_probs, axis=-1)

        # Mask tokens where cumsum (before current token) > p
        # Subtract sorted_probs to get cumulative sum before adding current token
        remove_mask = (cumsum_probs - sorted_probs) > p
        sorted_logits_masked = mx.where(remove_mask, -float("inf"), sorted_logits)

        # Scatter back to original order
        return mx.put_along_axis(
            mx.full(logits.shape, -float("inf"), logits.dtype),
            sorted_indices,
            sorted_logits_masked,
            axis=-1,
        )

    def _sample_with_temperature_topk_topp(
        self,
        logits: mx.array,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> tuple[mx.array, mx.array]:
        """Sample tokens with temperature, top-k, and top-p filtering."""
        # Handle greedy decoding (temperature=0) early
        if temperature == 0.0:
            token = mx.argmax(logits, axis=-1)
            probs = mx.softmax(logits, axis=-1)
            token_prob = mx.take_along_axis(probs, mx.expand_dims(token, -1), axis=-1)
            return token, token_prob.squeeze(-1)

        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Apply top-k filtering
        if top_k is not None and top_k > 0:
            logits = self._top_k_logits(logits, top_k)

        # Apply top-p filtering
        if top_p is not None and top_p < 1.0:
            logits = self._top_p_logits(logits, top_p)

        # Sample from logits (categorical expects unnormalized log-probs)
        token = mx.random.categorical(logits, axis=-1)

        # Get token probabilities for confidence scoring
        probs = mx.softmax(logits, axis=-1)
        token_prob = mx.take_along_axis(probs, mx.expand_dims(token, -1), axis=-1)

        return token, token_prob.squeeze(-1)

    def sanitize(self, weights):
        """
        Convert HuggingFace weights to MLX format.

        Stacks individual expert weights into batched tensors for MoE layers.
        """
        # Stack routed expert weights for MoE layers
        for l in range(self.args.first_k_dense_replace, self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"
            for proj_type in ["gate_proj", "up_proj", "down_proj"]:
                expert_key = f"{prefix}.mlp.experts.0.{proj_type}.weight"
                if expert_key in weights:
                    # Stack all expert weights
                    to_join = [
                        weights.pop(f"{prefix}.mlp.experts.{e}.{proj_type}.weight")
                        for e in range(self.args.num_experts)
                    ]
                    weights[f"{prefix}.mlp.switch_mlp.{proj_type}.weight"] = mx.stack(
                        to_join
                    )

        return weights
