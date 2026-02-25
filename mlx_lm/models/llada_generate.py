"""
LLaDA (Large Language Diffusion with mAsking) generation implementation.

LLaDA uses a masked diffusion process for text generation, which differs from
traditional autoregressive generation. This module implements the iterative
unmasking algorithm described in the LLaDA paper.

Reference: https://github.com/ML-GSAI/LLaDA
"""

import mlx.core as mx
import mlx.nn as nn


def add_gumbel_noise(logits: mx.array, temperature: float) -> mx.array:
    """
    Add Gumbel noise for sampling from categorical distributions.

    The Gumbel-max trick provides a way to sample from a categorical distribution.
    According to arXiv:2409.02908, for MDM, using higher precision improves
    generation quality.

    Args:
        logits: Model output logits of shape (batch, seq_len, vocab_size)
        temperature: Sampling temperature. 0 means deterministic (no noise).

    Returns:
        Noisy logits for sampling
    """
    if temperature == 0:
        return logits

    noise = mx.random.uniform(shape=logits.shape)
    # Avoid log(0) by clamping
    noise = mx.clip(noise, 1e-10, 1.0)
    gumbel_noise = (-mx.log(noise)) ** temperature
    return mx.exp(logits.astype(mx.float32)) / gumbel_noise


def get_num_transfer_tokens(mask_index: mx.array, steps: int) -> mx.array:
    """
    Compute the number of tokens to unmask at each step.

    LLaDA uses a linear noise schedule, so the number of tokens transitioned
    at each step should be roughly consistent.

    Args:
        mask_index: Boolean array indicating masked positions (batch, seq_len)
        steps: Number of denoising steps

    Returns:
        Array of shape (batch, steps) with number of tokens to unmask per step
    """
    mask_num = mask_index.sum(axis=1, keepdims=True)  # (batch, 1)

    base = mask_num // steps
    remainder = mask_num % steps

    # Create base allocation
    num_transfer_tokens = mx.zeros((mask_num.shape[0], steps), dtype=mx.int32) + base

    # Distribute remainder across first steps
    for i in range(mask_num.shape[0]):
        rem = int(remainder[i, 0].item())
        if rem > 0:
            # Add 1 to the first 'remainder' steps
            update = mx.concatenate([
                mx.ones((rem,), dtype=mx.int32),
                mx.zeros((steps - rem,), dtype=mx.int32)
            ])
            num_transfer_tokens = num_transfer_tokens.at[i].add(update)

    return num_transfer_tokens


def generate(
    model,
    prompt: mx.array,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
) -> mx.array:
    """
    Generate text using LLaDA's masked diffusion process.

    Args:
        model: LLaDA model instance
        prompt: Input token IDs of shape (batch, prompt_len)
        steps: Total number of denoising steps
        gen_length: Number of tokens to generate
        block_length: Block length for semi-autoregressive generation.
                     If less than gen_length, generates in multiple blocks.
        temperature: Sampling temperature for Gumbel noise. 0 = deterministic.
        cfg_scale: Classifier-free guidance scale. 0 = no guidance.
        remasking: Strategy for remasking. "low_confidence" or "random".
        mask_id: Token ID for [MASK] token (default 126336 for LLaDA)

    Returns:
        Generated token IDs of shape (batch, prompt_len + gen_length)
    """
    batch_size, prompt_len = prompt.shape
    total_len = prompt_len + gen_length

    # Initialize with mask tokens, then copy prompt
    x = mx.full((batch_size, total_len), mask_id, dtype=mx.int32)
    x = mx.concatenate([prompt, x[:, prompt_len:]], axis=1)

    # Track which positions are from the prompt (should not be modified)
    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0, "steps must be divisible by num_blocks"
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        block_start = prompt_len + num_block * block_length
        block_end = prompt_len + (num_block + 1) * block_length

        # Get mask positions within current block
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = (x == mask_id)

            # Get model predictions
            if cfg_scale > 0.0:
                # Classifier-free guidance: run with and without prompt
                un_x = mx.where(prompt_index, mask_id, x)
                x_combined = mx.concatenate([x, un_x], axis=0)
                logits = model(x_combined)
                logits, un_logits = mx.split(logits, 2, axis=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x)

            # Add Gumbel noise and sample
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = mx.argmax(logits_with_noise, axis=-1)  # (batch, seq_len)

            # Compute confidence scores
            if remasking == "low_confidence":
                p = mx.softmax(logits, axis=-1)
                # Gather probabilities of selected tokens
                x0_expanded = mx.expand_dims(x0, axis=-1)
                x0_p = mx.take_along_axis(p, x0_expanded, axis=-1)
                x0_p = mx.squeeze(x0_p, axis=-1)  # (batch, seq_len)
            elif remasking == "random":
                x0_p = mx.random.uniform(shape=(batch_size, total_len))
            else:
                raise ValueError(f"Unknown remasking strategy: {remasking}")

            # Don't consider positions beyond current block
            x0_p = mx.where(
                mx.arange(total_len) < block_end,
                x0_p,
                mx.array(float("-inf"))
            )

            # Only update masked positions
            x0 = mx.where(mask_index, x0, x)
            confidence = mx.where(mask_index, x0_p, mx.array(float("-inf")))

            # Select top-k tokens to unmask based on confidence
            transfer_index = mx.zeros(x.shape, dtype=mx.bool_)
            for j in range(batch_size):
                k = int(num_transfer_tokens[j, i].item())
                if k > 0:
                    # Get indices of top-k confident positions
                    conf_j = confidence[j]
                    # Use negative confidence for argsort (descending order)
                    sorted_indices = mx.argsort(-conf_j)
                    top_k_indices = sorted_indices[:k]

                    # Create mask for selected positions
                    mask_j = mx.zeros((total_len,), dtype=mx.bool_)
                    mask_j = mask_j.at[top_k_indices].add(mx.ones((k,), dtype=mx.bool_))
                    transfer_index = transfer_index.at[j].add(mask_j)

            # Update x with selected tokens
            x = mx.where(transfer_index, x0, x)
            mx.eval(x)  # Evaluate to avoid graph buildup

    return x


def stream_generate(
    model,
    tokenizer,
    prompt: str,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 32,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
):
    """
    Generate text with streaming output per block.

    This is a convenience function that yields generated text after each block
    is completed, providing a streaming-like experience.

    Args:
        model: LLaDA model instance
        tokenizer: Tokenizer instance
        prompt: Input text prompt
        steps: Total number of denoising steps
        gen_length: Number of tokens to generate
        block_length: Block length for semi-autoregressive generation
        temperature: Sampling temperature
        cfg_scale: Classifier-free guidance scale
        remasking: Remasking strategy

    Yields:
        Generated text after each block
    """
    mask_id = getattr(model.args, "mask_token_id", 126336)

    # Tokenize prompt
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    else:
        prompt_text = prompt

    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_tokens = mx.array([input_ids], dtype=mx.int32)

    batch_size, prompt_len = prompt_tokens.shape
    total_len = prompt_len + gen_length

    # Initialize with mask tokens
    x = mx.full((batch_size, total_len), mask_id, dtype=mx.int32)
    x = mx.concatenate([prompt_tokens, x[:, prompt_len:]], axis=1)

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        block_start = prompt_len + num_block * block_length
        block_end = prompt_len + (num_block + 1) * block_length

        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = (x == mask_id)

            if cfg_scale > 0.0:
                un_x = mx.where(prompt_index, mask_id, x)
                x_combined = mx.concatenate([x, un_x], axis=0)
                logits = model(x_combined)
                logits, un_logits = mx.split(logits, 2, axis=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x)

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = mx.argmax(logits_with_noise, axis=-1)

            if remasking == "low_confidence":
                p = mx.softmax(logits, axis=-1)
                x0_expanded = mx.expand_dims(x0, axis=-1)
                x0_p = mx.take_along_axis(p, x0_expanded, axis=-1)
                x0_p = mx.squeeze(x0_p, axis=-1)
            elif remasking == "random":
                x0_p = mx.random.uniform(shape=(batch_size, total_len))
            else:
                raise ValueError(f"Unknown remasking strategy: {remasking}")

            x0_p = mx.where(
                mx.arange(total_len) < block_end,
                x0_p,
                mx.array(float("-inf"))
            )

            x0 = mx.where(mask_index, x0, x)
            confidence = mx.where(mask_index, x0_p, mx.array(float("-inf")))

            transfer_index = mx.zeros(x.shape, dtype=mx.bool_)
            for j in range(batch_size):
                k = int(num_transfer_tokens[j, i].item())
                if k > 0:
                    conf_j = confidence[j]
                    sorted_indices = mx.argsort(-conf_j)
                    top_k_indices = sorted_indices[:k]

                    mask_j = mx.zeros((total_len,), dtype=mx.bool_)
                    mask_j = mask_j.at[top_k_indices].add(mx.ones((k,), dtype=mx.bool_))
                    transfer_index = transfer_index.at[j].add(mask_j)

            x = mx.where(transfer_index, x0, x)
            mx.eval(x)

        # Yield text generated so far (after this block)
        generated_tokens = x[0, prompt_len:block_end].tolist()
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        yield generated_text
