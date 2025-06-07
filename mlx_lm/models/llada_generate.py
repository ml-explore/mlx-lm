"""
LLaDA generation utilities for MLX
Based on the official PyTorch implementation
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Optional, Tuple


def add_gumbel_noise(logits: mx.array, temperature: float) -> mx.array:
    """
    Add Gumbel noise for sampling categorical distributions.
    According to the paper, low-precision Gumbel Max improves perplexity but reduces quality.
    """
    if temperature == 0:
        return logits
    
    # Generate uniform random noise
    noise = mx.random.uniform(logits.shape, dtype=mx.float32)
    noise = mx.maximum(noise, 1e-20)  # Avoid log(0)
    
    # Convert to Gumbel noise
    gumbel_noise = (-mx.log(noise)) ** temperature
    
    # Apply to logits
    return mx.exp(logits) / gumbel_noise


def get_num_transfer_tokens(mask_index: mx.array, steps: int) -> mx.array:
    """
    Compute number of tokens to transfer at each step.
    LLaDA uses a linear noise schedule, so tokens should be evenly distributed.
    """
    mask_num = mx.sum(mask_index, axis=1, keepdims=True)
    
    base = mask_num // steps
    remainder = mask_num % steps
    
    # Create array of shape (batch_size, steps)
    batch_size = mask_index.shape[0]
    num_transfer_tokens = mx.zeros((batch_size, steps), dtype=mx.int32) + base
    
    # Add remainders to first few steps
    for i in range(batch_size):
        if remainder[i] > 0:
            num_transfer_tokens[i, :remainder[i].item()] += 1
    
    return num_transfer_tokens


def llada_generate(
    model,
    prompt: mx.array,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = 'low_confidence',
    mask_id: int = 126336,
) -> mx.array:
    """
    Generate text using LLaDA's masked decoding approach.
    
    Args:
        model: The LLaDA model
        prompt: Input token IDs of shape (1, L)
        steps: Number of sampling steps (≤ gen_length)
        gen_length: Length of text to generate
        block_length: Block size for semi-autoregressive generation
        temperature: Sampling temperature
        cfg_scale: Classifier-free guidance scale
        remasking: Strategy for remasking ('low_confidence' or 'random')
        mask_id: Token ID for [MASK]
    
    Returns:
        Generated token IDs including the prompt
    """
    batch_size = prompt.shape[0]
    prompt_len = prompt.shape[1]
    total_len = prompt_len + gen_length
    
    # Initialize with masks
    x = mx.full((batch_size, total_len), mask_id, dtype=mx.int32)
    x[:, :prompt_len] = prompt
    
    prompt_index = (x != mask_id)
    
    # Ensure proper division
    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    num_blocks = gen_length // block_length
    
    assert steps % num_blocks == 0, "steps must be divisible by num_blocks"
    steps_per_block = steps // num_blocks
    
    for block_idx in range(num_blocks):
        block_start = prompt_len + block_idx * block_length
        block_end = prompt_len + (block_idx + 1) * block_length
        
        # Get mask indices for current block
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        
        for step in range(steps_per_block):
            mask_index = (x == mask_id)
            
            # Apply classifier-free guidance if needed
            if cfg_scale > 0:
                # Create unconditional input
                un_x = x.copy()
                un_x = mx.where(prompt_index, mask_id, un_x)
                
                # Concatenate conditional and unconditional
                x_combined = mx.concatenate([x, un_x], axis=0)
                logits = model(x_combined)
                
                # Split logits
                logits, un_logits = mx.split(logits, 2, axis=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x)
            
            # Add Gumbel noise and sample
            logits_with_noise = add_gumbel_noise(logits, temperature)
            x0 = mx.argmax(logits_with_noise, axis=-1)
            
            # Calculate confidence scores
            if remasking == 'low_confidence':
                # Get softmax probabilities
                probs = mx.softmax(logits, axis=-1)
                # Get probability of selected tokens
                x0_probs = mx.take_along_axis(probs, x0[:, :, None], axis=-1).squeeze(-1)
            elif remasking == 'random':
                x0_probs = mx.random.uniform(x0.shape, dtype=mx.float32)
            else:
                raise ValueError(f"Unknown remasking strategy: {remasking}")
            
            # Set confidence to -inf outside current block
            x0_probs = mx.where(
                mx.arange(total_len) < block_end,
                x0_probs,
                float('-inf')
            )
            
            # Only update masked positions
            x0 = mx.where(mask_index, x0, x)
            confidence = mx.where(mask_index, x0_probs, float('-inf'))
            
            # Select tokens to transfer based on confidence
            transfer_index = mx.zeros_like(x0, dtype=mx.bool_)
            
            for batch_idx in range(batch_size):
                k = num_transfer_tokens[batch_idx, step].item()
                if k > 0:
                    # Get top-k indices
                    batch_confidence = confidence[batch_idx]
                    # Sort in descending order and get indices
                    sorted_indices = mx.argsort(-batch_confidence)
                    top_k_indices = sorted_indices[:k]
                    
                    # Create transfer mask
                    batch_transfer = mx.zeros(total_len, dtype=mx.bool_)
                    batch_transfer[top_k_indices] = True
                    transfer_index[batch_idx] = batch_transfer
            
            # Transfer selected tokens
            x = mx.where(transfer_index, x0, x)
    
    return x