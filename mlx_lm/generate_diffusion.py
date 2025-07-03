# Copyright © 2025 Apple Inc.

import time
from dataclasses import dataclass
from typing import Any, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from transformers import PreTrainedTokenizer

from .models import cache
from .tokenizer_utils import TokenizerWrapper
from .utils import load


@dataclass
class DreamGenerationConfig:
    """Configuration for Dream diffusion generation."""
    temperature: float = 0.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_length: int = 20
    max_new_tokens: Optional[int] = None
    
    # Diffusion specific params
    eps: float = 1e-3
    steps: int = 32
    alg: str = 'origin'  # 'origin', 'maskgit_plus', 'topk_margin', 'entropy'
    alg_temp: Optional[float] = None
    
    # Special tokens
    mask_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    
    # Output control
    num_return_sequences: int = 1
    return_dict_in_generate: bool = False
    output_history: bool = False


@dataclass
class DreamModelOutput:
    """Output from Dream diffusion generation."""
    sequences: mx.array
    history: Optional[List[mx.array]] = None


@dataclass
class DreamGenerationResponse:
    """Response from Dream generation."""
    text: str
    sequences: mx.array
    prompt_tokens: int
    prompt_tps: float
    generation_tokens: int
    generation_tps: float
    peak_memory: float
    finish_reason: Optional[str] = None
    history: Optional[List[mx.array]] = None


def top_p_logits(logits: mx.array, top_p: float) -> mx.array:
    """Apply top-p filtering to logits."""
    sorted_logits = mx.sort(logits, axis=-1)[:, ::-1]  # Sort descending
    sorted_indices = mx.argsort(logits, axis=-1)[:, ::-1]
    
    cumulative_probs = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    
    # Shift indices to keep first token above threshold
    sorted_indices_to_remove = mx.concatenate([
        mx.zeros_like(sorted_indices_to_remove[:, :1]),
        sorted_indices_to_remove[:, :-1]
    ], axis=-1)
    
    # Create mask for original indices
    mask = mx.zeros_like(logits, dtype=mx.bool_)
    batch_indices = mx.arange(logits.shape[0])[:, None]
    # Note: This also needs to be fixed for MLX, but keeping for now
    # mask = mask.at[batch_indices, sorted_indices].set(sorted_indices_to_remove)
    
    return mx.where(mask, -mx.inf, logits)


def top_k_logits(logits: mx.array, top_k: int) -> mx.array:
    """Apply top-k filtering to logits."""
    top_k = min(top_k, logits.shape[-1])
    kth_largest = mx.sort(logits, axis=-1)[:, -top_k][:, None]
    return mx.where(logits < kth_largest, -mx.inf, logits)


def sample_tokens(
    logits: mx.array,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    margin_confidence: bool = False,
    neg_entropy: bool = False,
    key: Optional[mx.array] = None
) -> Tuple[mx.array, mx.array]:
    """Sample tokens from logits with various confidence measures."""
    
    if temperature > 0:
        logits = logits / temperature
    
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    
    probs = mx.softmax(logits, axis=-1)
    
    if temperature > 0:
        if key is None:
            key = mx.random.key(int(time.time()))
        x0 = mx.random.categorical(logits, axis=-1, key=key)
        confidence = probs[mx.arange(probs.shape[0]), x0]
    else:
        confidence = mx.max(probs, axis=-1)
        x0 = mx.argmax(probs, axis=-1)
    
    if margin_confidence:
        sorted_probs = mx.sort(probs, axis=-1)[:, ::-1]  # Descending
        top1_probs = sorted_probs[:, 0]
        top2_probs = sorted_probs[:, 1]
        confidence = top1_probs - top2_probs
    
    if neg_entropy:
        epsilon = 1e-10
        log_probs = mx.log(probs + epsilon)
        confidence = mx.sum(probs * log_probs, axis=-1)
    
    return confidence, x0


def diffusion_generate_step(
    prompt: mx.array,
    model: nn.Module,
    generation_config: DreamGenerationConfig,
    prompt_cache: Optional[Any] = None,
) -> Union[DreamModelOutput, mx.array]:
    """
    Generate text using diffusion sampling.
    """
    
    # Extract config parameters
    max_length = generation_config.max_length
    mask_token_id = generation_config.mask_token_id
    steps = generation_config.steps
    eps = generation_config.eps
    alg = generation_config.alg
    alg_temp = generation_config.alg_temp
    temperature = generation_config.temperature
    top_p = generation_config.top_p
    top_k = generation_config.top_k
    output_history = generation_config.output_history
    return_dict = generation_config.return_dict_in_generate
    
    batch_size = prompt.shape[0]
    
    # Pad input to max_length with mask tokens
    pad_length = max_length - prompt.shape[1]
    if pad_length > 0:
        mask_tokens = mx.full((batch_size, pad_length), mask_token_id, dtype=prompt.dtype)
        x = mx.concatenate([prompt, mask_tokens], axis=1)
    else:
        x = prompt[:, :max_length]
    
    # Create timestep schedule
    timesteps = mx.linspace(1, eps, steps + 1)
    
    # Store history if requested
    histories = [] if output_history else None
    
    # Initialize random key
    base_key = mx.random.key(int(time.time() * 1000))
    
    # Diffusion sampling loop
    for i in range(steps):
        # Find masked positions
        mask_index = (x == mask_token_id)
        
        # Check if there are any masked tokens left
        if not mx.any(mask_index):
            break  # No more masked tokens
        
        # Forward pass through model
        logits = model(x, cache=prompt_cache)
        
        # Shift logits to align with tokens (if needed)
        if logits.shape[1] == x.shape[1] + 1:
            logits = logits[:, :-1]  # Remove last position
        
        # Get current timestep values
        t = timesteps[i]
        s = timesteps[i + 1]
        
        # Generate unique key for this step
        step_key = mx.random.split(base_key, steps)[i]
        
        # Process each position individually
        seq_len = x.shape[1]
        
        # Convert to list for modification (MLX doesn't have .at[].set())
        x_list = x.tolist()
        
        for batch_idx in range(batch_size):
            for pos_idx in range(seq_len):
                # Check if this position is masked
                if x_list[batch_idx][pos_idx] == mask_token_id:
                    # Generate unique keys for this position
                    pos_keys = mx.random.split(step_key, batch_size * seq_len * 2)
                    key_idx = (batch_idx * seq_len + pos_idx) * 2
                    sample_key = pos_keys[key_idx]
                    transfer_key = pos_keys[key_idx + 1]
                    
                    # Get logits for this position
                    pos_logits = logits[batch_idx, pos_idx:pos_idx+1]  # Keep batch dimension
                    
                    if alg == 'origin':
                        # Original algorithm: probabilistic transfer
                        p_transfer = 1 - s / t if i < steps - 1 else 1
                        
                        # Sample new token
                        _, x0 = sample_tokens(
                            pos_logits, 
                            temperature=temperature, 
                            top_p=top_p, 
                            top_k=top_k,
                            key=sample_key
                        )
                        
                        # Generate random number for transfer decision
                        transfer_prob_array = mx.random.uniform(low=0.0, high=1.0, shape=(1,), key=transfer_key)
                        transfer_prob = transfer_prob_array[0]
                        
                        if transfer_prob < p_transfer:
                            # Transfer the token (modify list)
                            x_list[batch_idx][pos_idx] = int(x0[0])
                        
                    else:
                        # Confidence-based algorithms
                        if alg == 'maskgit_plus':
                            confidence, x0 = sample_tokens(
                                pos_logits, 
                                temperature=temperature, 
                                top_p=top_p, 
                                top_k=top_k,
                                key=sample_key
                            )
                        elif alg == 'topk_margin':
                            confidence, x0 = sample_tokens(
                                pos_logits, 
                                temperature=temperature, 
                                top_p=top_p, 
                                top_k=top_k,
                                margin_confidence=True,
                                key=sample_key
                            )
                        elif alg == 'entropy':
                            confidence, x0 = sample_tokens(
                                pos_logits, 
                                temperature=temperature, 
                                top_p=top_p, 
                                top_k=top_k,
                                neg_entropy=True,
                                key=sample_key
                            )
                        else:
                            raise ValueError(f"Unknown algorithm: {alg}")
                        
                        # Calculate transfer probability based on confidence
                        transfer_ratio = 1 - s / t if i < steps - 1 else 1
                        
                        # Use confidence as a probability for transfer
                        if alg_temp is None or alg_temp == 0:
                            # Deterministic: transfer if confidence is high enough
                            conf_threshold = 1.0 - transfer_ratio
                            if confidence[0] > conf_threshold:
                                x_list[batch_idx][pos_idx] = int(x0[0])
                        else:
                            # Stochastic: use confidence-weighted probability
                            conf_prob = mx.softmax(confidence / alg_temp)
                            transfer_sample = mx.random.uniform(low=0.0, high=1.0, shape=(), key=transfer_key)
                            
                            if transfer_sample < (transfer_ratio * conf_prob[0]):
                                x_list[batch_idx][pos_idx] = int(x0[0])
        
        # Convert back to mx.array
        x = mx.array(x_list, dtype=x.dtype)
        
        # Store history
        if histories is not None:
            histories.append(x.copy())
    
    if return_dict:
        return DreamModelOutput(sequences=x, history=histories)
    else:
        return x


def stream_diffusion_generate(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: Union[str, mx.array, List[int]],
    generation_config: Optional[DreamGenerationConfig] = None,
    **kwargs,
) -> Generator[DreamGenerationResponse, None, None]:
    """
    Generate text using Dream's diffusion process.
    
    Args:
        model: The Dream model
        tokenizer: Tokenizer for encoding/decoding
        prompt: Input prompt
        generation_config: Generation configuration
        **kwargs: Additional generation parameters
        
    Yields:
        DreamGenerationResponse with generated text and metadata
    """
    
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)
    
    # Prepare generation config
    if generation_config is None:
        generation_config = DreamGenerationConfig()
    
    # Update config with kwargs
    for key, value in kwargs.items():
        if hasattr(generation_config, key):
            setattr(generation_config, key, value)
    
    # Encode prompt
    if not isinstance(prompt, mx.array):
        if isinstance(prompt, str):
            add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(tokenizer.bos_token)
            prompt = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        prompt = mx.array(prompt)[None, :]  # Add batch dimension
    
    # Set up special tokens
    if generation_config.mask_token_id is None:
        generation_config.mask_token_id = tokenizer.mask_token_id or tokenizer.unk_token_id
    
    # Prepare max length
    input_length = prompt.shape[1]
    if generation_config.max_new_tokens is not None:
        generation_config.max_length = generation_config.max_new_tokens + input_length
    
    # Generate
    tic = time.perf_counter()
    
    # Create prompt cache
    prompt_cache = cache.make_prompt_cache(model)
    
    result = diffusion_generate_step(
        prompt=prompt,
        model=model,
        generation_config=generation_config,
        prompt_cache=prompt_cache,
    )
    
    generation_time = time.perf_counter() - tic
    
    # Extract sequences and history
    if isinstance(result, DreamModelOutput):
        sequences = result.sequences
        history = result.history
    else:
        sequences = result
        history = None
    
    # Decode generated text
    generated_tokens = sequences[0, input_length:]  # Remove prompt tokens
    generated_text = tokenizer.decode(generated_tokens.tolist(), skip_special_tokens=True)
    
    # Calculate metrics
    prompt_tps = input_length / generation_time if generation_time > 0 else 0
    generation_tokens = generated_tokens.shape[0]
    generation_tps = generation_tokens / generation_time if generation_time > 0 else 0
    
    yield DreamGenerationResponse(
        text=generated_text,
        sequences=sequences,
        prompt_tokens=input_length,
        prompt_tps=prompt_tps,
        generation_tokens=generation_tokens,
        generation_tps=generation_tps,
        peak_memory=mx.get_peak_memory() / 1e9,
        finish_reason="length",
        history=history,
    )


def diffusion_generate(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: Union[str, List[int]],
    verbose: bool = False,
    **kwargs,
) -> str:
    """
    Generate a complete response using Dream's diffusion process.
    
    Args:
        model: The Dream model
        tokenizer: Tokenizer for encoding/decoding
        prompt: Input prompt
        verbose: Whether to print timing information
        **kwargs: Generation parameters
        
    Returns:
        Generated text string
    """
    
    if verbose:
        print("=" * 10)
        print("Diffusion Generation")
        print("=" * 10)
    
    text = ""
    response = None
    
    for response in stream_diffusion_generate(model, tokenizer, prompt, **kwargs):
        text = response.text
        if verbose:
            print(f"Generated: {text}")
    
    if verbose and response:
        print("=" * 10)
        print(f"Prompt: {response.prompt_tokens} tokens")
        print(f"Generation: {response.generation_tokens} tokens")
        print(f"Generation TPS: {response.generation_tps:.3f}")
        print(f"Peak memory: {response.peak_memory:.3f} GB")
        if response.history:
            print(f"Diffusion steps: {len(response.history)}")
    
    return text

if __name__ == "__main__":
    model, tokenizer = load("/Users/gokdenizgulmez/Desktop/dream_grpo-4bit")

    response = diffusion_generate(model, tokenizer, "Write a quick sort in c++")

    print(response)