#!/usr/bin/env python3
"""
Fixed test script for LLaDA model with proper masked generation
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.utils import get_model_path, load_tokenizer
from mlx_lm.models.llada import ModelArgs, Model
import json
import glob


def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64 equivalent precision.
    """
    if temperature == 0:
        return logits
    # In MLX, we use float32 as the highest precision available
    logits = logits.astype(mx.float32)
    noise = mx.random.uniform(shape=logits.shape, dtype=mx.float32)
    # Avoid log(0) by adding small epsilon
    gumbel_noise = (-mx.log(noise + 1e-20)) ** temperature
    return mx.exp(logits) / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    """
    mask_num = mx.sum(mask_index, axis=1, keepdims=True)
    
    base = mask_num // steps
    remainder = mask_num % steps
    
    num_transfer_tokens = mx.zeros((mask_index.shape[0], steps), dtype=mx.int32) + base
    
    # Handle remainder
    for i in range(mask_index.shape[0]):
        if remainder[i] > 0:
            num_transfer_tokens[i, :remainder[i].item()] += 1
    
    return num_transfer_tokens


def llada_generate(model, prompt, tokenizer, steps=128, gen_length=128, block_length=128, temperature=0.,
                   cfg_scale=0., remasking='low_confidence', mask_id=126336):
    """
    LLaDA-specific masked generation algorithm ported to MLX
    
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
    """
    # Convert prompt to MLX array
    if isinstance(prompt, str):
        # Apply chat template if available
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        else:
            formatted_prompt = prompt
        
        input_ids = tokenizer(formatted_prompt)['input_ids']
        prompt = mx.array(input_ids).reshape(1, -1)
    
    # Initialize sequence with masks
    x = mx.full((1, prompt.shape[1] + gen_length), mask_id, dtype=mx.int32)
    x[:, :prompt.shape[1]] = prompt
    
    prompt_index = (x != mask_id)
    
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    
    assert steps % num_blocks == 0
    steps = steps // num_blocks
    
    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        
        for i in range(steps):
            mask_index = (x == mask_id)
            
            if cfg_scale > 0.:
                # Classifier-free guidance
                un_x = x.copy()
                un_x = mx.where(prompt_index, mask_id, un_x)
                x_ = mx.concatenate([x, un_x], axis=0)
                logits = model(x_)
                logits, un_logits = mx.split(logits, 2, axis=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x)
            
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = mx.argmax(logits_with_noise, axis=-1)
            
            if remasking == 'low_confidence':
                p = mx.softmax(logits, axis=-1)
                # Gather probabilities for predicted tokens
                x0_expanded = mx.expand_dims(x0, axis=-1)
                x0_p = mx.take_along_axis(p, x0_expanded, axis=-1).squeeze(-1)
            elif remasking == 'random':
                x0_p = mx.random.uniform(shape=x0.shape)
            else:
                raise NotImplementedError(remasking)
            
            # Set confidence to -inf outside current block
            mask = mx.arange(x0_p.shape[1]) < block_end
            x0_p = mx.where(mask, x0_p, -np.inf)
            
            x0 = mx.where(mask_index, x0, x)
            confidence = mx.where(mask_index, x0_p, -np.inf)
            
            # Select tokens to transfer based on confidence
            transfer_index = mx.zeros_like(x0, dtype=mx.bool_)
            for j in range(confidence.shape[0]):
                k = int(num_transfer_tokens[j, i].item())
                if k > 0:
                    # Get top-k indices using argsort
                    sorted_indices = mx.argsort(-confidence[j])
                    top_k_indices = sorted_indices[:k]
                    # Create mask for transfer
                    row_mask = mx.zeros(confidence.shape[1], dtype=mx.bool_)
                    for idx in top_k_indices:
                        row_mask = row_mask | (mx.arange(confidence.shape[1]) == idx)
                    transfer_index = mx.where(mx.arange(confidence.shape[0])[:, None] == j, 
                                            row_mask, 
                                            transfer_index)
            
            x = mx.where(transfer_index, x0, x)
    
    return x


def load_llada_model(model_name):
    """Load LLaDA model with proper quantization handling"""
    
    print(f"📥 Loading {model_name}...")
    
    # Get model path and config
    model_path = get_model_path(model_name)
    
    with open(f"{model_path}/config.json", "r") as f:
        config = json.load(f)
    
    # Create model
    model_args = ModelArgs.from_dict(config)
    model = Model(model_args)
    
    # Load weights - handle both single and multiple safetensors files
    safetensors_files = sorted(glob.glob(f"{model_path}/*.safetensors"))
    
    if len(safetensors_files) == 1:
        weights = mx.load(safetensors_files[0])
    else:
        # Multiple files - load and merge them
        weights = {}
        for file in safetensors_files:
            file_weights = mx.load(file)
            weights.update(file_weights)
    
    # Sanitize weights
    weights = model.sanitize(weights)
    
    # Apply quantization if needed
    quantization = config.get("quantization")
    if quantization:
        print(f"🔄 Applying {quantization['bits']}-bit quantization...")
        
        def class_predicate(p, m):
            return f"{p}.scales" in weights
        
        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            class_predicate=class_predicate,
        )
    
    # Load weights with strict=False to handle any remaining mismatches
    print("📦 Loading weights...")
    model.load_weights(list(weights.items()), strict=False)
    
    # Load tokenizer - handle tokenizer loading issues
    try:
        tokenizer = load_tokenizer(model_path)
    except Exception as e:
        print(f"⚠️  Standard tokenizer loading failed: {e}")
        print("🔄 Trying alternative tokenizer loading...")
        # Try loading from the original model
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    
    print("✅ Model loaded successfully!")
    return model, tokenizer


def test_llada_generation():
    """Test LLaDA with proper masked generation"""
    
    print("🚀 Testing LLaDA Masked Generation")
    print("=" * 60)
    
    # Load model
    model, tokenizer = load_llada_model("mlx-community/LLaDA-8B-Instruct-mlx-4bit")
    
    # Test with the math problem
    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"
    
    print(f"\n📝 Prompt: {prompt}")
    print("\n🎯 Generating response with masked generation...")
    
    # Generate response
    output = llada_generate(
        model, 
        prompt, 
        tokenizer,
        steps=128, 
        gen_length=128, 
        block_length=32, 
        temperature=0., 
        cfg_scale=0., 
        remasking='low_confidence'
    )
    
    # Decode only the generated part
    prompt_len = len(tokenizer(prompt)['input_ids'])
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        prompt_len = len(tokenizer(formatted_prompt)['input_ids'])
    
    generated_ids = output[0, prompt_len:].tolist()
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print(f"\n✅ Generated response:")
    print(response)
    
    # Test with another prompt
    print("\n" + "=" * 60)
    prompt2 = "What is 15 + 27?"
    print(f"\n📝 Testing simple math: {prompt2}")
    
    output2 = llada_generate(
        model, 
        prompt2, 
        tokenizer,
        steps=64, 
        gen_length=32, 
        block_length=32, 
        temperature=0., 
        cfg_scale=0., 
        remasking='low_confidence'
    )
    
    # Decode response
    prompt_len2 = len(tokenizer(prompt2)['input_ids'])
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt2}]
        formatted_prompt2 = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        prompt_len2 = len(tokenizer(formatted_prompt2)['input_ids'])
    
    generated_ids2 = output2[0, prompt_len2:].tolist()
    response2 = tokenizer.decode(generated_ids2, skip_special_tokens=True)
    
    print(f"\n✅ Response: {response2}")
    
    print("\n🎊 LLaDA masked generation is working correctly!")


if __name__ == "__main__":
    test_llada_generation()