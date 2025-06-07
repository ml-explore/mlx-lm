#!/usr/bin/env python3
"""
Final working test script for LLaDA with proper MLX operations
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.utils import get_model_path
from mlx_lm.models.llada import ModelArgs, Model
from transformers import AutoTokenizer
import json
import glob


def load_llada_model_and_tokenizer(model_name):
    """Load LLaDA model and tokenizer"""
    
    print(f"📥 Loading {model_name}...")
    
    # Get model path and config
    model_path = get_model_path(model_name)
    
    with open(f"{model_path}/config.json", "r") as f:
        config = json.load(f)
    
    # Create model
    model_args = ModelArgs.from_dict(config)
    model = Model(model_args)
    
    # Load weights
    safetensors_files = sorted(glob.glob(f"{model_path}/*.safetensors"))
    
    if len(safetensors_files) == 1:
        weights = mx.load(safetensors_files[0])
    else:
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
    
    # Load weights
    print("📦 Loading weights...")
    model.load_weights(list(weights.items()), strict=False)
    
    # Load tokenizer from original HF model
    print("🔄 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    
    print("✅ Model and tokenizer loaded!")
    return model, tokenizer


def llada_generate_mlx(model, tokenizer, prompt, steps=32, gen_length=50, temperature=0.0, mask_id=126336):
    """
    LLaDA generation adapted for MLX
    Simplified version that focuses on core functionality
    """
    
    # Format prompt with chat template
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    else:
        formatted_prompt = prompt
    
    # Tokenize
    input_ids = tokenizer(formatted_prompt, return_tensors='np')['input_ids'][0]
    prompt_len = len(input_ids)
    
    # Initialize sequence with masks
    total_len = prompt_len + gen_length
    sequence = np.full(total_len, mask_id, dtype=np.int32)
    sequence[:prompt_len] = input_ids
    x = mx.array(sequence).reshape(1, -1)
    
    # Generate tokens iteratively
    num_masks = gen_length
    tokens_per_step = max(1, num_masks // steps)
    
    for step in range(steps):
        # Get model predictions
        logits = model(x)
        
        # Find masked positions
        mask_condition = (x[0] == mask_id)
        
        # Get predictions at all positions
        if temperature == 0:
            predictions = mx.argmax(logits[0], axis=-1)
        else:
            # Add temperature-based sampling
            probs = mx.softmax(logits[0] / temperature, axis=-1)
            predictions = mx.argmax(probs, axis=-1)
        
        # Calculate confidence scores (using softmax probabilities)
        probs = mx.softmax(logits[0], axis=-1)
        max_probs = mx.max(probs, axis=-1)
        
        # Only consider masked positions
        confidence_at_masks = mx.where(mask_condition, max_probs, -1e10)
        
        # Find positions to unmask in this step
        # Convert to numpy for easier manipulation
        confidence_np = np.array(confidence_at_masks)
        mask_np = np.array(mask_condition)
        predictions_np = np.array(predictions)
        x_np = np.array(x)
        
        # Get indices of masked positions sorted by confidence
        masked_indices = np.where(mask_np)[0]
        if len(masked_indices) == 0:
            break
            
        # Sort by confidence and take top tokens_per_step
        confidences_at_masked = confidence_np[masked_indices]
        sorted_idx = np.argsort(-confidences_at_masked)
        
        # Unmask top confident predictions
        num_to_unmask = min(tokens_per_step, len(masked_indices))
        positions_to_unmask = masked_indices[sorted_idx[:num_to_unmask]]
        
        # Update sequence
        for pos in positions_to_unmask:
            x_np[0, pos] = predictions_np[pos]
        
        # Convert back to MLX
        x = mx.array(x_np)
    
    # Extract generated text
    output_ids = x[0, prompt_len:].tolist()
    # Remove remaining masks and special tokens
    output_ids = [t for t in output_ids if t != mask_id and t < tokenizer.vocab_size]
    
    return tokenizer.decode(output_ids, skip_special_tokens=True)


def test_llada_model():
    """Test the LLaDA model with various prompts"""
    
    print("🚀 Testing LLaDA Model with MLX")
    print("=" * 60)
    
    # Load model and tokenizer
    model, tokenizer = load_llada_model_and_tokenizer("mlx-community/LLaDA-8B-Instruct-mlx-4bit")
    
    # Test with the math problem that worked in PyTorch
    print("\n📝 Testing with math problem:")
    math_prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"
    
    print(f"Prompt: {math_prompt}")
    print("\n🎯 Generating response...")
    
    response = llada_generate_mlx(
        model, 
        tokenizer, 
        math_prompt,
        steps=128,
        gen_length=128,
        temperature=0.0
    )
    
    print(f"\n✅ Generated response:")
    print(response)
    
    # Test with simpler prompts
    print("\n" + "=" * 60)
    print("📝 Testing with simple math:")
    
    simple_prompts = [
        "What is 15 + 27?",
        "Calculate: 100 - 37 =",
        "What is 2 + 2?"
    ]
    
    for prompt in simple_prompts:
        print(f"\nPrompt: {prompt}")
        response = llada_generate_mlx(
            model, 
            tokenizer, 
            prompt,
            steps=16,
            gen_length=20,
            temperature=0.0
        )
        print(f"Response: {response}")
    
    print("\n🎊 Testing complete!")


if __name__ == "__main__":
    test_llada_model()