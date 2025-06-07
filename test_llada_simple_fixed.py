#!/usr/bin/env python3
"""
Simplified test script for LLaDA with masked generation
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.utils import get_model_path, load_tokenizer
from mlx_lm.models.llada import ModelArgs, Model
from transformers import AutoTokenizer
import json
import glob


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
    
    # Load tokenizer - fallback to original HF model if needed
    try:
        tokenizer = load_tokenizer(model_path)
    except:
        print("🔄 Loading tokenizer from original HF model...")
        tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    
    print("✅ Model loaded successfully!")
    return model, tokenizer


def simple_llada_generate(model, tokenizer, prompt, max_length=50, mask_id=126336):
    """
    Simplified LLaDA generation for testing
    Uses basic iterative mask prediction
    """
    
    # Tokenize prompt
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    else:
        formatted_prompt = prompt
    
    input_ids = tokenizer(formatted_prompt, return_tensors='np')['input_ids'][0]
    prompt_len = len(input_ids)
    
    # Initialize with masks
    total_len = prompt_len + max_length
    x = np.full(total_len, mask_id, dtype=np.int32)
    x[:prompt_len] = input_ids
    x = mx.array(x).reshape(1, -1)
    
    # Generate iteratively
    for step in range(max_length):
        # Get model predictions
        logits = model(x)
        
        # Only look at masked positions
        mask_positions = mx.where(x[0] == mask_id)[0]
        if len(mask_positions) == 0:
            break
        
        # Get predictions for first masked position
        first_mask_pos = mask_positions[0].item()
        pred_logits = logits[0, first_mask_pos]
        
        # Sample token
        pred_token = mx.argmax(pred_logits).item()
        
        # Update sequence
        x_np = np.array(x)
        x_np[0, first_mask_pos] = pred_token
        x = mx.array(x_np)
        
        # Check if we generated end token
        if pred_token == tokenizer.eos_token_id:
            break
    
    # Decode result
    output_ids = x[0, prompt_len:].tolist()
    # Remove mask tokens
    output_ids = [t for t in output_ids if t != mask_id]
    
    return tokenizer.decode(output_ids, skip_special_tokens=True)


def test_llada():
    """Test LLaDA model with simple generation"""
    
    print("🚀 Testing LLaDA Model")
    print("=" * 60)
    
    # Load model
    model, tokenizer = load_llada_model("mlx-community/LLaDA-8B-Instruct-mlx-4bit")
    
    # Test prompts
    prompts = [
        "What is 15 + 27?",
        "Complete this sentence: The capital of France is",
        "Hello, how are you today?",
    ]
    
    for prompt in prompts:
        print(f"\n📝 Prompt: {prompt}")
        print("🎯 Generating...")
        
        try:
            response = simple_llada_generate(model, tokenizer, prompt, max_length=30)
            print(f"✅ Response: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Test the math problem
    print("\n" + "=" * 60)
    math_prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"
    print(f"\n📝 Math Problem: {math_prompt}")
    print("🎯 Generating longer response...")
    
    try:
        response = simple_llada_generate(model, tokenizer, math_prompt, max_length=100)
        print(f"✅ Response: {response}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎊 Testing complete!")


if __name__ == "__main__":
    test_llada()