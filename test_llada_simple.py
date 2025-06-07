#!/usr/bin/env python3
"""
Simple test script for LLaDA with MLX
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load_model, get_model_path
from mlx_lm.models.llada_generate import llada_generate
from transformers import AutoTokenizer
import time


def main():
    print("🚀 Testing LLaDA with MLX")
    print("="*60)
    
    # Model name
    model_name = "mlx-community/LLaDA-8B-Instruct-mlx-4bit"
    
    # Load model
    print(f"📥 Loading {model_name}...")
    model_path = get_model_path(model_name)
    model, tokenizer = load_model(model_path)
    
    # If tokenizer loading failed, use HuggingFace
    if tokenizer is None:
        print("🔄 Loading tokenizer from HuggingFace...")
        tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    
    print("✅ Model loaded!")
    
    # Test prompt
    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"
    
    print(f"\n📝 Prompt: {prompt}")
    
    # Prepare input
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    
    # Tokenize
    input_ids = tokenizer(formatted_prompt, return_tensors='np')['input_ids']
    input_ids = mx.array(input_ids[0])
    
    print(f"📊 Input shape: {input_ids.shape}")
    print(f"🔤 Input tokens (first 10): {input_ids[:10].tolist()}")
    
    # Generate
    print("\n⏳ Generating response...")
    start_time = time.time()
    
    # Create a wrapper for the model that handles batching
    def model_forward(x):
        # Ensure batch dimension
        if len(x.shape) == 1:
            x = x[None, :]
        # Get logits
        output = model(x)
        # Return logits only
        if hasattr(output, 'logits'):
            return output.logits
        else:
            return output
    
    # Generate with LLaDA algorithm
    try:
        output_ids = llada_generate(
            model_forward,
            input_ids[None, :],  # Add batch dimension
            steps=64,  # Fewer steps for testing
            gen_length=64,  # Shorter generation for testing
            block_length=32,
            temperature=0.0,
            cfg_scale=0.0,
            remasking='low_confidence',
            mask_id=126336
        )
        
        elapsed = time.time() - start_time
        print(f"✅ Generation complete! ({elapsed:.1f}s)")
        
        # Decode response
        generated_ids = output_ids[0, len(input_ids):]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        print(f"\n💬 Response: {response}")
        
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("🎊 Test complete!")


if __name__ == "__main__":
    main()