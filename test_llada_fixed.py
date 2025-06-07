#!/usr/bin/env python3
"""
Test script for LLaDA with proper MLX implementation
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import get_model_path, load_tokenizer
from mlx_lm.models.llada import ModelArgs, Model
from mlx_lm.models.llada_generate import llada_generate
from transformers import AutoTokenizer
import time


def load_llada_model(model_path: str):
    """Load LLaDA model with weights"""
    print(f"📥 Loading {model_path}...")
    
    # Get model path
    model_path = get_model_path(model_path)
    
    # Load config and weights
    try:
        from mlx_lm.models.llada import load_config, load_model_by_shards
        config = load_config(model_path)
        
        # Create model
        model = Model(ModelArgs(**config))
        
        # Load weights
        weights = load_model_by_shards(model_path, lazy=False)
        
        # Apply quantization if needed
        print("🔄 Applying 4-bit quantization...")
        from mlx_lm.quant.utils import quantize_model
        weights = quantize_model(
            model=model,
            shards=weights,
            class_predicate=lambda _, m: hasattr(m, "to_quantized"),
            linear_class_predicate=lambda _, m: isinstance(m, nn.Linear) and m.weight.shape[0] != 126400,
            group_size=64,
            bits=4
        )
        
        print("📦 Loading weights...")
        model.load_weights(list(weights.items()), strict=False)
    except Exception as e:
        print(f"⚠️  Error loading with custom method: {e}")
        print("🔄 Trying alternative loading...")
        # Fallback to standard loading
        from mlx_lm.utils import load_model
        model, _ = load_model(model_path)
    
    # Load tokenizer with fallback
    print("🔄 Loading tokenizer...")
    try:
        tokenizer = load_tokenizer(model_path)
    except Exception as e:
        print(f"⚠️  Standard tokenizer loading failed: {e}")
        print("🔄 Loading tokenizer from HuggingFace...")
        tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    
    print("✅ Model and tokenizer loaded!")
    return model, tokenizer


def test_generation():
    """Test LLaDA generation"""
    # Load model
    model_name = "mlx-community/LLaDA-8B-Instruct-mlx-4bit"
    model, tokenizer = load_llada_model(model_name)
    
    # Test prompts
    test_prompts = [
        "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?",
        "What is 15 + 27?",
        "Calculate: 100 - 37 =",
        "What is 2 + 2?",
    ]
    
    print("\n" + "="*60)
    print("📝 Testing LLaDA generation with MLX")
    print("="*60)
    
    for prompt in test_prompts:
        print(f"\n🎯 Prompt: {prompt}")
        
        # Prepare input with chat template
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        # Tokenize
        input_ids = tokenizer(formatted_prompt, return_tensors='np')['input_ids']
        input_ids = mx.array(input_ids[0])
        
        # Generate
        print("⏳ Generating...", end='', flush=True)
        start_time = time.time()
        
        # Wrap model call
        def model_forward(x):
            return model(x[None, :])[0]  # Add batch dimension and remove it from output
        
        output_ids = llada_generate(
            model_forward,
            input_ids[None, :],  # Add batch dimension
            steps=128,
            gen_length=128,
            block_length=32,
            temperature=0.0,
            cfg_scale=0.0,
            remasking='low_confidence',
            mask_id=126336
        )
        
        elapsed = time.time() - start_time
        print(f" Done! ({elapsed:.1f}s)")
        
        # Decode only the generated part
        generated_ids = output_ids[0, len(input_ids):]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        print(f"✅ Response: {response}")
    
    print("\n" + "="*60)
    print("🎊 Testing complete!")
    print("="*60)


if __name__ == "__main__":
    test_generation()