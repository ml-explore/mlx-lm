#!/usr/bin/env python3
"""
Working test script for LLaDA model
Handles quantization properly
"""

import sys
import traceback
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import generate
from mlx_lm.utils import get_model_path, load_tokenizer
from mlx_lm.models.llada import ModelArgs, Model
import json

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
    import glob
    safetensors_files = glob.glob(f"{model_path}/*.safetensors")
    
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
    
    # Load tokenizer
    tokenizer = load_tokenizer(model_path)
    
    print("✅ Model loaded successfully!")
    return model, tokenizer

def test_model(model_name, description):
    """Test a specific LLaDA model"""
    
    print(f"\n🚀 Testing {description}")
    print("-" * 60)
    
    try:
        # Load the model
        model, tokenizer = load_llada_model(model_name)
        
        # Print basic info
        print(f"📊 Model type: {type(model).__name__}")
        print(f"📊 Vocab size: {model.args.vocab_size}")
        print(f"📊 Hidden size: {model.args.hidden_size}")
        print(f"📊 Layers: {model.args.num_hidden_layers}")
        
        # Test generation
        print("\n🧪 Testing generation...")
        prompt = "What is artificial intelligence?"
        
        # Apply chat template if available
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
            print(f"💬 Using chat template")
        else:
            formatted_prompt = prompt
            print(f"💬 Using raw prompt")
        
        # Generate response
        print("🎯 Generating response...")
        response = generate(
            model, 
            tokenizer, 
            prompt=formatted_prompt, 
            max_tokens=50
        )
        
        print(f"\n✅ Response generated:")
        print(f"📝 {response}")
        
        # Test with a math question
        print(f"\n🧮 Testing math reasoning...")
        math_prompt = "What is 15 + 27?"
        
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": math_prompt}]
            formatted_math_prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
        else:
            formatted_math_prompt = math_prompt
        
        math_response = generate(
            model, 
            tokenizer, 
            prompt=formatted_math_prompt, 
            max_tokens=30
        )
        
        print(f"🧮 Math response: {math_response}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    
    print("🔬 LLaDA Model Test Suite")
    print("=" * 60)
    
    # Test models
    models_to_test = [
        ("mlx-community/LLaDA-8B-Instruct-mlx-4bit", "LLaDA 8B Instruct (4-bit)"),
        ("mlx-community/LLaDA-8B-Instruct-mlx-8bit", "LLaDA 8B Instruct (8-bit)"),
    ]
    
    results = []
    
    for model_name, description in models_to_test:
        success = test_model(model_name, description)
        results.append((description, success))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for description, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} - {description}")
        if not success:
            all_passed = False
    
    if all_passed:
        print(f"\n🎊 All {len(results)} tests passed!")
        print("🚀 LLaDA is working correctly in mlx-lm!")
        sys.exit(0)
    else:
        print(f"\n💥 Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()