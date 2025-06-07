#!/usr/bin/env python3
"""
Test script for LLaDA model implementation in mlx-lm
"""

import sys
import traceback
from mlx_lm import load, generate

def test_llada_model():
    """Test the LLaDA model loading and generation"""
    
    print("🚀 Testing LLaDA model implementation...")
    print("-" * 50)
    
    try:
        # Load the model
        print("📥 Loading LLaDA model...")
        model, tokenizer = load("mlx-community/LLaDA-8B-Instruct-mlx-4bit")
        print("✅ Model loaded successfully!")
        
        # Print model info
        print(f"📊 Model type: {type(model)}")
        print(f"📊 Model args: {model.args}")
        print(f"📊 Vocab size: {model.args.vocab_size}")
        print(f"📊 Hidden size: {model.args.hidden_size}")
        print(f"📊 Num layers: {model.args.num_hidden_layers}")
        
        # Test basic generation
        print("\n🧪 Testing basic generation...")
        prompt = "Hello, how are you?"
        
        # Apply chat template if available
        if tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
            print(f"💬 Using chat template: {formatted_prompt[:100]}...")
        else:
            formatted_prompt = prompt
            print(f"💬 Using raw prompt: {formatted_prompt}")
        
        # Generate response
        print("🎯 Generating response...")
        response = generate(
            model, 
            tokenizer, 
            prompt=formatted_prompt, 
            verbose=True, 
            max_tokens=50,
            temp=0.7
        )
        
        print(f"\n🎉 Generated response:")
        print(f"📝 {response}")
        
        # Test with a more complex prompt
        print("\n🧪 Testing with complex prompt...")
        complex_prompt = "Explain the concept of artificial intelligence in simple terms."
        
        if tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": complex_prompt}]
            formatted_complex_prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
        else:
            formatted_complex_prompt = complex_prompt
            
        complex_response = generate(
            model, 
            tokenizer, 
            prompt=formatted_complex_prompt, 
            verbose=True, 
            max_tokens=100,
            temp=0.7
        )
        
        print(f"\n🎉 Complex response:")
        print(f"📝 {complex_response}")
        
        print("\n✅ All tests passed! LLaDA implementation is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print("\n🔍 Full traceback:")
        traceback.print_exc()
        return False

def test_model_architecture():
    """Test that the model architecture is correct"""
    
    print("\n🏗️  Testing model architecture...")
    
    try:
        from mlx_lm.models.llada import ModelArgs, Model
        
        # Test ModelArgs creation
        config = {
            "model_type": "llada",
            "d_model": 4096,
            "n_layers": 32,
            "mlp_hidden_size": 12288,
            "n_heads": 32,
            "vocab_size": 126464,
            "rms_norm_eps": 1e-05,
        }
        
        args = ModelArgs.from_dict(config)
        print(f"✅ ModelArgs created: {args}")
        
        # Test Model creation
        model = Model(args)
        print(f"✅ Model created successfully")
        print(f"📊 Model layers: {len(model.layers)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Architecture test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    
    print("🔬 LLaDA Model Test Suite")
    print("=" * 50)
    
    # Test architecture first
    arch_success = test_model_architecture()
    
    if not arch_success:
        print("❌ Architecture tests failed, skipping model loading tests")
        sys.exit(1)
    
    # Test full model loading and generation
    model_success = test_llada_model()
    
    if model_success:
        print("\n🎊 All tests completed successfully!")
        print("🚀 LLaDA model is ready for use!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()