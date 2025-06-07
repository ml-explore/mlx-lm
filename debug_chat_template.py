#!/usr/bin/env python3
"""
Debug script to check what's happening with chat templates
"""

import json
from mlx_lm.utils import get_model_path, load_tokenizer

def debug_chat_template(model_name):
    """Debug the chat template for a model"""
    
    print(f"\n🔍 Debugging chat template for: {model_name}")
    print("-" * 60)
    
    # Get model path
    model_path = get_model_path(model_name)
    
    # Load tokenizer config
    with open(f"{model_path}/tokenizer_config.json", "r") as f:
        tokenizer_config = json.load(f)
    
    print("📋 Tokenizer config:")
    print(f"  - BOS token: {tokenizer_config.get('bos_token')}")
    print(f"  - EOS token: {tokenizer_config.get('eos_token')}")
    print(f"  - Pad token: {tokenizer_config.get('pad_token')}")
    
    if 'chat_template' in tokenizer_config:
        print(f"\n📝 Chat template:")
        print(tokenizer_config['chat_template'])
    
    # Load tokenizer
    tokenizer = load_tokenizer(model_path)
    
    # Test message
    messages = [{"role": "user", "content": "Hello"}]
    
    # Apply chat template
    try:
        formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        print(f"\n✅ Chat template applied successfully")
        print(f"📊 Type of formatted output: {type(formatted)}")
        
        # If it's a string, tokenize it
        if isinstance(formatted, str):
            print(f"📝 Formatted string: '{formatted}'")
            token_ids = tokenizer.encode(formatted)
            print(f"🔢 Token IDs: {token_ids}")
        else:
            # It's already token IDs
            print(f"🔢 Token IDs: {formatted}")
            
        # Decode to see what it looks like
        if isinstance(formatted, str):
            decoded = formatted
        else:
            decoded = tokenizer.decode(formatted)
        print(f"📖 Decoded text: '{decoded}'")
        
    except Exception as e:
        print(f"❌ Error applying chat template: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function"""
    
    print("🔬 Chat Template Debug Tool")
    print("=" * 60)
    
    models = [
        "mlx-community/LLaDA-8B-Instruct-mlx-4bit",
        "mlx-community/LLaDA-8B-Instruct-mlx-8bit",
    ]
    
    for model in models:
        debug_chat_template(model)
    
    # Compare with a working model
    print("\n\n🔍 Comparing with a working model (Llama):")
    debug_chat_template("mlx-community/Llama-3.2-3B-Instruct-4bit")

if __name__ == "__main__":
    main()