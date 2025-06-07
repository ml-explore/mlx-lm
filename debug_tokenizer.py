#!/usr/bin/env python3
"""
Debug tokenizer special tokens
"""

import json
from mlx_lm.utils import get_model_path, load_tokenizer

def debug_tokenizer(model_name):
    """Debug the tokenizer special tokens"""
    
    print(f"\n🔍 Debugging tokenizer for: {model_name}")
    print("-" * 60)
    
    # Get model path
    model_path = get_model_path(model_name)
    
    # Load tokenizer
    tokenizer = load_tokenizer(model_path)
    
    # Check special tokens
    print("\n📋 Special tokens:")
    print(f"  - BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    print(f"  - EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"  - PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    
    # Check if special tokens used in chat template exist
    special_tokens = [
        "<|start_header_id|>",
        "<|end_header_id|>", 
        "<|eot_id|>",
        "<|startoftext|>",
        "<|endoftext|>"
    ]
    
    print("\n📋 Chat template special tokens:")
    for token in special_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"  - {token}: ID {token_id}")
    
    # Check vocab size
    print(f"\n📊 Vocab size: {len(tokenizer)}")
    
    # Test encoding/decoding
    test_text = "Hello world"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"\n🧪 Test encoding/decoding:")
    print(f"  - Original: '{test_text}'")
    print(f"  - Encoded: {encoded}")
    print(f"  - Decoded: '{decoded}'")

def main():
    """Main function"""
    
    print("🔬 Tokenizer Debug Tool")
    print("=" * 60)
    
    models = [
        "mlx-community/LLaDA-8B-Instruct-mlx-4bit",
        "mlx-community/Llama-3.2-3B-Instruct-4bit",
    ]
    
    for model in models:
        debug_tokenizer(model)

if __name__ == "__main__":
    main()