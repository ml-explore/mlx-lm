#!/usr/bin/env python3
"""
Debug script to fix LLaDA quantization issues
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import get_model_path, _get_classes
from mlx_lm.models.llada import ModelArgs, Model
import json

def debug_model_loading():
    """Debug the model loading process step by step"""
    
    print("🔍 Debugging LLaDA model loading...")
    
    # Get model path and config
    model_path = get_model_path("mlx-community/LLaDA-8B-Instruct-mlx-4bit")
    
    with open(f"{model_path}/config.json", "r") as f:
        config = json.load(f)
    
    print(f"📁 Model path: {model_path}")
    print(f"⚙️ Config keys: {list(config.keys())}")
    print(f"🔧 weight_tying: {config.get('weight_tying')}")
    print(f"🔧 quantization: {config.get('quantization')}")
    
    # Create model args
    model_args_class = ModelArgs
    model_args = model_args_class.from_dict(config)
    print(f"✅ ModelArgs created: tie_word_embeddings={model_args.tie_word_embeddings}")
    
    # Create model
    model = Model(model_args)
    print(f"✅ Model created")
    print(f"🏗️ Has lm_head: {hasattr(model, 'lm_head')}")
    
    if hasattr(model, 'lm_head'):
        print(f"🏗️ lm_head type: {type(model.lm_head)}")
    
    # Load weights
    weights = mx.load(f"{model_path}/model.safetensors")
    print(f"📦 Total weights: {len(weights)}")
    
    # Check lm_head weights
    lm_head_keys = [k for k in weights.keys() if 'lm_head' in k]
    print(f"🎯 lm_head weight keys: {lm_head_keys}")
    
    for key in lm_head_keys:
        print(f"  {key}: {weights[key].shape}")
    
    # Check if quantization is expected
    has_scales = any('.scales' in k for k in weights.keys())
    has_biases = any('.biases' in k for k in weights.keys())
    print(f"🔢 Has quantized weights (scales): {has_scales}")
    print(f"🔢 Has quantized weights (biases): {has_biases}")
    
    # Try to understand the quantization structure
    if has_scales:
        scales_keys = [k for k in weights.keys() if '.scales' in k][:5]
        print(f"📏 Example scales keys: {scales_keys}")
    
    # Sanitize weights
    sanitized_weights = model.sanitize(weights)
    print(f"🧼 Sanitized weights: {len(sanitized_weights)}")
    
    # Check what's missing or extra using tree_flatten
    from mlx.utils import tree_flatten
    
    # Get model parameter names
    model_params = tree_flatten(model.parameters())
    # This gives us a list of (key_path, parameter) tuples
    model_param_names = set(model_params[1])  # Get the paths
    
    weight_names = set(sanitized_weights.keys())
    
    print(f"🏗️ Model parameter count: {len(model_param_names)}")
    print(f"📦 Weight count: {len(weight_names)}")
    
    # Show first few model parameters
    if model_param_names:
        print(f"🔍 First few model params: {list(model_param_names)[:5]}")
    
    # Show first few weight names  
    print(f"🔍 First few weight names: {list(weight_names)[:5]}")
    
    missing = model_param_names - weight_names
    extra = weight_names - model_param_names
    
    if missing:
        print(f"❌ Missing from weights: {list(missing)[:5]}")
    if extra:
        print(f"➕ Extra in weights: {list(extra)[:5]}")
    
    return model, sanitized_weights

def test_manual_quantization():
    """Test manual quantization approach"""
    
    print("\n🔧 Testing manual quantization...")
    
    try:
        model, weights = debug_model_loading()
        
        # Apply quantization manually
        quantization = {"group_size": 64, "bits": 4}
        
        # Define class predicate for quantization
        def class_predicate(p, m):
            return f"{p}.scales" in weights
        
        print("🔄 Applying quantization...")
        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            class_predicate=class_predicate,
        )
        
        print("✅ Quantization applied")
        
        # Now try to load weights
        print("📥 Loading weights...")
        model.load_weights(list(weights.items()), strict=False)
        print("✅ Weights loaded!")
        
        return model
        
    except Exception as e:
        print(f"❌ Manual quantization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    model = test_manual_quantization()
    if model:
        print("\n🎉 Success! Model loading works!")
    else:
        print("\n💥 Failed to load model")