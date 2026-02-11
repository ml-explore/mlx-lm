
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.sarvam_moe import Model, ModelArgs

def print_model_structure():
    args = ModelArgs(
        model_type="sarvam_moe",
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=2,  # Small number for debug
        num_attention_heads=32,
        num_key_value_heads=8,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=1024,
    )
    
    model = Model(args)
    
    print("\n--- Model Parameter Keys ---")
    flat_params = dict(nn.utils.tree_flatten(model.parameters()))
    keys = sorted(flat_params.keys())
    for k in keys:
        print(k)
        
    print("\n--- Check Specific Key ---")
    target_key = "model.layers.0.attention.dense.weight"
    if target_key in keys:
        print(f"FOUND: {target_key}")
    else:
        print(f"MISSING: {target_key}")

if __name__ == "__main__":
    print_model_structure()
