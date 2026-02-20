
import os
import torch
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.sarvam_moe import SarvamMoEExperts, ModelArgs
from mlx_lm.models.sarvam_moe_transformers import SarvamMoEExperts as PTSarvamMoEExperts, SarvamMoEConfig

def set_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    mx.random.seed(seed)

def torch_to_mlx(tensor):
    return mx.array(tensor.detach().cpu().numpy())

def check_close(a, b, atol=1e-3, rtol=1e-3, name="Tensor"):
    if isinstance(a, mx.array):
        a = np.array(a)
    if isinstance(b, torch.Tensor):
        b = b.detach().cpu().numpy()
    
    if np.allclose(a, b, atol=atol, rtol=rtol):
        print(f"✅ {name} matches!")
        return True
    else:
        diff = np.abs(a - b).max()
        print(f"❌ {name} mismatch! Max diff: {diff}")
        print(f"  MLX shape: {a.shape}")
        print(f"  PT shape:  {b.shape}")
        return False

def test_experts():
    print("\n--- Testing Experts ---")
    set_seeds()
    
    # Configuration
    hidden_size = 64
    moe_intermediate_size = 128
    num_experts = 4
    num_experts_per_tok = 2
    
    config = SarvamMoEConfig(
        hidden_size=hidden_size,
        moe_intermediate_size=moe_intermediate_size,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
    )
    
    args = ModelArgs(
        model_type="sarvam_moe",
        hidden_size=hidden_size,
        moe_intermediate_size=moe_intermediate_size,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        # Required dummies
        vocab_size=1000,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128
    )
    
    # Instantiate
    print("Instantiating PyTorch Experts...")
    pt_experts = PTSarvamMoEExperts(config).eval()
    print("Instantiating MLX Experts...")
    mlx_experts = SarvamMoEExperts(args)
    
    # Sync Weights
    print("Syncing weights...")
    # PT: experts is list of MLPs (gate, up, down)
    # MLX: switch_mlp (gate, up, down - stacked)
    gate_w, up_w, down_w = [], [], []
    for i in range(config.num_experts):
        gate_w.append(pt_experts[i].gate_proj.weight)
        up_w.append(pt_experts[i].up_proj.weight)
        down_w.append(pt_experts[i].down_proj.weight)
    
    # Stack and assign to MLX SwitchGLU
    # SwitchGLU weights are (Experts, Out, In) matches PT simple stack?
    # PT Linear weight: (Out, In)
    # Stack dim 0: (Experts, Out, In)
    # SwitchLinear weight: (Experts, Out, In)
    # MLX SwitchLinear expects (Experts, Out, In)
    # let's verify stacking direction.
    
    mlx_experts.switch_mlp.gate_proj.weight = torch_to_mlx(torch.stack(gate_w))
    mlx_experts.switch_mlp.up_proj.weight = torch_to_mlx(torch.stack(up_w))
    mlx_experts.switch_mlp.down_proj.weight = torch_to_mlx(torch.stack(down_w))
    
    # Verify weights are loaded correctly (basic check)
    if not np.allclose(np.array(mlx_experts.switch_mlp.gate_proj.weight[0]), pt_experts[0].gate_proj.weight.detach().numpy(), atol=1e-5):
        print("WARNING: Weight sync might be incorrect!")
        
    # Inputs
    B, L = 2, 10
    total_tokens = B * L
    topk = config.num_experts_per_tok
    num_ex = config.num_experts
    
    # Random inputs
    x_pt = torch.randn(B, L, hidden_size)
    x_mlx = torch_to_mlx(x_pt)
    
    # Logic to generate reasonable indices
    logits = torch.randn(B, L, num_ex)
    _, inds_pt = torch.topk(logits, topk, dim=-1)
    inds_mlx = torch_to_mlx(inds_pt)
    
    # Random weights (normalized)
    weights_pt = torch.rand(B, L, topk)
    weights_pt = weights_pt / weights_pt.sum(dim=-1, keepdim=True)
    weights_mlx = torch_to_mlx(weights_pt)
    
    # Flatten for PT (Required by implementation)
    x_pt_flat = x_pt.view(-1, hidden_size)
    inds_pt_flat = inds_pt.view(-1, topk)
    weights_pt_flat = weights_pt.view(-1, topk)
    
    # Flatten for MLX (Optional but safe for comparison)
    x_mlx_flat = x_mlx.reshape(-1, hidden_size)
    inds_mlx_flat = inds_mlx.reshape(-1, topk)
    weights_mlx_flat = weights_mlx.reshape(-1, topk)
    
    print("Running PyTorch Forward...")
    with torch.no_grad():
        out_pt = pt_experts(x_pt_flat, inds_pt_flat, weights_pt_flat)
        
    print("Running MLX Forward...")
    out_mlx = mlx_experts(x_mlx_flat, inds_mlx_flat, weights_mlx_flat)
    
    print("Comparing Results...")
    check_close(out_mlx, out_pt, name="Experts Output (Flattened)")
    
    # Test Unflattened MLX input capability
    print("\nTesting MLX Unflattened Input Capability...")
    out_mlx_unflat = mlx_experts(x_mlx, inds_mlx, weights_mlx)
    out_pt_reshaped = out_pt.view(B, L, hidden_size)
    
    check_close(out_mlx_unflat, out_pt_reshaped, name="Experts Output (Unflattened)")
    
if __name__ == "__main__":
    test_experts()
