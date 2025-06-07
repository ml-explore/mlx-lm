#!/usr/bin/env python3
"""
Debug script to identify key differences between PyTorch and MLX implementations
"""

import torch
import mlx.core as mx
import numpy as np

print("🔍 Comparing key operations between PyTorch and MLX\n")

# Test 1: Gumbel noise generation
print("1. Gumbel Noise Generation:")
print("-" * 40)

# PyTorch version
torch.manual_seed(42)
logits_pt = torch.randn(1, 5)
temperature = 1.0

def pt_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

result_pt = pt_gumbel_noise(logits_pt, temperature)
print(f"PyTorch logits: {logits_pt}")
print(f"PyTorch result: {result_pt}")

# MLX version
mx.random.seed(42)
logits_mx = mx.random.normal((1, 5))

def mx_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.astype(mx.float32)
    noise = mx.random.uniform(shape=logits.shape, dtype=mx.float32)
    # Avoid log(0) by adding small epsilon
    gumbel_noise = (-mx.log(noise + 1e-20)) ** temperature
    return mx.exp(logits) / gumbel_noise

result_mx = mx_gumbel_noise(logits_mx, temperature)
print(f"\nMLX logits: {logits_mx}")
print(f"MLX result: {result_mx}")

# Test 2: Confidence calculation
print("\n\n2. Confidence Calculation:")
print("-" * 40)

# PyTorch
logits_pt = torch.randn(1, 10, 100)  # batch x sequence x vocab
x0 = torch.argmax(logits_pt, dim=-1)
p = torch.nn.functional.softmax(logits_pt, dim=-1)
x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
print(f"PyTorch x0 shape: {x0.shape}")
print(f"PyTorch confidence shape: {x0_p.shape}")
print(f"PyTorch confidence sample: {x0_p[0, :5]}")

# MLX
logits_mx = mx.random.normal((1, 10, 100))
x0_mx = mx.argmax(logits_mx, axis=-1)
p_mx = mx.softmax(logits_mx, axis=-1)
x0_expanded = mx.expand_dims(x0_mx, axis=-1)
x0_p_mx = mx.take_along_axis(p_mx, x0_expanded, axis=-1).squeeze(-1)
print(f"\nMLX x0 shape: {x0_mx.shape}")
print(f"MLX confidence shape: {x0_p_mx.shape}")
print(f"MLX confidence sample: {x0_p_mx[0, :5]}")

# Test 3: Top-k selection
print("\n\n3. Top-k Selection:")
print("-" * 40)

# PyTorch
confidence = torch.rand(1, 10)
k = 3
_, select_index = torch.topk(confidence[0], k=k)
print(f"PyTorch confidence: {confidence[0]}")
print(f"PyTorch top-{k} indices: {select_index}")

# MLX
confidence_mx = mx.random.uniform((1, 10))
# MLX doesn't have topk, so we use argsort
sorted_indices = mx.argsort(-confidence_mx[0])
top_k_indices = sorted_indices[:k]
print(f"\nMLX confidence: {confidence_mx[0]}")
print(f"MLX top-{k} indices: {top_k_indices}")

# Test 4: Masking operations
print("\n\n4. Masking Operations:")
print("-" * 40)

mask_id = 126336
# PyTorch
x_pt = torch.tensor([[1, 2, mask_id, 4, mask_id]])
mask_index = (x_pt == mask_id)
x0_pt = torch.tensor([[10, 20, 30, 40, 50]])
result_pt = torch.where(mask_index, x0_pt, x_pt)
print(f"PyTorch x: {x_pt}")
print(f"PyTorch mask: {mask_index}")
print(f"PyTorch result: {result_pt}")

# MLX
x_mx = mx.array([[1, 2, mask_id, 4, mask_id]])
mask_index_mx = (x_mx == mask_id)
x0_mx = mx.array([[10, 20, 30, 40, 50]])
result_mx = mx.where(mask_index_mx, x0_mx, x_mx)
print(f"\nMLX x: {x_mx}")
print(f"MLX mask: {mask_index_mx}")
print(f"MLX result: {result_mx}")

print("\n\n🔍 Key Differences Found:")
print("-" * 40)
print("1. PyTorch uses float64 for Gumbel noise, MLX limited to float32")
print("2. PyTorch has torch.topk, MLX needs argsort workaround")
print("3. gather/take_along_axis syntax differs slightly")
print("4. Random number generation may produce different distributions")