#!/usr/bin/env python3
"""Diagnostic: Check if ssd_chunk_state Metal kernel is being used."""
import sys
sys.path.insert(0, '/Users/galbloch/Desktop/work/git/mlx-lm')

import mlx.core as mx
from mlx_lm.models.ssd_chunk_state import _kernel_cache, _make_kernel

print("="*60)
print("Metal Kernel Diagnostics")
print("="*60)

print(f"\nEnvironment:")
print(f"  Metal available: {mx.metal.is_available()}")
print(f"  Default device: {mx.default_device()}")
print(f"  Kernel cache size: {len(_kernel_cache)}")

# Try to make a kernel
print(f"\nTesting kernel compilation...")
batch, nchunks, chunk_size, nheads, headdim, dstate = 1, 4, 256, 32, 64, 16
kernel = _make_kernel(batch, nchunks, chunk_size, nheads, headdim, dstate)

if kernel is not None:
    print(f"✅ Metal kernel compilation SUCCESSFUL")
else:
    print(f"❌ Metal kernel compilation FAILED - will use reference einsum")

# Now test actual usage
print(f"\nTesting actual ssd_chunk_state call...")
try:
    from mlx_lm.models.ssd_chunk_state import run as ssd_chunk_state
    
    # Create small test data
    B = mx.random.normal((batch, nchunks * chunk_size, 4, dstate))
    x = mx.random.normal((batch, nchunks * chunk_size, nheads, headdim))
    dt = mx.random.uniform(shape=(batch, nheads, nchunks, chunk_size))
    dA_cumsum = mx.cumsum(mx.random.normal((batch, nheads, nchunks, chunk_size)), axis=-1)
    
    result = ssd_chunk_state(B, x, dt, dA_cumsum)
    print(f"✅ ssd_chunk_state executed successfully")
    print(f"   Output shape: {result.shape}")
    
except Exception as e:
    print(f"❌ ssd_chunk_state execution failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
