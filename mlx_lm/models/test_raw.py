# Copyright © 2023-2024 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import mlx.core as mx

from base import sliding_window_attention_mlx


B, L, D = 2, 2048, 512  # Batch size, Sequence length, Embedding dim
Hq, Hkv = 8, 2  # Query heads, Key/Value heads (GQA: fewer KV heads)

q = mx.random.normal((B, Hq, L, D // Hq))
k = mx.random.normal((B, Hkv, L, D // Hq))
v = mx.random.normal((B, Hkv, L, D // Hq))
mask = None  # Optional causal mask

# Compute attention
attn_output = sliding_window_attention_mlx(q, k, v, mask, window_size=512)
print(attn_output.shape)  # Expected: (B, Hq, L, D//Hq)