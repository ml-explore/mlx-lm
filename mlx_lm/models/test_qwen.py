from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from base import BaseModelArgs, scaled_dot_product_attention, create_attention_mask
from rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    sliding_window: int
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = True
    use_sliding_window: bool = True


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        assert args.num_key_value_heads is not None
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.use_sliding_window = args.use_sliding_window
        self.sliding_window = args.sliding_window

        self.head_dim = args.hidden_size // n_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.rope = initialize_rope(
            self.head_dim,
            base=args.rope_theta,
            traditional=args.rope_traditional,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        sliding_window = None if self.use_sliding_window == False else self.sliding_window

        output = scaled_dot_product_attention(
            queries, keys, values, cache=None, mask=mask, sliding_window=sliding_window, scale=self.scale
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)

# Create test inputs
batch_size = 1
seq_len = 1024
hidden_dim = 128
num_heads = 8
num_kv_heads = 4


# Test with Attention module from the original code
args = ModelArgs(
    hidden_size=hidden_dim,
    num_attention_heads=num_heads,
    num_key_value_heads=num_kv_heads,
    sliding_window=12
)

attention = Attention(args)

# Create input for the attention module
x = mx.random.normal((batch_size, seq_len, hidden_dim))

B, L, D = x.shape

attn_mask = create_attention_mask(x, cache=None, return_array=True, sliding_window=12)

print(attn_mask.shape)

# Call the attention module
output_module = attention(x, mask=attn_mask)

print(f"Attention module input shape: {x.shape}")
print(f"Attention module output shape: {output_module.shape}")
print(f"Attention module output sample: {output_module[0, 0, :5]}")
