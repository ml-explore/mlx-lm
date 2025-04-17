# Copyright © 2023-2024 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    head_dim: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    attention_bias: bool = False
    mlp_bias: bool = False
    rope_theta: float = 10000
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


import mlx.core as mx
import mlx.nn as nn
import logging

logger = logging.getLogger(__name__)

# the weights are ternary so can be represented with 2 bits, and they are packed in uint8 tensors, hence the number of values per item is 4
VALUES_PER_ITEM = 4


def pack_weights(quantized_weights):
    """
    Packs a tensor of quantized weights into a compact format using 2 bits per value.

    Parameters:
    -----------
    quantized_weights : mx.array
        A tensor containing ternary quantized weights with values in {-1, 0, 1}. These values are adjusted to
        {0, 1, 2} before being packed.

    Returns:
    --------
    mx.array
        A packed tensor where each element stores 4 quantized values (each using 2 bits) in an 8-bit format.
    """
    original_shape = quantized_weights.shape

    row_dim = (original_shape[0] + VALUES_PER_ITEM - 1) // VALUES_PER_ITEM

    if len(original_shape) == 1:
        packed_tensor_shape = (row_dim,)
    else:
        packed_tensor_shape = (row_dim, *original_shape[1:])

    quantized_weights = quantized_weights + 1
    packed = mx.zeros(packed_tensor_shape, dtype=mx.uint8)
    unpacked = quantized_weights.astype(mx.uint8)

    it = min(VALUES_PER_ITEM, (original_shape[0] // row_dim) + 1)
    for i in range(it):
        start = i * row_dim
        end = min(start + row_dim, original_shape[0])
        if end > start:
            # MLX doesn't have in-place operations like |= so we need to recreate the tensor
            shift_value = mx.left_shift(unpacked[start:end], 2 * i)
            # Need to handle slice assignment differently in MLX
            # Create a list of indices for the update
            indices = mx.arange(end - start)
            packed = mx.indexed_update(packed, indices, mx.bitwise_or(packed[:end-start], shift_value))

    return packed


def unpack_weights(packed, dtype=mx.float32):
    """
    Unpacks a tensor of quantized weights that were stored in a packed format using 2 bits per value.

    Parameters:
    -----------
    packed : mx.array
        A tensor containing packed weights where each element represents 4 quantized values (using 2 bits per value).
    dtype : mx.dtype
        The dtype of the returned array

    Returns:
    --------
    mx.array
        A tensor of unpacked weights, where each value is converted from its packed 2-bit representation.
    """
    packed_shape = packed.shape

    if len(packed_shape) == 1:
        original_row_dim = packed_shape[0] * VALUES_PER_ITEM
        unpacked_shape = (original_row_dim,)
    else:
        original_row_dim = packed_shape[0] * VALUES_PER_ITEM
        unpacked_shape = (original_row_dim, *packed_shape[1:])

    unpacked = mx.zeros(unpacked_shape, dtype=mx.uint8)

    for i in range(VALUES_PER_ITEM):
        start = i * packed_shape[0]
        end = start + packed_shape[0]
        mask = 3 << (2 * i)
        # In MLX, we need to handle slicing differently
        mask_tensor = mx.array(mask, dtype=mx.uint8)
        masked_values = mx.bitwise_and(packed, mask_tensor)
        shifted_values = mx.right_shift(masked_values, 2 * i)

        # Update the slice in unpacked
        if start < unpacked_shape[0]:
            end = min(end, unpacked_shape[0])
            # Since neither indexed_update nor array_update exist in mlx.core,
            # we need to create a new array manually
            new_unpacked = mx.zeros_like(unpacked)

            # Copy the original values
            if start > 0:
                new_unpacked = mx.concatenate([unpacked[:start], new_unpacked[start:]])

            # Update the slice with new values
            slice_length = end - start
            middle_section = shifted_values[:slice_length]

            # Combine the parts: before slice + updated slice + after slice
            if end < unpacked_shape[0]:
                new_unpacked = mx.concatenate([
                    new_unpacked[:start],
                    middle_section,
                    unpacked[end:]
                ])
            else:
                new_unpacked = mx.concatenate([
                    new_unpacked[:start],
                    middle_section
                ])

            unpacked = new_unpacked


    return unpacked.astype(dtype) - 1


class BitLinear(nn.Module):
    """
    BitLinear module for MLX that uses 1.58-bit quantization.

    Mimics the functionality of the PyTorch BitLinear class.
    """
    def __init__(self, in_features, out_features, bias=True, dtype=mx.float32):
        super().__init__()
        self.dtype = dtype
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights in packed format
        self.weight = mx.zeros((out_features // VALUES_PER_ITEM, in_features), dtype=mx.uint8)
        self.weight_scale = mx.array([1.0], dtype=dtype)

        if bias:
            self.bias = mx.zeros((out_features,), dtype=dtype)
        else:
            self.bias = None

    def activation_quant(self, x, num_bits=8):
        """
        Performs symmetric, per-token quantization on the input activations.

        Parameters:
        -----------
        x : mx.array
            Input activations to be quantized.
        num_bits : int, optional (default=8)
            Number of bits to use for quantization.

        Returns:
        --------
        result : mx.array
            Quantized activation tensor.
        scale : mx.array
            The per-channel scaling factors used for quantization.
        """
        Qn = -(2 ** (num_bits - 1))
        Qp = 2 ** (num_bits - 1) - 1

        # Find max value along last dimension
        max_abs = mx.max(mx.abs(x), axis=-1, keepdims=True)
        max_abs = mx.maximum(max_abs, 1e-5)  # Clamp to avoid division by zero

        scale = Qp / max_abs
        result = mx.clip(mx.round(x * scale), Qn, Qp)

        return result.astype(mx.int8), scale

    def post_quant_process(self, x, weight_scale, input_scale):
        """
        Rescales the output after quantized matrix multiplication.

        Parameters:
        -----------
        x : mx.array
            Result of the quantized matrix multiplication.
        weight_scale : mx.array
            Scaling factor for the weights.
        input_scale : mx.array
            Scaling factor for the inputs.

        Returns:
        --------
        mx.array
            Rescaled output.
        """
        return x / (input_scale * weight_scale)

    def __call__(self, x):
        """
        Forward pass of the BitLinear layer.

        Parameters:
        -----------
        x : mx.array
            Input tensor.

        Returns:
        --------
        mx.array
            Output after linear transformation with quantized weights.
        """
        # Unpack the quantized weights
        w_quant = unpack_weights(self.weight, dtype=self.dtype)

        # Quantize the input
        input_quant, input_scale = self.activation_quant(x)

        # Convert to same dtype for matrix multiplication
        input_quant = input_quant.astype(self.dtype)

        # Perform the linear transformation
        y = mx.matmul(input_quant, w_quant.T)

        # Rescale the output
        y = self.post_quant_process(y, self.weight_scale, input_scale)

        # Add bias if present
        if self.bias is not None:
            y = y + self.bias

        return y


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        self.head_dim = head_dim = args.head_dim or args.hidden_size // n_heads

        self.scale = head_dim**-0.5
        if hasattr(args, "attention_bias"):
            attention_bias = args.attention_bias
        else:
            attention_bias = False

        self.q_proj = BitLinear(dim, n_heads * head_dim, bias=attention_bias)
        self.k_proj = BitLinear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = BitLinear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = BitLinear(n_heads * head_dim, dim, bias=attention_bias)

        self.rope = initialize_rope(
            self.head_dim,
            args.rope_theta,
            args.rope_traditional,
            args.rope_scaling,
            args.max_position_embeddings,
        )
        self.attn_sub_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

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

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        output = self.attn_sub_norm(output)
        output = self.o_proj(output)

        return output


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        hidden_dim = args.intermediate_size
        if hasattr(args, "mlp_bias"):
            mlp_bias = args.mlp_bias
        else:
            mlp_bias = False

        self.gate_proj = BitLinear(dim, hidden_dim, bias=mlp_bias)
        self.down_proj = BitLinear(hidden_dim, dim, bias=mlp_bias)
        self.up_proj = BitLinear(dim, hidden_dim, bias=mlp_bias)
        self.ffn_sub_norm = nn.RMSNorm(args.intermediate_size, eps=args.rms_norm_eps)

    def __call__(self, x) -> mx.array:
        x = nn.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.ffn_sub_norm(x)
        x = self.down_proj(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


class LlamaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
    ):
        h = self.embed_tokens(inputs)

        if mask is None:
            mask = create_attention_mask(h, cache)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LlamaModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
    ):
        out = self.model(inputs, mask, cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        # Remove unused precomputed rotary freqs
        weights = {
            k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
        }
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

    @property
    def layers(self):
        return self.model.layers
