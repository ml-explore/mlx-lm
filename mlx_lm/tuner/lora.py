# Copyright © 2024 Apple Inc.

import math

import mlx.core as mx
import mlx.nn as nn

from ..models.switch_layers import QuantizedSwitchLinear, SwitchLinear


def _pad_last_dim(x, size):
    padding = size - x.shape[-1]
    if padding == 0:
        return x
    return mx.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, padding)])


def _padded_size(size, group_size):
    return ((size + group_size - 1) // group_size) * group_size


def _quantize_matrix(weight, group_size, bits, mode):
    values = mx.quantize(weight, group_size=group_size, bits=bits, mode=mode)
    return values if len(values) == 3 else (*values, None)


class LoRALinear(nn.Module):
    @staticmethod
    def from_base(
        linear: nn.Linear,
        r: int = 8,
        dropout: float = 0.0,
        scale: float = 20.0,
    ):
        # TODO remove when input_dims and output_dims are attributes
        # on linear and quantized linear
        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            input_dims = input_dims * 32 // linear.bits
        lora_lin = LoRALinear(
            input_dims=input_dims,
            output_dims=output_dims,
            r=r,
            dropout=dropout,
            scale=scale,
        )
        lora_lin.linear = linear
        return lora_lin

    def fuse(self, dequantize: bool = False):
        linear = self.linear
        bias = "bias" in linear
        weight = linear.weight
        is_quantized = isinstance(linear, nn.QuantizedLinear)

        if is_quantized:
            weight = mx.dequantize(
                weight,
                linear.scales,
                linear.biases,
                group_size=linear.group_size,
                bits=linear.bits,
                mode=linear.mode,
            )
        output_dims, input_dims = weight.shape
        fused_linear = nn.Linear(input_dims, output_dims, bias=bias)

        delta = ((self.scale * self.lora_b.T) @ self.lora_a.T).astype(weight.dtype)
        fused_linear.weight = weight + delta
        if bias:
            fused_linear.bias = linear.bias

        if is_quantized and not dequantize:
            fused_linear = nn.QuantizedLinear.from_linear(
                fused_linear,
                linear.group_size,
                linear.bits,
                mode=linear.mode,
            )

        return fused_linear

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        r: int = 8,
        dropout: float = 0.0,
        scale: float = 20.0,
        bias: bool = False,
    ):
        super().__init__()

        # Regular linear layer weights
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)

        self.dropout = nn.Dropout(p=dropout)

        # Scale for low-rank update
        self.scale = scale

        # Low rank lora weights
        scale = 1 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(input_dims, r),
        )
        self.lora_b = mx.zeros(shape=(r, output_dims))

    def __call__(self, x):
        y = self.linear(x)
        z = (self.dropout(x) @ self.lora_a) @ self.lora_b
        return y + (self.scale * z).astype(x.dtype)

    def to_quantized(
        self,
        group_size: int = 64,
        bits: int = 8,
        rank_group_size: int = 32,
        mode: str = "affine",
    ):
        """Quantize this linear layer's LoRA matrices for inference."""
        return QuantizedLoRALinear.from_lora(
            self,
            group_size=group_size,
            bits=bits,
            rank_group_size=rank_group_size,
            mode=mode,
        )


class QuantizedLoRALinear(nn.Module):
    """A linear layer with quantized low-rank adapter matrices.

    The base linear layer is left unchanged. Adapter dimensions are padded as
    needed because MLX quantization requires the final matrix dimension to be
    divisible by the quantization group size.
    """

    @staticmethod
    def from_lora(
        lora: LoRALinear,
        group_size: int = 64,
        bits: int = 8,
        rank_group_size: int = 32,
        mode: str = "affine",
    ):
        input_dims, rank = lora.lora_a.shape
        rank_b, output_dims = lora.lora_b.shape
        if rank != rank_b:
            raise ValueError(f"LoRA rank mismatch between A ({rank}) and B ({rank_b})")

        quantized = QuantizedLoRALinear()
        quantized.linear = lora.linear
        quantized.dropout = lora.dropout
        quantized.scale = lora.scale
        quantized.bits = bits
        quantized.group_size = group_size
        quantized.rank_group_size = rank_group_size
        quantized.mode = mode
        quantized.input_dims = input_dims
        quantized.output_dims = output_dims
        quantized.rank = rank
        quantized.padded_input_dims = _padded_size(input_dims, group_size)
        quantized.padded_rank = _padded_size(rank, rank_group_size)

        lora_a = _pad_last_dim(lora.lora_a.T, quantized.padded_input_dims)
        lora_b = _pad_last_dim(lora.lora_b.T, quantized.padded_rank)
        (
            quantized.lora_a,
            quantized.lora_a_scales,
            quantized.lora_a_biases,
        ) = _quantize_matrix(lora_a, group_size, bits, mode)
        (
            quantized.lora_b,
            quantized.lora_b_scales,
            quantized.lora_b_biases,
        ) = _quantize_matrix(lora_b, rank_group_size, bits, mode)
        quantized.freeze()
        return quantized

    def lora_delta(self, x):
        x = _pad_last_dim(self.dropout(x), self.padded_input_dims)
        z = mx.quantized_matmul(
            x,
            self.lora_a,
            self.lora_a_scales,
            self.lora_a_biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )
        z = _pad_last_dim(z, self.padded_rank)
        return mx.quantized_matmul(
            z,
            self.lora_b,
            self.lora_b_scales,
            self.lora_b_biases,
            transpose=True,
            group_size=self.rank_group_size,
            bits=self.bits,
            mode=self.mode,
        )

    def fuse(self, dequantize: bool = False):
        linear = self.linear
        weight = linear.weight
        is_quantized = isinstance(linear, nn.QuantizedLinear)
        if is_quantized:
            weight = mx.dequantize(
                weight,
                linear.scales,
                linear.biases,
                group_size=linear.group_size,
                bits=linear.bits,
                mode=linear.mode,
            )

        lora_a = mx.dequantize(
            self.lora_a,
            self.lora_a_scales,
            self.lora_a_biases,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )[: self.rank, : self.input_dims]
        lora_b = mx.dequantize(
            self.lora_b,
            self.lora_b_scales,
            self.lora_b_biases,
            group_size=self.rank_group_size,
            bits=self.bits,
            mode=self.mode,
        )[:, : self.rank]

        fused = nn.Linear(self.input_dims, self.output_dims, bias="bias" in linear)
        fused.weight = weight + (self.scale * lora_b @ lora_a).astype(weight.dtype)
        if "bias" in linear:
            fused.bias = linear.bias
        if is_quantized and not dequantize:
            fused = nn.QuantizedLinear.from_linear(
                fused,
                group_size=linear.group_size,
                bits=linear.bits,
                mode=linear.mode,
            )
        return fused

    def __call__(self, x):
        y = self.linear(x)
        z = self.lora_delta(x)
        return y + (self.scale * z).astype(x.dtype)


class LoRASwitchLinear(nn.Module):
    @staticmethod
    def from_base(
        linear: nn.Module,
        r: int = 8,
        dropout: float = 0.0,
        scale: float = 20.0,
    ):
        lora_lin = LoRASwitchLinear(
            input_dims=linear.input_dims,
            output_dims=linear.output_dims,
            num_experts=linear.num_experts,
            r=r,
            dropout=dropout,
            scale=scale,
        )
        lora_lin.linear = linear
        return lora_lin

    def fuse(self, dequantize: bool = False):
        linear = self.linear
        bias = "bias" in linear
        weight = linear.weight
        is_quantized = isinstance(linear, QuantizedSwitchLinear)

        if is_quantized:
            weight = mx.dequantize(
                weight,
                linear.scales,
                linear.biases,
                group_size=linear.group_size,
                bits=linear.bits,
                mode=linear.mode,
            )
        num_experts, output_dims, input_dims = weight.shape
        fused_linear = SwitchLinear(input_dims, output_dims, num_experts, bias=bias)

        lora_b = self.scale * self.lora_b
        lora_a = self.lora_a.reshape(num_experts, -1, input_dims)
        fused_linear.weight = weight + (lora_b @ lora_a).astype(weight.dtype)
        if bias:
            fused_linear.bias = linear.bias

        if is_quantized and not dequantize:
            fused_linear = fused_linear.to_quantized(
                group_size=linear.group_size, bits=linear.bits, mode=linear.mode
            )

        return fused_linear

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        r: int = 8,
        dropout: float = 0.0,
        scale: float = 20.0,
        bias: bool = False,
    ):
        super().__init__()

        # Regular linear layer weights
        self.linear = SwitchLinear(input_dims, output_dims, num_experts, bias=bias)

        self.dropout = nn.Dropout(p=dropout)

        # Scale for low-rank update
        self.scale = scale

        # Low rank lora weights
        scale = 1 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(num_experts, r, input_dims),
        )
        self.lora_b = mx.zeros(shape=(num_experts, output_dims, r))
        self.num_experts = num_experts

    def __call__(self, x, indices, sorted_indices=False):
        y = self.linear(x, indices, sorted_indices=sorted_indices)
        z = mx.gather_mm(
            self.dropout(x),
            self.lora_a.swapaxes(-1, -2),
            rhs_indices=indices,
            sorted_indices=sorted_indices,
        )
        z = mx.gather_mm(
            z,
            self.lora_b.swapaxes(-1, -2),
            rhs_indices=indices,
            sorted_indices=sorted_indices,
        )
        return y + (self.scale * z).astype(x.dtype)


class LoRAEmbedding(nn.Module):
    @staticmethod
    def from_base(
        embedding: nn.Embedding,
        r: int = 8,
        dropout: float = 0.0,
        scale: float = 20.0,
    ):
        num_embeddings, dims = embedding.weight.shape
        if isinstance(embedding, nn.QuantizedEmbedding):
            dims = dims * 32 // embedding.bits
        lora_embedding = LoRAEmbedding(
            num_embeddings=num_embeddings,
            dims=dims,
            r=r,
            dropout=dropout,
            scale=scale,
        )
        lora_embedding.embedding = embedding
        return lora_embedding

    def fuse(self, dequantize: bool = False):
        embedding = self.embedding
        weight = embedding.weight
        is_quantized = isinstance(embedding, nn.QuantizedEmbedding)

        if is_quantized:
            weight = mx.dequantize(
                weight,
                embedding.scales,
                embedding.biases,
                group_size=embedding.group_size,
                bits=embedding.bits,
                mode=embedding.mode,
            )
        num_embeddings, dims = weight.shape
        fused_embedding = nn.Embedding(num_embeddings, dims)

        lora_a = self.scale * self.lora_a
        lora_b = self.lora_b
        fused_embedding.weight = weight + (lora_a @ lora_b).astype(weight.dtype)

        if is_quantized and not dequantize:
            fused_embedding = nn.QuantizedEmbedding.from_embedding(
                fused_embedding,
                group_size=embedding.group_size,
                bits=embedding.bits,
                mode=embedding.mode,
            )

        return fused_embedding

    def __init__(
        self,
        num_embeddings: int,
        dims: int,
        r: int = 8,
        dropout: float = 0.0,
        scale: float = 20.0,
    ):
        super().__init__()

        # Regular embedding layer
        self.embedding = nn.Embedding(num_embeddings, dims)
        self.dropout = nn.Dropout(p=dropout)

        # Scale for low-rank update
        self.scale = scale

        # Low rank lora weights
        scale = 1 / math.sqrt(num_embeddings)
        self.lora_a = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(num_embeddings, r),
        )
        self.lora_b = mx.zeros(shape=(r, dims))

    def __call__(self, x):
        y = self.embedding(x)
        z = self.dropout(self.lora_a[x] @ self.lora_b)
        out = y + (self.scale * z).astype(y.dtype)
        return out

    def as_linear(self, x):
        y = self.embedding.as_linear(x)
        z = (self.dropout(x) @ self.lora_b.T) @ self.lora_a.T
        return y + (self.scale * z).astype(x.dtype)
