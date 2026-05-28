# Copyright © 2023-2024 Apple Inc.

import math
from functools import partial

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu


def _gather_sort(x, indices):
    *_, M = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    return x.flatten(0, -3)[order // M], indices[order], inv_order


def _scatter_unsort(x, inv_order, shape=None):
    x = x[inv_order]
    if shape is not None:
        x = mx.unflatten(x, 0, shape)
    return x


class QuantizedSwitchLinear(nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
    ):
        super().__init__()

        scale = math.sqrt(1 / input_dims)
        self.weight, self.scales, *biases = mx.quantize(
            mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(num_experts, output_dims, input_dims),
            ),
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        self.biases = biases[0] if biases else None

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

        self.group_size = group_size
        self.bits = bits
        self.mode = mode

        # Freeze this model's parameters
        self.freeze()

    @property
    def input_dims(self):
        return self.scales.shape[2] * self.group_size

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def __call__(self, x, indices, sorted_indices=False):
        x = mx.gather_qmm(
            x,
            self["weight"],
            self["scales"],
            self.get("biases"),
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x


class SwitchLinear(nn.Module):
    def __init__(
        self, input_dims: int, output_dims: int, num_experts: int, bias: bool = True
    ):
        super().__init__()
        scale = math.sqrt(1 / input_dims)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(num_experts, output_dims, input_dims),
        )

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

    @property
    def input_dims(self):
        return self.weight.shape[2]

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def __call__(self, x, indices, sorted_indices=False):
        x = mx.gather_mm(
            x,
            self["weight"].swapaxes(-1, -2),
            rhs_indices=indices,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x

    def to_quantized(self, group_size: int = 64, bits: int = 4, mode: str = "affine"):
        num_experts, output_dims, input_dims = self.weight.shape
        ql = QuantizedSwitchLinear(
            input_dims,
            output_dims,
            num_experts,
            False,
            group_size,
            bits,
            mode=mode,
        )
        ql.weight, ql.scales, *biases = mx.quantize(
            self.weight, group_size, bits, mode=mode
        )
        ql.biases = biases[0] if biases else None

        if "bias" in self:
            ql.bias = self.bias
        return ql


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x, gate):
        return swiglu(gate, x)


class SwitchGLU(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=SwiGLU(),
        bias: bool = False,
        fuse_gate_up: bool = False,
    ):
        super().__init__()

        self.gate_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.up_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation
        self.fuse_gate_up = fuse_gate_up
        self._fused_gate_up_cache = None

    def _can_fuse_gate_up(self):
        if not self.fuse_gate_up or self.training:
            return False
        if type(self.up_proj) is not type(self.gate_proj):
            return False
        if not isinstance(self.up_proj, (SwitchLinear, QuantizedSwitchLinear)):
            return False
        if self.up_proj.input_dims != self.gate_proj.input_dims:
            return False
        if self.up_proj.output_dims != self.gate_proj.output_dims:
            return False
        if self.up_proj.num_experts != self.gate_proj.num_experts:
            return False
        if ("bias" in self.up_proj) != ("bias" in self.gate_proj):
            return False
        if isinstance(self.up_proj, QuantizedSwitchLinear):
            if self.up_proj.group_size != self.gate_proj.group_size:
                return False
            if self.up_proj.bits != self.gate_proj.bits:
                return False
            if self.up_proj.mode != self.gate_proj.mode:
                return False
            if (self.up_proj.get("biases") is None) != (
                self.gate_proj.get("biases") is None
            ):
                return False
        return True

    def _fused_gate_up_params(self):
        up = self.up_proj
        gate = self.gate_proj
        key = (
            type(up),
            id(up["weight"]),
            id(gate["weight"]),
            up["weight"].shape,
            gate["weight"].shape,
        )
        if self._fused_gate_up_cache is not None:
            cached_key, params = self._fused_gate_up_cache
            if cached_key == key:
                return params

        weight = mx.concatenate([up["weight"], gate["weight"]], axis=1)
        bias = None
        if "bias" in up:
            bias = mx.concatenate([up["bias"], gate["bias"]], axis=1)
        if isinstance(up, QuantizedSwitchLinear):
            scales = mx.concatenate([up["scales"], gate["scales"]], axis=1)
            up_biases = up.get("biases")
            gate_biases = gate.get("biases")
            biases = None
            if up_biases is not None:
                biases = mx.concatenate([up_biases, gate_biases], axis=1)
            params = (weight, scales, biases, bias)
        else:
            params = (weight, bias)
        self._fused_gate_up_cache = (key, params)
        return params

    def _fused_gate_up(self, x, indices, sorted_indices=False):
        hidden_dims = self.up_proj.output_dims
        if isinstance(self.up_proj, QuantizedSwitchLinear):
            weight, scales, biases, bias = self._fused_gate_up_params()
            x = mx.gather_qmm(
                x,
                weight,
                scales,
                biases,
                rhs_indices=indices,
                transpose=True,
                group_size=self.up_proj.group_size,
                bits=self.up_proj.bits,
                mode=self.up_proj.mode,
                sorted_indices=sorted_indices,
            )
        else:
            weight, bias = self._fused_gate_up_params()
            x = mx.gather_mm(
                x,
                weight.swapaxes(-1, -2),
                rhs_indices=indices,
                sorted_indices=sorted_indices,
            )
        if bias is not None:
            x = x + mx.expand_dims(bias[indices], -2)
        return x[..., :hidden_dims], x[..., hidden_dims:]

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        # When we have many tokens, then sort them to make sure that the access
        # of different experts is in order.
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        if self._can_fuse_gate_up():
            x_up, x_gate = self._fused_gate_up(x, idx, sorted_indices=do_sort)
        else:
            x_up = self.up_proj(x, idx, sorted_indices=do_sort)
            x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
        x = self.down_proj(
            self.activation(x_up, x_gate),
            idx,
            sorted_indices=do_sort,
        )

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)


class SwitchMLP(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=nn.GELU(approx="precise"),
        bias: bool = False,
    ):
        super().__init__()

        self.fc1 = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.fc2 = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        # When we have many tokens, then sort them to make sure that the access
        # of different experts is in order.
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        x = self.fc1(x, idx, sorted_indices=do_sort)
        x = self.activation(x)
        x = self.fc2(x, idx, sorted_indices=do_sort)

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)
