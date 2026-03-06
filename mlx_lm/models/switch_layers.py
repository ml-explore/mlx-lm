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
    ):
        super().__init__()

        self.gate_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.up_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def _fuse_projections(self):
        """Build a cached fused gate+up weight tensor for inference.

        Concatenates gate_proj and up_proj weights along the output
        dimension so that one gather_qmm (or gather_mm) dispatch replaces
        two.  The result is split at the midpoint -- a zero-copy slice.

        The original submodules are kept for state_dict compatibility and
        training.  The fused cache is only used during inference.
        """
        gp = self.gate_proj
        up = self.up_proj

        if isinstance(gp, QuantizedSwitchLinear) and isinstance(
            up, QuantizedSwitchLinear
        ):
            # Require matching quantization config
            if (
                gp.group_size != up.group_size
                or gp.bits != up.bits
                or gp.mode != up.mode
                or gp.num_experts != up.num_experts
                or gp.input_dims != up.input_dims
            ):
                self._fused = False
                return

            # Require symmetric bias configuration
            if ("bias" in gp) != ("bias" in up):
                self._fused = False
                return
            if (gp.biases is not None) != (up.biases is not None):
                self._fused = False
                return

            self._fused_weight = mx.concatenate([gp.weight, up.weight], axis=1)
            self._fused_scales = mx.concatenate([gp.scales, up.scales], axis=1)

            self._fused_biases = None
            if gp.biases is not None:
                self._fused_biases = mx.concatenate([gp.biases, up.biases], axis=1)

            self._fused_bias = None
            if "bias" in gp:
                self._fused_bias = mx.concatenate([gp.bias, up.bias], axis=1)

            arrays = [self._fused_weight, self._fused_scales]
            if self._fused_biases is not None:
                arrays.append(self._fused_biases)
            if self._fused_bias is not None:
                arrays.append(self._fused_bias)
            mx.eval(*arrays)

            self._fused_out_dims = gp.output_dims
            self._fused_group_size = gp.group_size
            self._fused_bits = gp.bits
            self._fused_mode = gp.mode
            self._fused = "qmm"

        elif isinstance(gp, SwitchLinear) and isinstance(up, SwitchLinear):
            # Require compatible shapes
            if gp.num_experts != up.num_experts or gp.input_dims != up.input_dims:
                self._fused = False
                return

            if ("bias" in gp) != ("bias" in up):
                self._fused = False
                return

            self._fused_weight = mx.concatenate([gp.weight, up.weight], axis=1)

            self._fused_bias = None
            if "bias" in gp:
                self._fused_bias = mx.concatenate([gp.bias, up.bias], axis=1)

            arrays = [self._fused_weight]
            if self._fused_bias is not None:
                arrays.append(self._fused_bias)
            mx.eval(*arrays)

            self._fused_out_dims = gp.output_dims
            self._fused = "mm"

        else:
            self._fused = False

    def __call__(self, x, indices) -> mx.array:
        # Build fused weight cache on first inference-mode forward pass.
        if not self.training and not hasattr(self, "_fused"):
            self._fuse_projections()

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

        fused = getattr(self, "_fused", False)

        if not self.training and fused == "qmm":
            combined = mx.gather_qmm(
                x,
                self._fused_weight,
                self._fused_scales,
                self._fused_biases,
                rhs_indices=idx,
                transpose=True,
                group_size=self._fused_group_size,
                bits=self._fused_bits,
                mode=self._fused_mode,
                sorted_indices=do_sort,
            )
            if self._fused_bias is not None:
                combined = combined + mx.expand_dims(self._fused_bias[idx], -2)
            x_gate = combined[..., : self._fused_out_dims]
            x_up = combined[..., self._fused_out_dims :]
        elif not self.training and fused == "mm":
            combined = mx.gather_mm(
                x,
                self._fused_weight.swapaxes(-1, -2),
                rhs_indices=idx,
                sorted_indices=do_sort,
            )
            if self._fused_bias is not None:
                combined = combined + mx.expand_dims(self._fused_bias[idx], -2)
            x_gate = combined[..., : self._fused_out_dims]
            x_up = combined[..., self._fused_out_dims :]
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
