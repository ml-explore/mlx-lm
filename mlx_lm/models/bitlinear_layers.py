# Copyright © 2025 Apple Inc.

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.quantized import QuantizedLinear
from mlx.utils import tree_flatten, tree_unflatten


def bitnet_quantize(model, quantization_config: dict):
    quantize_layers = []
    modules_to_not_convert = quantization_config.get("modules_to_not_convert", [])
    invert_weight_scales = (
        quantization_config.get("linear_class", "") != "autobitlinear"
    )

    for name, module in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module):

        # Replace nn.Linear layers, but skip any layer from the `modules_to_not_convert` list
        if name not in modules_to_not_convert and isinstance(module, nn.Linear):
            old_weight = module.weight
            out_features, in_features = old_weight.shape
            bias = "bias" in module
            new_layer = BitLinear(
                in_features,
                out_features,
                bias=bias,
                invert_weight_scales=invert_weight_scales,
            )
            quantize_layers.append((name, new_layer))
    if len(quantize_layers) > 0:
        model.update_modules(tree_unflatten(quantize_layers))
    return model


def make_bitlinear_kernel():
    """
    Custom Metal kernel that performs matrix multiplication directly on
    packed weights and scales the output. This eliminates the need to
    store unpacked weights in memory.
    """
    source = """
    constexpr int M = 4;
    constexpr int BLOCK = 32;

    uint tid = thread_position_in_grid.y;
    uint in_offset = thread_position_in_grid.x;

    uint batch_idx = tid / (out_features / 4);
    uint row_idx = tid % (out_features / 4);

    float sum[4] = {0.0};

    for (uint i = in_offset * M; i < in_features; i += BLOCK * M) {
        float v[M];
        for (int j=0; j<M; j++) {
            v[j] = x[batch_idx * in_features + i + j];
        }

        for (int j=0; j<M; j++) {
            uint8_t w = packed_weights[row_idx * in_features + i + j];
            sum[0] += v[j] * ((w & 3) - 1);
            sum[1] += v[j] * (((w >> 2) & 3) - 1);
            sum[2] += v[j] * (((w >> 4) & 3) - 1);
            sum[3] += v[j] * (((w >> 6) & 3) - 1);
        }
    }

    for (int j=0; j<4; j++) {
        sum[j] = simd_sum(sum[j]);
    }

    // Apply weight scaling by diving them or multiplying them
    if (in_offset == 0) {
        float scale = invert_weight_scales ? 1 / weight_scale[0] : weight_scale[0];
        for (int i=0; i<4; i++) {
            out[batch_idx * out_features + row_idx + i * (out_features/4)] = static_cast<T>(sum[i] * scale);
        }
    }
    """

    return mx.fast.metal_kernel(
        name="bitlinear_matmul",
        input_names=["x", "packed_weights", "weight_scale"],
        output_names=["out"],
        source=source,
    )


_bitlinear_kernel = make_bitlinear_kernel()


class BitLinear(nn.Module):
    """
    BitLinear module with memory-efficient weight handling.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        invert_weight_scales=False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Calculate packed dimensions - the first dimension gets packed 4:1
        # The weights are ternary so can be represented with 2 bits, and they
        # are packed in uint8 tensors, hence the number of values per item is 4
        packed_out_features = (out_features + 3) // 4
        self.weight = mx.zeros((packed_out_features, in_features), dtype=mx.uint8)

        self.invert_weight_scales = invert_weight_scales
        self.weight_scale = mx.array([1.0])

        if bias:
            self.bias = mx.zeros((out_features,))
        else:
            self.bias = None

    def execute_matmul_kernel(self, x, packed_weights):
        original_shape = x.shape
        if len(original_shape) > 2:
            x = x.reshape(-1, original_shape[-1])
        total_batch_elements, in_features = x.shape

        out_features = self.out_features

        dtype = self.weight_scale.dtype
        assert x.dtype == dtype, "Wrong type for input."
        out = _bitlinear_kernel(
            inputs=[
                x,
                packed_weights,
                self.weight_scale,
            ],
            template=[
                ("T", dtype),
                ("invert_weight_scales", self.invert_weight_scales),
                ("in_features", in_features),
                ("out_features", out_features),
            ],
            grid=(32, total_batch_elements * out_features // 4, 1),
            threadgroup=(32, 1, 1),  # SIMD width is 32 threads
            output_shapes=[(total_batch_elements, out_features)],
            output_dtypes=[dtype],
        )[0]

        if len(original_shape) > 2:
            out = out.reshape(*original_shape[:-1], out_features)
        return out

    def __call__(self, x):
        y = self.execute_matmul_kernel(x, self.weight)

        if self.bias is not None:
            y = mx.add(y, self.bias)
        return y
