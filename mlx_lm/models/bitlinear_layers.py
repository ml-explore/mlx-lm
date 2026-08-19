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


def _bitlinear_matmul_ops(x, packed_weights, out_features, weight_scale, invert_weight_scales):
    """
    Ops-based reference/fallback for the packed-ternary matmul above, used
    whenever a Metal backend is unavailable (e.g. CPU, CUDA). Slower and
    less memory-efficient than the Metal kernel (it materializes the
    unpacked ternary weight matrix), but numerically equivalent: each byte
    of `packed_weights` holds four 2-bit codes in {0, 1, 2} (bias +1, so
    ternary value = code - 1) for four different output rows, laid out the
    same way the kernel reads them -- row ``row_idx`` of the packed matrix,
    2-bit field ``i``, decodes to output row ``row_idx + i * packed_out_features``.
    """
    packed_out_features, in_features = packed_weights.shape
    if packed_weights.dtype != mx.uint8:
        # `weight` is a plain module parameter, so generic parameter-dtype
        # casts (e.g. model.update(tree_map(lambda p: p.astype(t), ...)))
        # can turn it into a float array of the same small integer values
        # (0-255) it always holds. mx.right_shift rejects floats outright,
        # whereas the Metal kernel's C++ `uint8_t w = packed_weights[...]`
        # narrows a float to the same values implicitly -- normalize here
        # so this fallback matches that behavior instead of raising.
        packed_weights = packed_weights.astype(mx.uint8)
    shifts = mx.array([0, 2, 4, 6], dtype=mx.uint8)
    # (packed_out_features, 4, in_features), each entry a 2-bit code in {0..3}
    codes = (packed_weights[:, None, :] >> shifts[None, :, None]) & 3
    ternary = codes.astype(x.dtype) - 1
    # -> output row o = row_idx + i * packed_out_features, i.e. `i` is the
    # slower-varying index once flattened.
    weight = ternary.transpose(1, 0, 2).reshape(-1, in_features)[:out_features]
    y = x @ weight.T
    scale = (1 / weight_scale) if invert_weight_scales else weight_scale
    return y * scale


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
        if mx.metal.is_available():
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
        else:
            # No Metal back-end (CPU, CUDA, ...) -- fall back to plain ops.
            out = _bitlinear_matmul_ops(
                x,
                packed_weights,
                out_features,
                self.weight_scale,
                self.invert_weight_scales,
            )

        if len(original_shape) > 2:
            out = out.reshape(*original_shape[:-1], out_features)
        return out

    def __call__(self, x):
        y = self.execute_matmul_kernel(x, self.weight)

        if self.bias is not None:
            y = mx.add(y, self.bias)
        return y
