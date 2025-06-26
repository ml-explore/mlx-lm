import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.quantized import QuantizedLinear


class BitLinear(nn.Module):
    """
    BitLinear module with memory-efficient weight handling.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        dtype=mx.float16,
        invert_weight_scales=False,
    ):
        super().__init__()
        self.dtype = dtype
        self.in_features = in_features
        self.out_features = out_features
        # Calculate packed dimensions - the first dimension gets packed 4:1
        # The weights are ternary so can be represented with 2 bits,
        # and they are packed in uint8 tensors, hence the number of values per item is 4
        packed_out_features = (out_features + 3) // 4
        self.weight = mx.zeros((packed_out_features, in_features), dtype=mx.uint8)

        self.invert_weight_scales = invert_weight_scales
        self.weight_scale = mx.array([1.0], dtype=dtype)

        if bias:
            self.bias = mx.zeros((out_features,), dtype=dtype)
        else:
            self.bias = None

        # Compile kernel
        self._compiled_kernel = self.compile_matmul_kernel()

    def compile_matmul_kernel(self):
        """
        Custom Metal kernel that performs matrix multiplication directly on packed weights and scales the output.
        This eliminates the need to store unpacked weights in memory.
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
                out[batch_idx * out_features + row_idx + i * (out_features/4)] = sum[i] * scale;
            }
        }
        """

        return mx.fast.metal_kernel(
            name="bitlinear_matmul",
            input_names=["x", "packed_weights", "weight_scale", "invert_weight_scales"],
            output_names=["out"],
            source=source,
        )

    def execute_matmul_kernel(self, x, packed_weights):
        # Handle multi-dimensional inputs by flattening all but the last dimension
        original_shape = x.shape
        if len(original_shape) > 2:
            # Flatten to (total_batch_elements, in_features)
            x_flattened = x.reshape(-1, original_shape[-1])
            total_batch_elements = x_flattened.shape[0]
            in_features = x_flattened.shape[1]
        else:
            x_flattened = x
            total_batch_elements, in_features = x_flattened.shape

        out_features = self.out_features

        outputs = self._compiled_kernel(
            inputs=[
                x_flattened.astype(self.dtype),
                packed_weights,
                self.weight_scale,
                self.invert_weight_scales,
            ],
            template=[
                ("batch_size", total_batch_elements),
                ("in_features", in_features),
                ("out_features", out_features),
            ],
            grid=(32, total_batch_elements * out_features // 4, 1),
            threadgroup=(32, 1, 1),  # SIMD width is 32 threads
            output_shapes=[(total_batch_elements, out_features)],
            output_dtypes=[self.dtype],
        )

        # Reshape output back to match input shape but with out_features as last dimension
        if len(original_shape) > 2:
            output_shape = original_shape[:-1] + (out_features,)
            return outputs[0].reshape(output_shape)
        else:
            return outputs[0]

    def __call__(self, x):
        """
        Forward pass with weight scaling applied correctly.
        """
        org_dtype = x.dtype

        # Use custom kernel for matrix multiplication directly on packed weights
        y = self.execute_matmul_kernel(x, self.weight)

        # Add bias if present
        if self.bias is not None:
            y = mx.add(y, self.bias)
        return y.astype(org_dtype)


class QuantAndBitLinear(nn.Linear):
    """
    A Linear layer that can be converted to a quantized and bitlinear version.
    """

    def to_quantized(self, method: str = None, group_size: int = 64, bits: int = 4, **kwargs):

        if method is None or group_size is None or bits is None:
            return QuantizedLinear.from_linear(self, group_size, bits)

        if method == "bitnet":
            bitlinear = BitLinear(
                in_features=self.weight.shape[1],
                out_features=self.weight.shape[0],
                bias= getattr(self, "bias", None) is not None,
                invert_weight_scales=True,
                **kwargs
            )
            return bitlinear
        else:
            raise ValueError(f"Unknown quantization method: {method}")

