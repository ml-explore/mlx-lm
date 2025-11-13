import mlx.core as mx

import numpy as np
import torch

# Refer to this https://www.shashankshekhar.com/blog/apple-metal-vs-nvidia-cuda for Metal kernel terminology
def make_sub_1bit_streamk_kernel():
    """
    Custom Metal kernel that performs matrix multiplication directly on
    packed weights and scales the output. This eliminates the need to
    store unpacked weights in memory.
    """
    source = """
    const uint pid = threadgroup_position_in_grid.x;

    const uint lane_id = thread_position_in_threadgroup.x;
    const uint warp_id = thread_position_in_threadgroup.y;

    uint total_threads = threads_per_threadgroup.x * threads_per_threadgroup.y;

    for (uint tile_id=pid; tile_id < GRID_MNK; tile_id += NUM_METAL_CORES) {
        const uint pid_k = tile_id / GRID_MN;
        const uint tile_id_mn = tile_id % GRID_MN;

        const uint pid_m = tile_id_mn / NUM_PID_N;
        const uint pid_n = tile_id_mn % NUM_PID_N;

        const uint offs_xq_m = pid_m * BLOCK_SIZE_M;
        const uint offs_wq_n = pid_n * BLOCK_SIZE_N;

        const uint offs_k = pid_k * BLOCK_SIZE_K;

        // Reg Accumulator located by (m_coord, n_coord), where acc[m][n] per thread sotres a partial sum
        half acc[reg_tile_m][reg_tile_n] = {0.0};

        constexpr uint BLOCK_SIZE_K_PACKED = BLOCK_SIZE_K / packed_size;

        for (uint k_start = offs_k; k_start < K; k_start += BLOCK_SIZE_K * split_k) {
            // load tile_xq and tile_w from global memory to threadgroup memory
            threadgroup half tile_xq[BLOCK_SIZE_M][BLOCK_SIZE_K];
            
            // load tile_wq in packed format from global memory to private memory
            threadgroup uint16_t tile_wq[BLOCK_SIZE_N][BLOCK_SIZE_K_PACKED];

            // TODO (yiakw) : using BlockMMA load
            // load xq cooperatively
            {
                for (uint i=lane_id + warp_id * threads_per_threadgroup.x; i < BLOCK_SIZE_M * BLOCK_SIZE_K; i+=total_threads) {
                    const uint m = i / BLOCK_SIZE_K;
                    const uint k = i % BLOCK_SIZE_K;

                    tile_xq[m][k] = x[(offs_xq_m + m) * stride_x_m + (k_start + k) * stride_x_k];
                }
            } // end of load xq

            threadgroup_barrier(mem_flags::mem_threadgroup);


            // load wq cooperatively
            {
                for (uint i=lane_id + warp_id * threads_per_threadgroup.x; i < BLOCK_SIZE_N * BLOCK_SIZE_K_PACKED; i+=total_threads) {
                    const uint n = i / BLOCK_SIZE_K_PACKED;
                    const uint k = i % BLOCK_SIZE_K_PACKED;

                    tile_wq[n][k] = packed_weights[(offs_wq_n + n) * stride_w_n + (k_start / packed_size + k) * stride_w_k];
                }
            } // end of load wq

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // TODO (yiakw) : using BlockMMA mma and simdgroup_multiply_accumulate, see https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
            // compute acc
            {

                uint m_thr_coord = lane_id / threads_per_threadgroup.x + warp_id * reg_tile_m;
                uint n_thr_coord = lane_id % threads_per_threadgroup.x;

                uint i=0;
                uint j=0;

                for (uint n = n_thr_coord; n < BLOCK_SIZE_N; n += reg_tile_n * threads_per_threadgroup.x) {
                    // generate partial sum
                    for (uint k = 0; k < BLOCK_SIZE_K && k_start + k < K; k++) {

                        threadgroup uint16_t& wq_i = tile_wq[n][k / packed_size];
                        half wq_depacked_i = half((wq_i >> (k % packed_size)) & 1) * 2 - 1;
                        //uint16_t wq_depacked_i = ((wq_i << (15 - k % packed_size)) ^ 0x8000);
                        for (uint m = m_thr_coord, i=0; m < BLOCK_SIZE_M; m += reg_tile_m * threads_per_threadgroup.y) {

                            // generate partial sum
                            half xq_i = tile_xq[m][k];

                            /*
                            // "reinterpret_cast" is not supported in Metal Shading Language
                            union {
                                uint16_t u;
                                half f;
                            } datum;

                            datum.f = xq_i;
                            datum.u ^= wq_depacked_i;
                            acc[i][j] += datum.f;
                             */

                            acc[i][j] += xq_i * wq_depacked_i;

                            i+=1;
                        } // end of m loop
                    }
                    
                    // horizontal sum within simd lane
                    acc[i][j] = simd_sum(acc[i][j]);

                    j+=1;
                }
 
            } // end of compute acc

            threadgroup_barrier(mem_flags::mem_threadgroup);

        } // end of stream-k loop

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // store acc to back to global memory
        {
            
                uint m_thr_coord = lane_id / threads_per_threadgroup.x + warp_id * reg_tile_m;
                uint n_thr_coord = lane_id % threads_per_threadgroup.x;

                uint i=0;
                uint j=0;

                for (uint m = m_thr_coord; m < BLOCK_SIZE_M; m += reg_tile_m * threads_per_threadgroup.y) {

                    // apply group/channel wise scale
                    float scale = invert_weight_scales ? 1.0 / weight_scale[(offs_xq_m + m)] : weight_scale[(offs_xq_m + m)];

                    for (uint n = n_thr_coord; n < BLOCK_SIZE_N; n += reg_tile_n * threads_per_threadgroup.x) {
                        if ( offs_xq_m + m < M && offs_wq_n + n < N ) {
                            out[(offs_xq_m + m) * stride_o_m + (offs_wq_n + n) * stride_o_n] += acc[i][j] * scale;
                        }
                       j+=1;
                    }
                    i+=1;
                }

        } // end of store acc back to global memory

    }
    """

    return mx.fast.metal_kernel(
        name="sub_1bit_streamk_matmul",
        input_names=["x", "packed_weights", "weight_scale"],
        output_names=["out"],
        source=source,
    )


_sub_1bit_streamk_kernel = make_sub_1bit_streamk_kernel()


# NOTE (yiakwy) : pack sub-1bit model weights in 1-bit format
def pack_1_bit_weights(weights, dtype=torch.int16):
    N, K = weights.shape
    pack_size = dtype.itemsize * 8
    assert K % pack_size == 0, f"K must be divisible by {pack_size}, but got K={K}"

    binary = ((weights + 1) // 2).to(dtype)

    binary = binary.reshape(N, K // pack_size, pack_size)
    packed = torch.zeros((N, K // pack_size), dtype=dtype, device=weights.device)

    for i in range(pack_size):
        packed |= binary[:, :, i] << i

    return packed


def ceil_div(a, b):
    return (a + b - 1) // b

# NOTE (yiakwy): implement fp8 group scaled gemm using sub-1bit kernel
def fp8_group_scaled_gemm(xq, packed_weights, o=None, weight_scale=None, **tuning_config):
    """
    Perform matrix multiplication using packed 1-bit weights with scaling.

    Inputs:
        xq: (M, K) float16 for mps backend
        packed_weights: (N, K / packed_size) uint8
    Outputs:
        out: (M, N) float16/float32
    """
    if weight_scale is None:
        weight_scale = mx.ones((xq.shape[0],))
    
    M, K = xq.shape
    N = packed_weights.shape[0]

    # split K dimension into chunks for streaming to ensure minimum work done per metal threadgroup
    SPLIT_K = 1

    assert K % SPLIT_K == 0, "K must be divisible by SPLIT_K"

    BLOCK_SIZE_M = tuning_config.get("BLOCK_SIZE_M", 2)
    BLOCK_SIZE_N = tuning_config.get("BLOCK_SIZE_N", 2)

    BLOCK_SIZE_K = tuning_config.get("BLOCK_SIZE_K", 16)
    assert K % BLOCK_SIZE_K == 0, "K must be divisible by BLOCK_SIZE_K"

    NUM_METAL_CORES = 20

    # NOTE (yiakwy) : apple fast kernel does not support inplace of output
    if o is None:
        outDtype = mx.float16

    NUM_PID_M = ceil_div(M, BLOCK_SIZE_M)
    NUM_PID_N = ceil_div(N, BLOCK_SIZE_N)

    GRID_MN = NUM_PID_M * NUM_PID_N
    GRID_MNK = GRID_MN * SPLIT_K

    grid = (
        min(NUM_METAL_CORES, ceil_div(M, BLOCK_SIZE_M) * ceil_div(N, BLOCK_SIZE_N) * SPLIT_K),
        1,
        1
    )

    threadgroup = (32, 4, 1)

    # NOTE (yiakwy) : sub 1-bit kernel only supports 1-bit weights
    bits_per_weight = 1
    packed_size = packed_weights.dtype.size * 8 // bits_per_weight
    assert BLOCK_SIZE_K % packed_size == 0, "BLOCK_SIZE_K must be divisible by packed_size"

    o = _sub_1bit_streamk_kernel(
        inputs=[
            xq,
            packed_weights,
            weight_scale,
        ],
        template = [
            ("T", mx.float16),
            ("invert_weight_scales", False),
            ("M", M),
            ("K", K),
            ("N", N),
            ("stride_x_m", K),
            ("stride_x_k", 1),
            ("stride_w_n", K // packed_size),
            ("stride_w_k", 1),
            ("stride_o_m", N),
            ("stride_o_n", 1),
            ("bits_per_weight", bits_per_weight),
            ("packed_size", packed_size),
            ("BLOCK_SIZE_M", BLOCK_SIZE_M),
            ("BLOCK_SIZE_N", BLOCK_SIZE_N),
            ("BLOCK_SIZE_K", BLOCK_SIZE_K),
            ("split_k", SPLIT_K),
            ("reg_tile_m", ceil_div(BLOCK_SIZE_M, threadgroup[1])),
            ("reg_tile_n", ceil_div(BLOCK_SIZE_N, threadgroup[0])),
            ("NUM_PID_M", NUM_PID_M),
            ("NUM_PID_N", NUM_PID_N),
            ("GRID_MNK", GRID_MNK),
            ("GRID_MN", GRID_MN),
            ("NUM_METAL_CORES", NUM_METAL_CORES),
        ],
        grid=grid,
        threadgroup=threadgroup, 
        output_shapes=[(M, N)],
        output_dtypes=[outDtype],
        verbose=False,
    )[0]

    return o


# support torch mps backend, mlx backend, and ANE backend
def test_fp8_group_scaled_gemm():
    test_configs = [
        # (4096, 16384, 4096),
        (2, 16, 2)
    ]

    for M, K, N in test_configs:
        print(f"\n{'='*60}")
        print(f"Testing M={M}, K={K}, N={N}")
        print(f"{'='*60}")

        torch.manual_seed(42)
        input_fp16 = torch.randn(M, K, dtype=torch.float16, device="mps")
        weights = torch.randint(0, 2, (K, N), device="mps").float() * 2 - 1

        weights_fp16 = weights.half()

        print(f"input_fp16 : {input_fp16}")
        print(f"weights_fp16 : {weights_fp16}")

        # correctness check
        output_torch = torch.matmul(input_fp16, weights_fp16)

        # packed version
        weights_int16 = torch.zeros((N, K), dtype=torch.float16, device="mps")
        weights_int16[:] = weights_fp16.T[:]
        packed_weights = pack_1_bit_weights(weights_int16, dtype=torch.uint16)

        input_fp16_mxl = mx.array(input_fp16.cpu().numpy())
        packed_weights_mxl = mx.array(packed_weights.cpu().numpy())
        output_mlx = fp8_group_scaled_gemm(input_fp16_mxl, packed_weights_mxl, weight_scale=None)
        mx.eval(output_mlx)

        max_error = np.max(np.abs(output_torch.cpu().numpy() - output_mlx))
        print(f"Max error between torch and mlx: {max_error:.6f}")

        print(f"output_torch : {output_torch}")
        print(f"output_mlx : {output_mlx}")

        # Performance benchmark for MLX

        # import time

        # # Warm up
        # for _ in range(10):
        #     _ = torch.matmul(input_fp16, weights_fp16)
        #     _ = fp8_group_scaled_gemm(input_fp16_mxl, packed_weights_mxl, weight_scale=None)
        #     pass

        # torch.mps.synchronize()

        # # Benchmark Pytorch MPS backend
        # start_time = time.time()
        # for _ in range(10):
        #     _ = torch.matmul(input_fp16, weights_fp16)
        # torch.mps.synchronize()
        # torch_elpase = (time.time() - start_time)/ 10 * 1000
        
        # # Benchmark MLX backend
        # start_time = time.time()
        # outs = []
        # for _ in range(10):
        #     out = fp8_group_scaled_gemm(input_fp16_mxl, packed_weights_mxl, weight_scale=None)
        #     outs.append(out)
        # mx.eval(outs)
        # mlx_elpase = (time.time() - start_time)/ 10 * 1000

        # print(f"Pytorch MPS backend average time over 10 runs: {torch_elpase:.2f} ms")
        # print(f"MLX backend average time over 10 runs: {mlx_elpase:.2f} ms")


if __name__ == "__main__":
    test_fp8_group_scaled_gemm()