import mlx.core as mx

import numpy as np
import torch

import subprocess

def get_mlx_gpu_cores():
    result = subprocess.run(
        ["/usr/sbin/system_profiler SPDisplaysDataType | grep 'Total Number\ of\ Cores:\ ' | awk '{print $NF}'"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        shell=True,
    )

    if result.returncode != 0:
        return False, None

    gpu_cores = int(result.stdout.strip())
    return True, [gpu_cores]


# Refer to this https://www.shashankshekhar.com/blog/apple-metal-vs-nvidia-cuda for Metal kernel terminology
def make_sub_1bit_streamk_kernel():
    """
    Custom Metal kernel that performs matrix multiplication directly on
    packed weights and scales the output. This eliminates the need to
    store unpacked weights in memory.
    """

    header = """
    #include <metal_atomic>
    #include <metal_simdgroup>
    """

    source = """
    // total blocks threads_per_grid.x / threads_per_threadgroup.x
    const uint pid = threadgroup_position_in_grid.x;

    const uint lane_id = thread_position_in_threadgroup.x;
    const uint warp_id = thread_position_in_threadgroup.y;

    const uint tid = lane_id + warp_id * threads_per_threadgroup.x;

    const uint total_threads = threads_per_threadgroup.x * threads_per_threadgroup.y;

    for (uint tile_id=pid; tile_id < kGRID_MNK; tile_id += NUM_METAL_CORES) {
        const uint pid_k = tile_id / kGRID_MN;
        const uint tile_id_mn = tile_id % kGRID_MN;

        const uint pid_m = tile_id_mn / NUM_PID_N;
        const uint pid_n = tile_id_mn % NUM_PID_N;

        const uint offs_xq_m = pid_m * BLOCK_SIZE_M;
        const uint offs_wq_n = pid_n * BLOCK_SIZE_N;

        const uint offs_k = pid_k * BLOCK_SIZE_K;

        // Reg Accumulator located by (m_coord, n_coord), where acc[m][n] per thread sotres a partial sum
        thread AccT acc[reg_tile_m][reg_tile_n] = {0.0};

        constexpr uint BLOCK_SIZE_K_PACKED = BLOCK_SIZE_K / packed_size;

        for (uint k_start = offs_k; k_start < K; k_start += BLOCK_SIZE_K * split_k) {
            // load tile_xq and tile_w from global memory to threadgroup memory
            threadgroup half tile_xq[BLOCK_SIZE_M][BLOCK_SIZE_K];
            
            // load tile_wq in packed format from global memory to private memory
            threadgroup uint16_t tile_wq[BLOCK_SIZE_N][BLOCK_SIZE_K_PACKED];

            // TODO (yiakwy) : using BlockMMA load with vectorized loads
            // load xq cooperatively
            {
                for (uint i=lane_id + warp_id * threads_per_threadgroup.x; i < BLOCK_SIZE_M * BLOCK_SIZE_K; i+=total_threads) {
                    const uint m = i / BLOCK_SIZE_K;
                    const uint k = i % BLOCK_SIZE_K;

                    if (offs_xq_m + m < M && k_start + k < K) {
                        tile_xq[m][k] = x[(offs_xq_m + m) * stride_x_m + (k_start + k) * stride_x_k];
                    }
                }
            } // end of load xq

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // load wq cooperatively
            {
                for (uint i=lane_id + warp_id * threads_per_threadgroup.x; i < BLOCK_SIZE_N * BLOCK_SIZE_K_PACKED; i+=total_threads) {
                    const uint n = i / BLOCK_SIZE_K_PACKED;
                    const uint k = i % BLOCK_SIZE_K_PACKED;

                    if (offs_wq_n + n < N && (k_start / packed_size) + k < K) {
                        tile_wq[n][k] = packed_weights[(offs_wq_n + n) * stride_w_n + ((k_start / packed_size) + k) * stride_w_k];
                    }
                }
            } // end of load wq

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // TODO (yiakwy) : using BlockMMA mma and simdgroup_multiply_accumulate, see https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
            // compute acc
            {

                // TODO (yiakwy) : warp size will be optimized later, this is for 1x32 frag
                constexpr uint warp_frag_size_n = WARP_SIZE_N;
                constexpr uint warp_frag_size_m = WARP_SIZE_M;

                thread const uint& warps = threads_per_threadgroup.y;

                uint m_thr_coord = lane_id / warp_frag_size_n + warp_id * warp_frag_size_m;
                uint n_thr_coord = lane_id % warp_frag_size_n;

                for (uint j=0, n = n_thr_coord; j < reg_tile_n; j++) {

                    // generate partial sum
                    for (uint k = 0; k < BLOCK_SIZE_K && k_start + k < K; k++) {

                        uint16_t wq_nk = tile_wq[n + j * warp_frag_size_n][k / packed_size];
                        thread half wq_depacked_nk = half((wq_nk >> (k % packed_size)) & 1) * 2 - 1;
                        //uint16_t wq_depacked_nk = ((wq_nk << (15 - k % packed_size)) ^ 0x8000);
                        
                        // for (uint m = m_thr_coord; m < BLOCK_SIZE_M; m += reg_tile_m * warps) {
                        for (uint i=0, m = m_thr_coord; i < reg_tile_m; i++) {
                            // generate partial sum

                            thread half xq_mk = tile_xq[m + i * warps * warp_frag_size_m][k];

                            /*
                            // "reinterpret_cast" is not supported in Metal Shading Language
                            union {
                                uint16_t u;
                                half f;
                            } datum;

                            datum.f = xq_mk;
                            datum.u ^= wq_depacked_nk;
                            acc[i][j] += datum.f;
                             */

                            acc[i][j] += xq_mk * wq_depacked_nk;

                        } // end of reg_tile_n

                    } // end of k loop

                } // end of reg_tile_m
 
            } // end of compute acc

            threadgroup_barrier(mem_flags::mem_threadgroup);

        } // end of stream-k loop

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // store acc to back to global memory
        {
            constexpr uint warp_frag_size_n = WARP_SIZE_N;
            constexpr uint warp_frag_size_m = WARP_SIZE_M;

            thread const uint& warps = threads_per_threadgroup.y;

            uint m_thr_coord = lane_id / warp_frag_size_n + warp_id * warp_frag_size_m;
            uint n_thr_coord = lane_id % warp_frag_size_n;

            // out[(offs_xq_m) * stride_o_m + offs_wq_n * stride_o_n] += tile_id;
            // atomic_fetch_add_explicit((device atomic<float> *)(out + (offs_xq_m) * stride_o_m + offs_wq_n * stride_o_n), pid, memory_order_relaxed);

            // NOTE (yiakwy) : find a min func
            uint estimated_boundary_m = offs_xq_m + BLOCK_SIZE_M;
            uint estimated_boundary_n = offs_wq_n + BLOCK_SIZE_N;

            uint boundary_m = M < estimated_boundary_m ? M : estimated_boundary_m ;
            uint boundary_n = N < estimated_boundary_n ? N : estimated_boundary_n;

            for (uint i=0, m = m_thr_coord; i < reg_tile_m; i++) {

                // apply group/channel wise scale
                float scale = invert_weight_scales ? 1.0 / weight_scale[(offs_xq_m + m)] : weight_scale[(offs_xq_m + m)];

                for (uint j=0, n = n_thr_coord; j < reg_tile_n; j++) {
                    if ( offs_xq_m + m + i * warps * warp_frag_size_m < boundary_m && offs_wq_n + n + j * warp_frag_size_n < boundary_n ) {
                        out[(offs_xq_m + m + i * warps * warp_frag_size_m) * stride_o_m + (offs_wq_n + n + j * warp_frag_size_n) * stride_o_n] = acc[i][j] * scale;
                    }
                }
            }
        } // end of store acc back to global memory

    }
    """

    return mx.fast.metal_kernel(
        name="sub_1bit_streamk_matmul",
        input_names=["x", "packed_weights", "weight_scale"],
        output_names=["out"],
        source=source,
        header=header,
        ensure_row_contiguous=True,
        atomic_outputs=False
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


NUM_METAL_CORES = get_mlx_gpu_cores()[1][0]


print(f"NUM_METAL_CORES : {NUM_METAL_CORES}")

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

    BLOCK_SIZE_M = tuning_config.get("BLOCK_SIZE_M", 4)
    BLOCK_SIZE_N = tuning_config.get("BLOCK_SIZE_N", 4)

    WARP_SIZE_M = tuning_config.get("WARP_SIZE_M", 1)
    WARP_SIZE_N = tuning_config.get("WARP_SIZE_N", 32) # 1 el / thread

    BLOCK_SIZE_K = tuning_config.get("BLOCK_SIZE_K", 16)
    assert K % BLOCK_SIZE_K == 0, "K must be divisible by BLOCK_SIZE_K"

    # NOTE (yiakwy) : apple fast kernel does not support inplace of output
    if o is None:
        outDtype = mx.float16

    NUM_PID_M = ceil_div(M, BLOCK_SIZE_M)
    NUM_PID_N = ceil_div(N, BLOCK_SIZE_N)

    kGRID_MN = NUM_PID_M * NUM_PID_N
    kGRID_MNK = kGRID_MN * SPLIT_K

    cugrid = (
        min(NUM_METAL_CORES, ceil_div(M, BLOCK_SIZE_M) * ceil_div(N, BLOCK_SIZE_N) * SPLIT_K),
        1,
        1
    )

    # SIMD 
    simd_group_size = 32

    # concurrent warps
    warps = 4

    threadgroup = (simd_group_size, warps, 1)
    grid = (cugrid[0]* threadgroup[0], threadgroup[1], threadgroup[2])

    # determin threads level repetition
    reg_tile_m = ceil_div(BLOCK_SIZE_M, WARP_SIZE_M * warps)
    reg_tile_n = ceil_div(BLOCK_SIZE_N, WARP_SIZE_N)

    print(f"grids : {grid}")
    print(f"threadgroup : {threadgroup}")
    print(f"(reg_tile_m, reg_tile_n) = (WARP_SIZE_M, WAPR_SIZE_N) / (BLOCK_SIZE_M, BLOCK_SIZE_N): ({reg_tile_m}, {reg_tile_n}) = ({WARP_SIZE_M}, {WARP_SIZE_N}) / ({BLOCK_SIZE_M}, {BLOCK_SIZE_N})")

    # NOTE (yiakwy) : sub 1-bit kernel only supports 1-bit weights
    bits_per_weight = 1
    packed_size = packed_weights.dtype.size * 8 // bits_per_weight
    assert BLOCK_SIZE_K % packed_size == 0, "BLOCK_SIZE_K must be divisible by packed_size"

    print(f"packed_weights : {packed_weights.shape}")

    x = xq

    o = _sub_1bit_streamk_kernel(
        inputs=[
            x,
            packed_weights,
            weight_scale,
        ],
        template = [
            ("AccT", mx.float32),
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
            ("WARP_SIZE_M", WARP_SIZE_M),
            ("WARP_SIZE_N", WARP_SIZE_N),
            ("split_k", SPLIT_K),
            ("reg_tile_m", reg_tile_m),
            ("reg_tile_n", reg_tile_n),
            ("NUM_PID_M", NUM_PID_M),
            ("NUM_PID_N", NUM_PID_N),
            ("kGRID_MNK", kGRID_MNK),
            ("kGRID_MN", kGRID_MN),
            ("NUM_METAL_CORES", NUM_METAL_CORES),
        ],
        grid=grid,
        threadgroup=threadgroup, 
        output_shapes=[(M, N)],
        output_dtypes=[outDtype],
        verbose=True,
    )[0]

    return o


# support torch mps backend, mlx backend, and ANE backend
def test_fp8_group_scaled_gemm():
    test_configs = [
        # (4096, 16384, 4096),
        (8, 32, 8)
    ]

    for M, K, N in test_configs:
        print(f"\n{'='*60}")
        print(f"Testing M={M}, K={K}, N={N}")
        print(f"{'='*60}")

        torch.manual_seed(42)
        input_fp16 = torch.randn(M, K, dtype=torch.float16, device="mps")
        weights = torch.randint(0, 2, (K, N), device="mps").float() * 2 - 1

        weights_fp16 = weights.half()

        # print(f"input_fp16 : {input_fp16}")
        # print(f"weights_fp16 : {weights_fp16}")

        assert(weights_fp16.is_contiguous())

        # correctness check
        output_torch = torch.matmul(input_fp16, weights_fp16)

        # packed version
        weights_int16 = torch.zeros((N, K), dtype=torch.float16, device="mps")
        assert(weights_int16.is_contiguous())
        weights_int16[:] = weights_fp16.T[:]
        packed_weights = pack_1_bit_weights(weights_int16, dtype=torch.uint16)

        print(f"input_fp16 strides: {input_fp16.stride(0), input_fp16.stride(1)}, shape={input_fp16.shape}")
        print(f"weights_int16 strides : {weights_int16.stride(0), weights_int16.stride(1)}, shape={weights_int16.shape}")

        input_fp16_mxl = mx.array(input_fp16.cpu().numpy())
        packed_weights_mxl = mx.array(packed_weights.cpu().numpy())
        output_mlx = fp8_group_scaled_gemm(input_fp16_mxl, packed_weights_mxl, weight_scale=None)
        mx.eval(output_mlx)

        max_error = np.max(np.abs(output_torch.cpu().numpy() - output_mlx))
        print(f"Max error between torch and mlx: {max_error:.6f}")

        # print(f"output_torch : {output_torch}")

        # np.set_printoptions(128)
        # print(f"output_mlx : {np.array(output_mlx)}")
        # print(f"output_mlx : {output_mlx}")

        print(f"diff : {output_torch.cpu().numpy() - output_mlx}")

        # # Performance benchmark for MLX

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