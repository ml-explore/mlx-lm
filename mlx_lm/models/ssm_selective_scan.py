from typing import Optional, Tuple

import mlx.core as mx

_kernel_cache = {}


def _kernel_key(
    batch: int,
    dim: int,
    seqlen: int,
    dstate: int,
    b_ndim: int,
    c_ndim: int,
    ngroups_b: int,
    ngroups_c: int,
    has_d: bool,
    has_delta_bias: bool,
    has_initial_state: bool,
    delta_softplus: bool,
):
    return (
        batch,
        dim,
        seqlen,
        dstate,
        b_ndim,
        c_ndim,
        ngroups_b,
        ngroups_c,
        has_d,
        has_delta_bias,
        has_initial_state,
        delta_softplus,
    )


def selective_scan_fwd_with_state(
    u: mx.array,
    delta: mx.array,
    A: mx.array,
    B: mx.array,
    C: mx.array,
    D: Optional[mx.array] = None,
    delta_bias: Optional[mx.array] = None,
    initial_state: Optional[mx.array] = None,
    delta_softplus: bool = True,
) -> Tuple[mx.array, mx.array]:
    """Selective scan forward kernel returning (output, final_state).

    Shapes:
    - u: (batch, dim, seqlen)
    - delta: (batch, dim, seqlen)
    - A: (dim, dstate)
    - B: variable B in { (dim, dstate), (batch, dstate, seqlen), (batch, groups, dstate, seqlen) }
    - C: variable C in { (dim, dstate), (batch, dstate, seqlen), (batch, groups, dstate, seqlen) }
    - D: (dim,)
    - delta_bias: (dim,)
    """
    u_f32 = u.astype(mx.float32)
    delta_f32 = delta.astype(mx.float32)
    A_f32 = A.astype(mx.float32)
    B_f32 = B.astype(mx.float32)
    C_f32 = C.astype(mx.float32)

    batch, dim, seqlen = u.shape
    dstate = A.shape[1]

    has_d = D is not None
    has_delta_bias = delta_bias is not None
    has_initial_state = initial_state is not None

    b_ndim = B.ndim
    c_ndim = C.ndim
    ngroups_b = B.shape[1] if b_ndim == 4 else 1
    ngroups_c = C.shape[1] if c_ndim == 4 else 1

    num_threads = 128
    items_per_thread = 4
    chunk_size = num_threads * items_per_thread
    num_warps = num_threads // 32

    key = _kernel_key(
        batch,
        dim,
        seqlen,
        dstate,
        b_ndim,
        c_ndim,
        ngroups_b,
        ngroups_c,
        has_d,
        has_delta_bias,
        has_initial_state,
        delta_softplus,
    )

    if key not in _kernel_cache:
        header = f"""
        #define NUM_THREADS {num_threads}
        #define ITEMS_PER_THREAD {items_per_thread}
        #define CHUNK_SIZE {chunk_size}
        #define NUM_WARPS {num_warps}
        #define DSTATE_MAX 256
        #define HAS_D {1 if has_d else 0}
        #define HAS_DELTA_BIAS {1 if has_delta_bias else 0}
        #define HAS_INITIAL_STATE {1 if has_initial_state else 0}
        #define DELTA_SOFTPLUS {1 if delta_softplus else 0}
        #define IS_VARIABLE_B {1 if b_ndim >= 3 else 0}
        #define IS_VARIABLE_C {1 if c_ndim >= 3 else 0}
        #define B_NDIM {b_ndim}
        #define C_NDIM {c_ndim}
        #define NGROUPS_B {ngroups_b}
        #define NGROUPS_C {ngroups_c}
        #define SEQLEN {seqlen}
        #define DSTATE {dstate}
        #define DIM {dim}
        """

        source = """
        uint batch_id = threadgroup_position_in_grid.x;
        uint dim_id   = threadgroup_position_in_grid.y;
        uint tid      = thread_position_in_threadgroup.x;
        uint warp_id  = tid / 32;
        uint lane_id  = thread_index_in_simdgroup;

        threadgroup float smem_a[2][NUM_WARPS];
        threadgroup float smem_b[2][NUM_WARPS];
        threadgroup float smem_running_a[DSTATE_MAX];
        threadgroup float smem_running_b[DSTATE_MAX];

        if (tid < DSTATE) {
            smem_running_a[tid] = 1.0f;
            smem_running_b[tid] = HAS_INITIAL_STATE ? initial_state[batch_id * DIM * DSTATE + dim_id * DSTATE + tid] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float d_val = HAS_D ? D_skip[dim_id] : 0.0f;
        float delta_bias_val = HAS_DELTA_BIAS ? delta_bias[dim_id] : 0.0f;

        uint num_chunks = (SEQLEN + CHUNK_SIZE - 1) / CHUNK_SIZE;

        auto get_B_val = [&](uint b, uint n, uint l) -> float {
            if (!IS_VARIABLE_B) {
                return B[dim_id * DSTATE + n];
            } else if (B_NDIM == 3) {
                return B[b * DSTATE * SEQLEN + n * SEQLEN + l];
            } else {
                uint group_id = dim_id / (DIM / NGROUPS_B);
                return B[b * NGROUPS_B * DSTATE * SEQLEN + group_id * DSTATE * SEQLEN + n * SEQLEN + l];
            }
        };

        auto get_B_vals4 = [&](uint b, uint n, uint l_start) -> float4 {
            if (!IS_VARIABLE_B) {
                float v = B[dim_id * DSTATE + n];
                return float4(v, v, v, v);
            } else if (B_NDIM == 3) {
                uint offset = b * DSTATE * SEQLEN + n * SEQLEN + l_start;
                return *((device const float4*)(B + offset));
            } else {
                uint group_id = dim_id / (DIM / NGROUPS_B);
                uint offset = b * NGROUPS_B * DSTATE * SEQLEN + group_id * DSTATE * SEQLEN + n * SEQLEN + l_start;
                return *((device const float4*)(B + offset));
            }
        };

        auto get_C_val = [&](uint b, uint n, uint l) -> float {
            if (!IS_VARIABLE_C) {
                return C[dim_id * DSTATE + n];
            } else if (C_NDIM == 3) {
                return C[b * DSTATE * SEQLEN + n * SEQLEN + l];
            } else {
                uint group_id = dim_id / (DIM / NGROUPS_C);
                return C[b * NGROUPS_C * DSTATE * SEQLEN + group_id * DSTATE * SEQLEN + n * SEQLEN + l];
            }
        };

        auto get_C_vals4 = [&](uint b, uint n, uint l_start) -> float4 {
            if (!IS_VARIABLE_C) {
                float v = C[dim_id * DSTATE + n];
                return float4(v, v, v, v);
            } else if (C_NDIM == 3) {
                uint offset = b * DSTATE * SEQLEN + n * SEQLEN + l_start;
                return *((device const float4*)(C + offset));
            } else {
                uint group_id = dim_id / (DIM / NGROUPS_C);
                uint offset = b * NGROUPS_C * DSTATE * SEQLEN + group_id * DSTATE * SEQLEN + n * SEQLEN + l_start;
                return *((device const float4*)(C + offset));
            }
        };

        for (uint chunk = 0; chunk < num_chunks; chunk++) {
            uint chunk_start = chunk * CHUNK_SIZE;
            uint start = chunk_start + tid * ITEMS_PER_THREAD;

            float delta_vals[ITEMS_PER_THREAD];
            float delta_u_vals[ITEMS_PER_THREAD];
            float out_vals[ITEMS_PER_THREAD];

            #pragma unroll
            for (uint i = 0; i < ITEMS_PER_THREAD; i++) {
                delta_vals[i] = 0.0f;
                delta_u_vals[i] = 0.0f;
                out_vals[i] = 0.0f;
            }

            if (start + ITEMS_PER_THREAD <= SEQLEN) {
                uint offset = batch_id * DIM * SEQLEN + dim_id * SEQLEN + start;
                float4 u_vec = *((device const float4*)(u + offset));
                float4 delta_vec = *((device const float4*)(delta + offset));
                float u_arr[4] = {u_vec.x, u_vec.y, u_vec.z, u_vec.w};
                float d_arr[4] = {delta_vec.x, delta_vec.y, delta_vec.z, delta_vec.w};

                #pragma unroll
                for (uint i = 0; i < ITEMS_PER_THREAD; i++) {
                    float dv = d_arr[i] + delta_bias_val;
                    if (DELTA_SOFTPLUS) {
                        dv = (dv <= 20.0f) ? log(1.0f + exp(dv)) : dv;
                    }
                    delta_vals[i] = dv;
                    float uv = u_arr[i];
                    delta_u_vals[i] = dv * uv;
                    if (HAS_D) out_vals[i] = d_val * uv;
                }
            } else {
                #pragma unroll
                for (uint i = 0; i < ITEMS_PER_THREAD; i++) {
                    uint l = start + i;
                    if (l < SEQLEN) {
                        float dv = delta[batch_id * DIM * SEQLEN + dim_id * SEQLEN + l] + delta_bias_val;
                        if (DELTA_SOFTPLUS) {
                            dv = (dv <= 20.0f) ? log(1.0f + exp(dv)) : dv;
                        }
                        delta_vals[i] = dv;
                        float uv = u[batch_id * DIM * SEQLEN + dim_id * SEQLEN + l];
                        delta_u_vals[i] = dv * uv;
                        if (HAS_D) out_vals[i] = d_val * uv;
                    }
                }
            }

            for (uint n = 0; n < DSTATE; n++) {
                uint buf_idx = n & 1;
                float A_val = A[dim_id * DSTATE + n];
                float run_a = smem_running_a[n], run_b = smem_running_b[n];

                float exp_deltaA[ITEMS_PER_THREAD];
                #pragma unroll
                for (uint i = 0; i < ITEMS_PER_THREAD; i++) {
                    exp_deltaA[i] = exp(delta_vals[i] * A_val);
                }

                float la = 1.0f, lb = 0.0f;
                if (start + ITEMS_PER_THREAD <= SEQLEN) {
                    float4 B_vec = get_B_vals4(batch_id, n, start);
                    float B_arr[4] = {B_vec.x, B_vec.y, B_vec.z, B_vec.w};
                    #pragma unroll
                    for (uint i = 0; i < 4; i++) {
                        float a_bar = exp_deltaA[i];
                        float b_bar = delta_u_vals[i] * B_arr[i];
                        lb = a_bar * lb + b_bar;
                        la = a_bar * la;
                    }
                } else {
                    #pragma unroll
                    for (uint i = 0; i < ITEMS_PER_THREAD; i++) {
                        uint l = start + i;
                        float a_bar = (l < SEQLEN) ? exp_deltaA[i] : 1.0f;
                        float b_bar = (l < SEQLEN) ? delta_u_vals[i] * get_B_val(batch_id, n, l) : 0.0f;
                        lb = a_bar * lb + b_bar;
                        la = a_bar * la;
                    }
                }

                float sa = la, sb = lb;
                #pragma unroll
                for (uint off = 1; off < 32; off <<= 1) {
                    float pa = simd_shuffle_up(sa, off);
                    float pb = simd_shuffle_up(sb, off);
                    if (lane_id >= off) {
                        sb = sa * pb + sb;
                        sa = sa * pa;
                    }
                }

                if (lane_id == 31) {
                    smem_a[buf_idx][warp_id] = sa;
                    smem_b[buf_idx][warp_id] = sb;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (warp_id == 0) {
                    float wa = (lane_id < NUM_WARPS) ? smem_a[buf_idx][lane_id] : 1.0f;
                    float wb = (lane_id < NUM_WARPS) ? smem_b[buf_idx][lane_id] : 0.0f;
                    #pragma unroll
                    for (uint off = 1; off < 32; off <<= 1) {
                        float pa = simd_shuffle_up(wa, off);
                        float pb = simd_shuffle_up(wb, off);
                        if (lane_id >= off) {
                            wb = wa * pb + wb;
                            wa = wa * pa;
                        }
                    }
                    if (lane_id < NUM_WARPS) {
                        smem_a[buf_idx][lane_id] = wa;
                        smem_b[buf_idx][lane_id] = wb;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                float excl_a = (lane_id == 0) ? 1.0f : simd_shuffle_up(sa, 1);
                float excl_b = (lane_id == 0) ? 0.0f : simd_shuffle_up(sb, 1);

                float warp_prefix_a = (warp_id > 0) ? smem_a[buf_idx][warp_id - 1] : 1.0f;
                float warp_prefix_b = (warp_id > 0) ? smem_b[buf_idx][warp_id - 1] : 0.0f;

                float intra_prefix_a = excl_a * warp_prefix_a;
                float intra_prefix_b = excl_a * warp_prefix_b + excl_b;

                float final_prefix_a = intra_prefix_a * run_a;
                float final_prefix_b = intra_prefix_a * run_b + intra_prefix_b;

                if (tid == 0) {
                    float ca = smem_a[buf_idx][NUM_WARPS - 1];
                    float cb = smem_b[buf_idx][NUM_WARPS - 1];
                    smem_running_a[n] = ca * run_a;
                    smem_running_b[n] = ca * run_b + cb;
                }

                float h = final_prefix_b;
                if (start + ITEMS_PER_THREAD <= SEQLEN) {
                    float4 B_vec = get_B_vals4(batch_id, n, start);
                    float4 C_vec = get_C_vals4(batch_id, n, start);
                    float B_arr[4] = {B_vec.x, B_vec.y, B_vec.z, B_vec.w};
                    float C_arr[4] = {C_vec.x, C_vec.y, C_vec.z, C_vec.w};
                    #pragma unroll
                    for (uint i = 0; i < 4; i++) {
                        float a_bar = exp_deltaA[i];
                        float b_bar = delta_u_vals[i] * B_arr[i];
                        h = a_bar * h + b_bar;
                        out_vals[i] += h * C_arr[i];
                    }
                } else {
                    #pragma unroll
                    for (uint i = 0; i < ITEMS_PER_THREAD; i++) {
                        uint l = start + i;
                        if (l < SEQLEN) {
                            float a_bar = exp_deltaA[i];
                            float b_bar = delta_u_vals[i] * get_B_val(batch_id, n, l);
                            h = a_bar * h + b_bar;
                            out_vals[i] += h * get_C_val(batch_id, n, l);
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (start + ITEMS_PER_THREAD <= SEQLEN) {
                float y_arr[4];
                #pragma unroll
                for (uint i = 0; i < ITEMS_PER_THREAD; i++) {
                    y_arr[i] = out_vals[i];
                }
                uint offset = batch_id * DIM * SEQLEN + dim_id * SEQLEN + start;
                *((device float4*)(out + offset)) = float4(y_arr[0], y_arr[1], y_arr[2], y_arr[3]);
            } else {
                #pragma unroll
                for (uint i = 0; i < ITEMS_PER_THREAD; i++) {
                    uint l = start + i;
                    if (l < SEQLEN) {
                        out[batch_id * DIM * SEQLEN + dim_id * SEQLEN + l] = out_vals[i];
                    }
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < DSTATE) {
            final_state[batch_id * DIM * DSTATE + dim_id * DSTATE + tid] = smem_running_b[tid];
        }
        """

        kernel = mx.fast.metal_kernel(
            name="selective_scan_fwd_with_state",
            input_names=[
                "u",
                "delta",
                "A",
                "B",
                "C",
                "D_skip",
                "delta_bias",
                "initial_state",
            ],
            output_names=["out", "final_state"],
            source=source,
            header=header,
        )
        _kernel_cache[key] = kernel

    kernel = _kernel_cache[key]
    empty = mx.zeros((1,), dtype=mx.float32)
    inputs = [
        u_f32,
        delta_f32,
        A_f32,
        B_f32,
        C_f32,
        D.astype(mx.float32) if D is not None else empty,
        delta_bias.astype(mx.float32) if delta_bias is not None else empty,
        initial_state.astype(mx.float32) if initial_state is not None else empty,
    ]

    out, final_state = kernel(
        inputs=inputs,
        grid=(batch * num_threads, dim, 1),
        threadgroup=(num_threads, 1, 1),
        output_shapes=[u.shape, (batch, dim, dstate)],
        output_dtypes=[mx.float32, mx.float32],
    )

    return out.astype(u.dtype), final_state
