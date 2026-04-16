import mlx.core as mx


def run(
    states: mx.array, dA_chunk_cumsum: mx.array, initial_states: mx.array | None = None
) -> tuple[mx.array, mx.array]:
    batch, nchunks, nheads, dim = states.shape
    if dim % 4 != 0:
        raise ValueError("ssd_state_passing requires dim to be divisible by 4")

    if initial_states is None:
        has_initial = 0
        # Must be large enough that MLX passes it as a device buffer, not constant
        initial_states = mx.zeros((batch, nheads, dim), dtype=mx.float32)
    else:
        has_initial = 1
        initial_states = initial_states.astype(mx.float32)

    VEC_SIZE = 4
    dim_vec = dim // VEC_SIZE
    THREADS_PER_GROUP = 64
    grid_dim_z = (dim_vec + THREADS_PER_GROUP - 1) // THREADS_PER_GROUP

    header = f"""
    #define BATCH {batch}
    #define NCHUNKS {nchunks}
    #define NHEADS {nheads}
    #define DIM {dim}
    #define DIM_VEC {dim_vec}
    #define HAS_INITIAL {has_initial}
    #define TPG {THREADS_PER_GROUP}
    """

    source = """
    uint batch_id = threadgroup_position_in_grid.x;
    uint head_id = threadgroup_position_in_grid.y;
    uint block_id = threadgroup_position_in_grid.z;
    uint tid = thread_position_in_threadgroup.x;
    uint vec_id = block_id * TPG + tid;

    if (vec_id >= DIM_VEC) return;

    device const float4* states_vec = (device const float4*)states;
    device float4* out_vec = (device float4*)out;
    device float4* final_vec = (device float4*)final_states;
    device const float4* initial_vec = (device const float4*)initial_states;

    float4 state = float4(0.0f);
    if (HAS_INITIAL) {
        uint init_idx = batch_id * (NHEADS * DIM_VEC) + head_id * DIM_VEC + vec_id;
        state = initial_vec[init_idx];
    }

    uint out_idx = batch_id * (NCHUNKS * NHEADS * DIM_VEC) + 0 * (NHEADS * DIM_VEC) + head_id * DIM_VEC + vec_id;
    out_vec[out_idx] = state;

    uint states_base = batch_id * (NCHUNKS * NHEADS * DIM_VEC) + head_id * DIM_VEC + vec_id;
    uint dA_base = batch_id * (NHEADS * NCHUNKS) + head_id * NCHUNKS;
    
    #pragma unroll 4
    for (uint c = 0; c < NCHUNKS; ++c) {
        uint state_idx = states_base + c * (NHEADS * DIM_VEC);
        float4 new_state = states_vec[state_idx];
        
        float dA = dA_cs[dA_base + c];
        float scale = exp(dA);
        
        state = scale * state + new_state;
        
        if (c < NCHUNKS - 1) {
            uint next_out_idx = states_base + (c + 1) * (NHEADS * DIM_VEC);
            out_vec[next_out_idx] = state;
        } else {
            uint final_idx = batch_id * (NHEADS * DIM_VEC) + head_id * DIM_VEC + vec_id;
            final_vec[final_idx] = state;
        }
    }
    """

    kernel = mx.fast.metal_kernel(
        name="ssd_state_passing_fwd_vec4",
        input_names=["states", "dA_cs", "initial_states"],
        output_names=["out", "final_states"],
        source=source,
        header=header,
    )

    outputs = kernel(
        inputs=[states, dA_chunk_cumsum, initial_states],
        template=[],
        grid=(batch * THREADS_PER_GROUP, nheads, grid_dim_z),
        threadgroup=(THREADS_PER_GROUP, 1, 1),
        output_shapes=[(batch, nchunks, nheads, dim), (batch, nheads, dim)],
        output_dtypes=[mx.float32, mx.float32],
    )

    return outputs[0], outputs[1]
