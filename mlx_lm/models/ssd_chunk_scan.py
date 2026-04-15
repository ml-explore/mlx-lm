import mlx.core as mx


def run(
    B: mx.array,
    C: mx.array,
    x: mx.array,
    dt: mx.array,
    dA_cumsum: mx.array,
    prev_states: mx.array,
    D: mx.array | None = None,
    z: mx.array | None = None,
) -> mx.array:
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    _, _, nchunks, chunk_size = dt.shape

    # Precompute CB and out_prev using highly optimized MLX matmul
    B_f = B.astype(mx.float32)
    C_f = C.astype(mx.float32)
    B_expanded = mx.repeat(B_f, nheads // ngroups, axis=2)
    C_expanded = mx.repeat(C_f, nheads // ngroups, axis=2)

    B_c = B_expanded.reshape(batch, nchunks, chunk_size, nheads, dstate)
    C_c = C_expanded.reshape(batch, nchunks, chunk_size, nheads, dstate)

    C_ct = mx.transpose(C_c, (0, 1, 3, 2, 4))
    B_ct = mx.transpose(B_c, (0, 1, 3, 2, 4))

    CB = C_ct @ mx.transpose(B_ct, (0, 1, 2, 4, 3))

    prev_states_f = prev_states.astype(mx.float32)
    prev_st_t = mx.transpose(prev_states_f, (0, 1, 2, 4, 3))
    out_prev = C_ct @ prev_st_t

    has_D = D is not None
    if has_D:
        if D.ndim == 1:
            D = mx.expand_dims(D, axis=-1)
        D = mx.broadcast_to(D, (nheads, headdim))
        D = mx.array(D)  # Force contiguous allocation (avoid constant address space)
    else:
        D = mx.zeros((nheads, headdim), dtype=mx.float32)

    has_z = z is not None
    if not has_z:
        # Must be large enough for MLX to pass as device buffer, not constant
        z = mx.zeros((batch, seqlen, nheads, headdim), dtype=mx.float32)

    header = f"""
    #define CHUNK_SIZE {chunk_size}
    #define HEADDIM {headdim}
    #define DSTATE {dstate}
    #define NHEADS {nheads}
    #define NCHUNKS {nchunks}
    #define NGROUPS {ngroups}
    #define HAS_D {1 if has_D else 0}
    #define HAS_Z {1 if has_z else 0}
    """

    source = """
    uint l = thread_position_in_grid.x;
    uint p = thread_position_in_grid.y * 4;
    uint bch = thread_position_in_grid.z;

    if (l >= CHUNK_SIZE || p >= HEADDIM) return;

    uint h = bch % NHEADS;
    uint bc = bch / NHEADS;
    uint c = bc % NCHUNKS;
    uint b = bc / NCHUNKS;
    
    uint seq_idx = c * CHUNK_SIZE + l;

    uint dt_da_base = b * NHEADS * NCHUNKS * CHUNK_SIZE + h * NCHUNKS * CHUNK_SIZE + c * CHUNK_SIZE;
    float da_l = dA_cumsum[dt_da_base + l];
    
    uint out_prev_idx = b * (NCHUNKS * NHEADS * CHUNK_SIZE * HEADDIM) + 
                        c * (NHEADS * CHUNK_SIZE * HEADDIM) + 
                        h * (CHUNK_SIZE * HEADDIM) + 
                        l * HEADDIM + p;
                        
    float4 prev_val = *(device const float4*)(out_prev + out_prev_idx);
    float4 acc = prev_val * metal::fast::exp(da_l);

    uint cb_base = b * NCHUNKS * NHEADS * CHUNK_SIZE * CHUNK_SIZE + c * NHEADS * CHUNK_SIZE * CHUNK_SIZE + h * CHUNK_SIZE * CHUNK_SIZE + l * CHUNK_SIZE;
    uint x_base = b * (NCHUNKS * CHUNK_SIZE * NHEADS * HEADDIM) + c * CHUNK_SIZE * (NHEADS * HEADDIM) + h * HEADDIM + p;
    uint x_stride = NHEADS * HEADDIM;

    uint s = 0;
    for (; s + 3 <= l; s += 4) {
        float4 cb_4 = *(device const float4*)(CB + cb_base + s);
        float4 da_s_4 = *(device const float4*)(dA_cumsum + dt_da_base + s);
        float4 dt_s_4 = *(device const float4*)(dt + dt_da_base + s);
        float4 decay_4 = metal::fast::exp(metal::min(da_l - da_s_4, 0.0f));
        float4 weight_4 = cb_4 * decay_4 * dt_s_4;
        
        acc += weight_4.x * *(device const float4*)(x + x_base + (s + 0) * x_stride);
        acc += weight_4.y * *(device const float4*)(x + x_base + (s + 1) * x_stride);
        acc += weight_4.z * *(device const float4*)(x + x_base + (s + 2) * x_stride);
        acc += weight_4.w * *(device const float4*)(x + x_base + (s + 3) * x_stride);
    }
    for (; s <= l; ++s) {
        float cb = CB[cb_base + s];
        float da_s = dA_cumsum[dt_da_base + s];
        float dt_s = dt[dt_da_base + s];
        float decay = metal::fast::exp(metal::min(da_l - da_s, 0.0f));
        float weight = cb * decay * dt_s;
        
        float4 x_vec = *(device const float4*)(x + x_base + s * x_stride);
        acc += weight * x_vec;
    }

    uint out_idx = b * (NCHUNKS * CHUNK_SIZE * NHEADS * HEADDIM) + seq_idx * (NHEADS * HEADDIM) + h * HEADDIM + p;

    if (HAS_D) {
        float4 d_vec = *(device const float4*)(D + h * HEADDIM + p);
        float4 x_vec = *(device const float4*)(x + out_idx);
        acc += x_vec * d_vec;
    }

    if (HAS_Z) {
        float4 z_vec = *(device const float4*)(z + out_idx);
        float4 sigmoid_z = 1.0f / (1.0f + metal::fast::exp(-z_vec));
        acc *= z_vec * sigmoid_z;
    }

    *(device float4*)(out + out_idx) = acc;
    """

    kernel = mx.fast.metal_kernel(
        name="ssd_chunk_scan_fwd",
        input_names=["x", "dt", "dA_cumsum", "D", "z", "CB", "out_prev"],
        output_names=["out"],
        source=source,
        header=header,
    )

    x = x.astype(mx.float32)
    dt = dt.astype(mx.float32)
    dA_cumsum = dA_cumsum.astype(mx.float32)
    D = D.astype(mx.float32)
    z = z.astype(mx.float32)
    out_prev = out_prev.astype(mx.float32)

    grid = (chunk_size, headdim // 4, batch * nchunks * nheads)
    threadgroup = (min(32, chunk_size), min(8, headdim // 4), 1)

    out = kernel(
        inputs=[x, dt, dA_cumsum, D, z, CB, out_prev],
        template=[],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]

    return out
