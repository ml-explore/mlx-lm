"""Optimized MLX SSD chunk state computation with Metal kernel fallback.

Computes for each chunk:
  decay = exp(dA_cumsum_last - dA_cumsum) * dt
  states[b,c,h,p,n] = sum_l  B[b,c,l,g,n] * decay[b,h,c,l] * x[b,c,l,h,p]

where g = h // (nheads // ngroups).

The Metal kernel accelerates the einsum by parallelizing over heads and headdim.
Entrypoint: ``run(B, x, dt, dA_cumsum)``
"""

import mlx.core as mx

_kernel_cache = {}


def _make_kernel(batch, nchunks, chunk_size, nheads, headdim, dstate):
    """Create Metal kernel for ssd_chunk_state computation.
    
    Computes states[b,c,h,p,n] = sum_l B[b,c,l,h,n] * scale[b,h,c,l] * x[b,c,l,h,p]
    
    Parallelization:
    - grid.x: headdim (output parameter dimension)
    - grid.y: nheads (output head dimension)  
    - grid.z: batch (batch dimension)
    Each thread accumulates over all chunks and chunk positions.
    """
    if not mx.metal.is_available():
        return None
        
    header = f"""
    #define BATCH {batch}
    #define NCHUNKS {nchunks}
    #define CHUNK_SIZE {chunk_size}
    #define NHEADS {nheads}
    #define HEADDIM {headdim}
    #define DSTATE {dstate}
    """

    # Kernel that computes the batched outer product reduction
    source = """
    uint p = thread_position_in_grid.x;  // headdim index
    uint h = thread_position_in_grid.y;  // head index
    uint b = thread_position_in_grid.z;  // batch index
    
    if (p >= HEADDIM || h >= NHEADS || b >= BATCH) return;
    
    // For each (batch, chunk, head, headdim), accumulate over all positions and dstate
    for (uint c = 0; c < NCHUNKS; ++c) {
        for (uint n = 0; n < DSTATE; ++n) {
            float acc = 0.0f;
            
            // sum over l: position within chunk
            for (uint l = 0; l < CHUNK_SIZE; ++l) {
                // scale[b,h,c,l] has shape (batch, nheads, nchunks, chunk_size)
                uint scale_idx = (((b * NHEADS + h) * NCHUNKS + c) * CHUNK_SIZE) + l;
                float scale_val = scale[scale_idx];
                
                // x[b,c,l,h,p] has shape (batch, nchunks, chunk_size, nheads, headdim)
                // Maps to linear: ((b * NCHUNKS + c) * CHUNK_SIZE + l) * NHEADS * HEADDIM + h * HEADDIM + p
                uint x_idx = (((b * NCHUNKS + c) * CHUNK_SIZE + l) * NHEADS + h) * HEADDIM + p;
                float x_val = x[x_idx];
                
                // B[b,c,l,h,n] has shape (batch, nchunks, chunk_size, nheads, dstate)
                // Maps to linear: ((b * NCHUNKS + c) * CHUNK_SIZE + l) * NHEADS * DSTATE + h * DSTATE + n
                uint B_idx = (((b * NCHUNKS + c) * CHUNK_SIZE + l) * NHEADS + h) * DSTATE + n;
                float B_val = B[B_idx];
                
                // Accumulate: states += B[b,c,l,h,n] * scale[b,h,c,l] * x[b,c,l,h,p]
                acc += B_val * scale_val * x_val;
            }
            
            // Write to states[b,c,h,p,n]
            // states shape: (batch, nchunks, nheads, headdim, dstate)
            // Maps to linear: ((b * NCHUNKS + c) * NHEADS + h) * HEADDIM * DSTATE + p * DSTATE + n
            uint out_idx = (((b * NCHUNKS + c) * NHEADS + h) * HEADDIM + p) * DSTATE + n;
            out[out_idx] = acc;
        }
    }
    """
    
    try:
        kernel = mx.fast.metal_kernel(
            name="ssd_chunk_state_fwd",
            input_names=["x", "B", "scale"],
            output_names=["out"],
            source=source,
            header=header,
        )
        return kernel
    except Exception as e:
        # If kernel compilation fails, fall back to reference
        print(f"Warning: Failed to compile ssd_chunk_state kernel: {e}")
        return None


def run(
    B: mx.array,
    x: mx.array,
    dt: mx.array,
    dA_cumsum: mx.array,
) -> mx.array:
    """Optimized SSD chunk state computation with Metal kernel.

    Args:
        B:          (batch, seqlen, ngroups, dstate)
        x:          (batch, seqlen, nheads, headdim)
        dt:         (batch, nheads, nchunks, chunk_size)
        dA_cumsum:  (batch, nheads, nchunks, chunk_size)

    Returns:
        states: (batch, nchunks, nheads, headdim, dstate)
    """
    batch, seqlen, nheads, headdim = x.shape
    dstate = B.shape[-1]
    ngroups = B.shape[2]
    _, _, nchunks, chunk_size = dt.shape
    original_dtype = x.dtype

    assert seqlen <= nchunks * chunk_size
    assert nheads % ngroups == 0

    nheads_per_group = nheads // ngroups

    # Pad sequence if needed
    if seqlen < nchunks * chunk_size:
        pad_len = nchunks * chunk_size - seqlen
        x_pad = mx.concatenate([x, mx.zeros((batch, pad_len, nheads, headdim), dtype=x.dtype)], axis=1)
        B_pad = mx.concatenate([B, mx.zeros((batch, pad_len, ngroups, dstate), dtype=B.dtype)], axis=1)
    else:
        x_pad = x
        B_pad = B

    # Reshape into chunks
    # x: (batch, nchunks, chunk_size, nheads, headdim)
    x_reshaped = x_pad.reshape(batch, nchunks, chunk_size, nheads, headdim)
    # B: (batch, nchunks, chunk_size, ngroups, dstate) -> expand groups
    B_reshaped = B_pad.reshape(batch, nchunks, chunk_size, ngroups, dstate)
    B_expanded = mx.repeat(B_reshaped, repeats=nheads_per_group, axis=3)
    
    # Compute decay and scale
    dA_cumsum_last = dA_cumsum[:, :, :, -1:]  # (batch, nheads, nchunks, 1)
    decay = mx.exp(dA_cumsum_last - dA_cumsum)  # (batch, nheads, nchunks, chunk_size)
    scale = (decay * dt).astype(mx.float32)  # (batch, nheads, nchunks, chunk_size)
    
    # Try to use Metal kernel if available
    use_kernel = False
    if mx.metal.is_available() and mx.default_device() == mx.gpu:
        kernel_key = (batch, nchunks, chunk_size, nheads, headdim, dstate)
        if kernel_key not in _kernel_cache:
            _kernel_cache[kernel_key] = _make_kernel(batch, nchunks, chunk_size, nheads, headdim, dstate)
        
        kernel = _kernel_cache[kernel_key]
        if kernel is not None:
            use_kernel = True
    
    if use_kernel:
        try:
            # Convert to float32 for kernel computation
            x_f32 = x_reshaped.astype(mx.float32)
            B_f32 = B_expanded.astype(mx.float32)
            
            # No need for explicit contiguous - arrays are naturally contiguous after reshape/astype
            
            out_shape = (batch, nchunks, nheads, headdim, dstate)
            
            states = kernel(
                inputs=[x_f32, B_f32, scale],
                output_shapes=[out_shape],
                output_dtypes=[mx.float32],
                grid=(headdim, nheads, batch),
                threadgroup=(1, 1, 1),
            )
            
            # Convert back to original dtype
            states = states[0].astype(original_dtype)
            mx.eval(states)
            return states
        except Exception as e:
            # Fall back to reference if kernel execution fails
            print(f"Warning: Metal kernel execution failed: {e}. Using reference implementation.")
            use_kernel = False
            use_kernel = False
    
    # Reference einsum implementation (fallback)
    # Transpose scale to (b, c, l, h) for broadcasting
    scale = mx.transpose(scale, (0, 2, 3, 1))  # (batch, nchunks, chunk_size, nheads)
    
    # Compute weighted x: (b, c, l, h, p) * (b, c, l, h, 1)
    x_scaled = x_reshaped.astype(mx.float32) * mx.expand_dims(scale, axis=-1)
    B_f32 = B_expanded.astype(mx.float32)
    
    # einsum: bclhp,bclhn->bchpn
    states = mx.einsum("bclhp,bclhn->bchpn", x_scaled, B_f32)
    
    # Cast back to original dtype
    states = states.astype(original_dtype)
    
    mx.eval(states)
    return states
