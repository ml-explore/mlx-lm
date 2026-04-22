"""Naive MLX SSD chunk state computation — correct but slow reference.

Port of ``chunk_state_ref`` from the Mamba-2 repository
(mamba_ssm/ops/triton/ssd_chunk_state.py) translated to MLX array ops.

For each chunk, computes:
  decay = exp(dA_cumsum_last - dA_cumsum) * dt
  states[b,c,h,p,n] = sum_l  B[b,c,l,g,n] * decay[b,h,c,l] * x[b,c,l,h,p]

where g = h // (nheads // ngroups).

Entrypoint: ``run(B, x, dt, dA_cumsum)``
"""

import mlx.core as mx


def run(
    B: mx.array,
    x: mx.array,
    dt: mx.array,
    dA_cumsum: mx.array,
) -> mx.array:
    """Naive SSD chunk state computation.

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

    assert seqlen <= nchunks * chunk_size
    assert nheads % ngroups == 0

    # Expand B groups: (batch, seqlen, ngroups, dstate) -> (batch, seqlen, nheads, dstate)
    nheads_per_group = nheads // ngroups
    # Repeat each group nheads_per_group times along axis=2
    B_expanded = mx.repeat(B, repeats=nheads_per_group, axis=2)

    # Pad sequence if needed
    if seqlen < nchunks * chunk_size:
        pad_len = nchunks * chunk_size - seqlen
        x = mx.concatenate(
            [x, mx.zeros((batch, pad_len, nheads, headdim), dtype=x.dtype)], axis=1
        )
        B_expanded = mx.concatenate(
            [
                B_expanded,
                mx.zeros((batch, pad_len, nheads, dstate), dtype=B_expanded.dtype),
            ],
            axis=1,
        )

    # Reshape into chunks
    # x: (batch, nchunks, chunk_size, nheads, headdim)
    x = x.reshape(batch, nchunks, chunk_size, nheads, headdim)
    # B_expanded: (batch, nchunks, chunk_size, nheads, dstate)
    B_expanded = B_expanded.reshape(batch, nchunks, chunk_size, nheads, dstate)

    # decay_states = exp(dA_cumsum[:, :, :, -1:] - dA_cumsum) * dt
    # dA_cumsum: (batch, nheads, nchunks, chunk_size)
    dA_cumsum_last = dA_cumsum[:, :, :, -1:]  # (batch, nheads, nchunks, 1)
    decay_states = mx.exp(
        dA_cumsum_last - dA_cumsum
    )  # (batch, nheads, nchunks, chunk_size)

    # einsum: bclhn, bhcl, bhcl, bclhp -> bchpn
    # where:
    #   B_expanded is (b, c, l, h, n)
    #   decay_states is (b, h, c, l)
    #   dt is (b, h, c, l)
    #   x is (b, c, l, h, p)
    # Combine decay_states and dt: (batch, nheads, nchunks, chunk_size)
    scale = (decay_states * dt).astype(mx.float32)  # (b, h, c, l)

    # Transpose scale to (b, c, l, h) for broadcasting
    scale = mx.transpose(scale, (0, 2, 3, 1))  # (b, c, l, h)

    # Compute weighted x: (b, c, l, h, p) * (b, c, l, h, 1)
    x_scaled = x.astype(mx.float32) * mx.expand_dims(scale, axis=-1)  # (b, c, l, h, p)

    # Compute outer product sum: sum over l of B[b,c,l,h,n] * x_scaled[b,c,l,h,p]
    # B_expanded: (b, c, l, h, n), x_scaled: (b, c, l, h, p)
    # Want: (b, c, h, p, n) = einsum('bclhp,bclhn->bchpn', x_scaled, B_expanded)
    B_f32 = B_expanded.astype(mx.float32)

    # Use einsum for clarity
    # x_scaled: (b, c, l, h, p)
    # B_f32:    (b, c, l, h, n)
    # result:   (b, c, h, p, n)
    states = mx.einsum("bclhp,bclhn->bchpn", x_scaled, B_f32)
    return states
