import mlx.core as mx
from typing import Optional, Tuple

from .ssd_chunk_state import run as ssd_chunk_state
from .ssd_state_passing import run as ssd_state_passing
from .ssd_chunk_scan import run as ssd_chunk_scan

def ssd_prefill_kernel(
    hidden_states: mx.array,
    A_log: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    dt: mx.array,
    dt_bias: mx.array,
    state: Optional[mx.array],
    time_step_limit: Tuple[float, float],
):
    batch, seq_len, nheads, headdim = hidden_states.shape
    _, _, ngroups, dstate = B.shape
    
    # Adaptive chunk size based on sequence length for better GPU utilization
    # For longer sequences, larger chunks reduce overhead; for shorter, smaller chunks improve cache locality
    if seq_len <= 256:
        chunk_size = min(128, seq_len)  # Smaller chunks for short sequences
    elif seq_len <= 1024:
        chunk_size = 256  # Standard size for medium sequences
    else:
        chunk_size = 512  # Larger chunks for long sequences reduce kernel dispatch overhead

    # Pad sequence to multiple of chunk_size
    pad_len = 0
    if seq_len % chunk_size != 0:
        pad_len = chunk_size - seq_len % chunk_size
        hidden_states = mx.pad(hidden_states, [(0, 0), (0, pad_len), (0, 0), (0, 0)])
        B = mx.pad(B, [(0, 0), (0, pad_len), (0, 0), (0, 0)])
        C = mx.pad(C, [(0, 0), (0, pad_len), (0, 0), (0, 0)])
        dt = mx.pad(dt, [(0, 0), (0, pad_len), (0, 0)])

    padded_len = seq_len + pad_len
    nchunks = padded_len // chunk_size

    # dt softplus
    # dt_chunks: (B, L, nheads) -> (B, nchunks, chunk_size, nheads) -> (B, nheads, nchunks, chunk_size)
    dt_chunks = dt.reshape(batch, nchunks, chunk_size, nheads)
    dt_chunks = mx.transpose(dt_chunks, (0, 3, 1, 2))
    dt_f = dt_chunks.astype(mx.float32)
    dt_f = dt_f + dt_bias.astype(mx.float32)[:, None, None]
    # In ssm.py compute_dt it uses custom limit:
    dt_f = mx.where(dt_f <= 20.0, mx.log(1.0 + mx.exp(dt_f)), dt_f)
    dt_f = mx.clip(dt_f, time_step_limit[0], time_step_limit[1])
    
    A = -mx.exp(A_log.astype(mx.float32))
    dA = dt_f * A[:, None, None]
    dA_cumsum = mx.cumsum(dA, axis=-1)

    # 1. State
    states = ssd_chunk_state(B, hidden_states, dt_f, dA_cumsum)
    
    # 2. State Passing
    states_flat = states.reshape(batch, nchunks, nheads, headdim * dstate)
    dA_chunk_last = dA_cumsum[:, :, :, -1]
    
    if state is not None:
        # User initial state is (B, nheads, headdim, dstate)
        initial_state_flat = state.reshape(batch, nheads, headdim * dstate)
        passed_states, final_states = ssd_state_passing(states_flat, dA_chunk_last, initial_state_flat)
    else:
        passed_states, final_states = ssd_state_passing(states_flat, dA_chunk_last)
    
    prev_states = passed_states.reshape(batch, nchunks, nheads, headdim, dstate)
    
    # 3. Chunk Scan
    out = ssd_chunk_scan(
        B, C, hidden_states, dt_f, dA_cumsum, prev_states, D=D, z=None
    )
    
    if pad_len > 0:
        out = out[:, :seq_len, :, :]
        
    out = out.astype(hidden_states.dtype)
    
    # Final state
    last_chunk_state = states[:, -1, :, :, :]
    last_prev = prev_states[:, -1, :, :, :]
    last_decay = mx.exp(dA_cumsum[:, :, -1, -1])
    final_ssm_state = last_decay[:, :, None, None] * last_prev + last_chunk_state
    final_ssm_state = final_ssm_state.astype(hidden_states.dtype)
    
    return out, final_ssm_state
