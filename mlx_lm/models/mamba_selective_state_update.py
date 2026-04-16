from typing import Optional, Tuple

import mlx.core as mx

_kernel_cache = {}


def _kernel_key(
    dim: int, dstate: int, has_d: bool, has_dt_bias: bool, dt_softplus: bool
):
    return (dim, dstate, has_d, has_dt_bias, dt_softplus)


def selective_state_update(
    state: Optional[mx.array],
    x: mx.array,
    dt: mx.array,
    A: mx.array,
    B: mx.array,
    C: mx.array,
    D: Optional[mx.array] = None,
    dt_bias: Optional[mx.array] = None,
    dt_softplus: bool = True,
) -> Tuple[mx.array, mx.array]:
    """Single-step selective state update for Mamba decode.

    Shapes:
    - state: (batch, dim, dstate) or None
    - x: (batch, dim)
    - dt: (batch, dim)
    - A: (dim, dstate)
    - B: (batch, dstate)
    - C: (batch, dstate)
    - D: (dim,)
    - dt_bias: (dim,)
    """
    x_f32 = x.astype(mx.float32)
    dt_f32 = dt.astype(mx.float32)
    A_f32 = A.astype(mx.float32)
    B_f32 = B.astype(mx.float32)
    C_f32 = C.astype(mx.float32)

    batch, dim = x.shape
    dstate = A.shape[1]

    has_d = D is not None
    has_dt_bias = dt_bias is not None

    if state is None:
        state_f32 = mx.zeros((batch, dim, dstate), dtype=mx.float32)
    else:
        state_f32 = state.astype(mx.float32)

    key = _kernel_key(dim, dstate, has_d, has_dt_bias, dt_softplus)

    block_size = 64
    grid_x = (dim + block_size - 1) // block_size

    if key not in _kernel_cache:
        header = f"""
        #define DIM {dim}
        #define DSTATE {dstate}
        #define BLOCK_SIZE {block_size}
        #define HAS_D {1 if has_d else 0}
        #define HAS_DT_BIAS {1 if has_dt_bias else 0}
        #define DT_SOFTPLUS {1 if dt_softplus else 0}
        #define DSTATE_MAX 256
        """

        source = """
        uint tid = thread_position_in_threadgroup.x;
        uint dim_idx = threadgroup_position_in_grid.x * BLOCK_SIZE + tid;
        uint b = threadgroup_position_in_grid.y;

        threadgroup float B_smem[DSTATE_MAX];
        threadgroup float C_smem[DSTATE_MAX];

        for (uint i = tid; i < DSTATE; i += BLOCK_SIZE) {
            B_smem[i] = B[b * DSTATE + i];
            C_smem[i] = C[b * DSTATE + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (dim_idx >= DIM) return;

        float dt_val = dt[b * DIM + dim_idx];
        if (HAS_DT_BIAS) {
            dt_val += dt_bias[dim_idx];
        }
        if (DT_SOFTPLUS) {
            dt_val = dt_val <= 20.0f ? log(1.0f + fast::exp(dt_val)) : dt_val;
        }

        float x_val = x[b * DIM + dim_idx];
        float dt_x = dt_val * x_val;

        uint state_base = b * DIM * DSTATE + dim_idx * DSTATE;
        uint a_base = dim_idx * DSTATE;

        float out_val = 0.0f;

        for (uint i = 0; i < DSTATE; ++i) {
            float s = state[state_base + i];
            float dA = fast::exp(A[a_base + i] * dt_val);
            s = fma(s, dA, B_smem[i] * dt_x);
            state_out[state_base + i] = s;
            out_val += s * C_smem[i];
        }

        if (HAS_D) {
            out_val += x_val * D_skip[dim_idx];
        }

        out[b * DIM + dim_idx] = out_val;
        """

        kernel = mx.fast.metal_kernel(
            name="mamba_selective_state_update",
            input_names=["state", "x", "dt", "A", "B", "C", "D_skip", "dt_bias"],
            output_names=["out", "state_out"],
            header=header,
            source=source,
        )
        _kernel_cache[key] = kernel

    kernel = _kernel_cache[key]
    empty = mx.zeros((1,), dtype=mx.float32)

    out_f32, state_out_f32 = kernel(
        inputs=[
            state_f32,
            x_f32,
            dt_f32,
            A_f32,
            B_f32,
            C_f32,
            D.astype(mx.float32) if D is not None else empty,
            dt_bias.astype(mx.float32) if dt_bias is not None else empty,
        ],
        grid=(grid_x * block_size, batch, 1),
        threadgroup=(block_size, 1, 1),
        output_shapes=[x.shape, state_f32.shape],
        output_dtypes=[mx.float32, mx.float32],
    )

    return out_f32.astype(x.dtype), state_out_f32.astype(x.dtype)
