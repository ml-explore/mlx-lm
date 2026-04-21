# Copyright © 2026 Apple Inc.

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_ssm_mask
from .cache import ArraysCache
from .ssm import ssm_update_trap  # mamba3 uses trapezoidal update instead of standard 2-term recurrence


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    num_heads: int
    head_dim: int
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    state_size: int
    num_hidden_layers: int
    layer_norm_epsilon: float
    n_groups: int
    use_bias: bool
    tie_word_embeddings: bool
    time_step_limit: Tuple[float, float]
    time_step_rank: Union[int, str]
    ssm_state_size: Optional[int] = None
    max_position_embeddings: int = 2056

    def __post_init__(self):
        if self.time_step_rank == "auto":
            self.time_step_rank = math.ceil(self.hidden_size / 16)
        if self.ssm_state_size is None:
            self.ssm_state_size = self.state_size


def apply_bc_rope(
    bc: mx.array,
    theta: mx.array,
    dt: mx.array,
    offset: int = 0,
) -> mx.array:
    """Apply data-dependent rotary embeddings to B or C.

    The cumulative rotation product  ∏_{s=0}^{t} R_s^⊤  is equivalent to
    standard RoPE with *data-dependent* angles:
        φ_t = Σ_{s=0}^{t} Δs * θ_s[i]   (cumulative sum of Δt * θ)

    In the SSD parallel form, each position t receives the cumulative angle
    up to that position, so we compute the prefix sum of (dt * θ) and rotate
    B/C with it.  During decode (seq_len == 1), the cache offset carries the
    cumulative angle from prior tokens.

    Note on state_size pairing:
        Each pair of state dimensions (2i, 2i+1) shares one rotation angle.
        If state_size // 2 > num_heads * n_groups_per_head, the angles are
        tiled/repeated to fill all pairs (same convention as Transformer RoPE).
    """
    b, l, g, d = bc.shape
    assert d % 2 == 0, "state_size must be even for RoPE pairing"

    # Incremental angle: Δt * θ per head  →  (b, l, num_heads)
    angle_step = dt * mx.softplus(theta) # keep θ > 0 for stable rotation

    # Cumulative sum along sequence → φ_t = Σ_{s=0}^{t} angle_step_s
    # During decode (l=1) this is just angle_step; prefill uses cumsum.
    angle_cum = mx.cumsum(angle_step, axis=1)   # (b, l, num_heads)

    # Add decode offset (stores cumulative angle from previous tokens)
    # offset is a scalar int representing the *count* of past steps; we pass
    # the actual cumulative angle tensor through cache[2] instead (see _ssm).
    # Here offset is the pre-accumulated angle array of shape (b, 1, num_heads)
    # or 0 for prefill from scratch.
    if not isinstance(offset, (int, float)):
        angle_cum = angle_cum + offset

    # Broadcast angles from num_heads → n_groups * (state_size // 2)
    # angle_cum: (b, l, num_heads) → (b, l, g, n_pairs)
    # Strategy: tile along the last axis to cover g*n_pairs slots, then reshape.
    # mx.broadcast_to requires equal or size-1 dims, so we tile explicitly.
    n_pairs = d // 2
    h = angle_cum.shape[-1] # num_heads
    target = g * n_pairs
    reps = math.ceil(target / h)
    angle = mx.tile(angle_cum, (1, 1, reps))[:, :, :target].reshape(b, l, g, n_pairs)

    # Build rotation: interleave cos/sin across paired dims
    bc_r = bc.reshape(b, l, g, n_pairs, 2) # (..., 2) = (even, odd) pairs
    cos = mx.cos(angle)[:, :, :, :, None] # (b, l, g, n_pairs, 1)
    sin = mx.sin(angle)[:, :, :, :, None]

    # Rotate: [x0, x1] → [x0*cos - x1*sin,  x0*sin + x1*cos]
    x0 = bc_r[..., 0:1]
    x1 = bc_r[..., 1:2]
    bc_rot = mx.concatenate([x0 * cos - x1 * sin, x0 * sin + x1 * cos], axis=-1)
    return bc_rot.reshape(b, l, g, d)


class Mamba3Block(nn.Module):
    """Mamba-3 SSM block (SISO).

    Changes from Mamba-2:
      - Conv1d removed (made redundant by BCNorm + B/C biases + Exp-Trap)
      - BCNorm: RMSNorm applied to B and C (replaces post-gate MambaRMSNormGated)
      - Learnable B_bias, C_bias (ones-init, shape n_groups × state_size)
      - Θ (imaginary A) projected per-head for data-dependent RoPE on B and C
      - ssm_update_trap (3-term recurrence) instead of ssm_update (2-term)
      - Gate applied as plain SiLU multiply (no post-gate norm)
      - cache: slot 0 = SSM state, slot 1 = prev_Bx, slot 2 = cumulative RoPE angle

    in_proj splits:  gate | x | B | C | dt | lam | theta
    """

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = args.num_heads
        self.hidden_size = args.hidden_size
        self.ssm_state_size = args.ssm_state_size
        self.intermediate_size = args.num_heads * args.head_dim
        self.n_groups = args.n_groups
        self.head_dim = args.head_dim
        self.time_step_limit = args.time_step_limit
        self.use_bias = args.use_bias

        assert args.ssm_state_size % 2 == 0, (
            "ssm_state_size must be even for data-dependent RoPE pairing"
        )

        bc_dim = self.n_groups * self.ssm_state_size

        # in_proj splits: gate, x, B, C, dt, lam, theta
        projection_size = (
            2 * self.intermediate_size # gate + x
            + 2 * bc_dim # B + C
            + 3 * self.num_heads # dt + lam + theta (imaginary A)
        )
        self.in_proj = nn.Linear(self.hidden_size, projection_size, bias=args.use_bias)

        # SSM parameters (real part of A — unchanged from Mamba-2)
        self.dt_bias = mx.ones(self.num_heads)
        self.A_log = mx.log(mx.arange(1, self.num_heads + 1, dtype=mx.float32))
        self.D = mx.ones(self.num_heads)

        # BCNorm: one RMSNorm over state_size, applied per-group
        self.bc_norm_B = nn.RMSNorm(self.ssm_state_size, eps=args.layer_norm_epsilon)
        self.bc_norm_C = nn.RMSNorm(self.ssm_state_size, eps=args.layer_norm_epsilon)

        # Learnable B/C biases (ones-init)
        self.B_bias = mx.ones((self.n_groups, self.ssm_state_size))
        self.C_bias = mx.ones((self.n_groups, self.ssm_state_size))

        self.out_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=args.use_bias
        )

    def _apply_bc_norm(self, raw: mx.array, norm: nn.RMSNorm, bias: mx.array) -> mx.array:
        """BCNorm + bias.  raw: (b, l, n_groups * state_size) → (b, l, n_groups, state_size)."""
        b, l, _ = raw.shape
        x = raw.reshape(b, l, self.n_groups, self.ssm_state_size)
        x = norm(x) # over last dim
        x = x + bias # broadcast over (b, l)
        return x

    def _ssm(
        self,
        x: mx.array,
        B: mx.array,
        C: mx.array,
        dt: mx.array,
        lam: mx.array,
        theta: mx.array,
        cache: Optional[ArraysCache],
        mask: Optional[mx.array],
    ) -> mx.array:
        batch_size, seq_len, _ = x.shape

        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        # B, C already (b, l, n_groups, state_size) from _apply_bc_norm

        # Retrieve or init cache slots
        state = cache[0] if cache else None
        prev_Bx = cache[1] if cache else None
        angle_offset = cache[2] if cache else 0 # cumulative RoPE angle

        lengths = cache.lengths if cache else None

        # dt is raw logits here; process for RoPE angle scaling
        from .ssm import compute_dt
        dt_proc = compute_dt(dt, self.dt_bias, self.time_step_limit) # (b, l, h)

        # Apply data-dependent RoPE to B and C (Proposition 4)
        # angle_offset: (b, 1, num_heads) cumulative angle from prior tokens, or 0
        B = apply_bc_rope(B, theta, dt_proc, offset=angle_offset)
        C = apply_bc_rope(C, theta, dt_proc, offset=angle_offset)

        # Update cumulative angle for next decode step
        if cache:
            new_angle = mx.sum(dt_proc * mx.softplus(theta), axis=1, keepdims=True)
            if isinstance(angle_offset, mx.array):
                cache[2] = angle_offset + new_angle
            else:
                cache[2] = new_angle

        y, new_state, new_Bx = ssm_update_trap(
            x,
            self.A_log,
            B,
            C,
            self.D,
            dt,
            self.dt_bias,
            lam,
            state,
            prev_Bx,
            self.time_step_limit,
            mask,
            lengths,
        )

        if cache:
            cache[0] = new_state
            cache[1] = new_Bx

        return y.reshape(batch_size, seq_len, self.intermediate_size)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array],
        cache: Optional[ArraysCache] = None,
    ) -> mx.array:
        projected = self.in_proj(hidden_states)

        bc_dim = self.n_groups * self.ssm_state_size
        gate, x, B_raw, C_raw, dt, lam, theta = mx.split(
            projected,
            [
                self.intermediate_size,
                2 * self.intermediate_size,
                2 * self.intermediate_size + bc_dim,
                2 * self.intermediate_size + 2 * bc_dim,
                2 * self.intermediate_size + 2 * bc_dim + self.num_heads,
                2 * self.intermediate_size + 2 * bc_dim + 2 * self.num_heads,
            ],
            axis=-1,
        )

        # BCNorm + learnable biases
        B = self._apply_bc_norm(B_raw, self.bc_norm_B, self.B_bias)
        C = self._apply_bc_norm(C_raw, self.bc_norm_C, self.C_bias)

        y = self._ssm(x, B, C, dt, lam, theta, cache, mask=mask)

        if cache:
            cache.advance(y.shape[1])

        # Gate: plain SiLU multiply (post-gate RMSNorm removed in Mamba-3)
        y = y * nn.silu(gate)

        return self.out_proj(y)


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.mixer = Mamba3Block(args, layer_idx)
        self.norm = nn.RMSNorm(args.hidden_size)

    def __call__(
        self, x: mx.array, mask: Optional[mx.array], cache: Optional[ArraysCache] = None
    ) -> mx.array:
        return self.mixer(self.norm(x), mask, cache) + x


class Mamba3(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [ResidualBlock(args, i) for i in range(args.num_hidden_layers)]
        self.norm_f = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)

    def __call__(
        self, x: mx.array, cache: Optional[list[ArraysCache]] = None
    ) -> mx.array:
        hidden = self.embeddings(x)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_ssm_mask(hidden, cache[0])
        for layer, c in zip(self.layers, cache):
            hidden = layer(hidden, mask, c)

        return self.norm_f(hidden)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.backbone = Mamba3(args)

        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self, inputs: mx.array, cache: Optional[list[ArraysCache]] = None
    ) -> mx.array:
        hidden = self.backbone(inputs, cache)

        if self.args.tie_word_embeddings:
            logits = self.backbone.embeddings.as_linear(hidden)
        else:
            logits = self.lm_head(hidden)
        return logits

    def make_cache(self, batch_size: int = 1) -> list[ArraysCache]:
        # slot 0: SSM state
        # slot 1: prev_Bx  (trapezoidal β term)
        # slot 2: cumulative RoPE angle  (data-dependent, per-head)
        return [ArraysCache(size=3) for _ in range(self.args.num_hidden_layers)]

    @property
    def layers(self):
        return self.backbone.layers

    def sanitize(self, weights):
        # No conv1d in Mamba-3
        return weights