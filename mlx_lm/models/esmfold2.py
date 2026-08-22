"""MLX implementation of ESMFold2 (pure-reference path).

ESMFold2 is a protein structure predictor: a pair-tensor refiner conditioned on
a frozen ESMC-6B language model, followed by a diffusion atom decoder. This port
targets ONLY the pure-PyTorch reference path of the `transformers` ESMFold2
implementation — no Triton/fused kernels, no flash-attention, no transformer-
engine/fp8, no tensor/sequence parallelism (all of which have pure-PyTorch
fallbacks selected by `set_kernel_backend(None)`, the default when those optional
libraries are absent).
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

_EPS = 1e-5


# ---------------------------------------------------------------------------
# Feed-forward
# ---------------------------------------------------------------------------


class SwiGLUMLP(nn.Module):
    """SwiGLU MLP with packed w12 and output w3 (bias-free).

    hidden = expansion_ratio * d_model (the reference's _compute_swiglu_hidden_size).
    """

    def __init__(self, d_model: int, expansion_ratio: int = 4):
        super().__init__()
        self.hidden_features = expansion_ratio * d_model
        self.w12 = nn.Linear(d_model, 2 * self.hidden_features, bias=False)
        self.w3 = nn.Linear(self.hidden_features, d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        x1, x2 = mx.split(self.w12(x), 2, axis=-1)
        return self.w3(nn.silu(x1) * x2)


class Transition(nn.Module):
    """Pre-norm SwiGLU FFN with residual: x + ffn(norm(x))."""

    def __init__(self, d_model: int, expansion_ratio: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=_EPS)
        self.ffn = SwiGLUMLP(d_model, expansion_ratio=expansion_ratio)

    def __call__(self, x: mx.array) -> mx.array:
        return x + self.ffn(self.norm(x))


# ---------------------------------------------------------------------------
# Triangle multiplicative update
# ---------------------------------------------------------------------------


class TriangleMultiplicativeUpdate(nn.Module):
    """Triangle multiplicative update with gated signal routing.

    Flattens the reference's TriangleMultiplicativeUpdate -> _engine
    (TriangleMultiplicativeBlock); the `._engine.` key segment is stripped in
    `Model.sanitize`. input_channels == latent_channels == dim.
    """

    def __init__(self, dim: int = 256, outgoing: bool = True):
        super().__init__()
        self.dim = dim
        self.outgoing = outgoing
        self.norm_start = nn.LayerNorm(dim, eps=_EPS)
        self.norm_mix = nn.LayerNorm(dim, eps=_EPS)
        self.proj_bundle = nn.Linear(dim, 4 * dim, bias=False)
        self.proj_emit = nn.Linear(dim, dim, bias=False)
        self.proj_gate = nn.Linear(dim, dim, bias=False)

    def _contract(self, left: mx.array, right: mx.array) -> mx.array:
        # left/right: (B, L, L, D). Reference einsum:
        #   outgoing: out[b,i,j,d] = sum_k left[b,i,k,d] * right[b,j,k,d]
        #   incoming: out[b,i,j,d] = sum_k left[b,k,i,d] * right[b,k,j,d]
        # Move channel D to the batch axis and reduce over k with a matmul.
        if self.outgoing:
            l = left.transpose(0, 3, 1, 2)   # (B, D, i, k)
            r = right.transpose(0, 3, 1, 2)  # (B, D, j, k)
        else:
            l = left.transpose(0, 3, 2, 1)   # (B, D, i, k)
            r = right.transpose(0, 3, 2, 1)  # (B, D, j, k)
        out = l @ r.transpose(0, 1, 3, 2)    # (B, D, i, j)
        return out.transpose(0, 2, 3, 1)     # (B, i, j, D)

    def __call__(self, z: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        normalized = self.norm_start(z)
        bundled = self.proj_bundle(normalized)
        signal, gate_logits = mx.split(bundled, 2, axis=-1)
        routed = signal * mx.sigmoid(gate_logits)
        if mask is not None:
            routed = routed * mask[..., None]
        # Contraction in fp32 (matches the reference).
        left, right = mx.split(routed.astype(mx.float32), 2, axis=-1)
        contracted = self._contract(left, right).astype(z.dtype)
        mixed = self.proj_emit(self.norm_mix(contracted))
        out_gate = mx.sigmoid(self.proj_gate(normalized))
        return mixed * out_gate


# ---------------------------------------------------------------------------
# Folding trunk
# ---------------------------------------------------------------------------


class PairUpdateBlock(nn.Module):
    """tri_mul_out -> tri_mul_in -> pair_transition (row dropout is identity at inference)."""

    def __init__(self, d_pair: int = 256, expansion_ratio: int = 4):
        super().__init__()
        self.tri_mul_out = TriangleMultiplicativeUpdate(dim=d_pair, outgoing=True)
        self.tri_mul_in = TriangleMultiplicativeUpdate(dim=d_pair, outgoing=False)
        self.pair_transition = Transition(d_pair, expansion_ratio=expansion_ratio)

    def __call__(self, pair: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        pair = pair + self.tri_mul_out(pair, mask=mask)
        pair = pair + self.tri_mul_in(pair, mask=mask)
        pair = self.pair_transition(pair)
        return pair


class FoldingTrunk(nn.Module):
    """Stack of PairUpdateBlocks."""

    def __init__(self, n_layers: int = 24, d_pair: int = 256, expansion_ratio: int = 4):
        super().__init__()
        self.blocks = [
            PairUpdateBlock(d_pair=d_pair, expansion_ratio=expansion_ratio)
            for _ in range(n_layers)
        ]
        self._compiled = None

    def _apply_blocks(self, pair, mask):
        for block in self.blocks:
            pair = block(pair, mask=mask)
        return pair

    def __call__(self, pair: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        # Fuse the whole block stack with mx.compile (cached per shape); only when
        # a concrete mask is present (compile needs array arguments).
        if mask is not None:
            if self._compiled is None:
                self._compiled = mx.compile(self._apply_blocks)
            return self._compiled(pair, mask)
        return self._apply_blocks(pair, mask)


# ---------------------------------------------------------------------------
# Language-model integration (ELMo mix of ESMC's 81 hidden states -> pair)
# ---------------------------------------------------------------------------


class SingleToPair(nn.Module):
    """Lift per-residue features to pair space via outer product + outer difference."""

    def __init__(self, input_dim: int, downproject_dim: int, output_dim: int):
        super().__init__()
        self.downproject = nn.Linear(input_dim, downproject_dim)
        self.output_mlp = [
            nn.Linear(2 * downproject_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.downproject(x)
        prod = x[:, :, None, :] * x[:, None, :, :]
        diff = x[:, :, None, :] - x[:, None, :, :]
        x = mx.concatenate([prod, diff], axis=3)
        for layer in self.output_mlp:
            x = layer(x)
        return x


class LanguageModelShim(nn.Module):
    """Project ESMC's (num_layers+1) hidden states to a pair representation.

    base_z_linear: [LayerNorm(d_model), Linear(d_model, d_z, bias=False)]
    base_z_combine: learnable softmax weights over the (num_layers+1) layers
    base_z_mlp: [SingleToPair(d_z), LayerNorm(d_z)]
    """

    def __init__(self, d_z: int = 256, d_model: int = 2560, num_layers: int = 80):
        super().__init__()
        self.base_z_mlp = [SingleToPair(d_z, d_z, d_z), nn.LayerNorm(d_z, eps=_EPS)]
        self.base_z_linear = [
            nn.LayerNorm(d_model, eps=_EPS),
            nn.Linear(d_model, d_z, bias=False),
        ]
        self.base_z_combine = mx.zeros((num_layers + 1,))

    def __call__(self, hidden_states: mx.array) -> mx.array:
        # hidden_states: (B, L, num_layers+1, d_model)
        lm_z = hidden_states
        for layer in self.base_z_linear:
            lm_z = layer(lm_z)  # (B, L, 81, d_z)
        weights = mx.softmax(self.base_z_combine, axis=0)  # (81,)
        lm_z = mx.sum(weights[:, None] * lm_z, axis=-2)  # (B, L, d_z)
        for layer in self.base_z_mlp:
            lm_z = layer(lm_z)  # (B, L, L, d_z)
        return lm_z


# ---------------------------------------------------------------------------
# Relative position / chain / entity encoding
# ---------------------------------------------------------------------------


def _one_hot(idx: mx.array, n: int) -> mx.array:
    return (idx[..., None] == mx.arange(n)).astype(mx.float32)


class RelativePositionEncoding(nn.Module):
    """ResIdx / asym / sym / entity relative encoding -> pair bias (bias-free Linear)."""

    def __init__(
        self,
        n_relative_residx_bins: int = 32,
        n_relative_chain_bins: int = 2,
        d_pair: int = 256,
    ):
        super().__init__()
        self.r = n_relative_residx_bins
        self.c = n_relative_chain_bins
        n_res = 2 * self.r + 2
        n_tok = 2 * self.r + 2
        n_chain = 2 * self.c + 2
        total = n_res + n_tok + n_chain + 1
        self.embed = nn.Linear(total, d_pair, bias=False)

    def __call__(self, residue_index, asym_id, sym_id, entity_id, token_index):
        same_chain = asym_id[:, :, None] == asym_id[:, None, :]
        same_residue = residue_index[:, :, None] == residue_index[:, None, :]
        same_entity = entity_id[:, :, None] == entity_id[:, None, :]

        dij_res = residue_index[:, :, None] - residue_index[:, None, :]
        dij_res = mx.clip(dij_res + self.r, 0, 2 * self.r)
        dij_res = mx.where(same_chain, dij_res, 2 * self.r + 1)
        aij_res = _one_hot(dij_res, 2 * self.r + 2)

        dij_tok = mx.clip(
            token_index[:, :, None] - token_index[:, None, :] + self.r, 0, 2 * self.r
        )
        dij_tok = mx.where(same_chain & same_residue, dij_tok, 2 * self.r + 1)
        aij_tok = _one_hot(dij_tok, 2 * self.r + 2)

        dij_chain = mx.clip(
            sym_id[:, :, None] - sym_id[:, None, :] + self.c, 0, 2 * self.c
        )
        dij_chain = mx.where(same_chain, 2 * self.c + 1, dij_chain)
        aij_chain = _one_hot(dij_chain, 2 * self.c + 2)

        feats = mx.concatenate(
            [aij_res, aij_tok, same_entity.astype(mx.float32)[..., None], aij_chain],
            axis=-1,
        )
        return self.embed(feats)


# ---------------------------------------------------------------------------
# Sliding-window attention with 3D RoPE (atom encoder / decoder / diffusion)
# ---------------------------------------------------------------------------

# F.rms_norm with eps=None uses finfo(float32).eps for fp32 inputs.
_RMS_EPS_F32 = 1.1920929e-07

# The reference forces bf16 inside the SWA atom attention.
_SWA_DTYPE = mx.bfloat16


def _rms_norm(x: mx.array, eps: float = _RMS_EPS_F32) -> mx.array:
    return x * mx.rsqrt(mx.mean(x.astype(mx.float32) ** 2, axis=-1, keepdims=True) + eps).astype(x.dtype)


def _rotate_half(x: mx.array) -> mx.array:
    x1, x2 = mx.split(x, 2, axis=-1)
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_emb_3d(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    # x: (B, L, H, D); cos/sin: (B, L, D/2). Tile cos/sin to D by repetition.
    ro_dim = cos.shape[-1] * 2
    cos_t = mx.concatenate([cos, cos], axis=-1)[:, :, None, :]
    sin_t = mx.concatenate([sin, sin], axis=-1)[:, :, None, :]
    xr = x[..., :ro_dim]
    rotated = xr * cos_t + _rotate_half(xr) * sin_t
    return mx.concatenate([rotated, x[..., ro_dim:]], axis=-1)


def build_3d_rope(
    ref_pos: mx.array,
    ref_space_uid: mx.array,
    head_dim: int,
    n_spatial_per_axis: int = 4,
    n_uid_pairs: int = 2,
    spatial_base_freq: float = 10000.0,
    uid_base_freq: float = 10.0,
):
    """cos/sin for 3D spatial + UID RoPE. Returns bf16 (matches the reference)."""
    B, N = ref_pos.shape[:2]
    half_dim = head_dim // 2
    n_spatial_total = 3 * n_spatial_per_axis

    spatial_inv = 1.0 / (
        spatial_base_freq ** (mx.arange(n_spatial_per_axis, dtype=mx.float32) / n_spatial_per_axis)
    )
    uid_inv = 1.0 / (
        uid_base_freq ** (mx.arange(n_uid_pairs, dtype=mx.float32) / n_uid_pairs)
    )

    pos = ref_pos.astype(mx.float32)
    # einsum("bna,k->bnak") -> (B, N, 3, n_spatial) -> (B, N, 3*n_spatial)
    spatial = (pos[..., None] * spatial_inv).reshape(B, N, n_spatial_total)
    uid = ref_space_uid.astype(mx.float32)[..., None] * uid_inv  # (B, N, n_uid)

    n_active = n_spatial_total + n_uid_pairs
    freqs = mx.concatenate([spatial, uid], axis=-1)
    if n_active < half_dim:
        freqs = mx.concatenate(
            [freqs, mx.zeros((B, N, half_dim - n_active), dtype=mx.float32)], axis=-1
        )
    return mx.cos(freqs).astype(_SWA_DTYPE), mx.sin(freqs).astype(_SWA_DTYPE)


class SWA3DRoPEAttention(nn.Module):
    """Sliding-window self-attention with 3D RoPE; runs internally in bf16."""

    def __init__(self, d_model: int, n_heads: int, half_window: int = 64):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5
        self.half_window = half_window
        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, d_model, bias=False)

    def __call__(self, x, cos, sin, valid=None):
        B, N = x.shape[:2]
        x_input = x
        qkv = self.Wqkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # (B, N, H, hd)
        q, k = _rms_norm(q), _rms_norm(k)
        q = apply_rotary_emb_3d(q, cos, sin)
        k = apply_rotary_emb_3d(k, cos, sin)

        input_dtype = q.dtype
        q, k, v = (t.astype(_SWA_DTYPE) for t in (q, k, v))

        # Rank-based sliding-window mask over valid atoms (reference no-flash path).
        if valid is None:
            valid = mx.ones((B, N), dtype=mx.bool_)
        rank = mx.cumsum(valid.astype(mx.int32), axis=1) - 1  # (B, N)
        within = mx.abs(rank[:, :, None] - rank[:, None, :]) <= self.half_window
        allowed = within & valid[:, :, None] & valid[:, None, :]
        eye = mx.arange(N)[:, None] == mx.arange(N)[None, :]
        allowed = allowed | eye[None]  # (B, N, N)

        qt, kt, vt = (t.transpose(0, 2, 1, 3) for t in (q, k, v))  # (B, H, N, hd)
        out = mx.fast.scaled_dot_product_attention(
            qt, kt, vt, scale=self.scale, mask=allowed[:, None]
        )
        out = out.transpose(0, 2, 1, 3)  # (B, N, H, hd)
        out = out * valid.astype(out.dtype)[:, :, None, None]
        out = out.astype(input_dtype).reshape(B, N, -1)
        out = out * mx.sigmoid(self.gate_proj(x_input))
        return self.out_proj(out)


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN with hardware-aligned hidden size (atom blocks)."""

    def __init__(self, d_model: int, expansion_ratio: int = 2):
        super().__init__()
        hidden = ((expansion_ratio * (d_model // 3) * 2) + 255) // 256 * 256
        self.w_up = nn.Linear(d_model, 2 * hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        x1, x2 = mx.split(self.w_up(x), 2, axis=-1)
        return self.w_down(nn.silu(x1) * x2)


class SWAAtomBlock(nn.Module):
    """adaLN-Zero conditioning + SWA 3D-RoPE attention + SwiGLU FFN."""

    def __init__(self, d_atom: int, n_heads: int, half_window: int = 64, expansion_ratio: int = 2):
        super().__init__()
        # adaln_modulation = [SiLU, Linear(d, 6d, bias=False)] -> key adaln_modulation.1.weight
        self.adaln_modulation = [nn.SiLU(), nn.Linear(d_atom, 6 * d_atom, bias=False)]
        self.attn = SWA3DRoPEAttention(d_atom, n_heads, half_window=half_window)
        self.ffn = SwiGLUFFN(d_atom, expansion_ratio)

    def __call__(self, x, c_l, cos, sin, valid=None):
        mod = c_l
        for layer in self.adaln_modulation:
            mod = layer(mod)
        if mod.ndim == 2:
            mod = mod[:, None]
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mx.split(mod, 6, axis=-1)
        attn_in = _rms_norm(x) * (1 + scale_a) + shift_a
        x = x + gate_a * self.attn(attn_in, cos, sin, valid)
        ffn_in = _rms_norm(x) * (1 + scale_f) + shift_f
        x = x + gate_f * self.ffn(ffn_in)
        return x


class SWAAtomTransformer(nn.Module):
    """Stack of SWAAtomBlocks with shared 3D RoPE."""

    def __init__(
        self,
        d_atom: int = 128,
        n_blocks: int = 3,
        n_heads: int = 4,
        swa_window_size: int = 128,
        expansion_ratio: int = 2,
        spatial_rope_base_frequency: float = 20.0,
        n_spatial_rope_pairs_per_axis: int = 2,
        n_uid_rope_pairs: int = 10,
        uid_rope_base_frequency: float = 10000.0,
    ):
        super().__init__()
        self.head_dim = d_atom // n_heads
        self.swa_window_size = swa_window_size
        self.spatial_rope_base_frequency = spatial_rope_base_frequency
        self.n_spatial_rope_pairs_per_axis = n_spatial_rope_pairs_per_axis
        self.n_uid_rope_pairs = n_uid_rope_pairs
        self.uid_rope_base_frequency = uid_rope_base_frequency
        self.blocks = [
            SWAAtomBlock(d_atom, n_heads, half_window=swa_window_size // 2, expansion_ratio=expansion_ratio)
            for _ in range(n_blocks)
        ]

    def build_rope(self, ref_pos, ref_space_uid):
        return build_3d_rope(
            ref_pos, ref_space_uid, self.head_dim,
            n_spatial_per_axis=self.n_spatial_rope_pairs_per_axis,
            n_uid_pairs=self.n_uid_rope_pairs,
            spatial_base_freq=self.spatial_rope_base_frequency,
            uid_base_freq=self.uid_rope_base_frequency,
        )

    def __call__(self, q_l, c_l, cos, sin, valid=None):
        for block in self.blocks:
            q_l = block(q_l, c_l, cos, sin, valid)
        return q_l


# ---------------------------------------------------------------------------
# Atom <-> token aggregation + atom encoder + inputs embedder
# ---------------------------------------------------------------------------

ATOM_FEATURE_DIM = 389  # 3 (pos) + 1 (charge) + 1 (mask) + 128 (element) + 256 (name chars)
MAX_CHARS = 4
CHAR_VOCAB_SIZE = 64


def scatter_atom_to_token(atom_features, atom_to_token_idx, n_tokens, atom_mask=None):
    """Mean-aggregate per-atom features to per-token (masked atoms excluded)."""
    B, A, d = atom_features.shape
    if atom_mask is None:
        atom_mask = mx.ones((B, A), dtype=mx.bool_)
    # one-hot over tokens; masked atoms contribute to no token.
    onehot = (atom_to_token_idx[:, :, None] == mx.arange(n_tokens)[None, None, :])
    onehot = (onehot & atom_mask[:, :, None]).astype(atom_features.dtype)  # (B, A, L)
    summed = onehot.transpose(0, 2, 1) @ atom_features  # (B, L, d)
    counts = mx.sum(onehot, axis=1)[:, :, None]  # (B, L, 1)
    return summed / mx.maximum(counts, 1.0)


class ESMFold2AtomEncoder(nn.Module):
    """SWA atom encoder -> per-token features (inputs path: structure_prediction=False)."""

    def __init__(
        self,
        d_atom: int = 128,
        d_token: int = 768,
        n_blocks: int = 3,
        n_heads: int = 4,
        swa_window_size: int = 128,
        expansion_ratio: int = 2,
        structure_prediction: bool = True,
        spatial_rope_base_frequency: float = 20.0,
        n_spatial_rope_pairs_per_axis: int = 2,
        n_uid_rope_pairs: int = 10,
        uid_rope_base_frequency: float = 10000.0,
    ):
        super().__init__()
        self.structure_prediction = structure_prediction
        self.atom_linear = nn.Linear(ATOM_FEATURE_DIM, d_atom, bias=False)
        self.atom_norm = nn.LayerNorm(d_atom, eps=_EPS)
        if structure_prediction:
            self.coords_linear = nn.Linear(6, d_atom, bias=False)
        self.atom_transformer = SWAAtomTransformer(
            d_atom=d_atom, n_blocks=n_blocks, n_heads=n_heads,
            swa_window_size=swa_window_size, expansion_ratio=expansion_ratio,
            spatial_rope_base_frequency=spatial_rope_base_frequency,
            n_spatial_rope_pairs_per_axis=n_spatial_rope_pairs_per_axis,
            n_uid_rope_pairs=n_uid_rope_pairs,
            uid_rope_base_frequency=uid_rope_base_frequency,
        )
        out_dim = d_token if structure_prediction else d_token // 2
        self.atom_to_token_linear = nn.Linear(d_atom, out_dim, bias=False)

    def __call__(
        self, ref_pos, atom_attention_mask, ref_space_uid, ref_charge,
        ref_element, ref_atom_name_chars, atom_to_token, n_tokens,
        r_l=None, num_diffusion_samples=1, return_skip=False,
    ):
        B, N = ref_pos.shape[:2]
        nds = num_diffusion_samples
        atom_feats = mx.concatenate(
            [
                ref_pos,
                ref_charge[..., None],
                atom_attention_mask[..., None],
                ref_element,
                ref_atom_name_chars.reshape(B, N, MAX_CHARS * CHAR_VOCAB_SIZE),
            ],
            axis=-1,
        )
        c_base = self.atom_norm(self.atom_linear(atom_feats))
        cos, sin = self.atom_transformer.build_rope(ref_pos, ref_space_uid)
        if nds > 1:
            cos = mx.repeat(cos, nds, axis=0)
            sin = mx.repeat(sin, nds, axis=0)
            mask_exp = mx.repeat(atom_attention_mask, nds, axis=0)
            a2t_exp = mx.repeat(atom_to_token, nds, axis=0)
        else:
            mask_exp, a2t_exp = atom_attention_mask, atom_to_token
        valid = mask_exp.astype(mx.bool_)

        c = c_base
        q = c
        if self.structure_prediction and r_l is not None:
            if nds > 1:
                q = mx.repeat(q, nds, axis=0)
            pred_r1 = mx.zeros_like(r_l)
            q = q + self.coords_linear(mx.concatenate([r_l, pred_r1], axis=-1))
        if nds > 1:
            c = mx.repeat(c, nds, axis=0)

        q = self.atom_transformer(q, c, cos, sin, valid)
        q_to_a = nn.relu(self.atom_to_token_linear(q))
        a = scatter_atom_to_token(q_to_a, a2t_exp, n_tokens, atom_mask=valid)
        if return_skip:
            return a, q, c, (cos, sin, valid)
        return a


class InputsEmbedder(nn.Module):
    """Atom encoding concatenated with aatype, profile, deletion_mean -> x_inputs[B,L,451]."""

    def __init__(self, atom_encoder: ESMFold2AtomEncoder):
        super().__init__()
        self.atom_attention_encoder = atom_encoder

    def __call__(
        self, aatype, profile, deletion_mean, ref_pos, atom_attention_mask,
        ref_space_uid, ref_charge, ref_element, ref_atom_name_chars,
        atom_to_token, n_tokens,
    ):
        a = self.atom_attention_encoder(
            ref_pos, atom_attention_mask, ref_space_uid, ref_charge,
            ref_element, ref_atom_name_chars, atom_to_token, n_tokens,
        )
        return mx.concatenate([a, aatype, profile, deletion_mean[..., None]], axis=-1)


# ---------------------------------------------------------------------------
# Parcae contractive-SSM recurrence
# ---------------------------------------------------------------------------


def _softplus(x: mx.array) -> mx.array:
    # Numerically-stable softplus matching torch.nn.functional.softplus.
    return mx.maximum(x, 0) + mx.log1p(mx.exp(-mx.abs(x)))


class ParcaeRecurrence(nn.Module):
    """Diagonal contractive SSM step: z <- a⊙z + Linear(input_norm(z_inject), b).

    a = exp(-softplus(log_delta) ⊙ exp(log_a)) ∈ (0,1) channel-wise (contractive).
    b = softplus(log_delta)[:,None] ⊙ b_cont.
    """

    def __init__(self, d_pair: int = 256):
        super().__init__()
        self.parcae_log_a = mx.zeros((d_pair,))
        self.parcae_log_delta = mx.zeros((d_pair,))
        self.parcae_b_cont = mx.eye(d_pair)
        self.parcae_input_norm = nn.LayerNorm(d_pair)  # default eps 1e-5
        self.parcae_readout = nn.Linear(d_pair, d_pair, bias=False)

    def discretized_dynamics(self):
        delta = _softplus(self.parcae_log_delta)
        a = mx.exp(-delta * mx.exp(self.parcae_log_a))
        b = delta[:, None] * self.parcae_b_cont
        return a, b

    def step(self, z, z_inject, a, b):
        injected = self.parcae_input_norm(z_inject)
        return a * z + injected @ b.T  # F.linear(injected, b)

    def readout(self, z):
        return self.parcae_readout(z)


# ---------------------------------------------------------------------------
# Diffusion token transformer (denoiser core)
# ---------------------------------------------------------------------------


def _layer_norm(x, weight=None, bias=None, eps=1e-5):
    """Functional layer norm matching F.layer_norm (biased variance)."""
    xf = x.astype(mx.float32)
    mean = mx.mean(xf, axis=-1, keepdims=True)
    var = mx.mean((xf - mean) ** 2, axis=-1, keepdims=True)
    out = ((xf - mean) * mx.rsqrt(var + eps)).astype(x.dtype)
    if weight is not None:
        out = out * weight
    if bias is not None:
        out = out + bias
    return out


class AdaptiveLayerNorm(nn.Module):
    """adaLN-Zero: normalize a, modulate by conditioning s."""

    def __init__(self, d_model: int, d_cond: int, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.d_cond = d_cond
        self.eps = eps
        self.s_scale = mx.ones((d_cond,))
        self.s_gate = nn.Linear(d_cond, d_model, bias=True)
        self.s_shift = nn.Linear(d_cond, d_model, bias=False)

    def __call__(self, a, s):
        a_norm = _layer_norm(a, eps=self.eps)
        s_norm = _layer_norm(s, weight=self.s_scale, eps=self.eps)
        return mx.sigmoid(self.s_gate(s_norm)) * a_norm + self.s_shift(s_norm)


class FourierEmbedding(nn.Module):
    """cos(2*pi*(t*w + b)); w,b are persistent buffers (loaded from checkpoint)."""

    def __init__(self, c: int):
        super().__init__()
        self.w = mx.zeros((c,))
        self.b = mx.zeros((c,))

    def __call__(self, t_hat):
        t = mx.array(t_hat).reshape(-1) if not isinstance(t_hat, mx.array) else t_hat.reshape(-1)
        return mx.cos(2.0 * mx.pi * (t[:, None] * self.w[None, :] + self.b[None, :]))


class TransitionLayer(nn.Module):
    """SwiGLU transition: norm -> silu(a_proj)*b_proj -> out_proj (no residual)."""

    def __init__(self, d_model: int, n: int, eps: float = 1e-5):
        super().__init__()
        hidden = n * d_model
        self.norm = nn.LayerNorm(d_model, eps=eps)
        self.a_proj = nn.Linear(d_model, hidden, bias=False)
        self.b_proj = nn.Linear(d_model, hidden, bias=False)
        self.out_proj = nn.Linear(hidden, d_model, bias=False)

    def __call__(self, x):
        x = self.norm(x)
        return self.out_proj(nn.silu(self.a_proj(x)) * self.b_proj(x))


class DiffusionConditioning(nn.Module):
    """Condition pair (z) and single (s) reps on the diffusion noise level."""

    def __init__(self, c_z=256, c_s=768, c_s_inputs=451, sigma_data=16.0,
                 fourier_dim=256, transition_multiplier=2, layer_norm_eps=1e-5):
        super().__init__()
        self.sigma_data = float(sigma_data)
        self.z_input_norm = nn.LayerNorm(2 * c_z, eps=layer_norm_eps)
        self.z_proj = nn.Linear(2 * c_z, c_z, bias=False)
        self.z_transitions = [TransitionLayer(c_z, transition_multiplier, layer_norm_eps) for _ in range(2)]
        self.s_input_norm = nn.LayerNorm(c_s_inputs, eps=layer_norm_eps)
        self.s_proj = nn.Linear(c_s_inputs, c_s, bias=False)
        self.fourier = FourierEmbedding(fourier_dim)
        self.noise_norm = nn.LayerNorm(fourier_dim, eps=layer_norm_eps)
        self.noise_proj = nn.Linear(fourier_dim, c_s, bias=False)
        self.s_transitions = [TransitionLayer(c_s, transition_multiplier, layer_norm_eps) for _ in range(2)]

    def static_part(self, s_inputs, z_trunk, relative_position_encoding):
        """t-independent conditioning: the pair rep ``z`` and the base single rep
        ``s_base``. These are identical across all diffusion steps, so the sampler
        computes them once and reuses them."""
        z = mx.concatenate([z_trunk, relative_position_encoding], axis=-1)
        z = self.z_proj(self.z_input_norm(z))
        for block in self.z_transitions:
            z = z + block(z)
        s_base = self.s_proj(self.s_input_norm(s_inputs))
        return z, s_base

    def dynamic_part(self, t_hat, s_base, sigma_data=None):
        """t-dependent single rep: add the noise embedding to ``s_base`` and run
        the s-transitions. Cheap (O(L)); recomputed per step."""
        sigma = self.sigma_data if sigma_data is None else float(sigma_data)
        B = s_base.shape[0]
        t = mx.array(t_hat).reshape(-1) if not isinstance(t_hat, mx.array) else t_hat.reshape(-1)
        if t.shape[0] == 1:
            t = mx.broadcast_to(t, (B,))
        t_noise = 0.25 * mx.log(mx.maximum(t / sigma, 1e-20))
        n = self.noise_proj(self.noise_norm(self.fourier(t_noise)))
        s = s_base + n[:, None]
        for block in self.s_transitions:
            s = s + block(s)
        return s

    def __call__(self, t_hat, s_inputs, z_trunk, relative_position_encoding, sigma_data=None):
        z, s_base = self.static_part(s_inputs, z_trunk, relative_position_encoding)
        s = self.dynamic_part(t_hat, s_base, sigma_data)
        return s, z


class AttentionPairBias(nn.Module):
    """Gated multi-head attention with pair bias + optional adaLN conditioning."""

    def __init__(self, d_model, d_pair, num_heads, d_cond=None, use_conditioning=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5
        self.d_model = d_model
        self.use_conditioning = use_conditioning
        d_cond = d_cond or d_model
        if use_conditioning:
            self.adaln = AdaptiveLayerNorm(d_model, d_cond, eps=1e-5)
            self.out_gate = nn.Linear(d_cond, d_model, bias=True)
        else:
            self.pre_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.g_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        if d_pair > 0:
            self.pair_norm = nn.LayerNorm(d_pair, eps=1e-5)
            self.pair_bias_proj = nn.Linear(d_pair, num_heads, bias=False)

    def pair_bias(self, z, attention_mask=None):
        """Additive attention bias from the pair rep (+ padding), as an SDPA mask
        (B,H,Nq,Nk). Depends only on z + mask, so the sampler precomputes it once
        and passes it back in via ``__call__(pair_bias=...)``."""
        if z.ndim == 4:
            bias = self.pair_bias_proj(self.pair_norm(z)).transpose(0, 3, 1, 2)  # (B,H,Nq,Nk)
        else:
            bias = z[:, None]  # (B,1,Nq,Nk), broadcast over heads
        if attention_mask is not None:
            keep = attention_mask.astype(mx.bool_)[:, None, None, :]  # (B,1,1,Nk)
            bias = mx.where(keep, bias, mx.array(-3.4e38, dtype=bias.dtype))
        return bias

    def __call__(self, a, s, z, attention_mask=None, pair_bias=None):
        B, Nq, _ = a.shape
        x = self.adaln(a, s) if s is not None else self.pre_norm(a)
        H, hd = self.num_heads, self.head_dim
        q = self.q_proj(x).reshape(B, Nq, H, hd)
        k, v = mx.split(self.kv_proj(x), 2, axis=-1)
        Nk = x.shape[1]
        k = k.reshape(B, Nk, H, hd)
        v = v.reshape(B, Nk, H, hd)
        g = mx.sigmoid(self.g_proj(x)).reshape(B, Nq, H, hd)

        qt = q.transpose(0, 2, 1, 3)  # (B, H, Nq, hd)
        kt = k.transpose(0, 2, 1, 3)
        vt = v.transpose(0, 2, 1, 3)

        # Additive attention bias: pair bias + padding. When folding, ``pair_bias``
        # is precomputed once (the pair rep is fixed across diffusion steps) and
        # passed in, skipping the per-step O(L²) bias projection.
        bias = self.pair_bias(z, attention_mask) if pair_bias is None else pair_bias
        logits = (qt @ kt.transpose(0, 1, 3, 2)) * self.scale + bias
        ctx = mx.softmax(logits, axis=-1) @ vt
        ctx = ctx.transpose(0, 2, 1, 3)  # (B, Nq, H, hd)
        ctx = g * ctx
        out = self.out_proj(ctx.reshape(B, Nq, self.d_model))
        if s is not None:
            out = mx.sigmoid(self.out_gate(s)) * out
        return out


class ConditionedTransitionBlock(nn.Module):
    """Conditioned SwiGLU transition with adaLN."""

    def __init__(self, d_model, d_cond=None, transition_multiplier=2, use_conditioning=True):
        super().__init__()
        d_cond = d_cond or d_model
        hidden = transition_multiplier * d_model
        self.use_conditioning = use_conditioning
        if use_conditioning:
            self.adaln = AdaptiveLayerNorm(d_model, d_cond, eps=1e-5)
            self.output_gate = nn.Linear(d_cond, d_model, bias=True)
        else:
            self.pre_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.lin_swish = nn.Linear(d_model, 2 * hidden, bias=False)
        self.lin_out = nn.Linear(hidden, d_model, bias=False)

    def __call__(self, a, s):
        x = self.adaln(a, s) if s is not None else self.pre_norm(a)
        sa, sb = mx.split(self.lin_swish(x), 2, axis=-1)
        out = self.lin_out(nn.silu(sa) * sb)
        if s is not None:
            out = mx.sigmoid(self.output_gate(s)) * out
        return out


class DiffusionTransformer(nn.Module):
    """Stack of (AttentionPairBias + ConditionedTransitionBlock) with residuals."""

    def __init__(self, d_model, d_pair, num_heads, num_blocks, d_cond=None,
                 transition_multiplier=2, use_conditioning=True):
        super().__init__()
        d_cond = d_cond or d_model
        self.attn_blocks = [
            AttentionPairBias(d_model, d_pair, num_heads, d_cond, use_conditioning)
            for _ in range(num_blocks)
        ]
        self.transition_blocks = [
            ConditionedTransitionBlock(d_model, d_cond, transition_multiplier, use_conditioning)
            for _ in range(num_blocks)
        ]

    def precompute_pair_bias(self, z, attention_mask=None):
        """Per-block additive attention bias: fixed across diffusion steps
        because it depends only on the (fixed) pair rep z."""
        return [attn.pair_bias(z, attention_mask) for attn in self.attn_blocks]

    def __call__(self, a, s, z, attention_mask=None, pair_biases=None):
        x = a
        for i, (attn, transition) in enumerate(zip(self.attn_blocks, self.transition_blocks)):
            pb = None if pair_biases is None else pair_biases[i]
            x = x + attn(x, s, z, attention_mask=attention_mask, pair_bias=pb)
            x = x + transition(x, s)
        return x


# ---------------------------------------------------------------------------
# Diffusion module (atom decoder + full denoiser)
# ---------------------------------------------------------------------------


def gather_token_to_atom(token_features, atom_to_token_idx):
    """Broadcast per-token features to per-atom features. (B,L,d),(B,A)->(B,A,d)."""
    idx = mx.broadcast_to(
        atom_to_token_idx[..., None],
        (*atom_to_token_idx.shape, token_features.shape[-1]),
    ).astype(mx.int32)
    return mx.take_along_axis(token_features, idx, axis=1)


class ESMFold2AtomDecoder(nn.Module):
    """Token features -> per-atom coordinate update via SWA atom transformer."""

    def __init__(self, d_atom=128, d_token=768, n_blocks=3, n_heads=4,
                 swa_window_size=128, expansion_ratio=2,
                 spatial_rope_base_frequency=20.0, n_spatial_rope_pairs_per_axis=2,
                 n_uid_rope_pairs=10, uid_rope_base_frequency=10000.0):
        super().__init__()
        self.token_to_atom_linear = nn.Linear(d_token, d_atom, bias=False)
        self.atom_transformer = SWAAtomTransformer(
            d_atom=d_atom, n_blocks=n_blocks, n_heads=n_heads,
            swa_window_size=swa_window_size, expansion_ratio=expansion_ratio,
            spatial_rope_base_frequency=spatial_rope_base_frequency,
            n_spatial_rope_pairs_per_axis=n_spatial_rope_pairs_per_axis,
            n_uid_rope_pairs=n_uid_rope_pairs,
            uid_rope_base_frequency=uid_rope_base_frequency,
        )
        self.norm = nn.LayerNorm(d_atom, eps=_EPS)
        self.output_linear = nn.Linear(d_atom, 3, bias=False)

    def __call__(self, a_i, q_l, c_l, p_lm, atom_to_token, num_diffusion_samples=1):
        cos, sin, valid = p_lm
        a2t = (mx.repeat(atom_to_token, num_diffusion_samples, axis=0)
               if num_diffusion_samples > 1 else atom_to_token)
        a_to_q = gather_token_to_atom(self.token_to_atom_linear(a_i), a2t)
        q_l = q_l + a_to_q
        q_l = self.atom_transformer(q_l, c_l, cos, sin, valid)
        return self.output_linear(self.norm(q_l))


class DiffusionModule(nn.Module):
    """Full diffusion denoiser: conditioning -> atom enc -> token tx -> atom dec -> denoise."""

    def __init__(self, c_atom=128, c_token=768, c_z=256, c_s_inputs=451,
                 sigma_data=16.0, fourier_dim=256, atom_num_blocks=3, atom_num_heads=4,
                 token_num_blocks=12, token_num_heads=16, transition_multiplier=2,
                 swa_window_size=128, spatial_rope_base_frequency=20.0,
                 n_spatial_rope_pairs_per_axis=2, n_uid_rope_pairs=10,
                 uid_rope_base_frequency=10000.0):
        super().__init__()
        self.sigma_data = float(sigma_data)
        rope = dict(swa_window_size=swa_window_size,
                    spatial_rope_base_frequency=spatial_rope_base_frequency,
                    n_spatial_rope_pairs_per_axis=n_spatial_rope_pairs_per_axis,
                    n_uid_rope_pairs=n_uid_rope_pairs,
                    uid_rope_base_frequency=uid_rope_base_frequency)
        self.conditioning = DiffusionConditioning(
            c_z=c_z, c_s=c_token, c_s_inputs=c_s_inputs, sigma_data=sigma_data,
            fourier_dim=fourier_dim, transition_multiplier=transition_multiplier)
        self.atom_encoder = ESMFold2AtomEncoder(
            d_atom=c_atom, d_token=c_token, n_blocks=atom_num_blocks,
            n_heads=atom_num_heads, expansion_ratio=2, structure_prediction=True, **rope)
        self.atom_decoder = ESMFold2AtomDecoder(
            d_atom=c_atom, d_token=c_token, n_blocks=atom_num_blocks,
            n_heads=atom_num_heads, expansion_ratio=2, **rope)
        self.s_to_token = nn.Linear(c_token, c_token, bias=False)
        self.token_transformer = DiffusionTransformer(
            d_model=c_token, d_pair=c_z, num_heads=token_num_heads,
            num_blocks=token_num_blocks, d_cond=c_token,
            transition_multiplier=transition_multiplier, use_conditioning=True)
        self.s_step_norm = nn.LayerNorm(c_token, eps=_EPS)
        self.token_norm = nn.LayerNorm(c_token, eps=_EPS)

    def precompute_conditioning(self, s_inputs, z_trunk, relative_position_encoding,
                                token_attention_mask=None):
        """Compute everything t-independent ONCE — the pair rep z, the base single
        rep s_base, and the per-block token-attention pair biases — so the sampler
        reuses them across all diffusion steps."""
        z, s_base = self.conditioning.static_part(s_inputs, z_trunk, relative_position_encoding)
        pair_biases = self.token_transformer.precompute_pair_bias(z, token_attention_mask)
        return z, s_base, pair_biases

    def __call__(self, x_noisy, t_hat, ref_pos, ref_charge, ref_mask, ref_element,
                 ref_atom_name_chars, ref_space_uid, tok_idx, s_inputs, z_trunk,
                 relative_position_encoding, n_tokens, sigma_data=None,
                 token_attention_mask=None, num_diffusion_samples=1, cond=None):
        bsz = x_noisy.shape[0]
        sigma = self.sigma_data if sigma_data is None else float(sigma_data)
        t = mx.array(t_hat).reshape(-1) if not isinstance(t_hat, mx.array) else t_hat.reshape(-1)
        if t.shape[0] == 1:
            t = mx.broadcast_to(t, (bsz,))

        # ``cond`` holds the precomputed t-independent conditioning (pair rep,
        # base single rep, per-block pair biases), reused across all steps.
        if cond is not None:
            z, s_base, pair_biases = cond
            s = self.conditioning.dynamic_part(t, s_base, sigma)
        else:
            pair_biases = None
            s, z = self.conditioning(t, s_inputs, z_trunk, relative_position_encoding, sigma)
        denom = mx.sqrt(t * t + sigma * sigma)
        r_noisy = x_noisy / denom[:, None, None]

        a, q_skip, c_skip, p_skip = self.atom_encoder(
            ref_pos, ref_mask, ref_space_uid, ref_charge, ref_element,
            ref_atom_name_chars, tok_idx, n_tokens, r_l=r_noisy,
            num_diffusion_samples=num_diffusion_samples, return_skip=True)
        a = a + self.s_to_token(self.s_step_norm(s))
        a = self.token_transformer(
            a, s, z, attention_mask=token_attention_mask, pair_biases=pair_biases)
        a = self.token_norm(a)
        r_update = self.atom_decoder(
            a, q_skip, c_skip, p_skip, tok_idx, num_diffusion_samples=num_diffusion_samples)

        sigma2, t2 = sigma * sigma, t * t
        out = (sigma2 / (sigma2 + t2))[:, None, None] * x_noisy
        out = out + ((sigma * t) / mx.sqrt(sigma2 + t2))[:, None, None] * r_update
        return out


# ---------------------------------------------------------------------------
# Diffusion sampler (Karras schedule + Euler-Maruyama + Kabsch alignment)
# ---------------------------------------------------------------------------


def _det3(M):
    """Batched determinant of (..., 3, 3) matrices (no mx.linalg.det)."""
    a, b, c = M[..., 0, 0], M[..., 0, 1], M[..., 0, 2]
    d, e, f = M[..., 1, 0], M[..., 1, 1], M[..., 1, 2]
    g, h, i = M[..., 2, 0], M[..., 2, 1], M[..., 2, 2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def weighted_rigid_align(x, x_gt, w, mask):
    """Kabsch: rotate/translate x onto x_gt with per-point weights (svd on CPU)."""
    w = (mask * w)[..., None]  # (B, N, 1)
    denom = mx.maximum(mx.sum(w, axis=-2, keepdims=True), 1e-8)
    mu = mx.sum(x * w, axis=-2, keepdims=True) / denom
    mu_gt = mx.sum(x_gt * w, axis=-2, keepdims=True) / denom
    x_c, xgt_c = x - mu, x_gt - mu_gt
    H = (w * xgt_c).transpose(0, 2, 1) @ x_c  # (B, 3, 3)
    H32 = H.astype(mx.float32)
    # Guard: LAPACK sgesvdx aborts (uncatchable C++ terminate) on NaN/Inf input.
    # Sanitize so an upstream overflow surfaces as NaN coords instead of a crash.
    H32 = mx.where(mx.isnan(H32) | mx.isinf(H32), mx.zeros_like(H32), H32)
    U, _, Vh = mx.linalg.svd(H32, stream=mx.cpu)
    det = _det3(U @ Vh)
    ones = mx.ones_like(det)
    diag = mx.stack([ones, ones, det], axis=-1)
    D = mx.eye(3) * diag[..., None, :]  # diag_embed([1,1,det])
    R = (U @ D @ Vh).astype(H.dtype)
    return x_c @ R.transpose(0, 2, 1) + mu_gt


def quat_to_rotation(q):
    """Quaternions (n,4) -> rotation matrices (n,3,3), matching the reference."""
    scale = mx.sqrt(mx.sum(q * q, axis=1))
    signs = mx.where(q[:, 0] < 0, -scale, scale)
    q = q / signs[:, None]
    r, i, j, k = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    two_s = 2.0 / mx.sum(q * q, axis=-1)
    rot = mx.stack([
        1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
        two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
        two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
    ], axis=-1)
    return rot.reshape(-1, 3, 3)


class DiffusionSampler(nn.Module):
    """Wraps DiffusionModule with the Karras + Euler-Maruyama sampling loop.

    Sampling is stochastic; `noise` lets callers inject pre-drawn random tensors
    (init, per-step augmentation quaternion/translation, per-step eps) so the
    loop is reproducible / parity-checkable. Without `noise`, draws via mx.random.
    """

    def __init__(self, diffusion_module: DiffusionModule, *, sigma_data=16.0,
                 gamma_0=0.605, gamma_min=1.107, noise_scale=0.0, step_scale=1.0,
                 inference_s_max=160.0, inference_s_min=4e-4, inference_p=8.0,
                 inference_num_steps=68):
        super().__init__()
        self.diffusion_module = diffusion_module
        self.sigma_data = sigma_data
        self.gamma_0, self.gamma_min = gamma_0, gamma_min
        self.noise_scale, self.step_scale = noise_scale, step_scale
        self.inference_s_max, self.inference_s_min = inference_s_max, inference_s_min
        self.inference_p, self.inference_num_steps = inference_p, inference_num_steps

    def noise_schedule(self, num_steps=None):
        steps = self.inference_num_steps if num_steps is None else int(num_steps)
        if steps == 1:
            sched = mx.array([self.inference_s_max * self.sigma_data, 0.0])
            return sched
        p, inv_p = float(self.inference_p), 1.0 / float(self.inference_p)
        k = mx.arange(steps, dtype=mx.float32)
        base = self.inference_s_max**inv_p + (k / (steps - 1)) * (
            self.inference_s_min**inv_p - self.inference_s_max**inv_p)
        sched = self.sigma_data * (base ** p)
        return mx.concatenate([sched, mx.zeros((1,))])

    def _center_random_augmentation(self, x, atom_mask, second=None, quat=None, trans=None):
        B = x.shape[0]
        m = atom_mask[..., None]
        mean = mx.sum(x * m, axis=1, keepdims=True) / mx.maximum(mx.sum(m, axis=1, keepdims=True), 1)
        x = x - mean
        second = None if second is None else second - mean
        R = quat_to_rotation(mx.random.normal((B, 4)) if quat is None else quat)  # (B,3,3)
        x = x @ R
        second = None if second is None else second @ R
        t = mx.random.normal((B, 1, 3)) if trans is None else trans
        return x + t, (None if second is None else second + t)

    def sample(self, z_trunk, s_inputs, relative_position_encoding, ref_pos, ref_charge,
               ref_mask, ref_element, ref_atom_name_chars, ref_space_uid, tok_idx,
               n_tokens, token_attention_mask=None, num_diffusion_samples=1,
               num_sampling_steps=None, max_inference_sigma=256.0, injected=None):
        n_atoms = tok_idx.shape[1]
        tb = z_trunk.shape[0] * num_diffusion_samples
        steps = self.inference_num_steps if num_sampling_steps is None else int(num_sampling_steps)
        sl = self.noise_schedule(steps).tolist()
        if max_inference_sigma is not None:
            m = float(max_inference_sigma)
            sl = [m] + [s for s in sl if s <= m]
        lam, eta = self.noise_scale, self.step_scale
        atom_mask = (mx.repeat(ref_mask, num_diffusion_samples, 0) if num_diffusion_samples > 1 else ref_mask).astype(mx.float32)
        # injected noise (for parity): consume pre-drawn tensors in draw order
        _it = iter(injected) if injected is not None else None

        def draw(shape):
            return mx.array(next(_it)) if _it is not None else mx.random.normal(shape)

        # The t-independent conditioning (pair rep, base single rep, per-block pair
        # biases) is identical across all steps — compute it once, reuse each step.
        cond = self.diffusion_module.precompute_conditioning(
            s_inputs, z_trunk, relative_position_encoding, token_attention_mask)

        x = sl[0] * draw((tb, n_atoms, 3))
        gl = [self.gamma_0 if s > self.gamma_min else 0.0 for s in sl]
        x_prev = None
        for i in range(len(sl) - 1):
            sigma_tm, sigma_t, gamma = sl[i], sl[i + 1], gl[i + 1]
            q = draw((tb, 4)); tr = draw((tb, 1, 3))
            x, x_prev = self._center_random_augmentation(x, atom_mask, x_prev, quat=q, trans=tr)
            t_hat = sigma_tm * (1.0 + gamma)
            eps_std = lam * max(t_hat ** 2 - sigma_tm ** 2, 0.0) ** 0.5
            x_noisy = x + eps_std * draw(x.shape)
            x_den = self.diffusion_module(
                x_noisy=x_noisy, t_hat=mx.array([t_hat]), ref_pos=ref_pos,
                ref_charge=ref_charge, ref_mask=ref_mask, ref_element=ref_element,
                ref_atom_name_chars=ref_atom_name_chars, ref_space_uid=ref_space_uid,
                tok_idx=tok_idx, s_inputs=s_inputs, z_trunk=z_trunk,
                relative_position_encoding=relative_position_encoding, n_tokens=n_tokens,
                token_attention_mask=token_attention_mask,
                num_diffusion_samples=num_diffusion_samples, cond=cond)
            x_noisy = weighted_rigid_align(
                x_noisy.astype(mx.float32), x_den.astype(mx.float32), atom_mask, atom_mask)
            x = x_noisy + eta * (sigma_t - t_hat) * ((x_noisy - x_den) / t_hat)
            x_prev = x_den
        return x


# ---------------------------------------------------------------------------
# Trunk loop + distogram (z_init -> 21x contractive-SSM loop -> readout/coda -> distogram)
# ---------------------------------------------------------------------------


class ESMFold2Trunk(nn.Module):
    """The recurrent pair-refinement trunk through the distogram head.

    Parcae params are top-level (matching the reference ESMFold2Model layout).
    Excludes z_init construction's atom encoder; `run_loop` takes z_init / lm_z
    directly. lm_dropout / MSA are inference-deterministic only with lm_z=None
    or lm_dropout=0 (the reference applies per-loop LM dropout otherwise).
    """

    def __init__(self, d_pair: int = 256, n_trunk_layers: int = 24,
                 n_coda_layers: int = 2, distogram_bins: int = 128):
        super().__init__()
        self.parcae_log_a = mx.zeros((d_pair,))
        self.parcae_log_delta = mx.zeros((d_pair,))
        self.parcae_b_cont = mx.eye(d_pair)
        self.parcae_input_norm = nn.LayerNorm(d_pair)
        self.parcae_readout = nn.Linear(d_pair, d_pair, bias=False)
        self.folding_trunk = FoldingTrunk(n_layers=n_trunk_layers, d_pair=d_pair)
        self.parcae_coda = FoldingTrunk(n_layers=n_coda_layers, d_pair=d_pair)
        self.distogram_head = nn.Linear(d_pair, distogram_bins)

    def _dynamics(self):
        delta = _softplus(self.parcae_log_delta)
        a = mx.exp(-delta * mx.exp(self.parcae_log_a))
        b = delta[:, None] * self.parcae_b_cont
        return a.reshape(1, 1, 1, -1), b

    def run_loop(self, z, z_init, pair_mask, total_steps, lm_z=None):
        a, b = self._dynamics()
        for _ in range(total_steps):
            z_inject = z_init if lm_z is None else z_init + lm_z
            injected = self.parcae_input_norm(z_inject)
            z = a * z + injected @ b.T
            z = self.folding_trunk(z, mask=pair_mask)
        z = self.parcae_readout(z)
        z = self.parcae_coda(z, mask=pair_mask)
        return z

    def distogram(self, z):
        return self.distogram_head(z + z.transpose(0, 2, 1, 3))


# ---------------------------------------------------------------------------
# Confidence head (pLDDT / PAE / pTM / ipTM)
# ---------------------------------------------------------------------------

_NONPOLYMER_ID = 3


def _categorical_mean(logits, start, end):
    n = logits.shape[-1]
    edges = mx.linspace(start, end, n + 1)
    v = (edges[:-1] + edges[1:]) / 2
    return mx.sum(mx.softmax(logits.astype(mx.float32), axis=-1) * v, axis=-1)


def _cdist(a):  # (B, N, 3) -> (B, N, N)
    diff = a[:, :, None, :] - a[:, None, :, :]
    return mx.sqrt(mx.maximum(mx.sum(diff * diff, axis=-1), 0.0))


def _intra_token_idx(a2t):  # (B, A) contiguous -> local index within token
    B, A = a2t.shape
    same = mx.concatenate([mx.zeros((B, 1), dtype=mx.bool_), a2t[:, 1:] == a2t[:, :-1]], axis=1)
    cumsum = mx.cumsum(mx.ones_like(a2t), axis=-1)
    group_start = mx.where(same, mx.zeros_like(cumsum), cumsum)
    group_start = mx.cummax(group_start, axis=-1)
    return cumsum - group_start


class RowAttentionPooling(nn.Module):
    def __init__(self, d_pair, d_single):
        super().__init__()
        self.attn_proj = nn.Linear(d_pair, 1, bias=False)
        self.out_proj = nn.Linear(d_pair, d_single, bias=False)

    def __call__(self, z, mask):
        scores = self.attn_proj(z)[..., 0]  # (B, N, M)
        scores = mx.where(mask[:, None, :].astype(mx.bool_), scores, -1e9)
        weights = mx.softmax(scores, axis=-1)
        pooled = mx.sum(weights[..., None] * z, axis=2)  # (B, N, d_pair)
        return self.out_proj(pooled)


class ConfidenceHead(nn.Module):
    """Predicts per-atom pLDDT, PAE, and pTM/ipTM from coords + reps."""

    def __init__(self, d_single=384, d_pair=256, d_inputs=451, distogram_bins=128,
                 min_dist=2.0, max_dist=52.0, num_plddt_bins=50, num_pae_bins=64,
                 num_pde_bins=64, n_trunk_layers=4, max_atoms_per_token=23):
        super().__init__()
        self.boundaries = mx.linspace(min_dist, max_dist, distogram_bins - 1)
        self.dist_bin_pairwise_embed = nn.Embedding(distogram_bins, d_pair)
        self.s_norm = nn.LayerNorm(d_single)               # (unused in forward)
        self.s_inputs_to_single = nn.Linear(d_inputs, d_single, bias=False)  # (unused)
        self.s_to_z = nn.Linear(d_inputs, d_pair, bias=False)
        self.s_to_z_transpose = nn.Linear(d_inputs, d_pair, bias=False)
        self.s_to_z_prod_in1 = nn.Linear(d_inputs, d_pair, bias=False)
        self.s_to_z_prod_in2 = nn.Linear(d_inputs, d_pair, bias=False)
        self.s_to_z_prod_out = nn.Linear(d_pair, d_pair, bias=False)
        self.s_input_to_s = nn.Linear(d_inputs, d_single, bias=False)        # (unused)
        self.s_inputs_norm = nn.LayerNorm(d_inputs)
        self.z_norm = nn.LayerNorm(d_pair)
        self.row_attention_pooling = RowAttentionPooling(d_pair, d_single)
        self.folding_trunk = FoldingTrunk(n_layers=n_trunk_layers, d_pair=d_pair)
        self.plddt_ln = nn.LayerNorm(d_single)
        self.plddt_weight = mx.zeros((max_atoms_per_token, d_single, num_plddt_bins))
        self.pae_ln = nn.LayerNorm(d_pair)
        self.pae_head = nn.Linear(d_pair, num_pae_bins, bias=False)
        self.pde_ln = nn.LayerNorm(d_pair)
        self.pde_head = nn.Linear(d_pair, num_pde_bins, bias=False)
        self.resolved_ln = nn.LayerNorm(d_single)
        self.resolved_weight = mx.zeros((max_atoms_per_token, d_single, 2))

    def __call__(self, s_inputs, z, x_pred, distogram_atom_idx, token_attention_mask,
                 atom_to_token, atom_attention_mask, asym_id, mol_type,
                 relative_position_encoding=None, token_bonds_encoding=None):
        si = self.s_inputs_norm(s_inputs)
        zb = self.z_norm(z)
        if relative_position_encoding is not None:
            zb = zb + relative_position_encoding
        if token_bonds_encoding is not None:
            zb = zb + token_bonds_encoding
        zb = zb + self.s_to_z(si)[:, :, None]
        zb = zb + self.s_to_z_transpose(si)[:, None]
        zb = zb + self.s_to_z_prod_out(
            self.s_to_z_prod_in1(si)[:, :, None, :] * self.s_to_z_prod_in2(si)[:, None, :, :])
        pair = zb

        rep_idx = distogram_atom_idx.astype(mx.int32)
        rep_coords = gather_token_to_atom(x_pred, rep_idx)  # (B, L, 3)
        rep_d = _cdist(rep_coords)
        dbins = mx.sum((rep_d[..., None] > self.boundaries).astype(mx.int32), axis=-1)
        pair = pair + self.dist_bin_pairwise_embed(dbins)

        mask = token_attention_mask
        pair_mask = mask[:, :, None].astype(mx.float32) * mask[:, None, :].astype(mx.float32)
        pair = pair + self.folding_trunk(pair, mask=pair_mask)
        single = self.row_attention_pooling(pair, mask)

        atom_mask_f = atom_attention_mask.astype(mx.float32)
        s_at_atoms = gather_token_to_atom(single, atom_to_token.astype(mx.int32))
        s_at = self.plddt_ln(s_at_atoms)
        intra = mx.minimum(_intra_token_idx(atom_to_token), self.plddt_weight.shape[0] - 1)
        B, A = atom_to_token.shape
        w = mx.take(self.plddt_weight, intra.reshape(-1), axis=0).reshape(
            B, A, self.plddt_weight.shape[1], self.plddt_weight.shape[2])
        plddt_logits = mx.sum(s_at[..., :, None] * w, axis=-2)  # (B, A, nbins)
        plddt_per_atom = _categorical_mean(plddt_logits, 0.0, 1.0)  # (B, A)

        L = single.shape[1]
        oh = (atom_to_token.astype(mx.int32)[:, :, None] == mx.arange(L)[None, None, :]).astype(mx.float32)
        plddt_sum = mx.sum(oh * (plddt_per_atom * atom_mask_f)[..., None], axis=1)
        count = mx.sum(oh * atom_mask_f[..., None], axis=1)
        plddt = plddt_sum / mx.maximum(count, 1e-6)
        complex_plddt = mx.sum(plddt_per_atom * atom_mask_f, axis=-1) / (mx.sum(atom_mask_f, axis=-1) + _EPS)
        plddt_ca = mx.take_along_axis(plddt_per_atom, rep_idx, axis=1)

        pae_logits = self.pae_head(self.pae_ln(pair))
        pae = _categorical_mean(pae_logits, 0.0, 32.0)

        pde_logits = self.pde_head(self.pde_ln(pair))
        pde = _categorical_mean(pde_logits, 0.0, 32.0)

        s_at_res = self.resolved_ln(s_at_atoms)
        w_res = mx.take(self.resolved_weight, intra.reshape(-1), axis=0).reshape(
            B, A, self.resolved_weight.shape[1], self.resolved_weight.shape[2])
        resolved_logits = mx.sum(s_at_res[..., :, None] * w_res, axis=-2)  # (B, A, 2)

        n_bins = pae_logits.shape[-1]
        bw = 32.0 / n_bins
        bin_centers = (mx.arange(n_bins, dtype=mx.float32) + 0.5) * bw
        mask_f = mask.astype(mx.float32)
        N_res = mx.sum(mask_f, axis=-1, keepdims=True)
        d0 = 1.24 * (mx.maximum(N_res, 19) - 15) ** (1.0 / 3.0) - 1.8
        tm_per_bin = 1.0 / (1.0 + (bin_centers / d0) ** 2)  # (B, n_bins)
        tm_expected = mx.sum(mx.softmax(pae_logits, axis=-1) * tm_per_bin[:, None, None, :], axis=-1)
        pm2 = mask_f[:, :, None] * mask_f[:, None, :]
        ptm_row = mx.sum(tm_expected * pm2, axis=-1) / (mx.sum(pm2, axis=-1) + _EPS)
        ptm = mx.max(ptm_row, axis=-1)
        inter = (asym_id[:, :, None] != asym_id[:, None, :]).astype(mx.float32) * pm2
        iptm_row = mx.sum(tm_expected * inter, axis=-1) / (mx.sum(inter, axis=-1) + _EPS)
        iptm = mx.max(iptm_row, axis=-1)

        return {
            "plddt": plddt, "plddt_per_atom": plddt_per_atom, "plddt_ca": plddt_ca,
            "complex_plddt": complex_plddt, "plddt_logits": plddt_logits,
            "pae_logits": pae_logits, "pae": pae,
            "pde_logits": pde_logits, "pde": pde,
            "resolved_logits": resolved_logits,
            "ptm": ptm, "iptm": iptm,
        }


# ---------------------------------------------------------------------------
# MSA encoder (full model only; conditions the pair on the query MSA)
# ---------------------------------------------------------------------------


class PairTransition(nn.Module):
    """LayerNorm + SwiGLU FFN returning a DELTA (caller adds the residual)."""

    def __init__(self, d_model, expansion_ratio=4):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=_EPS)
        self.ffn = SwiGLUMLP(d_model, expansion_ratio=expansion_ratio)

    def __call__(self, x):
        return self.ffn(self.norm(x))


class OuterProductMean(nn.Module):
    def __init__(self, d_msa, d_hidden, d_pair):
        super().__init__()
        self.norm = nn.LayerNorm(d_msa, eps=_EPS)
        self.W = nn.Linear(d_msa, 2 * d_hidden, bias=False)
        self.Wout = nn.Linear(d_hidden * d_hidden, d_pair, bias=True)

    def __call__(self, m, msa_attention_mask):
        m_norm = self.norm(m)
        x = self.W(m_norm) * msa_attention_mask[..., None].astype(m_norm.dtype)
        a, b = mx.split(x, 2, axis=-1)  # (B,L,M,c),(B,L,M,d)
        mask_f = msa_attention_mask.astype(a.dtype)
        n_valid = mx.maximum((mask_f @ mask_f.transpose(0, 2, 1))[..., None], 1.0)
        outer = mx.einsum("bimc,bjmd->bijcd", a, b)  # (B,L,L,c,d)
        B, L = outer.shape[0], outer.shape[1]
        return self.Wout(outer.reshape(B, L, L, -1)) / n_valid


class MSAPairWeightedAveraging(nn.Module):
    def __init__(self, d_msa, d_pair, n_heads=8, head_width=32):
        super().__init__()
        self.n_heads, self.head_width = n_heads, head_width
        self.norm_single = nn.LayerNorm(d_msa, eps=_EPS)
        self.compute_bias = [nn.LayerNorm(d_pair, eps=_EPS), nn.Linear(d_pair, n_heads, bias=False)]
        self.Wv = nn.Linear(d_msa, n_heads * head_width, bias=False)
        self.Wgate = nn.Linear(d_msa, n_heads * head_width, bias=False)
        self.Wout = nn.Linear(n_heads * head_width, d_msa, bias=False)

    def __call__(self, msa_repr, pair_repr, pair_attention_mask):
        B, L, M, _ = msa_repr.shape
        h, dh = self.n_heads, self.head_width
        msa_normed = self.norm_single(msa_repr)
        bias = pair_repr
        for layer in self.compute_bias:
            bias = layer(bias)  # (B,L,L,h)
        bias = mx.where(pair_attention_mask[..., None].astype(mx.bool_), bias, -1e5)
        attn = mx.softmax(bias, axis=-2)  # softmax over j (dim=-2 of (B,i,j,h))
        v = self.Wv(msa_normed).reshape(B, L, M, h, dh)
        gate = mx.sigmoid(self.Wgate(msa_normed)).reshape(B, L, M, h, dh)
        # einsum("bijh,bjmhd,bimhd->bimhd") split: contract j, then elementwise gate
        ctx = mx.einsum("bijh,bjmhd->bimhd", attn, v)
        out = ctx * gate
        return self.Wout(out.reshape(B, L, M, h * dh))


class MSAEncoderBlock(nn.Module):
    def __init__(self, d_msa, d_pair, d_hidden, n_heads_msa, msa_head_width, is_final_block=False):
        super().__init__()
        self.is_final_block = is_final_block
        self.outer_product_mean = OuterProductMean(d_msa, d_hidden, d_pair)
        if not is_final_block:
            self.msa_pair_weighted_averaging = MSAPairWeightedAveraging(d_msa, d_pair, n_heads_msa, msa_head_width)
            self.msa_transition = PairTransition(d_msa, expansion_ratio=4)
        self.tri_mul_out = TriangleMultiplicativeUpdate(dim=d_pair, outgoing=True)
        self.tri_mul_in = TriangleMultiplicativeUpdate(dim=d_pair, outgoing=False)
        self.pair_transition = PairTransition(d_pair, expansion_ratio=4)

    def __call__(self, m, pair, msa_attention_mask, pair_attention_mask):
        pair = pair + self.outer_product_mean(m, msa_attention_mask)
        if not self.is_final_block:
            m = m + self.msa_pair_weighted_averaging(m, pair, pair_attention_mask)
            m = m + self.msa_transition(m)
        pair = pair + self.tri_mul_out(pair, mask=pair_attention_mask)
        pair = pair + self.tri_mul_in(pair, mask=pair_attention_mask)
        pair = pair + self.pair_transition(pair)
        return m, pair


class MSAEncoder(nn.Module):
    def __init__(self, d_msa, d_pair, d_inputs, d_hidden=32, n_layers=4, n_heads_msa=8, msa_head_width=16):
        super().__init__()
        self.embed = nn.Linear(35, d_msa, bias=False)
        self.project_inputs = nn.Linear(d_inputs, d_msa, bias=False)
        self.blocks = [
            MSAEncoderBlock(d_msa, d_pair, d_hidden, n_heads_msa, msa_head_width,
                            is_final_block=(i == n_layers - 1))
            for i in range(n_layers)
        ]

    def __call__(self, x_pair, x_inputs, msa_oh, has_deletion, deletion_value, msa_attention_mask):
        m_feat = mx.concatenate([msa_oh, has_deletion[..., None], deletion_value[..., None]], axis=-1)
        m = self.embed(m_feat) + self.project_inputs(x_inputs)[:, :, None]
        tok_mask = msa_attention_mask[:, :, 0].astype(mx.bool_)
        pair_am = (tok_mask[:, :, None] & tok_mask[:, None, :]).astype(mx.float32)
        for block in self.blocks:
            m, x_pair = block(m, x_pair, msa_attention_mask, pair_am)
        return x_pair


# ---------------------------------------------------------------------------
# Top-level ESMFold2Model (pure MLX). Consumes a features dict (mx arrays),
# returns a dict of mx arrays. torch<->mlx bridging lives in a separate adapter.
# ---------------------------------------------------------------------------

NUM_RES_TYPES = 33
MAX_ATOMIC_NUMBER = 128


def _cfg(d, *path, default=None):
    for p in path:
        d = d.get(p, {}) if isinstance(d, dict) else {}
    return d if d != {} else default


class ESMFold2Model(nn.Module):
    """MLX ESMFold2 (release). Pure-MLX; features in / dict out.

    `structure_head` holds the DiffusionSampler (its `.diffusion_module` matches
    the checkpoint key tree). `confidence_head` is optional (skipped here).
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self._esmc = None  # MLX ESMC encoder; attach AFTER strict weight load
        d_pair = config["d_pair"]
        d_inputs = _cfg(config, "inputs", "d_inputs", default=451)
        ae = config["inputs"]["atom_encoder"]
        dm = config["structure_head"]["diffusion_module"]
        sh = config["structure_head"]

        atom_enc = ESMFold2AtomEncoder(
            d_atom=ae["d_atom"], d_token=ae["d_token"], n_blocks=ae["n_blocks"],
            n_heads=ae["n_heads"], swa_window_size=ae["swa_window_size"],
            expansion_ratio=ae["expansion_ratio"], structure_prediction=False,
            spatial_rope_base_frequency=ae["spatial_rope_base_frequency"],
            n_spatial_rope_pairs_per_axis=ae["n_spatial_rope_pairs_per_axis"],
            n_uid_rope_pairs=ae["n_uid_rope_pairs"],
            uid_rope_base_frequency=ae["uid_rope_base_frequency"])
        self.inputs_embedder = InputsEmbedder(atom_enc)

        self.z_init_1 = nn.Linear(d_inputs, d_pair, bias=False)
        self.z_init_2 = nn.Linear(d_inputs, d_pair, bias=False)
        self.rel_pos = RelativePositionEncoding(
            config.get("n_relative_residx_bins", 32),
            config.get("n_relative_chain_bins", 2), d_pair)
        self.token_bonds = nn.Linear(1, d_pair, bias=False)
        self.language_model = LanguageModelShim(
            d_z=d_pair, d_model=config.get("lm_d_model", 2560),
            num_layers=config.get("lm_num_layers", 80))
        self.folding_trunk = FoldingTrunk(config["folding_trunk"]["n_layers"], d_pair)
        self.lm_encoder = FoldingTrunk(config["lm_encoder"]["n_layers"], d_pair)

        self.parcae_log_a = mx.zeros((d_pair,))
        self.parcae_log_delta = mx.zeros((d_pair,))
        self.parcae_b_cont = mx.eye(d_pair)
        self.parcae_input_norm = nn.LayerNorm(d_pair)
        self.parcae_readout = nn.Linear(d_pair, d_pair, bias=False)
        self.parcae_coda = FoldingTrunk(config["parcae"]["coda_n_layers"], d_pair)

        diffusion = DiffusionModule(
            c_atom=dm["c_atom"], c_token=dm["c_token"], c_z=dm["c_z"],
            c_s_inputs=dm["c_s_inputs"], sigma_data=dm["sigma_data"],
            fourier_dim=dm["fourier_dim"], atom_num_blocks=dm["atom_num_blocks"],
            atom_num_heads=dm["atom_num_heads"], token_num_blocks=dm["token_num_blocks"],
            token_num_heads=dm["token_num_heads"],
            transition_multiplier=dm["transition_multiplier"],
            swa_window_size=ae["swa_window_size"],
            spatial_rope_base_frequency=ae["spatial_rope_base_frequency"],
            n_spatial_rope_pairs_per_axis=ae["n_spatial_rope_pairs_per_axis"],
            n_uid_rope_pairs=ae["n_uid_rope_pairs"],
            uid_rope_base_frequency=ae["uid_rope_base_frequency"])
        self.structure_head = DiffusionSampler(
            diffusion, sigma_data=dm["sigma_data"],
            gamma_0=sh.get("gamma_0", 0.605), gamma_min=sh.get("gamma_min", 1.107),
            noise_scale=sh.get("noise_scale", 0.0), step_scale=sh.get("step_scale", 1.0),
            inference_s_max=sh.get("inference_s_max", 160.0),
            inference_s_min=sh.get("inference_s_min", 4e-4),
            inference_p=sh.get("inference_p", 8.0),
            inference_num_steps=sh.get("inference_num_steps", 68))
        self.distogram_head = nn.Linear(d_pair, sh.get("distogram_bins", 64))

        mc = config.get("msa_encoder", {}) or {}
        self.msa_encoder_overwrite = bool(config.get("msa_encoder_overwrite", True))
        if mc.get("enabled", False):
            self.msa_encoder = MSAEncoder(
                d_msa=mc["d_msa"], d_pair=d_pair, d_inputs=d_inputs,
                d_hidden=mc["d_hidden"], n_layers=mc["n_layers"],
                n_heads_msa=mc["n_heads_msa"], msa_head_width=mc["msa_head_width"])
        else:
            self.msa_encoder = None

        # Confidence head (opt-in). Built when the config carries a
        # `confidence_head` section so its weights load 1:1; running it is
        # gated behind `confidence()` / `fold(return_confidence=True)` so the
        # default coords path (and its timing) is unchanged.
        cc = config.get("confidence_head", {}) or {}
        if cc.get("enabled", False):
            ct = cc.get("folding_trunk", {}) or {}
            self.confidence_head = ConfidenceHead(
                d_single=config["d_single"], d_pair=d_pair, d_inputs=d_inputs,
                distogram_bins=cc.get("distogram_bins", 128),
                min_dist=cc.get("min_dist", 2.0), max_dist=cc.get("max_dist", 52.0),
                num_plddt_bins=cc.get("num_plddt_bins", 50),
                num_pae_bins=cc.get("num_pae_bins", 64),
                num_pde_bins=cc.get("num_pde_bins", 64),
                n_trunk_layers=ct.get("n_layers", 4))
        else:
            self.confidence_head = None

    def _dynamics(self):
        delta = _softplus(self.parcae_log_delta)
        a = mx.exp(-delta * mx.exp(self.parcae_log_a)).reshape(1, 1, 1, -1)
        b = delta[:, None] * self.parcae_b_cont
        return a, b

    def compute_lm_hidden_states(self, input_ids):
        """Single-chain: [cls] + residues + [eos] -> MLX ESMC -> (B, L, n_layers+1, D)."""
        B, L = input_ids.shape
        ids = input_ids.astype(mx.int32)
        lm_ids = mx.concatenate(
            [mx.zeros((B, 1), dtype=mx.int32), ids, mx.full((B, 1), 2, dtype=mx.int32)], axis=1)
        _, hs = self._esmc.encode(lm_ids, output_hidden_states=True)  # (n+1, B, L+2, D)
        hs = hs[:, :, 1:1 + L, :]                                      # strip BOS/EOS
        return hs.transpose(1, 2, 0, 3)                               # (B, L, n+1, D)

    def trunk(self, feats, lm_hidden_states=None, z0=None, num_loops=3):
        """Preprocess -> inputs_embedder -> z_init -> loop -> readout/coda. Returns (z, x_inputs, aux)."""
        if lm_hidden_states is None:
            lm_hidden_states = self.compute_lm_hidden_states(feats["input_ids"])
        tok_mask = feats["token_attention_mask"].astype(mx.float32)
        atm_mask = feats["atom_attention_mask"].astype(mx.float32)
        B, L = feats["res_type"].shape
        n_tokens = L

        res_type_oh = _one_hot(feats["res_type"], NUM_RES_TYPES) * tok_mask[..., None]
        profile = res_type_oh  # single-sequence query-only MSA -> profile == query
        deletion_mean = feats.get("deletion_mean", mx.zeros((B, L)))
        ref_element_oh = _one_hot(feats["ref_element"], MAX_ATOMIC_NUMBER) * atm_mask[..., None]
        ref_name_oh = _one_hot(feats["ref_atom_name_chars"], CHAR_VOCAB_SIZE) * atm_mask[..., None, None]
        atom_to_token = (feats["atom_to_token"] * atm_mask.astype(feats["atom_to_token"].dtype)).astype(mx.int32)

        x_inputs = self.inputs_embedder(
            res_type_oh, profile, deletion_mean, feats["ref_pos"], atm_mask,
            feats["ref_space_uid"], feats["ref_charge"].astype(mx.float32),
            ref_element_oh, ref_name_oh, atom_to_token, n_tokens)

        z_init = self.z_init_1(x_inputs)[:, :, None] + self.z_init_2(x_inputs)[:, None]
        z_init = z_init + self.rel_pos(
            feats["residue_index"], feats["asym_id"], feats["sym_id"],
            feats["entity_id"], feats["token_index"])
        z_init = z_init + self.token_bonds(feats["token_bonds"].astype(mx.float32))

        lm_z = self.language_model(lm_hidden_states)

        pair_mask = tok_mask[:, :, None] * tok_mask[:, None, :]
        z = mx.zeros_like(z_init) if z0 is None else z0
        a, b = self._dynamics()
        def _conditioning():
            # Loop-invariant injection term: msa/lm encoders -> input-norm -> @ b.
            z_inject = z_init
            if self.msa_encoder is not None and "msa" in feats:
                msa_oh = _one_hot(feats["msa"].transpose(0, 2, 1), NUM_RES_TYPES)  # (B,L,M,33)
                msa_attn = feats["msa_attention_mask"].transpose(0, 2, 1).astype(mx.float32)
                msa_oh = msa_oh * msa_attn[..., None]
                hd = feats["has_deletion"].transpose(0, 2, 1).astype(mx.float32)
                dv = feats["deletion_value"].transpose(0, 2, 1).astype(mx.float32)
                msa_pair = self.msa_encoder(z_inject, x_inputs, msa_oh, hd, dv, msa_attn)
                z_inject = msa_pair if self.msa_encoder_overwrite else z_inject + msa_pair
            if self.lm_encoder is not None:
                z_inject = z_inject + self.lm_encoder(lm_z, mask=pair_mask)
            return self.parcae_input_norm(z_inject) @ b.T

        # The conditioning is identical every loop — compute it once (hoisted).
        inj = _conditioning()
        for _ in range(max(1, num_loops + 1)):
            z = a * z + inj
            z = self.folding_trunk(z, mask=pair_mask)
        z = self.parcae_readout(z)
        z = self.parcae_coda(z, mask=pair_mask)
        aux = dict(
            relpos=self.rel_pos(feats["residue_index"], feats["asym_id"], feats["sym_id"],
                                feats["entity_id"], feats["token_index"]),
            ref_pos=feats["ref_pos"], ref_charge=feats["ref_charge"].astype(mx.float32),
            ref_mask=atm_mask, ref_element_oh=ref_element_oh, ref_name_oh=ref_name_oh,
            ref_space_uid=feats["ref_space_uid"], atom_to_token=atom_to_token,
            tok_mask=tok_mask, n_tokens=n_tokens)
        return z, x_inputs, aux

    def distogram(self, z):
        return self.distogram_head(z + z.transpose(0, 2, 1, 3))

    def confidence(self, feats, z, x_inputs, aux, coords, num_diffusion_samples=1):
        """Run the confidence head on already-computed trunk/structure outputs.

        Returns {plddt, plddt_ca, complex_plddt, pae, pde, ptm, iptm, ...}.
        Requires the model to have been built with a `confidence_head` section
        in the config (biohub/ESMFold2* have `confidence_head.enabled=True`).
        """
        if self.confidence_head is None:
            raise RuntimeError(
                "confidence_head not built; config has no enabled confidence_head section")
        token_bonds_encoding = self.token_bonds(feats["token_bonds"].astype(mx.float32))
        return self.confidence_head(
            s_inputs=x_inputs, z=z.astype(mx.float32), x_pred=coords,
            distogram_atom_idx=feats["distogram_atom_idx"].astype(mx.int32),
            token_attention_mask=aux["tok_mask"],
            atom_to_token=aux["atom_to_token"],
            atom_attention_mask=aux["ref_mask"],
            asym_id=feats["asym_id"], mol_type=feats["mol_type"],
            relative_position_encoding=aux["relpos"],
            token_bonds_encoding=token_bonds_encoding)

    def fold(self, feats, lm_hidden_states=None, num_loops=3, num_sampling_steps=50,
             num_diffusion_samples=1, z0=None, return_confidence=False):
        z, x_inputs, aux = self.trunk(feats, lm_hidden_states, z0=z0, num_loops=num_loops)
        coords = self.structure_head.sample(
            z_trunk=z, s_inputs=x_inputs, relative_position_encoding=aux["relpos"],
            ref_pos=aux["ref_pos"], ref_charge=aux["ref_charge"], ref_mask=aux["ref_mask"],
            ref_element=aux["ref_element_oh"], ref_atom_name_chars=aux["ref_name_oh"],
            ref_space_uid=aux["ref_space_uid"], tok_idx=aux["atom_to_token"],
            n_tokens=aux["n_tokens"], token_attention_mask=aux["tok_mask"],
            num_diffusion_samples=num_diffusion_samples, num_sampling_steps=num_sampling_steps)
        out = {"sample_atom_coords": coords, "distogram_logits": self.distogram(z)}
        if return_confidence:
            out.update(self.confidence(feats, z, x_inputs, aux, coords,
                                       num_diffusion_samples=num_diffusion_samples))
        return out


def sanitize_esmfold2(weights: dict) -> dict:
    """Strip the reference's `._engine.` trimul wrapper segment. `confidence_head.*`
    keys are kept (they map 1:1 onto the opt-in ConfidenceHead built when the config
    enables it); msa_encoder.* keys are likewise kept when that module is built."""
    return {k.replace("._engine.", "."): v for k, v in weights.items()}


def sanitize_trunk(weights: dict) -> dict:
    """Strip the reference's `._engine.` wrapper segment from trimul keys."""
    return {k.replace("._engine.", "."): v for k, v in weights.items()}
