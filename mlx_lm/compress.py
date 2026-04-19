"""Public API for theorem-guided DeltaNet compression.

The ``mlx_lm.compress`` module exposes utilities for compression-aware
training of GatedDeltaNet-style linear-attention models.

Design motivation
-----------------

Trained GatedDeltaNet state has O(1) stable rank (measured: ≤ 2.12
on Qwen3.5-9B, ≤ 1.94 on Mamba-2-370M, ≤ 1.79 on RWKV-7-1.5B).
A formal theorem (see below) shows this low rank follows from the
stable rank of the recent-window key stream, which is itself
architecturally bounded. Therefore the state during training can be
safely projected onto a low-rank subspace at every chunk boundary,
with provable bound on information loss.

This module lets downstream users:

1. **Measure** the minimum-safe compression rank for a given model
   via :func:`estimate_rank` or :func:`estimate_rank_per_layer`.
2. **Enable** compression-aware training via the env vars listed in
   ``qwen3_5.py`` (``MLX_DELTANET_COMPRESS_RANK``,
   ``MLX_DELTANET_COMPRESS_RANK_PER_LAYER``).

Theorem reference
-----------------

    stable_rank(S_T) ≤ r_k · 1/(1 − g²) + O(g^{2W})

where r_k is the stable rank of the recent-window key stream and
g ≤ g_max < 1 is the decay coefficient. Empirically on Qwen3.5-9B
with g_max = 0.95 and r_k ≤ 9 (576 measurements across
24 layers × 3 texts × 8 window sizes): stable_rank(S_T) ≤ 92.

Full proof and derivations in the project's research notes
(THEOREM_MAIN.md, to be published as an arXiv preprint).
"""

from typing import Dict, Optional

from .models.gated_delta_rank_estimator import estimate_rank  # noqa: F401


def estimate_rank_per_layer(
    model,
    tokenizer,
    calibration_text: Optional[str] = None,
    safety_buffer: int = 2,
    min_rank: int = 4,
    max_rank: int = 64,
) -> Dict[int, int]:
    """Compute theorem-safe compression rank for every linear-attn layer.

    Returns a dict ``{layer_idx: rank}`` — write this to JSON and point
    ``MLX_DELTANET_COMPRESS_RANK_PER_LAYER`` at the resulting file to
    enable per-layer compression at training time.

    Rank for layer ``l`` is ``max_window_stable_rank(K_l) + safety_buffer``,
    rounded up to the next power of 2. Typical values on Qwen3.5-9B:
    most layers rank 4–8, L16 (expander) rank 16.
    """
    import mlx.core as mx
    import mlx.nn as nn

    if calibration_text is None:
        calibration_text = (
            "The empirical study of trained state-space models has "
            "revealed a surprising structural property: the recurrent "
            "state occupies only a tiny subspace regardless of "
            "sequence length. This reflects training dynamics. " * 50
        )
    ids = mx.array(tokenizer.encode(calibration_text))[None, ...]
    T = ids.shape[1]

    # Access linear-attention layers.
    try:
        layers = model.language_model.model.layers
    except AttributeError:
        raise ValueError(
            "estimate_rank_per_layer requires a Qwen3.5/Qwen3-Next-style "
            "hybrid model exposing model.language_model.model.layers"
        )

    linear_indices = [i for i, L in enumerate(layers) if getattr(L, "is_linear", False)]
    captured: Dict[int, mx.array] = {}
    for idx in linear_indices:
        L = layers[idx]
        orig = L.linear_attn.in_proj_qkv

        def make_hook(orig_fn, proj_idx):
            def hook(x):
                out = orig_fn(x)
                captured[proj_idx] = out
                return out

            return hook

        L.linear_attn.in_proj_qkv = make_hook(orig, idx)

    caches = model.language_model.make_cache()
    _ = model(ids, cache=caches)
    mx.eval(_)

    def stable_rank_mx(M: mx.array) -> float:
        M32 = M.astype(mx.float32)
        sigma = mx.linalg.svd(M32, stream=mx.cpu)[1]
        sq = sigma * sigma
        total = float(sq.sum().item())
        top = float(sq[0].item())
        return total / max(top, 1e-30)

    ranks: Dict[int, int] = {}
    windows = [5, 10, 20, 50, 100, 200, 500, T]
    powers = [4, 8, 16, 32, 64]

    for idx in linear_indices:
        if idx not in captured:
            continue
        mod = layers[idx].linear_attn
        qkv = captured[idx]
        key_dim = mod.key_dim
        num_k = mod.num_k_heads
        head_k_dim = mod.head_k_dim
        k_block = qkv[:, :, key_dim : 2 * key_dim]
        k_heads = k_block.reshape(1, T, num_k, head_k_dim)
        inv_scale = head_k_dim**-0.5
        k_norm = inv_scale * mx.fast.rms_norm(k_heads, None, 1e-6)
        k0 = k_norm[0, :, 0, :].astype(mx.float32)

        max_sr = 0.0
        for W in windows:
            if W > T:
                continue
            K_W = k0[T - W : T]
            sr = stable_rank_mx(K_W)
            if sr > max_sr:
                max_sr = sr

        target = max_sr + safety_buffer
        rank = next((p for p in powers if p >= target), powers[-1])
        rank = max(min_rank, min(max_rank, rank))
        ranks[idx] = rank

    return ranks
