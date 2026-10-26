# Copyright © 2026 Apple Inc.

# Sparse prefill for DeepSeek sparse attention (DSA) models (deepseek_v32, glm_moe_dsa):
# attention over each query's top-k keys from the indexer.

import mlx.core as mx

SPARSE_QUERY_BLOCK = 1024


def sparse_attention(attn, q_nope, q_pe, kv_latent, k_pe, topk, mask, B, L):
    # Absorbed form: 576-wide keys (512 latent + 64 rope), 512-wide values. B is 1.
    block = SPARSE_QUERY_BLOCK
    if L > block:
        # Query blocks keep the gathered keys small, which is faster for long chunks.
        outs = []
        for s in range(0, L, block):
            e = min(L, s + block)
            m = None if mask is None else mask.reshape(L, -1)[s:e]
            q = (q_nope[:, :, s:e], q_pe[:, :, s:e])
            idx = topk[:, :, s:e]
            outs.append(sparse_attention(attn, *q, kv_latent, k_pe, idx, m, B, e - s))
        return mx.concatenate(outs, axis=1)
    q_full = mx.concatenate([attn.embed_q(q_nope), q_pe], axis=-1)[0]  # [H, L, 576]
    kv_full = mx.concatenate([kv_latent, k_pe], axis=-1)[0, 0]  # [N, 576]
    idx = topk[0, 0]  # [L, K]
    gathered = kv_full[idx]  # [L, K, 576], shared by all heads

    q_t = q_full.transpose(1, 0, 2)  # [L, H, 576]
    scores = (q_t @ gathered.swapaxes(-1, -2)).astype(mx.float32) * attn.scale
    if mask is not None:
        valid = mx.take_along_axis(mask.reshape(L, -1), idx, axis=-1)  # [L, K]
        scores = mx.where(valid[:, None, :], scores, -mx.inf)
    probs = mx.softmax(scores, axis=-1).astype(gathered.dtype)
    latent_out = probs @ gathered[..., : attn.kv_lora_rank]  # [L, H, 512]
    latent_out = latent_out.transpose(1, 0, 2)[None]  # [1, H, L, 512]
    out = attn.unembed_out(latent_out)  # [1, H, L, v_head_dim]
    out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return attn.o_proj(out)


def blocked_topk(indexer, x, qr, mask, cache, block=4096):
    # Stock indexer scores, one key block at a time, without the [b, H, s, N] tensor.
    # Heads are added in the stock order. A running top-k keeps the sorts short and
    # selects the same keys as stock (up to the order of equal scores).
    b, s, _ = x.shape
    q = indexer.wq_b(qr)
    q = q.reshape(b, s, indexer.n_heads, indexer.head_dim).swapaxes(1, 2)
    k = indexer.k_norm(indexer.wk(x))
    k = mx.reshape(k, (b, 1, s, indexer.head_dim))
    offset = cache.offset if cache is not None else 0
    q = indexer.rope(q, offset=offset)
    k = indexer.rope(k, offset=offset)
    if cache is not None:
        k, _ = cache.update_and_fetch(k, k)
        cache.values = mx.zeros_like(cache.keys)
    n = k.shape[2]
    if n <= indexer.index_topk:
        return None

    weights = indexer.weights_proj(x) * (indexer.n_heads**-0.5 * indexer.softmax_scale)
    weights = weights.swapaxes(-1, -2)[..., None]  # [b, H, s, 1]

    best_v = best_i = None
    for s0 in range(0, n, block):
        s1 = min(n, s0 + block)
        sc = mx.maximum(q @ k[:, :, s0:s1].swapaxes(-1, -2), 0) * weights
        sc = sum(sc[:, h : h + 1] for h in range(sc.shape[1]))  # [b, 1, s, blk]
        if mask is not None:
            sc = mx.where(mask[..., s0:s1], sc, -float("inf"))
        ids = mx.broadcast_to(mx.arange(s0, s1), sc.shape)
        if best_v is not None:
            sc = mx.concatenate([best_v, sc], axis=-1)
            ids = mx.concatenate([best_i, ids], axis=-1)
        k_top = indexer.index_topk
        pos = mx.stop_gradient(mx.argpartition(sc, kth=-k_top, axis=-1)[..., -k_top:])
        best_v = mx.take_along_axis(sc, pos, axis=-1)
        best_i = mx.take_along_axis(ids, pos, axis=-1)
    return best_i
