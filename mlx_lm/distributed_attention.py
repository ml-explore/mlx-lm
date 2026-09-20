"""Context-sharded (distributed) attention: decode step and sharded prefill.

Decode: each rank holds only a shard of the KV cache (see
:class:`mlx_lm.models.sharded_cache.ShardedKVCache`). A decode-step query
attends over the *entire* context with no causal mask needed -- every cached
key is already in the past. Each rank computes attention against its own
shard only and the exact full-context result is recovered with an online
softmax merge (the same combine rule used by Flash Attention / ring
attention), an O(1) communication cost in the context length: only
per-query max/sum statistics and the (L, D) partial outputs cross the wire,
never the full score matrix.

Prefill: ranks own *different* queries too (not just different KV), so
unlike decode, K/V data itself must move between ranks -- merge statistics
alone are not enough. This is true ring attention: each rank's KV block is
forwarded neighbor-to-neighbor around the ring (``send``/``recv``), so no
rank ever materializes more than its own shard plus one in-flight block --
O(N/P) memory per rank, matching the whole point of context-sharding for
contexts too large for one machine. As each block arrives, the rank folds
it into its running online-softmax result, applying a causal mask only to
its own (diagonal) block and skipping blocks that are entirely in the
future for it (contiguous, order-sharded context), while still forwarding
every block on for ranks further ahead in the ring.

Correctness of the core merge math is covered by
``experiments/test_online_softmax_merge.py`` (decode-style) and
``experiments/test_sharded_prefill_attention.py`` (causal, block-wise) in
the repo root.
"""

import os
import time
from typing import Any, Optional

import mlx.core as mx

from .fast_decode_attention import fast_partial_attention
from .fast_decode_attention import supported as _fast_supported


def _split_heads_for_gqa(queries: mx.array, n_kv_heads: int):
    """Reshape queries so KV heads can broadcast against repeated query groups.

    queries: (B, n_q_heads, L, D) -> (B, n_kv_heads, n_repeats, L, D) when
    n_q_heads > n_kv_heads, otherwise returned unchanged.
    """
    B, n_q_heads, L, D = queries.shape
    n_repeats = n_q_heads // n_kv_heads
    if n_repeats == 1:
        return queries, n_repeats
    return mx.reshape(queries, (B, n_kv_heads, n_repeats, L, D)), n_repeats


# Upper bound on the number of attention scores (batch x heads x queries x keys)
# materialised at once. Bigger inputs are processed in tiles and recombined with
# the online-softmax rule, so temporary memory stays flat instead of growing
# with (queries x keys). Override with MLX_LM_SHARD_SCORE_BUDGET.
SCORE_BUDGET = int(os.environ.get("MLX_LM_SHARD_SCORE_BUDGET", 2**26))


def _partial_attention_tile(
    queries, keys_shard, values_shard, scale, mask=None, softcap=None
):
    """Partial attention of queries against one rank's KV shard.

    Supports GQA (fewer KV heads than query heads). ``mask``, if given, is a
    boolean array broadcastable against (..., Lq, Lk_shard) with True meaning
    "attend"; used for the causal within-block mask during sharded prefill.
    Decode calls this with ``mask=None`` since a query only ever attends to
    past (already-cached) keys.

    Returns (local_max, local_sumexp, local_weighted_v), each broadcastable
    back to the original (B, n_q_heads, L, ...) query shape.
    """
    n_kv_heads = keys_shard.shape[1]
    q, n_repeats = _split_heads_for_gqa(queries, n_kv_heads)
    k, v = keys_shard, values_shard
    if n_repeats > 1:
        k = mx.expand_dims(k, axis=-3)
        v = mx.expand_dims(v, axis=-3)

    scores = (q @ mx.swapaxes(k, -1, -2)) * scale
    if softcap is not None:
        scores = softcap * mx.tanh(scores / softcap)
    if mask is not None:
        scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
    local_max = mx.max(scores, axis=-1, keepdims=True)
    probs = mx.exp(scores - local_max)
    local_sumexp = mx.sum(probs, axis=-1, keepdims=True)
    local_weighted_v = probs @ v

    if n_repeats > 1:
        B, n_q_heads, L, D = queries.shape
        local_max = mx.reshape(local_max, (B, n_q_heads, L, 1))
        local_sumexp = mx.reshape(local_sumexp, (B, n_q_heads, L, 1))
        local_weighted_v = mx.reshape(local_weighted_v, (B, n_q_heads, L, D))

    return local_max, local_sumexp, local_weighted_v


def _kv_len(x):
    """Number of stored tokens of a K or V shard (plain array or quantized tuple)."""
    return x[0].shape[2] if isinstance(x, tuple) else x.shape[2]


def _kv_slice(x, start, stop):
    if isinstance(x, tuple):
        return tuple(a[:, :, start:stop] for a in x)
    return x[:, :, start:stop]


def quant_layout(queries, keys_q):
    """(bits, group_size) of a quantized (packed, scales, biases) tuple."""
    D = queries.shape[-1]
    return keys_q[0].shape[-1] * 32 // D, D // keys_q[1].shape[-1]


def _partial_attention_tile_quantized(
    queries, keys_q, values_q, scale, mask=None, softcap=None
):
    """Partial attention against K/V stored quantized (mx.quantize format).

    Same result as dequantizing first, without materialising a full-precision
    copy: ``mx.quantized_matmul`` reads the packed data directly.
    """
    B, n_q_heads, L, D = queries.shape
    bits, group = quant_layout(queries, keys_q)
    n_kv = keys_q[0].shape[1]
    n_rep = n_q_heads // n_kv
    q = queries * scale
    if n_rep > 1:
        q = mx.reshape(q, (B, n_kv, n_rep, L, D))
        keys_q = tuple(mx.expand_dims(x, axis=-3) for x in keys_q)
        values_q = tuple(mx.expand_dims(x, axis=-3) for x in values_q)
    scores = mx.quantized_matmul(q, *keys_q, transpose=True, group_size=group, bits=bits)
    if softcap is not None:
        scores = softcap * mx.tanh(scores / softcap)
    if mask is not None:
        scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
    local_max = mx.max(scores, axis=-1, keepdims=True)
    probs = mx.exp(scores - local_max)
    local_sumexp = mx.sum(probs, axis=-1, keepdims=True)
    local_wv = mx.quantized_matmul(
        probs, *values_q, transpose=False, group_size=group, bits=bits
    )
    if n_rep > 1:
        local_max = mx.reshape(local_max, (B, n_q_heads, L, 1))
        local_sumexp = mx.reshape(local_sumexp, (B, n_q_heads, L, 1))
        local_wv = mx.reshape(local_wv, (B, n_q_heads, L, D))
    return local_max, local_sumexp, local_wv


def _attention_tile(queries, keys, values, scale, mask=None, softcap=None):
    if isinstance(keys, tuple):
        return _partial_attention_tile_quantized(
            queries, keys, values, scale, mask, softcap
        )
    return _partial_attention_tile(queries, keys, values, scale, mask, softcap)


def local_partial_attention(
    queries, keys_shard, values_shard, scale, mask=None, softcap=None
):
    """Partial attention (running max, sum of exp, weighted V) of ``queries``
    against a KV shard, with bounded temporary memory.

    Same result as computing all scores at once, but when
    ``batch * heads * queries * keys`` exceeds ``SCORE_BUDGET`` the work is split
    into query tiles x key tiles and recombined with ``combine_partial_local``.
    ``mask`` (2-D boolean, queries x keys, True = attend) is sliced per tile.
    """
    if softcap is None and _fast_supported(queries, keys_shard, values_shard, mask):
        return fast_partial_attention(queries, keys_shard, values_shard, scale)
    B, n_q_heads, L, _ = queries.shape
    S = _kv_len(keys_shard)
    if B * n_q_heads * L * S <= SCORE_BUDGET or (mask is not None and mask.ndim != 2):
        return _attention_tile(queries, keys_shard, values_shard, scale, mask, softcap)

    bh = B * n_q_heads
    tq = min(L, max(1, SCORE_BUDGET // (bh * min(S, 4096))))
    tk = min(S, max(1, SCORE_BUDGET // (bh * tq)))
    out_max, out_sum, out_wv = [], [], []
    for qs in range(0, L, tq):
        q = queries[:, :, qs : qs + tq]
        state = None
        for ks in range(0, S, tk):
            tile_mask = None if mask is None else mask[qs : qs + tq, ks : ks + tk]
            part = _attention_tile(
                q,
                _kv_slice(keys_shard, ks, ks + tk),
                _kv_slice(values_shard, ks, ks + tk),
                scale,
                tile_mask,
                softcap,
            )
            state = part if state is None else combine_partial_local(state, part)
            mx.eval(state)  # free this tile's scores before starting the next
        out_max.append(state[0])
        out_sum.append(state[1])
        out_wv.append(state[2])
    return (
        mx.concatenate(out_max, axis=2),
        mx.concatenate(out_sum, axis=2),
        mx.concatenate(out_wv, axis=2),
    )


def combine_partial_local(state_a, state_b):
    """Fold two partial-attention states together, purely locally (no comm).

    Used to accumulate contributions from several KV blocks already held in
    memory on this rank (see ``sharded_prefill_attention``), as opposed to
    ``merge_partial_attention`` which combines one state per rank via
    collectives.
    """
    max_a, sumexp_a, wv_a = state_a
    max_b, sumexp_b, wv_b = state_b

    combined_max = mx.maximum(max_a, max_b)
    factor_a = mx.exp(max_a - combined_max)
    factor_b = mx.exp(max_b - combined_max)

    combined_sumexp = sumexp_a * factor_a + sumexp_b * factor_b
    combined_wv = wv_a * factor_a + wv_b * factor_b

    return combined_max, combined_sumexp, combined_wv


# True: merge with one all_gather per layer (fast). False: the original
# three-collective version (all_max + two all_sum), kept as the simple
# reference for debugging. Both give the same result. Set the environment
# variable MLX_LM_SHARD_UNFUSED=1 to pick the reference version.
FUSE_COLLECTIVES = not os.environ.get("MLX_LM_SHARD_UNFUSED")


def _merge_fused(local_max, local_sumexp, local_weighted_v, group, simulated_rtt_s):
    """Pack the three partial results into one tensor, gather it from every
    rank in a single collective, then run the online-softmax merge locally.
    """
    if simulated_rtt_s:
        time.sleep(simulated_rtt_s)
    packed = mx.concatenate([local_max, local_sumexp, local_weighted_v], axis=-1)
    gathered = mx.distributed.all_gather(packed, group=group)
    size = group.size()
    gathered = mx.reshape(gathered, (size,) + packed.shape)
    maxes = gathered[..., 0:1]
    sumexps = gathered[..., 1:2]
    weighted_vs = gathered[..., 2:]

    global_max = mx.max(maxes, axis=0)
    factor = mx.exp(maxes - global_max)
    global_sumexp = mx.sum(sumexps * factor, axis=0)
    global_weighted_v = mx.sum(weighted_vs * factor, axis=0)
    return global_weighted_v / global_sumexp


def merge_partial_attention(
    local_max,
    local_sumexp,
    local_weighted_v,
    group: Any,
    simulated_rtt_s: float = 0.0,
    fused: Optional[bool] = None,
):
    """Combine partial-attention results from all ranks via online softmax.

    ``fused`` picks the implementation (default: module ``FUSE_COLLECTIVES``).

    ``simulated_rtt_s``, if nonzero, sleeps before each collective to
    approximate the round-trip latency of a real network (Thunderbolt/JACCL)
    when benchmarking on a single machine (see Stage 5 in
    parallel_context_plan.md) -- it has no effect on the numerical result.
    """
    if fused is None:
        fused = FUSE_COLLECTIVES
    if fused:
        return _merge_fused(
            local_max, local_sumexp, local_weighted_v, group, simulated_rtt_s
        )

    if simulated_rtt_s:
        time.sleep(simulated_rtt_s)
    global_max = mx.distributed.all_max(local_max, group=group)
    mx.eval(global_max)

    factor = mx.exp(local_max - global_max)
    rescaled_sumexp = local_sumexp * factor
    rescaled_weighted_v = local_weighted_v * factor

    if simulated_rtt_s:
        time.sleep(simulated_rtt_s)
    global_sumexp = mx.distributed.all_sum(rescaled_sumexp, group=group)
    global_weighted_v = mx.distributed.all_sum(rescaled_weighted_v, group=group)
    mx.eval(global_sumexp, global_weighted_v)

    return global_weighted_v / global_sumexp


def _empty_partial(queries):
    """Zero contribution of a rank that holds no keys yet (e.g. first blocks)."""
    B, H, L, D = queries.shape
    lowest = mx.finfo(queries.dtype).min
    return (
        mx.full((B, H, L, 1), lowest, dtype=queries.dtype),
        mx.zeros((B, H, L, 1), dtype=queries.dtype),
        mx.zeros((B, H, L, D), dtype=queries.dtype),
    )


def _has_keys(keys_shard):
    return keys_shard is not None and _kv_len(keys_shard) > 0


def sharded_scaled_dot_product_attention(
    queries,
    keys_shard,
    values_shard,
    scale: float,
    group: Any,
    simulated_rtt_s: float = 0.0,
    softcap: Optional[float] = None,
):
    """Decode-step distributed attention: local partial + cross-rank merge."""
    if _has_keys(keys_shard):
        local_max, local_sumexp, local_weighted_v = local_partial_attention(
            queries, keys_shard, values_shard, scale, softcap=softcap
        )
    else:
        local_max, local_sumexp, local_weighted_v = _empty_partial(queries)
    return merge_partial_attention(
        local_max, local_sumexp, local_weighted_v, group, simulated_rtt_s
    )


def sharded_query_attention(
    queries,
    keys_shard,
    values_shard,
    scale: float,
    group: Any,
    new_len: int,
    owns_new: bool,
    simulated_rtt_s: float = 0.0,
    softcap: Optional[float] = None,
):
    """``new_len`` replicated query tokens against a sharded, already-stored past.

    Every rank holds the same queries. The rank that owns the new tokens has
    just appended their K/V at the end of its shard, so it needs a causal mask
    among the new tokens (and full access to its own past); other ranks see
    only past keys and use no mask. Partial results are merged as in decode.
    """
    if not _has_keys(keys_shard):
        return merge_partial_attention(
            *_empty_partial(queries), group, simulated_rtt_s
        )
    mask = None
    if owns_new:
        total = _kv_len(keys_shard)
        past = total - new_len
        q_idx = mx.arange(new_len)[:, None]
        k_idx = mx.arange(total)[None, :]
        mask = (k_idx < past) | ((k_idx - past) <= q_idx)
    local_max, local_sumexp, local_wv = local_partial_attention(
        queries, keys_shard, values_shard, scale, mask=mask, softcap=softcap
    )
    return merge_partial_attention(
        local_max, local_sumexp, local_wv, group, simulated_rtt_s
    )


def _causal_block_mask(q_len: int, k_len: int):
    q_idx = mx.arange(q_len)[:, None]
    k_idx = mx.arange(k_len)[None, :]
    return q_idx >= k_idx


def compute_shard_lengths(total: int, weights):
    """Split ``total`` tokens across ranks proportionally to ``weights``.

    Largest-remainder rounding; every rank gets at least one token.
    """
    n = len(weights)
    if total < n:
        raise ValueError(f"cannot split {total} tokens across {n} ranks")
    weight_sum = float(sum(weights))
    raw = [total * w / weight_sum for w in weights]
    lengths = [max(1, int(r)) for r in raw]
    order = sorted(range(n), key=lambda i: raw[i] - int(raw[i]), reverse=True)
    while sum(lengths) < total:
        lengths[order[0]] += 1
        order = order[1:] + order[:1]
    while sum(lengths) > total:
        i = max(range(n), key=lambda j: lengths[j])
        lengths[i] -= 1
    return lengths


def plan_shards(total: int, size: int):
    """Per-rank shard lengths, or ``None`` for the default equal split.

    Unequal (proportional) shards are opt-in: set the environment variable
    ``MLX_LM_SHARD_WEIGHTS`` to one weight per rank, e.g. ``6,1`` gives rank 0
    six times the tokens of rank 1. Unset means the equal-shard path, which
    is unchanged.
    """
    raw = os.environ.get("MLX_LM_SHARD_WEIGHTS")
    if not raw:
        return None
    weights = [float(x) for x in raw.split(",")]
    if len(weights) != size or any(w <= 0 for w in weights):
        raise ValueError(
            f"MLX_LM_SHARD_WEIGHTS={raw!r} needs {size} positive weights"
        )
    return compute_shard_lengths(total, weights)


def shard_bounds(total: int, size: int, rank: int):
    """``(start, end, shard_lengths)`` of this rank's slice of ``total`` tokens.

    Equal split (``shard_lengths`` is ``None``) unless ``MLX_LM_SHARD_WEIGHTS``
    is set, see :func:`plan_shards`.
    """
    lengths = plan_shards(total, size)
    if lengths is None:
        assert total % size == 0, "equal shards need total divisible by ranks"
        shard = total // size
        return rank * shard, (rank + 1) * shard, None
    start = sum(lengths[:rank])
    return start, start + lengths[rank], lengths


def sharded_prefill_attention(
    local_queries,
    local_keys,
    local_values,
    scale: float,
    group: Any,
    simulated_rtt_s: float = 0.0,
    simulated_bandwidth_bytes_per_s: Optional[float] = None,
    shard_lengths: Optional[list] = None,
    softcap: Optional[float] = None,
):
    """Causal attention for one rank's chunk of prefill queries, over a
    contiguous, order-sharded context (rank 0 holds the earliest tokens).

    True ring attention: each rank's KV block is forwarded neighbor-to-
    neighbor around the ring (``send``/``recv``, not ``all_gather``), so at
    any instant a rank holds only its own shard plus one in-flight block --
    O(N/P) memory, never the full context. Each block is used by every rank
    it reaches; a rank contributes it to its running result only while the
    block can actually affect its queries under causality:
      - blocks originating before this rank: fully unmasked (past)
      - this rank's own block: causal mask within the block
      - blocks originating after this rank: skipped, but still forwarded on
        (needed by ranks further ahead in the ring)

    By default all ranks must hold equal-sized local shards. Pass
    ``shard_lengths`` (one length per rank, see ``plan_shards``) to allow
    unequal shards; each rank then sizes its receive buffers from that list.

    ``simulated_rtt_s`` and ``simulated_bandwidth_bytes_per_s``, together,
    approximate real Thunderbolt/JACCL transport cost on a single machine
    for Stage 5 benchmarking (parallel_context_plan.md): a fixed per-hop
    latency plus a transfer time proportional to the actual K/V block size
    (``current_k.nbytes + current_v.nbytes``) at a given link bandwidth --
    the part pure-latency simulation misses, since prefill ships real data
    (unlike decode's O(1) merge statistics). Neither has any effect on the
    numerical result.
    """
    rank = group.rank()
    size = group.size()
    shard_len = local_keys.shape[2]
    if shard_lengths is not None:
        if len(shard_lengths) != size or shard_lengths[rank] != shard_len:
            raise ValueError(
                f"shard_lengths={shard_lengths} does not match rank {rank} "
                f"of {size} with a local shard of {shard_len} tokens"
            )

    current_k, current_v = local_keys, local_values
    state = None

    for i in range(size):
        source_rank = (rank - i) % size
        if source_rank <= rank:
            mask = (
                _causal_block_mask(local_queries.shape[2], shard_len)
                if source_rank == rank
                else None
            )
            block_state = local_partial_attention(
                local_queries, current_k, current_v, scale, mask=mask, softcap=softcap
            )
            state = (
                block_state if state is None else combine_partial_local(state, block_state)
            )

        if i < size - 1:
            next_rank = (rank + 1) % size
            prev_rank = (rank - 1) % size
            hop_delay = simulated_rtt_s
            if simulated_bandwidth_bytes_per_s:
                block_bytes = current_k.nbytes + current_v.nbytes
                hop_delay += block_bytes / simulated_bandwidth_bytes_per_s
            if hop_delay:
                time.sleep(hop_delay)
            # Evaluating a simultaneous send+recv pair on every rank
            # deadlocks the ring backend (each rank's blocking send waits on
            # a receiver that is itself still blocked on its own send).
            # Alternating the order by parity breaks the cycle: on each
            # edge of the ring, one side sends first and the other side
            # recvs first.
            def recv_block(like):
                if shard_lengths is None:
                    return mx.distributed.recv_like(like, prev_rank, group=group)
                n = shard_lengths[(rank - i - 1) % size]
                shape = (*like.shape[:2], n, like.shape[3])
                return mx.distributed.recv(shape, like.dtype, prev_rank, group=group)

            if rank % 2 == 0:
                sent_k = mx.distributed.send(current_k, next_rank, group=group)
                sent_v = mx.distributed.send(current_v, next_rank, group=group)
                mx.eval(sent_k, sent_v)
                recv_k = recv_block(current_k)
                recv_v = recv_block(current_v)
                mx.eval(recv_k, recv_v)
            else:
                recv_k = recv_block(current_k)
                recv_v = recv_block(current_v)
                mx.eval(recv_k, recv_v)
                sent_k = mx.distributed.send(current_k, next_rank, group=group)
                sent_v = mx.distributed.send(current_v, next_rank, group=group)
                mx.eval(sent_k, sent_v)
            current_k, current_v = recv_k, recv_v

    _, sumexp, weighted_v = state
    return weighted_v / sumexp
