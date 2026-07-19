# Copyright © 2023-2024 Apple Inc.

import math
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu

# Default cap on experts materialized in one temporary stack, independent of
# resident_expert_fraction (see _offload_enable's docstring for why this must
# NOT scale with resident_slots). 76 (resident_expert_fraction=0.3 on
# Qwen3.6's 256 experts) completed a ~1458-token worst-case prefill without
# incident; 128 (fraction=0.5) hit a real Metal OOM — 64 leaves real margin
# below the observed failure point while still bounding well below a "nearly
# the whole table" chunk at any fraction.
_OFFLOAD_DEFAULT_MAX_STACK = 64

_OFFLOAD_EXECUTOR = None


def _offload_executor():
    """Lazily-created thread pool shared across every offload-enabled
    SwitchLinear/QuantizedSwitchLinear (one per process, not one per module —
    a model has ~120 of these, no reason to pay 120 pools' worth of thread
    overhead). 8 workers: disk I/O bound work, not CPU bound, so this is
    about overlapping blocking read() calls, not matching core count."""
    global _OFFLOAD_EXECUTOR
    if _OFFLOAD_EXECUTOR is None:
        _OFFLOAD_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="mira-expert-fetch")
    return _OFFLOAD_EXECUTOR


def _offload_to_mx(raw):
    """Convert one fetch_fn() result slot to an mx.array. Must run on the
    same thread the resulting array will be used on — MLX streams are
    thread-local, and mira-mlx's engine pins model execution to one
    dedicated thread, so an mx.array constructed on a ThreadPoolExecutor
    worker thread crashes the first time it's used here
    ("RuntimeError: There is no Stream(gpu, N) in current thread", a real
    crash hit during Phase C validation — every fetch_fn call used to build
    its own mx.array directly, inside the worker thread). This is why
    fetch_fn (core/inference/disk_expert_cache.py, mira-core) returns raw
    (np.ndarray, dtype_str) instead: the parallel part (disk I/O + numpy) is
    thread-safe, only this conversion isn't."""
    if raw is None:
        return None
    np_array, dtype_str = raw
    arr = mx.array(np_array)
    if dtype_str == "BF16":
        arr = arr.view(mx.bfloat16)
    return arr


def _offload_fetched_to_data(raw, quantized: bool):
    if quantized:
        w_raw, s_raw, b_raw = raw
        return (_offload_to_mx(w_raw), _offload_to_mx(s_raw), _offload_to_mx(b_raw))
    return _offload_to_mx(raw)


def _gather_sort(x, indices):
    *_, M = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    return x.flatten(0, -3)[order // M], indices[order], inv_order


def _scatter_unsort(x, inv_order, shape=None):
    x = x[inv_order]
    if shape is not None:
        x = mx.unflatten(x, 0, shape)
    return x


def _offload_enable(module, resident_slots: int, fetch_fn, quantized: bool, max_stack_size=None):
    """Switch a SwitchLinear/QuantizedSwitchLinear to a disk-backed,
    dict-keyed LRU expert cache (`expert_id -> weight data`) instead of
    keeping every expert's weights resident, fetching cold experts via
    `fetch_fn(expert_id)`.

    No-op (module keeps its default, fully-resident behavior) if
    `resident_slots >= num_experts` — this keeps the default/unset path
    provably unchanged, since callers only reach this method at all when
    offloading is explicitly requested.

    The cache is keyed by expert id rather than a fixed slot array: a fixed
    (num_experts_resident, ...) tensor with in-place row overwrites would let
    an eviction *later in the same forward call* silently corrupt a slot an
    *earlier* expert in that same call already resolved to (the actual
    gather happens only after every index in the call is resolved) — with a
    dict cache there are no shared slots to collide over, so this can't
    happen.

    `max_stack_size` (default: `min(resident_slots, _OFFLOAD_DEFAULT_MAX_STACK)`)
    bounds how many experts' weights `_offload_chunked_gather` is allowed to
    materialize into one temporary stacked tensor at once. Without this
    bound, a call whose unique-expert count is large — e.g. a long prefill,
    where tokens_in_call * top_k routinely exceeds num_experts, so nearly
    every expert gets touched in one call regardless of how skewed
    steady-state routing is — builds one huge temporary stack covering
    nearly the whole expert table, which is what caused a real Metal
    kIOGPUCommandBufferCallbackErrorOutOfMemory crash under a ~1458-token
    prompt (specs/moe-expert-offload-02-runtime-cache.md, Phase C). Chunking
    keeps every call's peak transient memory bounded regardless of call
    shape, with no prefill/decode signal needed from the caller.

    The default deliberately does NOT scale with resident_slots (i.e. isn't
    just `resident_slots` itself): a first version of this fix used exactly
    that and still crashed with the same OOM at resident_expert_fraction=0.5
    (resident_slots=128 there → chunks of ~128, nearly half the expert
    table again) — the whole point of a bound is that it has to hold
    independent of how generous the residency setting is, not shrink or
    grow with it. `_OFFLOAD_DEFAULT_MAX_STACK` is a fixed cap instead;
    callers that know their own headroom can still override it explicitly.
    """
    n_experts = module.num_experts
    if resident_slots >= n_experts:
        return
    module._offload_fetch = fetch_fn
    module._offload_quantized = quantized
    module._offload_capacity = resident_slots
    module._offload_max_stack_size = max(
        max_stack_size if max_stack_size is not None else min(resident_slots, _OFFLOAD_DEFAULT_MAX_STACK), 1
    )
    module._offload_cache = {}
    module._offload_lru = []
    # True expert count, stashed because the 1-row stand-in installed below
    # can no longer carry it in weight.shape[0] (see the num_experts property).
    module._offload_num_experts = n_experts

    # Seed the resident set by FETCHING FROM DISK, not by slicing the
    # eager-loaded tensor. `module.weight[:resident_slots]` is an mx *view*
    # that pins the ENTIRE parent buffer for as long as it lives — verified
    # directly: allocate (256, 1024, 1024), slice [:32], drop the parent, and 0
    # bytes free until the slice itself is dropped too. So the previous
    # `module.weight = module.weight[:resident_slots]` never released the other
    # experts at all — it kept the whole table resident and offload merely
    # piled cache + temp-stack memory on top, which is exactly why peak memory
    # exceeded the full-resident baseline at every fraction (spec Phase C
    # "still-open gap"). Experts fetched through the same disk path a cold miss
    # uses are genuinely independent buffers, so once they seed the cache the
    # full-size eager tensors have no remaining reference and are freed.
    seed_ids = list(range(resident_slots))
    raw_seed = list(_offload_executor().map(fetch_fn, seed_ids))
    for e, raw in zip(seed_ids, raw_seed):
        module._offload_cache[e] = _offload_fetched_to_data(raw, quantized)
        module._offload_lru.append(e)

    # Replace the full-size eager tensors with a 1-row stand-in (a view of a
    # now-resident expert) so input_dims/output_dims keep resolving and the
    # parameter tree stays intact, WITHOUT holding the other experts. Assigning
    # here drops the last reference to the full-size tensors, which is what
    # actually frees them. The stand-in pins exactly one expert's worth of
    # bytes and is never read for compute — the offload __call__ path reads
    # only the cache, never module.weight/scales/biases.
    seed0 = module._offload_cache[0]
    if quantized:
        w0, s0, b0 = seed0
        module.weight = mx.expand_dims(w0, 0)
        module.scales = mx.expand_dims(s0, 0)
        if b0 is not None:
            module.biases = mx.expand_dims(b0, 0)
        mx.eval(module.weight, module.scales)
    else:
        module.weight = mx.expand_dims(seed0, 0)
        mx.eval(module.weight)

    module._offload_hits = 0
    module._offload_misses = 0


def _offload_touch(module, expert_id):
    if expert_id in module._offload_lru:
        module._offload_lru.remove(expert_id)
    module._offload_lru.append(expert_id)


def _offload_stack_rows(rows, quantized):
    if quantized:
        return (
            mx.stack([d[0] for d in rows]),
            mx.stack([d[1] for d in rows]),
            mx.stack([d[2] for d in rows]) if rows[0][2] is not None else None,
        )
    return mx.stack(rows)


def _offload_chunked_gather(module, x, indices, gather_fn, sorted_indices=False):
    """Compute the same output as gathering against every needed expert at
    once, but never materialize more than `module._offload_max_stack_size`
    experts' weights in a single temporary stack.

    Partitions the call's unique experts into groups of at most
    max_stack_size, running one `gather_fn(x, stacked_data, local_indices)`
    call per group against the *entire* index tensor (positions outside the
    group get a dummy index and their output is discarded), then combines
    the groups' outputs with `mx.where`. This costs more matmul work than a
    single unchunked call when there's more than one group (every group's
    call touches every position, not just its own) — that trade is what
    buys a memory bound that holds for any call shape, prefill or decode,
    without needing to know which one it is.

    Fetches (and evicts) lazily, group by group, rather than resolving every
    unique expert up front: this is what keeps the *cache dict itself*
    bounded during a large call too, not just the final stacked tensor — the
    original design fetched every unique expert into the cache before doing
    any eviction, which meant the dict alone could balloon to near-full-table
    size on a diverse call, independent of the stacking cost. Evicting after
    each group's own gather is safe because each expert belongs to exactly
    one group: nothing evicted here is needed again later in this same call.
    An `mx.eval()` after each group's combine forces MLX to actually release
    the previous group's temporary stack before the next one is built —
    without it, laziness re-batches every group into one deferred graph
    anyway and the memory bound is fiction. That eval is skipped when there is
    only ONE group (the decode case: top_k experts fit a single stack), where
    there is no next group to bound against — see the guard at the eval below.
    """
    flat = indices.reshape(-1).tolist()
    unique = list(dict.fromkeys(flat))
    max_stack = module._offload_max_stack_size
    groups = [unique[i : i + max_stack] for i in range(0, len(unique), max_stack)]

    group_of = {}
    local_pos_of = {}
    for gi, group in enumerate(groups):
        for li, e in enumerate(group):
            group_of[e] = gi
            local_pos_of[e] = li

    cache = module._offload_cache

    def _resolve_group(group):
        """Fetch+stack this group's experts, updating hit/miss counters and
        LRU recency. Shared by both the mask and compact paths so the two
        differ ONLY in how they combine group outputs, never in fetch/eviction
        accounting (a hit is a hit regardless of which path computes it)."""
        missing = [e for e in group if e not in cache]
        module._offload_hits += len(group) - len(missing)
        module._offload_misses += len(missing)
        if missing:
            # Concurrent, not sequential: each fetch is a blocking disk
            # read (~0.3-0.6ms measured against the real Qwen3.6 shards),
            # but a large/diverse call can have tens of thousands of misses
            # in one forward pass — sequential reads there dominate wall
            # time far more than the chunking's extra matmul work does.
            # Each fetch opens its own file handle and only reads
            # already-resolved (read-only after setup) shard/offset
            # metadata, so concurrent calls need no locking. fetch_fn
            # returns raw (numpy, dtype) data, not mx.array — the actual
            # mx.array construction happens right after, back on THIS
            # thread (see _offload_to_mx's docstring for why that split is
            # required, not optional).
            raw_fetched = list(_offload_executor().map(module._offload_fetch, missing))
            for e, raw in zip(missing, raw_fetched):
                cache[e] = _offload_fetched_to_data(raw, module._offload_quantized)
        rows = []
        for e in group:
            rows.append(cache[e])
            _offload_touch(module, e)
        return _offload_stack_rows(rows, module._offload_quantized)

    def _evict():
        while len(module._offload_lru) > module._offload_capacity:
            victim = module._offload_lru.pop(0)
            cache.pop(victim, None)

    # --- Compact path -------------------------------------------------------
    # The mask path below runs EVERY group's gather over ALL positions and
    # discards the out-of-group ones with mx.where: G groups x full-width
    # matmul (the "G-fold" prefill tax). But when the caller has pre-sorted the
    # routing (SwitchGLU/SwitchMLP call _gather_sort before dispatch whenever
    # indices.size >= 64, i.e. exactly the multi-group prefill case), `flat` is
    # monotonic non-decreasing, so each group's positions form ONE contiguous
    # slice of `flat`. We can then slice x to that segment, gather once over
    # just those positions, and concatenate the segments back — each position
    # computed exactly once, no mx.where. Guarded by an actual monotonicity
    # check (not just the caller's flag) so a mislabelled `sorted_indices` can
    # never silently scatter output to the wrong positions: if the guarantee
    # doesn't hold we fall through to the always-correct mask path.
    use_compact = (
        sorted_indices
        and len(groups) > 1
        and indices.ndim == 1
        and all(flat[i] <= flat[i + 1] for i in range(len(flat) - 1))
    )
    if use_compact:
        # Contiguous per-group position counts (monotonic flat => group ids
        # appear in non-decreasing order, so cumulative counts are the segment
        # boundaries).
        counts = [0] * len(groups)
        for e in flat:
            counts[group_of[e]] += 1
        outputs = []
        pos = 0
        for gi, group in enumerate(groups):
            cnt = counts[gi]
            stacked = _resolve_group(group)
            seg = flat[pos : pos + cnt]
            local_indices = mx.array(
                [local_pos_of[e] for e in seg], dtype=indices.dtype
            ).reshape((cnt,) + tuple(indices.shape[1:]))
            seg_output = gather_fn(x[pos : pos + cnt], stacked, local_indices)
            outputs.append(seg_output)
            mx.eval(seg_output)
            _evict()
            pos += cnt
        return mx.concatenate(outputs, axis=0)

    # --- Mask path (always correct, any call shape) -------------------------
    result = None
    for gi, group in enumerate(groups):
        stacked = _resolve_group(group)

        local_flat = [local_pos_of[e] if group_of[e] == gi else 0 for e in flat]
        local_indices = mx.array(local_flat, dtype=indices.dtype).reshape(indices.shape)
        group_output = gather_fn(x, stacked, local_indices)

        if len(groups) == 1:
            result = group_output
        else:
            mask = mx.array([group_of[e] == gi for e in flat], dtype=mx.bool_).reshape(indices.shape)
            # gather_mm/gather_qmm's output has more trailing dims than
            # `indices` itself (e.g. indices.shape + (1, output_dims) for
            # SwitchGLU's usual (N, top_k) indices) — pad to whatever rank
            # this call's output actually came out at, rather than assuming
            # a fixed offset.
            while mask.ndim < group_output.ndim:
                mask = mx.expand_dims(mask, -1)
            result = group_output if result is None else mx.where(mask, group_output, result)
        # The per-group eval bounds transient memory by materializing this
        # group's contribution before the next group's gather allocates. With a
        # single group (always the case at decode: top_k experts < max_stack, so
        # every selection fits one group) there is no next group to bound
        # against, and the eval only forces a per-layer GPU sync with no memory
        # benefit — it defers the token's gather graph to the caller's next eval
        # boundary (the sampler evals each token anyway). Bit-identical either
        # way; measured +13% decode (7.15->8.09 t/s) at identical 12.73GB peak on
        # Qwen3.6-35B-A3B-8bit over-DRAM offload.
        if len(groups) > 1:
            mx.eval(result)
        _evict()

    return result


class QuantizedSwitchLinear(nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
    ):
        super().__init__()

        scale = math.sqrt(1 / input_dims)
        self.weight, self.scales, *biases = mx.quantize(
            mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(num_experts, output_dims, input_dims),
            ),
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        self.biases = biases[0] if biases else None

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

        self.group_size = group_size
        self.bits = bits
        self.mode = mode

        # Freeze this model's parameters
        self.freeze()

    @property
    def input_dims(self):
        return self.scales.shape[2] * self.group_size

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        # After enable_offload the full weight is replaced by a 1-row stand-in,
        # so shape[0] no longer reflects the true expert count; the real value
        # is stashed on the module (offload path only; unset otherwise).
        n = getattr(self, "_offload_num_experts", None)
        return n if n is not None else self.weight.shape[0]

    def enable_offload(self, resident_slots: int, fetch_fn, max_stack_size=None):
        _offload_enable(self, resident_slots, fetch_fn, quantized=True, max_stack_size=max_stack_size)

    def __call__(self, x, indices, sorted_indices=False):
        if hasattr(self, "_offload_fetch"):
            def gather_fn(x, data, local_indices):
                weight, scales, biases = data
                return mx.gather_qmm(
                    x,
                    weight,
                    scales,
                    biases,
                    rhs_indices=local_indices,
                    transpose=True,
                    group_size=self.group_size,
                    bits=self.bits,
                    mode=self.mode,
                    sorted_indices=False,
                )

            x = _offload_chunked_gather(self, x, indices, gather_fn, sorted_indices=sorted_indices)
        else:
            x = mx.gather_qmm(
                x,
                self["weight"],
                self["scales"],
                self.get("biases"),
                rhs_indices=indices,
                transpose=True,
                group_size=self.group_size,
                bits=self.bits,
                mode=self.mode,
                sorted_indices=sorted_indices,
            )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x


class SwitchLinear(nn.Module):
    def __init__(
        self, input_dims: int, output_dims: int, num_experts: int, bias: bool = True
    ):
        super().__init__()
        scale = math.sqrt(1 / input_dims)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(num_experts, output_dims, input_dims),
        )

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

    @property
    def input_dims(self):
        return self.weight.shape[2]

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        # See QuantizedSwitchLinear.num_experts: offload swaps the full weight
        # for a 1-row stand-in, so the true count is stashed on the module.
        n = getattr(self, "_offload_num_experts", None)
        return n if n is not None else self.weight.shape[0]

    def enable_offload(self, resident_slots: int, fetch_fn, max_stack_size=None):
        _offload_enable(self, resident_slots, fetch_fn, quantized=False, max_stack_size=max_stack_size)

    def __call__(self, x, indices, sorted_indices=False):
        if hasattr(self, "_offload_fetch"):
            def gather_fn(x, weight, local_indices):
                return mx.gather_mm(
                    x,
                    weight.swapaxes(-1, -2),
                    rhs_indices=local_indices,
                    sorted_indices=False,
                )

            x = _offload_chunked_gather(self, x, indices, gather_fn, sorted_indices=sorted_indices)
        else:
            x = mx.gather_mm(
                x,
                self["weight"].swapaxes(-1, -2),
                rhs_indices=indices,
                sorted_indices=sorted_indices,
            )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x

    def to_quantized(self, group_size: int = 64, bits: int = 4, mode: str = "affine"):
        num_experts, output_dims, input_dims = self.weight.shape
        ql = QuantizedSwitchLinear(
            input_dims,
            output_dims,
            num_experts,
            False,
            group_size,
            bits,
            mode=mode,
        )
        ql.weight, ql.scales, *biases = mx.quantize(
            self.weight, group_size, bits, mode=mode
        )
        ql.biases = biases[0] if biases else None

        if "bias" in self:
            ql.bias = self.bias
        return ql


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x, gate):
        return swiglu(gate, x)


class SwitchGLU(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=SwiGLU(),
        bias: bool = False,
    ):
        super().__init__()

        self.gate_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.up_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        # When we have many tokens, then sort them to make sure that the access
        # of different experts is in order.
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
        x = self.down_proj(
            self.activation(x_up, x_gate),
            idx,
            sorted_indices=do_sort,
        )

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)


class SwitchMLP(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=nn.GELU(approx="precise"),
        bias: bool = False,
    ):
        super().__init__()

        self.fc1 = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.fc2 = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        # When we have many tokens, then sort them to make sure that the access
        # of different experts is in order.
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        x = self.fc1(x, idx, sorted_indices=do_sort)
        x = self.activation(x)
        x = self.fc2(x, idx, sorted_indices=do_sort)

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)
