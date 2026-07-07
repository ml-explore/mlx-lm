# LLaDA-8B: masked-diffusion language model (MDM).
#
# Structurally OLMo-derived (pre-norm RMSNorm + SwiGLU MLP), but with two
# defining differences from a standard causal LM:
#   1. Attention is BIDIRECTIONAL / non-causal (mask=None everywhere).
#   2. Generation is diffusion-based, re-running a full forward over the whole
#      sequence at every denoising step, so there is NO KV cache.
#
# HF checkpoints prefix everything with ``model.transformer.``; ``sanitize``
# remaps those keys to the standard mlx-lm layout so the weights are
# quantization-friendly.

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import BaseModelArgs

# LLaDA-8B special token ids (from the official config.json).
DEFAULT_MASK_TOKEN_ID = 126336
DEFAULT_EOS_TOKEN_ID = 126081


@dataclass
class ModelArgs(BaseModelArgs):
    # Field names mirror the official config.json keys so ``from_dict`` maps
    # them with no manual translation.
    model_type: str = "llada"
    d_model: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 32
    mlp_hidden_size: int = 12288
    vocab_size: int = 126464
    embedding_size: Optional[int] = None
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    mask_token_id: int = DEFAULT_MASK_TOKEN_ID
    eos_token_id: int = DEFAULT_EOS_TOKEN_ID
    pad_token_id: int = DEFAULT_EOS_TOKEN_ID
    weight_tying: bool = False
    include_bias: bool = False
    include_qkv_bias: bool = False

    def __post_init__(self):
        # ``embedding_size`` is the true vocab / head size in LLaDA configs;
        # keep ``vocab_size`` as an alias when only one is provided.
        if self.embedding_size is None:
            self.embedding_size = self.vocab_size
        else:
            self.vocab_size = self.embedding_size

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.d_model
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = head_dim = args.d_model // args.n_heads
        self.scale = head_dim**-0.5

        bias = args.include_qkv_bias
        self.q_proj = nn.Linear(dim, self.n_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(self.n_heads * head_dim, dim, bias=args.include_bias)

        self.rope = nn.RoPE(head_dim, traditional=False, base=args.rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        prefix_kv: Optional[tuple] = None,
        pos_offset: int = 0,
        return_kv: bool = False,
        suffix_kv: Optional[tuple] = None,
    ):
        """Bidirectional attention with an optional Fast-dLLM prefix/suffix KV cache.

        Modes:

        * **Plain** (``prefix_kv=None``, ``return_kv=False``): the historical
          path — project Q/K/V over all of ``x``, RoPE at absolute position 0,
          attend with no cache. Byte-identical to the original implementation.

        * **Prime** (``return_kv=True``): same full forward, but ALSO return the
          post-RoPE ``(keys, values)`` for the whole ``x`` so the caller can
          slice out a prefix window ``[0, block_start)`` and (for DualCache) a
          suffix window ``[block_end, total_len)`` and cache them.

        * **Cached** (``prefix_kv`` given): ``x`` holds ONLY the active window's
          hidden states. Q/K/V are projected for the active positions, RoPE is
          applied at ``offset=pos_offset`` (their absolute start), the cached
          prefix K/V are prepended, and active queries attend over the full
          ``[prefix ++ active]`` K/V bidirectionally. Only active outputs are
          returned.

        * **DualCache** (``prefix_kv`` AND ``suffix_kv`` given): as cached, but
          the cached suffix K/V (post-RoPE at absolute positions
          ``[block_end, total_len)``) are ALSO appended, so active queries attend
          over ``[prefix ++ active ++ suffix]``. Attention is permutation-
          invariant over keys and RoPE is baked in at prime time, so the concat
          order is fine as long as each cached K carries its correct absolute
          position. This shrinks the active window to just the current block.
        """
        B, L, _ = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        # RoPE at the active window's absolute offset (0 in the plain/prime
        # paths; block_start in the cached path).
        queries = self.rope(queries, offset=pos_offset)
        keys = self.rope(keys, offset=pos_offset)

        if prefix_kv is not None or suffix_kv is not None:
            # Prepend the cached prefix K/V and append the cached suffix K/V
            # (each already post-RoPE at its absolute position) so active
            # queries see the whole sequence bidirectionally.
            key_parts, val_parts = [], []
            if prefix_kv is not None:
                key_parts.append(prefix_kv[0])
                val_parts.append(prefix_kv[1])
            key_parts.append(keys)
            val_parts.append(values)
            if suffix_kv is not None:
                key_parts.append(suffix_kv[0])
                val_parts.append(suffix_kv[1])
            attn_keys = mx.concatenate(key_parts, axis=2)
            attn_values = mx.concatenate(val_parts, axis=2)
        else:
            attn_keys, attn_values = keys, values

        # Bidirectional attention: no causal mask.
        output = mx.fast.scaled_dot_product_attention(
            queries, attn_keys, attn_values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        out = self.o_proj(output)
        if return_kv:
            # Post-RoPE K/V for all of x (caller slices the prefix window).
            return out, (keys, values)
        return out


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.d_model
        hidden_dim = args.mlp_hidden_size
        bias = args.include_bias
        # SwiGLU with three separate matrices (not a fused gate+up).
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.d_model, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.d_model, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        prefix_kv: Optional[tuple] = None,
        pos_offset: int = 0,
        return_kv: bool = False,
        suffix_kv: Optional[tuple] = None,
    ):
        # Pre-norm: attn_norm before attention, ff_norm before MLP.
        attn = self.self_attn(
            self.input_layernorm(x),
            mask,
            prefix_kv=prefix_kv,
            pos_offset=pos_offset,
            return_kv=return_kv,
            suffix_kv=suffix_kv,
        )
        if return_kv:
            attn, kv = attn
        h = x + attn
        out = h + self.mlp(self.post_attention_layernorm(h))
        if return_kv:
            return out, kv
        return out


class LLaDAModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.d_model)
        self.layers = [TransformerBlock(args) for _ in range(args.n_layers)]
        self.norm = nn.RMSNorm(args.d_model, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        prefix_kv: Optional[list] = None,
        pos_offset: int = 0,
        return_kv: bool = False,
        suffix_kv: Optional[list] = None,
    ):
        """Bidirectional forward with an optional Fast-dLLM prefix/suffix KV cache.

        * ``prefix_kv=None, return_kv=False``: the plain full forward.
        * ``return_kv=True``: prime — also return a list of per-layer post-RoPE
          ``(keys, values)`` for the whole ``inputs``.
        * ``prefix_kv`` given (a per-layer list of cached prefix ``(K, V)``):
          ``inputs`` holds only the active window; each layer prepends its
          cached prefix and RoPEs the active tokens at ``pos_offset``.
        * ``suffix_kv`` given (DualCache; a per-layer list of cached suffix
          ``(K, V)``): each layer ALSO appends its cached suffix, so the active
          window is just the current block ``[block_start, block_end)``.
        * ``return_kv=True`` WITH ``prefix_kv``/``suffix_kv`` (incremental
          cache): a cached forward that ALSO returns the per-layer post-RoPE
          ``(keys, values)`` for the ACTIVE window only. Used to capture a
          finalized block's K/V so it can be appended to the prefix cache
          instead of re-priming the next block.
        """
        h = self.embed_tokens(inputs)
        if return_kv:
            pkv_list = prefix_kv if prefix_kv is not None else [None] * len(self.layers)
            skv_list = suffix_kv if suffix_kv is not None else [None] * len(self.layers)
            kvs = []
            for layer, pkv, skv in zip(self.layers, pkv_list, skv_list):
                h, kv = layer(
                    h, mask=None, prefix_kv=pkv, pos_offset=pos_offset,
                    return_kv=True, suffix_kv=skv,
                )
                kvs.append(kv)
            return self.norm(h), kvs
        if prefix_kv is not None or suffix_kv is not None:
            pkv_list = prefix_kv if prefix_kv is not None else [None] * len(self.layers)
            skv_list = suffix_kv if suffix_kv is not None else [None] * len(self.layers)
            for layer, pkv, skv in zip(self.layers, pkv_list, skv_list):
                h = layer(
                    h, mask=None, prefix_kv=pkv, pos_offset=pos_offset, suffix_kv=skv
                )
            return self.norm(h)
        for layer in self.layers:
            h = layer(h, mask=None)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LLaDAModel(args)
        if not args.weight_tying:
            self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)

    def _head(self, out: mx.array) -> mx.array:
        if self.args.weight_tying:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def __call__(
        self,
        inputs: mx.array,
        prefix_kv: Optional[list] = None,
        pos_offset: int = 0,
        return_kv: bool = False,
        suffix_kv: Optional[list] = None,
    ):
        """Bidirectional forward. Returns logits ``[B, L, vocab]``.

        With ``return_kv=True`` also returns the per-layer prefix KV list (for
        priming the Fast-dLLM cache). With ``prefix_kv`` (and optionally
        ``suffix_kv`` for DualCache) given, ``inputs`` is the active window only
        and the returned logits cover just that window. ``return_kv=True``
        together with ``prefix_kv``/``suffix_kv`` is a cached forward that also
        returns the active window's per-layer K/V (incremental-cache capture).
        """
        if return_kv:
            out, kvs = self.model(
                inputs, prefix_kv=prefix_kv, pos_offset=pos_offset,
                return_kv=True, suffix_kv=suffix_kv,
            )
            return self._head(out), kvs
        out = self.model(
            inputs, prefix_kv=prefix_kv, pos_offset=pos_offset, suffix_kv=suffix_kv
        )
        return self._head(out)

    def sanitize(self, weights):
        """Remap HF ``model.transformer.*`` keys to the mlx-lm layout.

        A leading ``model.`` prefix is optional; we key off the ``transformer.``
        segment. The critical ambiguity is ``blocks.{N}.ff_out`` (the MLP
        down-proj, nested under ``.blocks.``) vs the top-level
        ``transformer.ff_out`` (the LM head), disambiguated by whether
        ``.blocks.`` appears in the path.
        """
        sanitized = {}
        for key, value in weights.items():
            new_key = self._remap_key(key)
            sanitized[new_key] = value
        return sanitized

    @staticmethod
    def _remap_key(key: str) -> str:
        # Strip everything up to and including the ``transformer.`` segment so
        # an optional leading ``model.`` prefix is tolerated.
        marker = "transformer."
        idx = key.find(marker)
        if idx == -1:
            return key
        suffix = key[idx + len(marker) :]

        if suffix.startswith("wte."):
            return "model.embed_tokens." + suffix[len("wte.") :]
        if suffix.startswith("ln_f."):
            return "model.norm." + suffix[len("ln_f.") :]
        if suffix.startswith("blocks."):
            rest = suffix[len("blocks.") :]
            n, _, tail = rest.partition(".")
            attn_map = {
                "q_proj.": "self_attn.q_proj.",
                "k_proj.": "self_attn.k_proj.",
                "v_proj.": "self_attn.v_proj.",
                "attn_out.": "self_attn.o_proj.",
                "ff_proj.": "mlp.gate_proj.",
                "up_proj.": "mlp.up_proj.",
                "ff_out.": "mlp.down_proj.",
                "attn_norm.": "input_layernorm.",
                "ff_norm.": "post_attention_layernorm.",
            }
            for src, dst in attn_map.items():
                if tail.startswith(src):
                    return f"model.layers.{n}.{dst}{tail[len(src):]}"
            return key
        # Top-level ``transformer.ff_out`` is the LM head (no ``.blocks.``).
        if suffix.startswith("ff_out."):
            return "lm_head." + suffix[len("ff_out.") :]
        return key

    @property
    def layers(self):
        return self.model.layers


# ---------------------------------------------------------------------------
# Diffusion sampler (LLaDA official generation reimplemented in pure MLX).
# ---------------------------------------------------------------------------


def add_gumbel_noise(logits: mx.array, temperature: float) -> mx.array:
    """Gumbel-max sampling helper (matches the official implementation).

    At ``temperature == 0`` this is a no-op (pure argmax downstream).
    """
    if temperature == 0.0:
        return logits
    logits = logits.astype(mx.float32)
    noise = mx.random.uniform(shape=logits.shape).astype(mx.float32)
    gumbel = (-mx.log(noise)) ** temperature
    return mx.exp(logits) / gumbel


def get_num_transfer_tokens(mask_index: mx.array, steps: int) -> mx.array:
    """Per-row token-reveal schedule over ``steps`` denoising steps.

    Each row unmasks ``base = mask_count // steps`` tokens per step, plus one
    extra for the first ``remainder = mask_count % steps`` steps. The schedule
    sums exactly to ``mask_count`` per row. Returns an int array [B, steps].
    """
    mask_count = mask_index.sum(axis=1, keepdims=True)  # [B, 1]
    base = mask_count // steps
    remainder = mask_count % steps

    num_transfer = mx.repeat(base, steps, axis=1)  # [B, steps]
    step_idx = mx.arange(steps).reshape(1, steps)
    num_transfer = num_transfer + (step_idx < remainder).astype(num_transfer.dtype)
    return num_transfer


def generate(
    model: Model,
    prompt: mx.array,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = DEFAULT_MASK_TOKEN_ID,
    tokenizer=None,
    parallel_threshold: Optional[float] = None,
    return_stats: bool = False,
    kv_cache: bool = False,
    dual_cache: bool = False,
    incremental_cache: bool = False,
    remask_refine: bool = False,
    remask_conf: float = 0.9,
    remask_rounds: int = 2,
):
    """Diffusion (MDM) generation for LLaDA.

    Two decoding schedules are supported:

    * **Fixed schedule** (``parallel_threshold=None``, the default): the
      original LLaDA path — each block runs a fixed ``steps_per_block`` number
      of forwards, revealing a pre-computed top-``k`` per step. This path is
      byte-identical to the historical behaviour; benches depend on it.

    * **Confidence-aware parallel** (``parallel_threshold`` set, Fast-dLLM
      style): each block runs a *dynamic* number of forwards, unmasking every
      eligible position whose clean-softmax confidence clears the threshold in
      one shot. Because it can reveal many tokens per forward, it cuts the
      number of full forwards (the ~98% cost) well below ``steps``. It is not
      bitwise-equal to the fixed schedule; see the bench for the drift.

    Args:
        model: a built ``Model``.
        prompt: token ids, shape ``[1, prompt_len]`` (or ``[prompt_len]``).
        steps: total denoising steps (split evenly across blocks). Only used by
            the fixed-schedule path.
        gen_length: number of tokens to generate.
        block_length: semi-autoregressive block size; ``gen_length`` must be
            divisible by it.
        temperature: Gumbel sampling temperature (0 == greedy/argmax).
        cfg_scale: classifier-free guidance scale (0 == disabled).
        remasking: ``"low_confidence"`` or ``"random"``. (Fixed-schedule path;
            the parallel path always uses clean-softmax confidence.)
        mask_id: the mask token id.
        tokenizer: optional; if given, the decoded text is returned alongside
            the generated ids.
        parallel_threshold: if set (e.g. ``0.9``), use the confidence-aware
            parallel path and unmask every eligible position whose confidence
            exceeds this value each forward. ``None`` keeps the fixed schedule.
        return_stats: if True, also return a stats dict
            ``{"forwards": int, "tokens_per_step_mean": float, "steps": int}``.
        kv_cache: if True, use a Fast-dLLM block-wise **prefix KV cache**. When
            denoising block ``b``, the prefix (prompt + finalized blocks
            ``0..b-1``, positions ``[0, block_start)``) is fixed across all of
            block ``b``'s denoising steps. Attention is bidirectional so the
            prefix K/V technically depend on the still-masked tail, but
            Fast-dLLM shows they are near-identical across steps, so we cache
            them once per block and reuse. Each denoising forward then processes
            ONLY the active window ``[block_start, total_len)`` and attends it
            against ``[cached_prefix_KV ++ active_KV]``. This cuts the forward
            length (the ~98% cost) as the prefix grows. The result is
            *approximate* (not bitwise-equal to ``kv_cache=False``) but stays
            coherent. **Restricted to ``cfg_scale == 0``** (greedy is the main
            use); asserts otherwise. Default False keeps behaviour identical.
        dual_cache: if True (only meaningful when ``kv_cache=True``), ALSO cache
            the **suffix** ``[block_end, total_len)`` K/V (all mask tokens, near-
            stable across the block's denoising steps) so each cached forward
            processes ONLY the current block ``[block_start, block_end)`` and
            attends it against ``[cached_prefix ++ active ++ cached_suffix]``.
            For early blocks the suffix is large (most of the sequence is still
            masked tail), so this captures the win the prefix-only cache misses.
            The suffix masks drift as the block reveals, so the approximation is
            MORE aggressive than prefix-only; it stays coherent on real weights.
            Default False keeps the prefix-only cache behaviour.
        incremental_cache: if True (requires ``kv_cache`` and ``dual_cache``),
            eliminate the per-block full-forward PRIME. Only block 0 is primed
            (a full forward — unavoidable, no prior cache). When a block
            finalizes, its now-revealed tokens' post-RoPE K/V (already computed
            in the block's last cached forward) are **appended** to the prefix
            cache, and the suffix cache is **sliced** to drop the next active
            block — so block ``b>0`` needs no prime at all. This turns ``N``
            full-length primes into 1 prime + append/slice bookkeeping, which is
            the dominant cost of DualCache at long context (gen>=512: 16 primes
            of ~137ms each). The approximation is the same class as DualCache's
            (prefix K/V assumed stable across the boundary; the appended block
            K/V and sliced suffix are one denoising-step staler than a fresh
            prime), and stays coherent on real weights. Composes with the fixed
            and parallel paths; ``remask_refine`` falls back to re-priming inside
            a block (rare path). Default False keeps DualCache's per-block prime.
        remask_refine: opt-in order-aware re-masking for the parallel path only.
            The parallel schedule can commit a low-reveal-confidence token in an
            ambiguous spot (a "reveal-ORDER" artifact — e.g. it reveals a
            competing branch's token before the greedy one). After a block fills,
            up to ``remask_rounds`` refinement rounds re-mask the block positions
            whose reveal-time confidence was below ``remask_conf`` (plus their
            immediate neighbours, since collisions span adjacent tokens) and
            re-decode them one-token-per-step in most-confident-first order,
            conditioned on the now-fixed high-confidence tokens. This re-commits
            the ambiguous span with more context locked, which can fix the order
            artifact. Only active when ``parallel_threshold`` is set AND this is
            True. Default False preserves the parallel path exactly.
        remask_conf: reveal-time confidence bar below which a token is re-masked
            in a refinement round (only used when ``remask_refine=True``).
        remask_rounds: max number of refinement rounds per block (only used when
            ``remask_refine=True``).

    Returns:
        Generated ids ``[1, gen_length]``; a trailing decoded ``text`` if a
        tokenizer was supplied; and a trailing stats dict if ``return_stats``.
        The return shape when ``return_stats=False`` is unchanged.
    """
    if prompt.ndim == 1:
        prompt = prompt[None, :]
    prompt_len = prompt.shape[1]

    total_len = prompt_len + gen_length
    x = mx.full((1, total_len), mask_id, dtype=prompt.dtype)
    x[:, :prompt_len] = prompt
    prompt_index = x != mask_id

    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    num_blocks = gen_length // block_length

    if kv_cache and cfg_scale > 0.0:
        raise ValueError(
            "kv_cache=True is only supported with cfg_scale==0.0 "
            "(the doubled-batch CFG path is not cached). Disable one of them."
        )
    if dual_cache and not kv_cache:
        raise ValueError("dual_cache=True requires kv_cache=True.")
    if incremental_cache and not (kv_cache and dual_cache):
        raise ValueError(
            "incremental_cache=True requires kv_cache=True and dual_cache=True."
        )

    neg_inf = mx.array(-float("inf"), dtype=mx.float32)
    col_index = mx.arange(total_len).reshape(1, total_len)

    # Count of full forwards actually run (the win metric — ~98% of step cost).
    forwards = 0

    def _forward_logits(x_cur: mx.array) -> mx.array:
        """One full forward (with optional CFG). Increments the forward count."""
        nonlocal forwards
        forwards += 1
        if cfg_scale > 0.0:
            un_x = mx.where(prompt_index, mask_id, x_cur)
            x_ = mx.concatenate([x_cur, un_x], axis=0)
            logits = model(x_)
            cond_logits, uncond_logits = logits[:1], logits[1:]
            return uncond_logits + (cfg_scale + 1.0) * (cond_logits - uncond_logits)
        return model(x_cur)

    # ------------------------------------------------------------------
    # Fast-dLLM block-KV cache (only used when kv_cache=True).
    #
    # ``_prefix_cache`` is a per-layer list of cached prefix ``(K, V)`` for
    # absolute positions ``[0, _prefix_len)``. With ``dual_cache=True``,
    # ``_suffix_cache`` is a per-layer list of cached suffix ``(K, V)`` for
    # absolute positions ``[_suffix_start, total_len)`` (the still-masked tail).
    # ``_last_full_logits`` holds the full-length logits from the most recent
    # prime so the sampler always sees a ``[1, total_len, vocab]`` tensor
    # (prefix/suffix logits are never consumed by the sampler).
    # ------------------------------------------------------------------
    _prefix_cache = None
    _prefix_len = 0
    _suffix_cache = None
    _suffix_start = total_len
    _last_full_logits = None
    # Incremental cache: the per-layer post-RoPE (K, V) of the CURRENT active
    # window from the most recent cached forward. When a block finalizes these
    # are the finalized block's K/V, appended to the prefix cache in place of a
    # re-prime for the next block. ``_active_kv_span`` records the absolute
    # ``(start, end)`` those K/V cover, so we only append when they exactly match
    # the just-finalized block (else we fall back to a prime).
    _active_kv = None
    _active_kv_span = (0, 0)

    def _prime_cache(x_cur: mx.array, block_start: int, block_end: int):
        """Full forward over the whole ``x``; cache prefix [0, block_start) and,
        with ``dual_cache``, suffix [block_end, total_len) K/V.

        Returns the full-length logits (also stashed for later scatter).
        """
        nonlocal _prefix_cache, _prefix_len, _suffix_cache, _suffix_start
        nonlocal _last_full_logits, forwards
        forwards += 1
        logits, kvs = model(x_cur, return_kv=True)  # kvs: per-layer (K, V) over all L
        # Slice out the fixed prefix window [0, block_start) along the seq axis.
        _prefix_cache = [
            (k[:, :, :block_start, :], v[:, :, :block_start, :]) for (k, v) in kvs
        ]
        _prefix_len = block_start
        if dual_cache:
            # Slice out the suffix window [block_end, total_len) — the masked
            # tail, which is near-stable across the block's denoising steps.
            _suffix_cache = [
                (k[:, :, block_end:, :], v[:, :, block_end:, :]) for (k, v) in kvs
            ]
            _suffix_start = block_end
        else:
            _suffix_cache = None
            _suffix_start = total_len
        _last_full_logits = logits
        return logits

    def _forward_logits_cached(x_cur: mx.array) -> mx.array:
        """Cached forward: process only the active window, scatter to full length.

        Uses the primed ``_prefix_cache`` (positions ``[0, _prefix_len)``) and,
        with ``dual_cache``, ``_suffix_cache`` (positions
        ``[_suffix_start, total_len)``). The active hidden states
        ``x_cur[:, _prefix_len:_suffix_start]`` are RoPE'd at absolute offset
        ``_prefix_len`` and attended against ``[prefix ++ active ++ suffix]``
        K/V. The returned logits are ``[1, total_len, vocab]``; the
        prefix/suffix rows are filled from the last prime (never consumed by the
        sampler).
        """
        nonlocal forwards, _last_full_logits, _active_kv, _active_kv_span
        forwards += 1
        active = x_cur[:, _prefix_len:_suffix_start]
        if incremental_cache:
            # Also return the active window's per-layer post-RoPE K/V so the
            # finalized block can be appended to the prefix cache next block.
            active_logits, _active_kv = model(
                active,
                prefix_kv=_prefix_cache,
                pos_offset=_prefix_len,
                suffix_kv=_suffix_cache,
                return_kv=True,
            )
            _active_kv_span = (_prefix_len, _suffix_start)
        else:
            active_logits = model(
                active,
                prefix_kv=_prefix_cache,
                pos_offset=_prefix_len,
                suffix_kv=_suffix_cache,
            )
        # Scatter active logits back into a full-length buffer. Prefix rows come
        # from the last prime; active rows are fresh; suffix rows (if any) come
        # from the last prime too.
        parts = [_last_full_logits[:, :_prefix_len, :], active_logits]
        if _suffix_cache is not None:
            parts.append(_last_full_logits[:, _suffix_start:, :])
        full = mx.concatenate(parts, axis=1)
        _last_full_logits = full
        return full

    def _extend_cache(block_start: int, block_end: int):
        """Incremental cache advance — NO full forward (the prime is eliminated).

        Called at the start of block ``b>0`` in place of ``_prime_cache``. The
        just-finalized block ``b-1`` occupied the active window
        ``[_prefix_len, _suffix_start)`` == ``[block_start-block_len, block_start)``;
        its post-RoPE K/V were captured in ``_active_kv`` by the block's last
        cached forward. Append them to the prefix cache so the prefix now covers
        ``[0, block_start)``, and slice the suffix cache to drop the new active
        block, leaving ``[block_end, total_len)``. Attention is permutation-
        invariant over keys and RoPE is baked into every cached K, so appending
        the finalized block's K/V is order-correct.
        """
        nonlocal _prefix_cache, _prefix_len, _suffix_cache, _suffix_start
        block_len = block_end - block_start
        # Append finalized block K/V to the prefix (per layer, along seq axis).
        _prefix_cache = [
            (
                mx.concatenate([pk, ak], axis=2),
                mx.concatenate([pv, av], axis=2),
            )
            for (pk, pv), (ak, av) in zip(_prefix_cache, _active_kv)
        ]
        _prefix_len = block_start
        # Slice the suffix: it covered [old block_end, total_len) == [block_start,
        # total_len); drop the first block_len positions (the new active block).
        _suffix_cache = [
            (k[:, :, block_len:, :], v[:, :, block_len:, :])
            for (k, v) in _suffix_cache
        ]
        _suffix_start = block_end

    if parallel_threshold is None:
        # -------------------------------------------------------------------
        # Fixed-schedule path (unchanged historical behaviour).
        # -------------------------------------------------------------------
        assert steps % num_blocks == 0, "steps must be divisible by num_blocks"
        steps_per_block = steps // num_blocks

        for b in range(num_blocks):
            block_start = prompt_len + b * block_length
            block_end = prompt_len + (b + 1) * block_length

            # Prime the block cache once per block (prefix [0, block_start);
            # with dual_cache also suffix [block_end, total_len)). The prime is
            # itself a full forward, so its logits serve step 0. With
            # incremental_cache, only block 0 is primed; later blocks advance the
            # cache by append/slice (no full forward), so step 0 has no free
            # prime logits and runs a real cached forward.
            primed_logits = None
            if kv_cache:
                # Extend only when the captured active K/V exactly cover the
                # just-finalized block [block_start-block_length, block_start);
                # otherwise (block 0, or a block that filled in its prime with no
                # cached forward) fall back to a full prime.
                can_extend = (
                    incremental_cache
                    and b > 0
                    and _active_kv is not None
                    and _active_kv_span == (block_start - block_length, block_start)
                )
                if can_extend:
                    _extend_cache(block_start, block_end)
                else:
                    primed_logits = _prime_cache(x, block_start, block_end)

            # Mask positions still masked within the current block.
            block_mask_index = x[:, block_start:block_end] == mask_id
            num_transfer_tokens = get_num_transfer_tokens(
                block_mask_index, steps_per_block
            )

            for i in range(steps_per_block):
                mask_index = x == mask_id
                if kv_cache:
                    logits = (
                        primed_logits
                        if (i == 0 and primed_logits is not None)
                        else _forward_logits_cached(x)
                    )
                else:
                    logits = _forward_logits(x)

                noised = add_gumbel_noise(logits, temperature)
                x0 = mx.argmax(noised, axis=-1)  # [1, total_len]

                if remasking == "low_confidence":
                    p = mx.softmax(logits.astype(mx.float32), axis=-1)
                    x0_p = mx.take_along_axis(p, x0[..., None], axis=-1).squeeze(-1)
                elif remasking == "random":
                    x0_p = mx.random.uniform(shape=x0.shape)
                else:
                    raise ValueError(f"Unknown remasking strategy: {remasking}")

                # Never reveal past the current block boundary.
                beyond_block = col_index >= block_end
                x0_p = mx.where(beyond_block, neg_inf, x0_p.astype(mx.float32))

                # Only masked positions are candidates; keep existing tokens.
                x0 = mx.where(mask_index, x0, x)
                confidence = mx.where(mask_index, x0_p, neg_inf)

                # Reveal the top-k highest-confidence positions per row.
                transfer_index = _select_topk(
                    confidence, num_transfer_tokens[:, i : i + 1]
                )
                x = mx.where(transfer_index, x0, x)
                mx.eval(x)
    else:
        # -------------------------------------------------------------------
        # Confidence-aware parallel path (Fast-dLLM style).
        #
        # Per block, loop forwards until the block has no masked positions
        # left. Each forward unmasks EVERY eligible position whose clean-
        # softmax confidence clears ``parallel_threshold`` at once. The loop is
        # capped at ``block_length`` iterations (worst case = reveal one token
        # per step), which guarantees termination.
        # -------------------------------------------------------------------
        # Per-position reveal-time confidence, [1, total_len]. Prompt / never-
        # masked positions stay at +inf; each parallel reveal records the clean-
        # softmax confidence the token had at the step it was committed. Only
        # consulted by the opt-in ``remask_refine`` refinement below.
        pos_inf = mx.array(float("inf"), dtype=mx.float32)
        reveal_conf = mx.full((1, total_len), pos_inf, dtype=mx.float32)

        def _parallel_fill(eligible_mask, max_steps):
            """Confidence-aware parallel fill over the given eligibility mask.

            Runs full forwards until every position in ``eligible_mask`` that is
            currently masked has been revealed (or ``max_steps`` is hit). Each
            forward unmasks every still-masked eligible position whose clean-
            softmax confidence clears ``parallel_threshold`` (progress-guarantee
            reveals the single most confident one otherwise). Records each
            revealed position's reveal-time confidence into ``reveal_conf`` and
            mutates the enclosing ``x``. Returns nothing.
            """
            nonlocal x, reveal_conf
            step = 0
            while step < max_steps:
                mask_index = x == mask_id
                eligible = mask_index & eligible_mask
                # Stop once every eligible position is filled.
                if int(eligible.sum()) == 0:
                    break

                if kv_cache:
                    logits = (
                        primed_logits
                        if (step == 0 and primed_logits is not None)
                        else _forward_logits_cached(x)
                    )
                else:
                    logits = _forward_logits(x)

                # Token choice honours temperature via Gumbel noise ...
                noised = add_gumbel_noise(logits, temperature)
                x0 = mx.argmax(noised, axis=-1)  # [1, total_len]

                # ... but confidence comes from the CLEAN softmax, not the
                # gumbel-noised logits.
                p = mx.softmax(logits.astype(mx.float32), axis=-1)
                conf = mx.take_along_axis(p, x0[..., None], axis=-1).squeeze(-1)
                conf = mx.where(eligible, conf.astype(mx.float32), neg_inf)

                # Unmask every eligible position over the confidence bar.
                reveal = eligible & (conf > parallel_threshold)

                # Progress guarantee: if nothing clears the bar, reveal the
                # single most-confident eligible position so the loop advances.
                if int(reveal.sum()) == 0:
                    top = mx.argmax(conf, axis=-1)  # [1]
                    reveal = col_index == top[:, None]

                # Record reveal-time confidence for freshly revealed positions.
                reveal_conf = mx.where(reveal, conf, reveal_conf)
                x = mx.where(reveal, x0, x)
                mx.eval(x, reveal_conf)
                step += 1

        def _parallel_fill_sequential(eligible_mask):
            """Re-decode: reveal exactly ONE token per forward, most-confident
            first, over ``eligible_mask``. This maximises the fixed context each
            re-committed token sees (order-aware greedy re-reveal), which is what
            corrects a reveal-order artifact. Records reveal-time confidence and
            mutates the enclosing ``x``.
            """
            nonlocal x, reveal_conf
            n_masked = int(((x == mask_id) & eligible_mask).sum())
            for _s in range(n_masked):
                mask_index = x == mask_id
                eligible = mask_index & eligible_mask
                if int(eligible.sum()) == 0:
                    break
                if kv_cache:
                    logits = _forward_logits_cached(x)
                else:
                    logits = _forward_logits(x)
                noised = add_gumbel_noise(logits, temperature)
                x0 = mx.argmax(noised, axis=-1)
                p = mx.softmax(logits.astype(mx.float32), axis=-1)
                conf = mx.take_along_axis(p, x0[..., None], axis=-1).squeeze(-1)
                conf = mx.where(eligible, conf.astype(mx.float32), neg_inf)
                # Single most-confident eligible position.
                top = mx.argmax(conf, axis=-1)
                reveal = col_index == top[:, None]
                reveal_conf = mx.where(reveal, conf, reveal_conf)
                x = mx.where(reveal, x0, x)
                mx.eval(x, reveal_conf)

        for b in range(num_blocks):
            block_start = prompt_len + b * block_length
            block_end = prompt_len + (b + 1) * block_length

            # Prime the block cache once per block (prefix [0, block_start);
            # with dual_cache also suffix [block_end, total_len)). The prime is a
            # full forward; reuse its logits for the block's first denoising step
            # so priming costs nothing extra. With incremental_cache, block b>0
            # advances the cache by append/slice instead of priming (no full
            # forward), so there is no free step-0 logit — step 0 runs a cached
            # forward. Extend only when the captured active K/V exactly cover the
            # just-finalized block; otherwise fall back to a prime.
            primed_logits = None
            if kv_cache:
                can_extend = (
                    incremental_cache
                    and b > 0
                    and _active_kv is not None
                    and _active_kv_span == (block_start - block_length, block_start)
                )
                if can_extend:
                    _extend_cache(block_start, block_end)
                else:
                    primed_logits = _prime_cache(x, block_start, block_end)

            # Eligible = masked AND inside the current block window.
            in_block = (col_index >= block_start) & (col_index < block_end)

            _parallel_fill(in_block, block_length)

            # --------------------------------------------------------------
            # Order-aware re-masking (opt-in). The parallel schedule can commit
            # a low-reveal-confidence token in an ambiguous spot before the
            # greedy branch's token is revealed. Re-mask the block's low-reveal-
            # confidence positions (+/- 1 neighbour, since collisions span
            # adjacent tokens) and re-decode them one-token-per-step in most-
            # confident-first order, now conditioned on the fixed high-
            # confidence tokens. Re-masking (not flipping) is what removes the
            # locked-in wrong token so the model re-predicts from more context.
            # --------------------------------------------------------------
            if remask_refine:
                for _r in range(remask_rounds):
                    # Uncertain block positions: revealed under low confidence.
                    low = in_block & (reveal_conf < remask_conf)
                    if int(low.sum()) == 0:
                        break
                    # Widen to the immediate neighbours (contiguous span).
                    low_left = mx.roll(low, 1, axis=1)
                    low_right = mx.roll(low, -1, axis=1)
                    span = (low | low_left | low_right) & in_block
                    if int(span.sum()) == 0:
                        break
                    # Re-mask the span and reset its reveal-time confidence.
                    x = mx.where(span, mask_id, x)
                    reveal_conf = mx.where(span, pos_inf, reveal_conf)
                    mx.eval(x, reveal_conf)
                    # Re-prime the cache if used (block prefix unchanged, but the
                    # active window now has fresh masks) and re-decode strictly
                    # one token per step so each re-commit sees the maximum fixed
                    # context (order-aware greedy re-reveal).
                    if kv_cache:
                        primed_logits = _prime_cache(x, block_start, block_end)
                    _parallel_fill_sequential(span)

    out = x[:, prompt_len:]
    mx.eval(out)

    ret = (out,)
    if tokenizer is not None:
        ret = ret + (tokenizer.decode(out[0].tolist()),)
    if return_stats:
        revealed = gen_length  # every gen position ends unmasked
        stats = {
            "forwards": forwards,
            "tokens_per_step_mean": (revealed / forwards) if forwards else 0.0,
            "steps": forwards,
        }
        ret = ret + (stats,)
    return ret[0] if len(ret) == 1 else ret


def _select_topk(confidence: mx.array, k_per_row: mx.array) -> mx.array:
    """Boolean mask selecting the top-``k`` positions per row of ``confidence``.

    ``confidence`` is [B, L] (with ``-inf`` for ineligible positions); ``k_per_row``
    is [B, 1]. Uses an exact rank via ``argsort`` so ties never cause more than
    ``k`` reveals in a row.
    """
    B, L = confidence.shape
    # Descending order: rank 0 = highest confidence.
    order = mx.argsort(-confidence, axis=1)  # [B, L] column indices, best first
    ranks = mx.zeros((B, L), dtype=mx.int32)
    col_positions = mx.arange(L).reshape(1, L)
    # Scatter ranks: ranks[b, order[b, r]] = r.
    ranks = mx.put_along_axis(
        ranks,
        order,
        mx.broadcast_to(col_positions.astype(mx.int32), (B, L)),
        axis=1,
    )
    return ranks < k_per_row.astype(mx.int32)
