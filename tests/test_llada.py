"""Offline smoke test for the LLaDA-8B MLX port (no real weights)."""

import mlx.core as mx
import numpy as np

from mlx_lm.models.llada import Model, ModelArgs, generate


def tiny_args():
    return ModelArgs(
        model_type="llada",
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        mlp_hidden_size=128,
        vocab_size=512,
        embedding_size=512,
        rms_norm_eps=1e-5,
        rope_theta=500000.0,
        weight_tying=False,
    )


def test_forward():
    mx.random.seed(0)
    model = Model(tiny_args())
    ids = mx.array(np.random.randint(0, 512, size=(2, 16)))
    logits = model(ids)
    assert logits.shape == (2, 16, 512), logits.shape
    assert bool(mx.all(mx.isfinite(logits))), "logits not finite"
    print("[ok] forward: logits", logits.shape, "finite")


def test_generate():
    mx.random.seed(0)
    model = Model(tiny_args())
    mask_id = 126336
    prompt = mx.array(np.random.randint(0, 500, size=(1, 6)))

    # temperature=0 (greedy)
    out = generate(
        model, prompt, steps=8, gen_length=16, block_length=8,
        temperature=0.0, cfg_scale=0.0, mask_id=mask_id,
    )
    assert out.shape == (1, 16), out.shape
    assert int(mx.sum(out == mask_id)) == 0, "mask tokens remain (temp=0)"
    print("[ok] generate temp=0:", out.shape, "no masks")

    # temperature=0.3, cfg_scale=1.5
    out2 = generate(
        model, prompt, steps=8, gen_length=16, block_length=8,
        temperature=0.3, cfg_scale=1.5, mask_id=mask_id,
    )
    assert out2.shape == (1, 16), out2.shape
    assert int(mx.sum(out2 == mask_id)) == 0, "mask tokens remain (temp=0.3,cfg)"
    print("[ok] generate temp=0.3 cfg=1.5:", out2.shape, "no masks")

    # confidence-aware parallel path (opt-in): output fully unmasked and the
    # dynamic loop must not run more forwards than the fixed schedule (steps).
    out3, stats = generate(
        model, prompt, steps=8, gen_length=16, block_length=8,
        temperature=0.0, cfg_scale=0.0, mask_id=mask_id,
        parallel_threshold=0.9, return_stats=True,
    )
    assert out3.shape == (1, 16), out3.shape
    assert int(mx.sum(out3 == mask_id)) == 0, "mask tokens remain (parallel)"
    # The dynamic loop is capped at block_length forwards per block, so it can
    # never run more forwards than gen_length (its 1-token-per-step worst case).
    # On random tiny weights confidence rarely clears 0.9, so this degenerates
    # to that worst case; on real weights it runs far fewer (see the bench).
    assert stats["forwards"] <= 16, f"forwards {stats['forwards']} > cap 16"
    print("[ok] generate parallel_threshold=0.9:", out3.shape,
          "no masks; forwards", stats["forwards"], "<= cap 16")

    # Order-aware re-masking refinement (opt-in, parallel path only). Must still
    # terminate with no masks left and stay in-vocab; defaults elsewhere are
    # untouched. On tiny random weights confidence rarely clears 0.9 so the
    # refinement re-masks and re-decodes a real span — exercises both paths.
    out_rm, stats_rm = generate(
        model, prompt, steps=8, gen_length=16, block_length=8,
        temperature=0.0, cfg_scale=0.0, mask_id=mask_id,
        parallel_threshold=0.9, remask_refine=True, remask_conf=0.9,
        remask_rounds=2, return_stats=True,
    )
    assert out_rm.shape == (1, 16), out_rm.shape
    assert int(mx.sum(out_rm == mask_id)) == 0, "mask tokens remain (remask_refine)"
    assert int(mx.min(out_rm)) >= 0 and int(mx.max(out_rm)) < 512, "OOB id"
    print("[ok] generate remask_refine=True:", out_rm.shape,
          "no masks; forwards", stats_rm["forwards"])

    # Fast-dLLM prefix KV cache (opt-in). The single cached forward is
    # bitwise-exact vs a full forward when primed on the same x (verified
    # separately); across a block the prefix goes slightly stale (bidirectional
    # attention → prefix K/V drift as the tail is revealed), so on tiny RANDOM
    # weights (near-uniform attention, maximal drift) we assert coherence — same
    # shape, no mask leakage, no NaN/garbage — not id equality. On real trained
    # weights the drift is negligible (see bench_llada_kvcache.py).
    for path_kw in (dict(parallel_threshold=None), dict(parallel_threshold=0.9)):
        out_c = generate(
            model, prompt, steps=8, gen_length=16, block_length=8,
            temperature=0.0, cfg_scale=0.0, mask_id=mask_id,
            kv_cache=True, **path_kw,
        )
        assert out_c.shape == (1, 16), out_c.shape
        assert int(mx.sum(out_c == mask_id)) == 0, "mask tokens remain (kv_cache)"
        # Every id must be a valid vocab token (no garbage / OOB).
        assert int(mx.min(out_c)) >= 0 and int(mx.max(out_c)) < 512, "OOB id"
    print("[ok] generate kv_cache=True (fixed + parallel): (1, 16) no masks")

    # DualCache (suffix cache; opt-in, requires kv_cache=True). Same coherence
    # gate as the prefix-only cache: shape, no mask leakage, in-vocab. The suffix
    # approximation is more aggressive (masked tail drifts as the block reveals),
    # so on tiny RANDOM weights we assert coherence, not id equality. Exercises
    # both fixed and parallel paths.
    for path_kw in (dict(parallel_threshold=None), dict(parallel_threshold=0.9)):
        out_d = generate(
            model, prompt, steps=8, gen_length=16, block_length=8,
            temperature=0.0, cfg_scale=0.0, mask_id=mask_id,
            kv_cache=True, dual_cache=True, **path_kw,
        )
        assert out_d.shape == (1, 16), out_d.shape
        assert int(mx.sum(out_d == mask_id)) == 0, "mask tokens remain (dual_cache)"
        assert int(mx.min(out_d)) >= 0 and int(mx.max(out_d)) < 512, "OOB id"
    print("[ok] generate dual_cache=True (fixed + parallel): (1, 16) no masks")

    # Incremental cache (opt-in, requires kv_cache + dual_cache). Eliminates the
    # per-block prime by appending each finalized block's K/V to the prefix cache
    # and slicing the suffix. Same coherence gate: shape, no mask leakage,
    # in-vocab. Exercises both fixed and parallel paths. gen=16/block=8 → 2 blocks
    # so block 1 exercises the append/slice path (not just block 0's prime).
    for path_kw in (dict(parallel_threshold=None), dict(parallel_threshold=0.9)):
        out_i = generate(
            model, prompt, steps=8, gen_length=16, block_length=8,
            temperature=0.0, cfg_scale=0.0, mask_id=mask_id,
            kv_cache=True, dual_cache=True, incremental_cache=True, **path_kw,
        )
        assert out_i.shape == (1, 16), out_i.shape
        assert int(mx.sum(out_i == mask_id)) == 0, "mask tokens remain (incremental)"
        assert int(mx.min(out_i)) >= 0 and int(mx.max(out_i)) < 512, "OOB id"
    print("[ok] generate incremental_cache=True (fixed + parallel): (1, 16) no masks")

    # incremental_cache requires kv_cache AND dual_cache; must raise otherwise.
    for bad_kw in (dict(kv_cache=False, dual_cache=False),
                   dict(kv_cache=True, dual_cache=False)):
        raised_i = False
        try:
            generate(model, prompt, steps=8, gen_length=16, block_length=8,
                     temperature=0.0, cfg_scale=0.0, mask_id=mask_id,
                     incremental_cache=True, **bad_kw)
        except ValueError:
            raised_i = True
        assert raised_i, f"incremental_cache without dual_cache should raise ({bad_kw})"
    print("[ok] incremental_cache asserts kv_cache==True and dual_cache==True")

    # DualCache composed with remask_refine (parallel path, sequential re-decode
    # also uses the cached forward). Must terminate cleanly.
    out_dr = generate(
        model, prompt, steps=8, gen_length=16, block_length=8,
        temperature=0.0, cfg_scale=0.0, mask_id=mask_id,
        kv_cache=True, dual_cache=True, parallel_threshold=0.9,
        remask_refine=True, remask_conf=0.9, remask_rounds=2,
    )
    assert out_dr.shape == (1, 16), out_dr.shape
    assert int(mx.sum(out_dr == mask_id)) == 0, "mask tokens remain (dual+remask)"
    assert int(mx.min(out_dr)) >= 0 and int(mx.max(out_dr)) < 512, "OOB id"
    print("[ok] generate dual_cache=True + remask_refine: (1, 16) no masks")

    # dual_cache requires kv_cache=True; must raise otherwise.
    raised_d = False
    try:
        generate(model, prompt, steps=8, gen_length=16, block_length=8,
                 temperature=0.0, cfg_scale=0.0, mask_id=mask_id,
                 kv_cache=False, dual_cache=True)
    except ValueError:
        raised_d = True
    assert raised_d, "dual_cache without kv_cache should raise"
    print("[ok] dual_cache asserts kv_cache==True")

    # kv_cache is restricted to cfg_scale==0; must raise otherwise.
    raised = False
    try:
        generate(model, prompt, steps=8, gen_length=16, block_length=8,
                 temperature=0.0, cfg_scale=1.5, mask_id=mask_id, kv_cache=True)
    except ValueError:
        raised = True
    assert raised, "kv_cache with cfg_scale>0 should raise"
    print("[ok] kv_cache asserts cfg_scale==0")


def make_synthetic_hf_weights(a: ModelArgs):
    """Build a dict with the REAL HF key names at tiny-config shapes."""
    hd = a.d_model // a.n_heads
    w = {}
    r = lambda *s: mx.array(np.random.randn(*s).astype(np.float32))

    w["model.transformer.wte.weight"] = r(a.vocab_size, a.d_model)
    w["model.transformer.ln_f.weight"] = r(a.d_model)
    # top-level LM head (NO .blocks. in path)
    w["model.transformer.ff_out.weight"] = r(a.vocab_size, a.d_model)

    for n in range(a.n_layers):
        p = f"model.transformer.blocks.{n}."
        w[p + "q_proj.weight"] = r(a.n_heads * hd, a.d_model)
        w[p + "k_proj.weight"] = r(a.n_kv_heads * hd, a.d_model)
        w[p + "v_proj.weight"] = r(a.n_kv_heads * hd, a.d_model)
        w[p + "attn_out.weight"] = r(a.d_model, a.n_heads * hd)
        w[p + "ff_proj.weight"] = r(a.mlp_hidden_size, a.d_model)
        w[p + "up_proj.weight"] = r(a.mlp_hidden_size, a.d_model)
        # blocks.N.ff_out == MLP down-proj (nested under .blocks.)
        w[p + "ff_out.weight"] = r(a.d_model, a.mlp_hidden_size)
        w[p + "attn_norm.weight"] = r(a.d_model)
        w[p + "ff_norm.weight"] = r(a.d_model)
    return w


def test_sanitize_and_load():
    a = tiny_args()
    model = Model(a)
    hf = make_synthetic_hf_weights(a)
    sanitized = model.sanitize(hf)

    # Same count in/out
    assert len(sanitized) == len(hf), (len(sanitized), len(hf))
    # No leftover raw HF markers
    leftover = [k for k in sanitized if "transformer." in k or ".blocks." in k]
    assert not leftover, f"leftover HF keys: {leftover[:5]}"
    # Both ff_out variants resolved distinctly
    assert "lm_head.weight" in sanitized, "top-level ff_out -> lm_head missing"
    assert "model.layers.0.mlp.down_proj.weight" in sanitized, "blocks ff_out -> down_proj missing"
    assert "model.embed_tokens.weight" in sanitized
    assert "model.norm.weight" in sanitized

    # Strict load must succeed
    model.load_weights(list(sanitized.items()), strict=True)
    print("[ok] sanitize: remapped", len(sanitized), "keys; strict load succeeded")


if __name__ == "__main__":
    test_forward()
    test_generate()
    test_sanitize_and_load()
    print("\nALL TESTS PASSED")
