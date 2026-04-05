"""End-to-end RPS benchmark: FP16 → 3-bit base + residuals → generate.

Compares:
1. Original FP16 quality (reference)
2. Standard 4-bit quantization
3. Standard 3-bit quantization
4. RPS: 3-bit base + 2-bit residuals on all projections
5. RPS Tier 2: 3-bit base + 2-bit residuals on v_proj + down_proj only
"""

import sys
import time

import mlx.core as mx
import mlx.nn as nn


def benchmark_rps(model_path: str, prompt: str = "Explain quantum computing in simple terms.", max_tokens: int = 100):
    from mlx_lm import load, generate
    from mlx_lm.rps.linear import RPSLinear
    from mlx_lm.rps.convert import compute_residuals

    results = {}

    # --- FP16 baseline ---
    print("=== FP16 (reference) ===")
    model, tok = load(model_path)
    generate(model, tok, prompt="Hi", max_tokens=3, verbose=False)
    t0 = time.perf_counter()
    out = generate(model, tok, prompt=prompt, max_tokens=max_tokens, verbose=False)
    t1 = time.perf_counter()
    toks = len(tok.encode(out))
    speed = toks / (t1 - t0)
    mem = mx.get_active_memory() / 1024**3
    print(f"  {speed:.1f} tok/s, {mem:.1f} GB")
    print(f"  {out[:200]}")
    results["fp16"] = (speed, out)

    # Collect FP16 weights for all projections
    fp16_weights = {}
    for li, layer in enumerate(model.layers):
        attn = layer.self_attn
        mlp = layer.mlp
        for an, parent in [("q_proj", attn), ("k_proj", attn), ("v_proj", attn),
                           ("o_proj", attn), ("gate_proj", mlp), ("up_proj", mlp),
                           ("down_proj", mlp)]:
            proj = getattr(parent, an, None)
            if proj is None or not hasattr(proj, "weight"):
                continue
            w = proj.weight
            if isinstance(proj, nn.QuantizedLinear):
                w = mx.dequantize(w, proj.scales, proj.biases,
                                  group_size=proj.group_size, bits=proj.bits)
            mx.eval(w)
            fp16_weights[(li, an)] = w.astype(mx.float16)

    def _test_config(label, base_bits, tier2_keys=None):
        """Apply RPS with given config and benchmark."""
        nonlocal model, tok
        del model
        mx.clear_cache()
        model, tok = load(model_path)

        for li, layer in enumerate(model.layers):
            attn = layer.self_attn
            mlp = layer.mlp
            for an, parent in [("q_proj", attn), ("k_proj", attn), ("v_proj", attn),
                               ("o_proj", attn), ("gate_proj", mlp), ("up_proj", mlp),
                               ("down_proj", mlp)]:
                proj = getattr(parent, an, None)
                if proj is None:
                    continue
                w = fp16_weights.get((li, an))
                if w is None:
                    continue

                if tier2_keys is None:
                    # Standard quantization (no RPS)
                    bq, bs, bb = mx.quantize(w, group_size=64, bits=base_bits)
                    mx.eval(bq, bs, bb)
                    if isinstance(proj, nn.QuantizedLinear):
                        proj.weight = bq
                        proj.scales = bs
                        proj.biases = bb
                        proj.bits = base_bits
                        proj.group_size = 64
                    else:
                        new_proj = nn.QuantizedLinear(
                            w.shape[1], w.shape[0], group_size=64, bits=base_bits
                        )
                        new_proj.weight = bq
                        new_proj.scales = bs
                        new_proj.biases = bb
                        setattr(parent, an, new_proj)
                else:
                    # RPS
                    (bq, bs, bb, rq, rs, rb, _, _) = compute_residuals(
                        w, base_bits=base_bits, residual_bits=2
                    )
                    mx.eval(bq, bs, bb, rq, rs, rb)
                    rps = RPSLinear(proj) if isinstance(proj, nn.QuantizedLinear) else RPSLinear(
                        nn.QuantizedLinear(w.shape[1], w.shape[0], group_size=64, bits=base_bits)
                    )
                    rps.weight = bq
                    rps.scales = bs
                    rps.biases = bb
                    rps.bits = base_bits
                    rps.group_size = 64
                    if tier2_keys == "all" or an in tier2_keys:
                        rps.attach_residual(rq, rs, rb, r_bits=2, r_group_size=64)
                    setattr(parent, an, rps)
        mx.eval()

        print(f"\n=== {label} ===")
        generate(model, tok, prompt="Hi", max_tokens=3, verbose=False)
        t0 = time.perf_counter()
        out = generate(model, tok, prompt=prompt, max_tokens=max_tokens, verbose=False)
        t1 = time.perf_counter()
        toks = len(tok.encode(out))
        speed = toks / (t1 - t0)
        mem = mx.get_active_memory() / 1024**3
        print(f"  {speed:.1f} tok/s, {mem:.1f} GB")
        print(f"  {out[:200]}")
        results[label] = (speed, out)

    # --- Standard quantization baselines ---
    _test_config("4-bit standard", base_bits=4)
    _test_config("3-bit standard", base_bits=3)

    # --- RPS configurations ---
    _test_config("RPS 3+2 (all projections)", base_bits=3, tier2_keys="all")
    _test_config("RPS 3+2 (v_proj+down_proj)", base_bits=3, tier2_keys={"v_proj", "down_proj"})
    _test_config("RPS 3+2 (attention only)", base_bits=3, tier2_keys={"q_proj", "k_proj", "v_proj", "o_proj"})

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for label, (speed, out) in results.items():
        quality = "OK" if len(out.strip()) > 20 and out.strip()[-1] != out.strip()[-10] else "DEGRADED"
        print(f"  {label:35s}  {speed:6.1f} tok/s  {quality}")

    return results


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/qwen2.5-3b-fp16"
    benchmark_rps(model_path)
