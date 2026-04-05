"""
Residual Precision Streaming — Full Validation Suite

No claims until the numbers speak. Tests:
1. Perplexity across multiple long passages (objective)
2. Generation quality across 15 diverse prompts
3. All quantization configs: FP16, 4-bit, 3-bit, 2-bit, RPS variants
4. Speed benchmarks (averaged over 3 runs)
5. Memory measurements
6. Selective residual testing (which projections matter)
7. Long generation (500 tokens) stability
8. MSE decomposition analysis
"""

import time
import json
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
from mlx_lm.utils import quantize_model
from mlx_lm.rps.linear import RPSLinear

MODEL = "/tmp/qwen2.5-3b-fp16"

EVAL_TEXTS = [
    (
        "The theory of general relativity, published by Albert Einstein in 1915, "
        "describes gravity not as a force but as a curvature of spacetime caused by mass "
        "and energy. This revolutionary framework replaced Newton's law of universal "
        "gravitation for extreme conditions. The field equations relate the geometry of "
        "spacetime to the distribution of matter within it. General relativity predicts "
        "phenomena such as gravitational time dilation, gravitational lensing, "
        "gravitational waves, and the expansion of the universe. The theory has been "
        "confirmed by numerous experiments and observations, from the perihelion precession "
        "of Mercury to the detection of gravitational waves by LIGO in 2015."
    ),
    (
        "Machine learning is a subset of artificial intelligence that focuses on building "
        "systems that learn from and make decisions based on data. Unlike traditional "
        "programming where rules are explicitly coded, machine learning algorithms identify "
        "patterns in training data and use them to make predictions on new data. The three "
        "main paradigms are supervised learning, unsupervised learning, and reinforcement "
        "learning. Deep learning, which uses neural networks with many layers, has driven "
        "recent breakthroughs in computer vision, natural language processing, and game "
        "playing. However, challenges remain in areas such as interpretability, data "
        "efficiency, and robustness to adversarial examples."
    ),
    (
        "The human immune system is a complex network of cells, tissues, and organs that "
        "work together to defend the body against harmful pathogens. The innate immune "
        "system provides immediate but non-specific defense, while the adaptive immune "
        "system develops targeted responses to specific threats. Key components include "
        "white blood cells such as T cells and B cells, antibodies, the complement system, "
        "and the lymphatic system. Vaccination works by training the adaptive immune system "
        "to recognize and fight specific pathogens without causing the disease itself."
    ),
]

PROMPTS = [
    # Factual
    "What is photosynthesis?",
    "Explain how a CPU works.",
    "What causes thunder and lightning?",
    "How does the internet work?",
    # Math/Logic
    "If a train travels 60 mph for 2.5 hours, how far does it go?",
    "What is 15% of 200?",
    "Is 0.3 + 0.3 + 0.3 equal to 0.9?",
    # Creative
    "Write a haiku about mountains.",
    "Tell me a joke about cats.",
    "Write a limerick about a programmer.",
    # Instruction
    "List the 4 seasons in order.",
    "Translate 'good morning' to Spanish.",
    "Name 3 planets in our solar system.",
    # Reasoning
    "Which is heavier, a pound of feathers or a pound of steel?",
    "If all roses are flowers and some flowers fade quickly, can we say all roses fade quickly?",
]


def compute_perplexity(model, tok, text):
    tokens = mx.array(tok.encode(text))
    n = min(len(tokens), 512)
    tokens = tokens[:n]
    logits = model(tokens[None, :-1])
    mx.eval(logits)
    targets = tokens[1:]
    log_probs = logits[0] - mx.logsumexp(logits[0], axis=-1, keepdims=True)
    nll = -log_probs[mx.arange(len(targets)), targets]
    mx.eval(nll)
    return mx.exp(mx.mean(nll)).item()


def setup_rps(model, fp16_w, base_bits, residual_bits, tier2_keys="all"):
    """Apply RPS to a pre-quantized model using cached FP16 weights."""
    attached = 0
    for li, layer in enumerate(model.layers):
        for an, p in [
            ("q_proj", layer.self_attn), ("k_proj", layer.self_attn),
            ("v_proj", layer.self_attn), ("o_proj", layer.self_attn),
            ("gate_proj", layer.mlp), ("up_proj", layer.mlp),
            ("down_proj", layer.mlp),
        ]:
            proj = getattr(p, an, None)
            w_orig = fp16_w.get((li, an))
            if proj is None or w_orig is None or not isinstance(proj, nn.QuantizedLinear):
                continue
            if tier2_keys != "all" and an not in tier2_keys:
                continue
            w_base = mx.dequantize(
                proj.weight, proj.scales, proj.biases,
                group_size=proj.group_size, bits=proj.bits,
            )
            residual = (w_orig - w_base.astype(mx.float16)).astype(mx.float16)
            rq, rs, rb = mx.quantize(residual, group_size=64, bits=residual_bits)
            mx.eval(rq, rs, rb)
            rps = RPSLinear(proj)
            rps.attach_residual(rq, rs, rb, r_bits=residual_bits, r_group_size=64)
            setattr(p, an, rps)
            attached += 1
            del residual, w_base
    mx.eval()
    return attached


def run_config(label, model, tok, prompts, max_tokens=80, speed_runs=3):
    """Run generation and perplexity for a config."""
    # Perplexity
    ppls = []
    for text in EVAL_TEXTS:
        ppl = compute_perplexity(model, tok, text)
        ppls.append(ppl)
    avg_ppl = sum(ppls) / len(ppls)

    # Speed (average over multiple runs)
    generate(model, tok, prompt="Hi", max_tokens=3, verbose=False)
    mx.eval()

    speeds = []
    for _ in range(speed_runs):
        total_toks = 0
        total_time = 0
        for prompt in prompts[:5]:
            t0 = time.perf_counter()
            out = generate(model, tok, prompt=prompt, max_tokens=max_tokens, verbose=False)
            t1 = time.perf_counter()
            total_toks += len(tok.encode(out))
            total_time += (t1 - t0)
        speeds.append(total_toks / total_time)
    avg_speed = sum(speeds) / len(speeds)

    # All prompts
    outputs = []
    for prompt in prompts:
        out = generate(model, tok, prompt=prompt, max_tokens=max_tokens, verbose=False)
        outputs.append(out)

    # Memory
    mx.eval()
    active = mx.get_active_memory() / 1024**3

    return {
        "label": label,
        "ppl": avg_ppl,
        "ppl_per_text": ppls,
        "speed": avg_speed,
        "speed_std": max(speeds) - min(speeds),
        "memory_gb": active,
        "outputs": outputs,
    }


def main():
    print("=" * 70)
    print("RPS FULL VALIDATION — Qwen2.5-3B-Instruct")
    print("=" * 70)

    # Load FP16 and cache weights
    print("\n[1/7] Loading FP16 model and caching weights...")
    model, tok = load(MODEL)
    fp16_w = {}
    for li, layer in enumerate(model.layers):
        for an, p in [
            ("q_proj", layer.self_attn), ("k_proj", layer.self_attn),
            ("v_proj", layer.self_attn), ("o_proj", layer.self_attn),
            ("gate_proj", layer.mlp), ("up_proj", layer.mlp),
            ("down_proj", layer.mlp),
        ]:
            proj = getattr(p, an, None)
            if proj and hasattr(proj, "weight"):
                fp16_w[(li, an)] = proj.weight.astype(mx.float16)
                mx.eval(fp16_w[(li, an)])

    # FP16 baseline
    print("\n[2/7] FP16 baseline...")
    r_fp16 = run_config("FP16", model, tok, PROMPTS)
    del model; mx.clear_cache()

    # Standard quantization baselines
    results = [r_fp16]

    for bits in [4, 3, 2]:
        print(f"\n[3/7] {bits}-bit standard...")
        model, tok = load(MODEL)
        config = model.args.__dict__ if hasattr(model, "args") else {}
        quantize_model(model, config, group_size=64, bits=bits)
        r = run_config(f"{bits}-bit", model, tok, PROMPTS)
        results.append(r)
        del model; mx.clear_cache()

    # RPS configurations
    rps_configs = [
        ("RPS 3+2 (all)", 3, 2, "all"),
        ("RPS 3+2 (attn)", 3, 2, {"q_proj", "k_proj", "v_proj", "o_proj"}),
        ("RPS 3+2 (v+down)", 3, 2, {"v_proj", "down_proj"}),
        ("RPS 3+3 (all)", 3, 3, "all"),
        ("RPS 2+2 (all)", 2, 2, "all"),
    ]

    for i, (label, base_b, res_b, keys) in enumerate(rps_configs):
        print(f"\n[{4+i}/7] {label}...")
        model, tok = load(MODEL)
        config = model.args.__dict__ if hasattr(model, "args") else {}
        quantize_model(model, config, group_size=64, bits=base_b)
        n = setup_rps(model, fp16_w, base_b, res_b, keys)
        r = run_config(label, model, tok, PROMPTS)
        r["residuals_attached"] = n
        results.append(r)
        del model; mx.clear_cache()

    # Long generation stability test
    print("\n[7/7] Long generation (500 tokens)...")
    model, tok = load(MODEL)
    config = model.args.__dict__ if hasattr(model, "args") else {}
    quantize_model(model, config, group_size=64, bits=3)
    setup_rps(model, fp16_w, 3, 2, "all")
    long_out = generate(model, tok,
                        prompt="Write a detailed essay about the history of computing.",
                        max_tokens=500, verbose=False)
    del model; mx.clear_cache()

    # === REPORT ===
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\n--- Perplexity (lower = better) ---")
    print(f"{'Config':<22} {'Avg PPL':>8} {'Text1':>8} {'Text2':>8} {'Text3':>8} {'vs FP16':>8}")
    fp16_ppl = results[0]["ppl"]
    for r in results:
        ppls = r["ppl_per_text"]
        ratio = r["ppl"] / fp16_ppl
        print(f"  {r['label']:<22} {r['ppl']:>8.2f} {ppls[0]:>8.2f} {ppls[1]:>8.2f} {ppls[2]:>8.2f} {ratio:>7.2f}x")

    print("\n--- Speed (tok/s, higher = better) ---")
    print(f"{'Config':<22} {'Avg':>8} {'Range':>8} {'vs FP16':>8}")
    fp16_speed = results[0]["speed"]
    for r in results:
        ratio = r["speed"] / fp16_speed
        print(f"  {r['label']:<22} {r['speed']:>8.1f} {r['speed_std']:>7.1f} {ratio:>7.2f}x")

    print("\n--- Memory (GB) ---")
    for r in results:
        extra = f"  ({r.get('residuals_attached', '-')} residuals)" if "residuals_attached" in r else ""
        print(f"  {r['label']:<22} {r['memory_gb']:>6.2f} GB{extra}")

    print("\n--- Generation Quality (first 80 chars per prompt) ---")
    for i, prompt in enumerate(PROMPTS):
        print(f"\n  P{i}: {prompt}")
        for r in results:
            out = r["outputs"][i][:80].replace("\n", " ")
            print(f"    {r['label']:<20}: {out}")

    print("\n--- Long Generation (500 tok, RPS 3+2) ---")
    words = long_out.split()
    print(f"  Length: {len(words)} words, {len(tok.encode(long_out))} tokens")
    print(f"  First 200 chars: {long_out[:200]}")
    print(f"  Last 200 chars: ...{long_out[-200:]}")

    # Check for repetition in long output
    chunks = [long_out[i:i+50] for i in range(0, len(long_out)-50, 50)]
    unique_ratio = len(set(chunks)) / max(len(chunks), 1)
    print(f"  Repetition check: {unique_ratio:.0%} unique 50-char chunks")

    print("\n--- Key Findings ---")
    ppl_3bit = next(r["ppl"] for r in results if r["label"] == "3-bit")
    ppl_4bit = next(r["ppl"] for r in results if r["label"] == "4-bit")
    ppl_rps = next(r["ppl"] for r in results if r["label"] == "RPS 3+2 (all)")

    print(f"  RPS 3+2 ppl ({ppl_rps:.2f}) vs 3-bit ({ppl_3bit:.2f}): "
          f"{'BETTER' if ppl_rps < ppl_3bit else 'WORSE'} by {abs(ppl_3bit - ppl_rps):.2f}")
    print(f"  RPS 3+2 ppl ({ppl_rps:.2f}) vs 4-bit ({ppl_4bit:.2f}): "
          f"{'BETTER' if ppl_rps < ppl_4bit else 'WORSE'} by {abs(ppl_4bit - ppl_rps):.2f}")
    print(f"  RPS 3+2 ppl ({ppl_rps:.2f}) vs FP16 ({fp16_ppl:.2f}): "
          f"{ppl_rps/fp16_ppl:.2f}x degradation")

    # Save raw results
    save_data = []
    for r in results:
        save_data.append({k: v for k, v in r.items() if k != "outputs"})
    with open("/tmp/rps_validation_results.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Raw data saved to /tmp/rps_validation_results.json")


if __name__ == "__main__":
    main()
