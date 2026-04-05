"""Comprehensive RPS validation suite.

Tests quality, speed, memory, and edge cases across multiple prompts
and configurations to validate Residual Precision Streaming.
"""

import time
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
from mlx_lm.utils import quantize_model
from mlx_lm.rps.linear import RPSLinear

MODEL_PATH = "/tmp/qwen2.5-3b-fp16"

PROMPTS = [
    # Factual / technical
    "Explain quantum computing in simple terms.",
    "What causes the seasons on Earth?",
    "How does a neural network learn?",
    # Creative
    "Write a haiku about the ocean.",
    "Tell me a short joke about programmers.",
    # Reasoning
    "If a train travels 60 mph for 2.5 hours, how far does it go?",
    "What is larger, 9.9 or 9.11?",
    # Instruction following
    "List exactly 3 colors of the rainbow.",
    "Translate 'hello world' to French.",
    # Long-form
    "Write a paragraph about why exercise is important for health.",
]


def load_and_save_fp16():
    """Load FP16 model and cache original weights."""
    model, tok = load(MODEL_PATH)
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
    return model, tok, fp16_w


def make_quantized(bits, fp16_w):
    """Load model and quantize at given bits."""
    model, tok = load(MODEL_PATH)
    config = model.args.__dict__ if hasattr(model, "args") else {}
    quantize_model(model, config, group_size=64, bits=bits)
    return model, tok


def make_rps(base_bits, residual_bits, fp16_w, tier2_keys="all"):
    """Load model, quantize to base_bits, attach residuals."""
    model, tok = load(MODEL_PATH)
    config = model.args.__dict__ if hasattr(model, "args") else {}
    quantize_model(model, config, group_size=64, bits=base_bits)

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
                group_size=64, bits=base_bits,
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
    return model, tok, attached


def measure_perplexity(model, tok, text):
    """Compute perplexity on a text string."""
    tokens = mx.array(tok.encode(text))
    n = min(len(tokens), 512)
    tokens = tokens[:n]

    logits = model(tokens[None, :-1])
    mx.eval(logits)

    targets = tokens[1:]
    log_probs = logits[0] - mx.logsumexp(logits[0], axis=-1, keepdims=True)
    token_log_probs = log_probs[mx.arange(len(targets)), targets]
    mx.eval(token_log_probs)

    avg_nll = -mx.mean(token_log_probs).item()
    return 2.718281828 ** avg_nll  # e^(avg NLL) = perplexity


def benchmark_generation(model, tok, prompts, max_tokens=80):
    """Generate for all prompts, return outputs and speed."""
    # Warmup
    generate(model, tok, prompt="Hi", max_tokens=3, verbose=False)
    mx.eval()

    outputs = []
    total_toks = 0
    total_time = 0

    for prompt in prompts:
        t0 = time.perf_counter()
        out = generate(model, tok, prompt=prompt, max_tokens=max_tokens, verbose=False)
        t1 = time.perf_counter()
        toks = len(tok.encode(out))
        total_toks += toks
        total_time += (t1 - t0)
        outputs.append(out)

    avg_speed = total_toks / total_time if total_time > 0 else 0
    return outputs, avg_speed


def run_all():
    print("=" * 70)
    print("RESIDUAL PRECISION STREAMING — COMPREHENSIVE VALIDATION")
    print("=" * 70)
    print(f"Model: {MODEL_PATH}")
    print(f"Prompts: {len(PROMPTS)}")
    print()

    # --- Phase 1: Load and cache FP16 weights ---
    print("Phase 1: Loading FP16 model and caching weights...")
    model_fp16, tok, fp16_w = load_and_save_fp16()

    # --- Phase 2: Perplexity test ---
    print("\nPhase 2: Perplexity measurement")
    ppl_text = (
        "The quick brown fox jumps over the lazy dog. "
        "Machine learning is a subset of artificial intelligence that focuses on "
        "building systems that learn from data. Neural networks are inspired by "
        "the structure of the human brain and consist of layers of interconnected "
        "nodes. Deep learning uses multiple layers to progressively extract higher "
        "level features from raw input. Quantum computing uses quantum mechanical "
        "phenomena to process information in fundamentally new ways."
    ) * 3

    configs = {}

    # FP16 perplexity
    ppl_fp16 = measure_perplexity(model_fp16, tok, ppl_text)
    print(f"  FP16:     ppl = {ppl_fp16:.2f}")
    del model_fp16
    mx.clear_cache()

    # 4-bit perplexity
    model_4, tok = make_quantized(4, fp16_w)
    ppl_4 = measure_perplexity(model_4, tok, ppl_text)
    print(f"  4-bit:    ppl = {ppl_4:.2f}")
    del model_4
    mx.clear_cache()

    # 3-bit perplexity
    model_3, tok = make_quantized(3, fp16_w)
    ppl_3 = measure_perplexity(model_3, tok, ppl_text)
    print(f"  3-bit:    ppl = {ppl_3:.2f}")
    del model_3
    mx.clear_cache()

    # RPS 3+2 all perplexity
    model_rps, tok, n = make_rps(3, 2, fp16_w, tier2_keys="all")
    ppl_rps = measure_perplexity(model_rps, tok, ppl_text)
    print(f"  RPS 3+2:  ppl = {ppl_rps:.2f}  ({n} residuals)")
    del model_rps
    mx.clear_cache()

    # RPS 3+2 selective (v_proj + down_proj only)
    model_rps2, tok, n2 = make_rps(3, 2, fp16_w, tier2_keys={"v_proj", "down_proj"})
    ppl_rps2 = measure_perplexity(model_rps2, tok, ppl_text)
    print(f"  RPS sel:  ppl = {ppl_rps2:.2f}  ({n2} residuals, v_proj+down_proj)")
    del model_rps2
    mx.clear_cache()

    # --- Phase 3: Generation quality across all prompts ---
    print("\nPhase 3: Generation quality and speed")

    all_results = {}

    for label, setup_fn in [
        ("FP16", lambda: (load(MODEL_PATH))),
        ("4-bit", lambda: make_quantized(4, fp16_w)),
        ("3-bit", lambda: make_quantized(3, fp16_w)),
        ("RPS 3+2", lambda: make_rps(3, 2, fp16_w, "all")[:2]),
    ]:
        print(f"\n  --- {label} ---")
        result = setup_fn()
        model = result[0]
        t = result[1]
        outputs, speed = benchmark_generation(model, t, PROMPTS, max_tokens=80)
        all_results[label] = (outputs, speed)
        print(f"  Speed: {speed:.1f} tok/s")
        for i, (prompt, out) in enumerate(zip(PROMPTS, outputs)):
            print(f"    P{i}: {out[:100]}")
        del model
        mx.clear_cache()

    # --- Phase 4: Memory measurement ---
    print("\n\nPhase 4: Memory usage")
    for label, setup_fn in [
        ("FP16", lambda: load(MODEL_PATH)),
        ("4-bit", lambda: make_quantized(4, fp16_w)),
        ("3-bit", lambda: make_quantized(3, fp16_w)),
        ("RPS 3+2", lambda: make_rps(3, 2, fp16_w, "all")[:2]),
    ]:
        mx.clear_cache()
        mx.reset_peak_memory()
        result = setup_fn()
        model = result[0]
        t = result[1]
        generate(model, t, prompt="Hello", max_tokens=50, verbose=False)
        mx.eval()
        peak = mx.get_peak_memory() / 1024**3
        print(f"  {label:10s}: peak = {peak:.2f} GB")
        del model
        mx.clear_cache()

    # --- Phase 5: Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Config':<12} {'PPL':>8} {'Tok/s':>8} {'vs FP16 PPL':>14}")
    for label, ppl in [
        ("FP16", ppl_fp16),
        ("4-bit", ppl_4),
        ("3-bit", ppl_3),
        ("RPS 3+2", ppl_rps),
        ("RPS sel", ppl_rps2),
    ]:
        speed = all_results.get(label, (None, 0))[1]
        ppl_ratio = ppl / ppl_fp16
        print(f"  {label:<12} {ppl:>8.2f} {speed:>8.1f} {ppl_ratio:>12.2f}x")

    print()
    print("Key questions:")
    print(f"  Does RPS 3+2 beat 3-bit in perplexity? {'YES' if ppl_rps < ppl_3 else 'NO'} ({ppl_rps:.2f} vs {ppl_3:.2f})")
    print(f"  Does RPS 3+2 approach 4-bit?           {'YES' if ppl_rps < ppl_4 * 1.1 else 'NO'} ({ppl_rps:.2f} vs {ppl_4:.2f})")
    print(f"  Is RPS 3+2 coherent on all prompts?    (check outputs above)")


if __name__ == "__main__":
    run_all()
