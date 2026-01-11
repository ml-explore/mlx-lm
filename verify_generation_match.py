
import sys
import argparse
import numpy as np
import torch
import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from mlx_lm import load, generate
from mlx_lm.utils import generate_step


import gc

def run_verification(model_path, prompt="What is the capital of India?", max_tokens=50):
    print(f"--- Verifying SarvamMoE Inference ---")
    print(f"Model Path: {model_path}")
    print(f"Prompt: {prompt}")

    # Store results here
    hf_results = {}
    mlx_results = {}

    # --- Transformers (Reference) ---
    print("\n[Transformers] Loading model...")
    device = torch.device("cpu") # Force CPU for strict deterministic comparison
    try:
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        hf_tokenizer = AutoTokenizer.from_pretrained(model_path)
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=hf_config,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            device_map="cpu"
        )
        hf_model.eval()
        
        print("[Transformers] Generating...")
        inputs = hf_tokenizer(prompt, return_tensors="pt").to(device)
        
        # 1. Logit Comparison (Next Token Prediction)
        with torch.no_grad():
            hf_out = hf_model(**inputs)
            hf_results['last_token_logits'] = hf_out.logits[0, -1, :].cpu().numpy()
            hf_topk = torch.topk(hf_out.logits[0, -1, :], 5)
            hf_results['topk_indices'] = hf_topk.indices.tolist()
            hf_results['topk_values'] = hf_topk.values.tolist()
            
            print(f"[Transformers] Top 5 logits (Last Prompt Token):")
            for idx, score in zip(hf_results['topk_indices'], hf_results['topk_values']):
                 print(f"  Token {idx}: {score:.4f} ('{hf_tokenizer.decode([idx])}')")

        # 2. Text Generation
        with torch.no_grad():
            hf_generated_ids = hf_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False, # Greedy
                temperature=None,
                top_p=None
            )
        hf_results['text'] = hf_tokenizer.decode(hf_generated_ids[0], skip_special_tokens=True)
        print(f"[Transformers] Output:\n{hf_results['text']}")

        # CLEANUP
        del hf_model
        del hf_tokenizer
        del hf_config
        del inputs
        del hf_out
        del hf_generated_ids
        gc.collect()
        
    except Exception as e:
        print(f"[Transformers] Error: {e}")
        return

    print("\n[Memory] Cleared Transformers contents.")


    # --- MLX ---
    print("\n[MLX] Loading model...")
    try:
        mlx_model, mlx_tokenizer = load(model_path)
        
        print("[MLX] Generating...")
        
        # 1. Logit Comparison (Next Token Prediction)
        prompt_tokens = mx.array(mlx_tokenizer.encode(prompt))
        
        def get_logits(model, w, tokens):
            return model(tokens[None, :])

        logits = get_logits(mlx_model, None, prompt_tokens)
        
        # Capture logits
        mlx_last_token_logits = logits[0, -1, :]
        
        # Use HF indices for direct comparison later, but get top 5 here for display
        mlx_topk_indices = mx.argpartition(mlx_last_token_logits, -5)[-5:]
        mlx_topk_scores = mlx_last_token_logits[mlx_topk_indices]
        
        sort_inds = mx.argsort(mlx_topk_scores)[::-1]
        mlx_results['topk_indices'] = mlx_topk_indices[sort_inds].tolist()
        mlx_results['topk_values'] = mlx_topk_scores[sort_inds].tolist()
        
        # Store all logits for comparison with HF specific indices
        # Converting entire logit vocab to cpu numpy might be heavy? Usually vocab is 32k-100k, float32, so ~400KB. It's fine.
        # But mlx_last_token_logits is on GPU if using metal.
        # We need specific values.
        mlx_results['logits_for_comparison'] = []
        for hf_idx in hf_results['topk_indices']:
            mlx_results['logits_for_comparison'].append(mlx_last_token_logits[hf_idx].item())

        print(f"[MLX] Top 5 logits (Last Prompt Token):")
        for idx, score in zip(mlx_results['topk_indices'], mlx_results['topk_values']):
            print(f"  Token {idx}: {score:.4f} ('{mlx_tokenizer.decode([idx])}')")


        # 2. Text Generation
        mlx_text = generate(mlx_model, mlx_tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False, temp=0.0)
        mlx_results['text'] = prompt + mlx_text 
        print(f"[MLX] Output:\n{mlx_results['text']}")
        
    except Exception as e:
        print(f"[MLX] Error: {e}")
        return

    # --- COMPARISON ---
    print("\n" + "="*30)
    print("      COMPARISON REPORT")
    print("="*30)
    
    # 1. Logits
    print("\n[Logits] Checking top-5 match relative to HF...")
    max_diff = 0.0
    for i, hf_idx in enumerate(hf_results['topk_indices']):
        hf_val = hf_results['topk_values'][i]
        mlx_val = mlx_results['logits_for_comparison'][i]
        diff = abs(hf_val - mlx_val)
        max_diff = max(max_diff, diff)
        print(f"  Token {hf_idx}: HF={hf_val:.4f}, MLX={mlx_val:.4f}, Diff={diff:.6f}")

    if max_diff < 1e-3:
        print(f">>> Logit Check: PASS (Max Diff < 1e-3)")
    else:
        print(f">>> Logit Check: WARNING (Max Diff = {max_diff:.6f})")

    # 2. Text
    print("\n[Text] Comparing generated strings...")
    hf_text = hf_results['text']
    mlx_text = mlx_results['text']
    
    if hf_text == mlx_text:
        print(">>> Text Check: PASS (Exact Match)")
    else:
        print(">>> Text Check: FAIL")
        print("Diff:")
        import difflib
        diff = difflib.ndiff(hf_text.splitlines(keepends=True), mlx_text.splitlines(keepends=True))
        print(''.join(diff))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_generation_match.py <model_path> [prompt]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "What is the capital of India?"
    
    run_verification(model_path, prompt)
