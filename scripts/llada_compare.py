#!/usr/bin/env python3
"""
Compare LLaDA implementations: PyTorch (GSAI-ML) vs MLX (mlx-lm).

This script runs the same prompts with identical parameters on both
implementations to verify that the MLX implementation produces
matching results.

Usage:
    python llada_compare.py
    python llada_compare.py --prompts "What is 2+2?" "Hello world"
    python llada_compare.py --steps 128 --gen-length 128
"""

import argparse
import time

import torch
import torch.nn.functional as F
import mlx.core as mx


# ============================================================================
# PyTorch Implementation (from GSAI-ML/LLaDA)
# ============================================================================

def torch_add_gumbel_noise(logits, temperature):
    """Add Gumbel noise for sampling (PyTorch version)."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def torch_get_num_transfer_tokens(mask_index, steps):
    """Compute number of tokens to unmask per step (PyTorch version)."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
    ) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens


@torch.no_grad()
def torch_generate(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    mask_id=126336,
):
    """Generate with PyTorch LLaDA model."""
    device = next(model.parameters()).device
    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long
    ).to(device)
    x[:, : prompt.shape[1]] = prompt.clone()
    prompt_index = x != mask_id

    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = x[:, block_start:block_end] == mask_id
        num_transfer_tokens = torch_get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = x == mask_id
            logits = model(x).logits

            logits_with_noise = torch_add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            p = F.softmax(logits, dim=-1)
            x0_p = torch.squeeze(
                torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
            )

            x0_p[:, block_end:] = -float("inf")

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(
                mask_index, x0_p, torch.tensor(-float("inf"), device=device)
            )

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


# ============================================================================
# Main Comparison
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare PyTorch and MLX LLaDA implementations"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=[
            "What is 2 + 2?",
            "What is the capital of France?",
        ],
        help="Test prompts to compare",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=64,
        help="Number of denoising steps (default: 64)",
    )
    parser.add_argument(
        "--gen-length",
        type=int,
        default=64,
        help="Number of tokens to generate (default: 64)",
    )
    parser.add_argument(
        "--block-length",
        type=int,
        default=32,
        help="Block length (default: 32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["mps", "cuda", "cpu"],
        help="PyTorch device (default: mps)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("LLaDA Implementation Comparison: PyTorch vs MLX")
    print("=" * 70)
    print(f"Parameters: steps={args.steps}, gen_length={args.gen_length}, "
          f"block_length={args.block_length}, temperature=0.0")
    print("=" * 70)

    # Load PyTorch model
    print("\n[1] Loading PyTorch model (GSAI-ML/LLaDA-8B-Instruct)...")
    from transformers import AutoTokenizer, AutoModel

    torch_model = (
        AutoModel.from_pretrained(
            "GSAI-ML/LLaDA-8B-Instruct",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        .to(args.device)
        .eval()
    )
    torch_tokenizer = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
    )
    torch_tokenizer.padding_side = "left"
    print("PyTorch model loaded!")

    # Load MLX model
    print("\n[2] Loading MLX model (mlx-community/LLaDA-8B-Instruct-mlx-fp16)...")
    from mlx_lm import load
    from mlx_lm.llada_generate import generate as mlx_generate

    mlx_model, mlx_tokenizer = load("mlx-community/LLaDA-8B-Instruct-mlx-fp16")
    print("MLX model loaded!")

    # Run comparisons
    print("\n[3] Running comparison tests...")
    print("-" * 70)

    results = []
    for prompt in args.prompts:
        print(f"\nPrompt: {prompt}")

        # Prepare input for PyTorch
        messages = [{"role": "user", "content": prompt}]
        torch_prompt_text = torch_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        torch_input_ids = torch_tokenizer(
            torch_prompt_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(args.device)

        # Prepare input for MLX
        mlx_prompt_text = mlx_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        mlx_input_ids = mx.array(
            [mlx_tokenizer.encode(mlx_prompt_text, add_special_tokens=False)]
        )

        # Generate with PyTorch
        print("  Running PyTorch...", end=" ", flush=True)
        t0 = time.time()
        torch_output = torch_generate(
            torch_model,
            torch_input_ids,
            steps=args.steps,
            gen_length=args.gen_length,
            block_length=args.block_length,
            temperature=0.0,
        )
        torch_time = time.time() - t0
        torch_generated = torch_tokenizer.decode(
            torch_output[0, torch_input_ids.shape[1] :], skip_special_tokens=True
        )
        print(f"({torch_time:.2f}s)")

        # Generate with MLX
        print("  Running MLX...", end=" ", flush=True)
        t0 = time.time()
        mlx_output = mlx_generate(
            mlx_model,
            mlx_input_ids,
            steps=args.steps,
            gen_length=args.gen_length,
            block_length=args.block_length,
            temperature=0.0,
        )
        mlx_time = time.time() - t0
        mlx_generated = mlx_tokenizer.decode(
            mlx_output[0, mlx_input_ids.shape[1] :].tolist(), skip_special_tokens=True
        )
        print(f"({mlx_time:.2f}s)")

        # Compare results
        match = torch_generated.strip() == mlx_generated.strip()
        results.append(match)

        print(f"  PyTorch output: {torch_generated}")
        print(f"  MLX output:     {mlx_generated}")
        print(f"  Match: {'✓ YES' if match else '✗ NO'}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    total = len(results)
    passed = sum(results)
    print(f"Passed: {passed}/{total}")
    if passed == total:
        print("All tests passed! MLX implementation matches PyTorch.")
    else:
        print("Some tests failed. Please check the outputs above.")


if __name__ == "__main__":
    main()
