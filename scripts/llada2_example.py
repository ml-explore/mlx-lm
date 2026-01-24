"""
LLaDA2 Diffusion-based Text Generation Example

LLaDA2 is a diffusion language model that generates text using block-wise
iterative refinement rather than standard autoregressive decoding.

This example demonstrates how to use mlx_lm.llada2_generate for proper
text generation with LLaDA2 models.

Usage (from the mlx-lm repository root):
    PYTHONPATH=. python examples/llada2_example.py --prompt "Your question here"

Or if mlx-lm is installed with LLaDA2 support:
    python examples/llada2_example.py --prompt "Your question here"

Available models:
    - mlx-community/LLaDA2.0-mini-4bit (default, ~4GB)
    - mlx-community/LLaDA2.0-1B-4bit
    - mlx-community/LLaDA2.0-8B-4bit
"""

import argparse
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm import llada2_generate


def main():
    parser = argparse.ArgumentParser(
        description="Generate text with LLaDA2 using diffusion-based decoding"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/LLaDA2.0-mini-4bit",
        help="Model path or HuggingFace repo",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What is the capital of France?",
        help="Input prompt",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--block-length",
        type=int,
        default=32,
        help="Block size for diffusion generation",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=32,
        help="Number of refinement steps per block",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p (nucleus) sampling threshold",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling threshold",
    )
    parser.add_argument(
        "--mask-id",
        type=int,
        default=156895,
        help="Mask token ID for LLaDA2",
    )
    parser.add_argument(
        "--eos-id",
        type=int,
        default=156892,
        help="EOS token ID for LLaDA2",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Don't apply chat template to prompt",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream output per block",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)
    print("Model loaded!")

    print(f"\nPrompt: {args.prompt}")
    print("-" * 50)

    if args.stream:
        # Streaming generation
        print("Response: ", end="", flush=True)
        start_time = time.time()
        last_text = ""
        for text in llada2_generate.stream_generate(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_tokens,
            block_length=args.block_length,
            steps=args.steps,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            mask_id=args.mask_id,
            eos_id=args.eos_id,
        ):
            # Print only new content
            new_content = text[len(last_text):]
            print(new_content, end="", flush=True)
            last_text = text
        print()
        elapsed = time.time() - start_time
        print("-" * 50)
        print(f"Generated in {elapsed:.2f}s")
    else:
        # Non-streaming generation
        # Format prompt
        if args.no_chat_template or tokenizer.chat_template is None:
            formatted_prompt = args.prompt
        else:
            messages = [{"role": "user", "content": args.prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )

        # Tokenize
        input_ids = mx.array([tokenizer.encode(formatted_prompt)])

        # Generate
        start_time = time.time()
        output = llada2_generate.generate(
            model,
            input_ids,
            max_new_tokens=args.max_tokens,
            block_length=args.block_length,
            steps=args.steps,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            mask_id=args.mask_id,
            eos_id=args.eos_id,
        )
        elapsed = time.time() - start_time

        # Decode full output
        full_text = tokenizer.decode(output[0].tolist())

        # Extract assistant response if chat template was used
        if not args.no_chat_template and tokenizer.chat_template is not None:
            assistant_marker = "<role>ASSISTANT</role>"
            if assistant_marker in full_text:
                response = full_text.split(assistant_marker)[-1]
                response = response.replace("<|role_end|>", "").replace("<|endoftext|>", "")
                response = response.strip()
            else:
                response = full_text
        else:
            response = full_text

        print(f"Response: {response}")
        print("-" * 50)

        # Stats
        num_tokens = output.shape[1]
        new_tokens = num_tokens - input_ids.shape[1]
        print(f"Generated {new_tokens} new tokens in {elapsed:.2f}s")
        print(f"Speed: {new_tokens / elapsed:.2f} tokens/sec")


if __name__ == "__main__":
    main()
