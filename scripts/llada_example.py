#!/usr/bin/env python3
"""
Example script for running LLaDA (Large Language Diffusion with mAsking) with mlx-lm.

LLaDA is a diffusion-based language model that uses iterative unmasking
instead of autoregressive token generation.

Usage:
    python llada_example.py --prompt "What is 2 + 2?"
    python llada_example.py --prompt "Write a haiku about coding" --steps 128 --gen-length 64
    python llada_example.py --interactive
"""

import argparse
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.llada_generate import generate, stream_generate


def main():
    parser = argparse.ArgumentParser(description="LLaDA text generation with mlx-lm")
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/LLaDA-8B-Instruct-mlx-fp16",
        help="Model path or Hugging Face repo",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Input prompt for generation",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=128,
        help="Number of denoising steps (default: 128)",
    )
    parser.add_argument(
        "--gen-length",
        type=int,
        default=128,
        help="Number of tokens to generate (default: 128)",
    )
    parser.add_argument(
        "--block-length",
        type=int,
        default=32,
        help="Block length for semi-autoregressive generation (default: 32)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for Gumbel noise (default: 0.0, deterministic)",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=0.0,
        help="Classifier-free guidance scale (default: 0.0, disabled)",
    )
    parser.add_argument(
        "--remasking",
        type=str,
        default="low_confidence",
        choices=["low_confidence", "random"],
        help="Remasking strategy (default: low_confidence)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming output (shows generation progress per block)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive chat mode",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)
    print("Model loaded successfully!\n")

    if args.interactive:
        interactive_mode(model, tokenizer, args)
    elif args.prompt:
        single_prompt(model, tokenizer, args)
    else:
        parser.print_help()
        print("\nError: Please provide --prompt or use --interactive mode")


def single_prompt(model, tokenizer, args):
    """Generate response for a single prompt."""
    prompt = args.prompt
    print(f"Prompt: {prompt}")
    print("-" * 50)

    start_time = time.time()

    if args.stream:
        # Streaming mode - show progress per block
        print("Generating (streaming):")
        for i, text in enumerate(stream_generate(
            model,
            tokenizer,
            prompt,
            steps=args.steps,
            gen_length=args.gen_length,
            block_length=args.block_length,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
            remasking=args.remasking,
        )):
            print(f"[Block {i+1}] {text}")
    else:
        # Non-streaming mode - return final result
        messages = [{"role": "user", "content": prompt}]
        if tokenizer.chat_template is not None:
            prompt_text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        else:
            prompt_text = prompt

        input_ids = mx.array([tokenizer.encode(prompt_text, add_special_tokens=False)])

        output = generate(
            model,
            input_ids,
            steps=args.steps,
            gen_length=args.gen_length,
            block_length=args.block_length,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
            remasking=args.remasking,
        )

        generated_tokens = output[0, input_ids.shape[1]:].tolist()
        result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"Response: {result}")

    elapsed = time.time() - start_time
    print("-" * 50)
    print(f"Generation time: {elapsed:.2f}s")


def interactive_mode(model, tokenizer, args):
    """Interactive chat mode."""
    print("=" * 50)
    print("LLaDA Interactive Mode")
    print("Type 'quit' or 'exit' to end the session")
    print("Type 'clear' to clear conversation history")
    print("=" * 50)
    print()

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not prompt:
            continue
        if prompt.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break
        if prompt.lower() == "clear":
            print("(Conversation cleared)")
            continue

        print("Assistant: ", end="", flush=True)

        # Generate response
        final_text = ""
        for text in stream_generate(
            model,
            tokenizer,
            prompt,
            steps=args.steps,
            gen_length=args.gen_length,
            block_length=args.block_length,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
            remasking=args.remasking,
        ):
            final_text = text

        print(final_text)
        print()


if __name__ == "__main__":
    main()
