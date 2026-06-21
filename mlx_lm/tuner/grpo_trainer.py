# Copyright © 2025 Apple Inc.

"""GRPO (Group Relative Policy Optimization) trainer for mlx-lm.

Generates K completions per prompt, scores them with a user-provided reward
function, and applies a REINFORCE-style policy gradient with group-normalized
advantages. Supports LoRA for parameter-efficient training of large models.

Example usage::

    from mlx_lm import load
    from mlx_lm.tuner import linear_to_lora_layers
    from mlx_lm.tuner.grpo_trainer import GRPOArgs, grpo_train

    model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    linear_to_lora_layers(model, num_layers=16, lora_config={"rank": 8})

    def reward_fn(completions, prompt):
        # Return list of float rewards, one per completion
        return [1.0 if "correct" in c else 0.0 for c in completions]

    grpo_train(
        model=model,
        tokenizer=tokenizer,
        prompts=["Solve 2+2", "What is 3*4"],
        reward_fn=reward_fn,
        args=GRPOArgs(num_completions=8, max_new_tokens=256),
    )
"""

import math
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map

from ..generate import generate
from ..sample_utils import make_sampler
from ..tokenizer_utils import TokenizerWrapper


@dataclass
class GRPOArgs:
    """Hyperparameters for GRPO training."""

    # Generation
    num_completions: int = 8
    """Number of completions (K) to generate per prompt."""

    max_new_tokens: int = 1024
    """Maximum tokens per completion."""

    temperature: float = 1.0
    """Sampling temperature for generation."""

    # Optimization
    learning_rate: float = 1e-5
    """Learning rate for AdamW."""

    iters: int = 20
    """Number of training iterations (one prompt per iter)."""

    grad_clip: float = 1.0
    """Maximum gradient norm."""

    kl_beta: float = 0.0
    """KL penalty coefficient (0 = no KL penalty)."""

    # Logging & checkpoints
    steps_per_report: int = 1
    """Report metrics every N steps."""

    steps_per_save: int = 10
    """Save adapter every N steps."""

    adapter_path: str = "grpo_adapters"
    """Path to save adapter weights."""

    seed: int = 42
    """Random seed for prompt shuffling."""


@dataclass
class GRPOMetrics:
    """Metrics from a single GRPO step."""

    step: int = 0
    reward_mean: float = 0.0
    reward_std: float = 0.0
    loss: float = 0.0
    grad_norm: float = 0.0
    gen_time: float = 0.0
    train_time: float = 0.0
    has_signal: bool = False


class GRPOCallback:
    """Callback for GRPO training events."""

    def on_step(self, metrics: GRPOMetrics):
        """Called after each training step."""
        pass

    def on_save(self, path: str, step: int):
        """Called when adapter weights are saved."""
        pass


def _compute_logprobs(model, tokenizer, prompt_text, completion_text):
    """Compute log probability of completion given prompt.

    Forward passes the full sequence and extracts log-probs for completion
    tokens only (prompt tokens are masked).
    """
    full_text = prompt_text + completion_text
    tokens = mx.array(tokenizer.encode(full_text))
    prompt_tokens = tokenizer.encode(prompt_text)
    prompt_len = len(prompt_tokens)

    # Forward pass
    logits = model(tokens[None, :])  # [1, seq_len, vocab]
    logits = logits[0]  # [seq_len, vocab]

    # Log-softmax
    shift_logits = logits[:-1]
    shift_targets = tokens[1:]
    log_probs = shift_logits - mx.logsumexp(shift_logits, axis=-1, keepdims=True)

    # Gather log-probs of actual tokens
    token_logps = mx.take_along_axis(
        log_probs, shift_targets[:, None], axis=-1
    ).squeeze(-1)

    # Sum only completion tokens (skip prompt)
    completion_logps = token_logps[prompt_len - 1 :].sum()
    return completion_logps


def _grpo_loss(model, tokenizer, prompt_text, completions, advantages, prompt_len):
    """GRPO policy gradient loss: -mean(advantage * log_prob)."""
    losses = []
    for comp, adv in zip(completions, advantages):
        if abs(adv) < 1e-8:
            continue

        full_text = prompt_text + comp
        tokens = mx.array(tokenizer.encode(full_text))

        logits = model(tokens[None, :])
        logits = logits[0]

        shift_logits = logits[:-1]
        shift_targets = tokens[1:]
        log_probs = shift_logits - mx.logsumexp(shift_logits, axis=-1, keepdims=True)
        token_logps = mx.take_along_axis(
            log_probs, shift_targets[:, None], axis=-1
        ).squeeze(-1)

        # Only completion tokens
        completion_logps = token_logps[prompt_len - 1 :].sum()
        losses.append(-adv * completion_logps)

    if not losses:
        return mx.array(0.0)
    return mx.stack(losses).mean()


def grpo_train(
    model: nn.Module,
    tokenizer,
    prompts: List[str],
    reward_fn: Callable[[List[str], str], List[float]],
    args: GRPOArgs = GRPOArgs(),
    optimizer: Optional[optim.Optimizer] = None,
    callback: Optional[GRPOCallback] = None,
):
    """
    Train a model using GRPO (Group Relative Policy Optimization).

    Each iteration:
    1. Selects a prompt
    2. Generates K completions using the current policy
    3. Scores completions with ``reward_fn``
    4. Computes group-normalized advantages
    5. Updates the policy with REINFORCE loss

    Args:
        model: The language model (with LoRA layers applied if desired).
        tokenizer: The tokenizer (HuggingFace or mlx-lm TokenizerWrapper).
        prompts: List of training prompt strings.
        reward_fn: Callable that takes (completions: List[str], prompt: str)
            and returns a list of float rewards (one per completion).
        args: GRPO hyperparameters.
        optimizer: Optional optimizer. If None, creates AdamW with args.learning_rate.
        callback: Optional callback for logging/monitoring.

    Returns:
        dict with training summary (total_steps, steps_with_signal, mean_reward).
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    if optimizer is None:
        optimizer = optim.Adam(learning_rate=args.learning_rate)

    sampler = make_sampler(temp=args.temperature)

    # Shuffle prompts
    rng = np.random.default_rng(args.seed)
    prompt_indices = np.arange(len(prompts))
    rng.shuffle(prompt_indices)

    # Training state
    all_rewards = []
    steps_with_signal = 0

    print(f"GRPO Training | K={args.num_completions} | iters={args.iters}")
    print(f"  LR={args.learning_rate} | temp={args.temperature} | grad_clip={args.grad_clip}")
    print(f"  prompts={len(prompts)} | max_new_tokens={args.max_new_tokens}")
    print("-" * 60)

    for step in range(args.iters):
        metrics = GRPOMetrics(step=step)
        step_t0 = time.perf_counter()

        # Select prompt
        idx = prompt_indices[step % len(prompt_indices)]
        prompt_text = prompts[idx]
        prompt_ids = tokenizer.encode(prompt_text)
        prompt_len = len(prompt_ids)

        # ── Generate K completions ──────────────────────────────────
        model.eval()
        gen_t0 = time.perf_counter()
        completions = []
        for _ in range(args.num_completions):
            result = generate(
                model, tokenizer,
                prompt=prompt_ids,
                max_tokens=args.max_new_tokens,
                sampler=sampler,
            )
            completions.append(result)
        metrics.gen_time = time.perf_counter() - gen_t0

        # ── Compute rewards ─────────────────────────────────────────
        rewards = reward_fn(completions, prompt_text)
        r = np.array(rewards)
        metrics.reward_mean = float(r.mean())
        metrics.reward_std = float(r.std())
        all_rewards.append(metrics.reward_mean)

        # ── Check for gradient signal ───────────────────────────────
        if r.std() < 1e-8:
            # All rewards same — no gradient signal
            if step % args.steps_per_report == 0:
                print(
                    f"  Step {step + 1}/{args.iters}: "
                    f"reward={metrics.reward_mean:.3f} (all-same, skip) | "
                    f"gen={metrics.gen_time:.0f}s"
                )
            if callback:
                callback.on_step(metrics)
            continue

        metrics.has_signal = True
        steps_with_signal += 1

        # ── Compute advantages ──────────────────────────────────────
        advantages = ((r - r.mean()) / (r.std() + 1e-8)).tolist()

        # ── Training step ───────────────────────────────────────────
        model.train()
        train_t0 = time.perf_counter()

        def loss_fn(model):
            return _grpo_loss(
                model, tokenizer, prompt_text, completions, advantages, prompt_len
            )

        loss_and_grad = nn.value_and_grad(model, loss_fn)
        loss_val, grads = loss_and_grad(model)
        mx.eval(loss_val, grads)

        # Gradient clipping
        grad_norm_sq = sum(
            (g * g).sum().item()
            for _, g in tree_flatten(grads)
            if g is not None
        )
        grad_norm = math.sqrt(grad_norm_sq)
        if grad_norm > args.grad_clip:
            scale = args.grad_clip / grad_norm
            grads = tree_map(lambda g: g * scale if g is not None else g, grads)

        # Optimizer step
        optimizer.update(model, grads)
        mx.eval(model.parameters())

        metrics.loss = loss_val.item()
        metrics.grad_norm = grad_norm
        metrics.train_time = time.perf_counter() - train_t0

        # ── Report ──────────────────────────────────────────────────
        if step % args.steps_per_report == 0:
            total_time = time.perf_counter() - step_t0
            print(
                f"  Step {step + 1}/{args.iters}: "
                f"reward={metrics.reward_mean:.3f} "
                f"loss={metrics.loss:.4f} "
                f"grad_norm={metrics.grad_norm:.4f} | "
                f"gen={metrics.gen_time:.0f}s "
                f"train={metrics.train_time:.1f}s "
                f"total={total_time:.0f}s"
            )

        if callback:
            callback.on_step(metrics)

        # ── Save checkpoint ─────────────────────────────────────────
        if (step + 1) % args.steps_per_save == 0:
            weights = dict(tree_flatten(model.trainable_parameters()))
            mx.save_safetensors(
                f"{args.adapter_path}/step_{step + 1}.safetensors", weights
            )
            if callback:
                callback.on_save(args.adapter_path, step + 1)

    # ── Final save ──────────────────────────────────────────────────
    import os

    os.makedirs(args.adapter_path, exist_ok=True)
    weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(f"{args.adapter_path}/final.safetensors", weights)

    # Summary
    summary = {
        "total_steps": args.iters,
        "steps_with_signal": steps_with_signal,
        "mean_reward": float(np.mean(all_rewards)) if all_rewards else 0.0,
    }
    print("=" * 60)
    print(f"GRPO complete: {steps_with_signal}/{args.iters} steps with signal")
    if all_rewards:
        print(f"Mean reward: {summary['mean_reward']:.3f}")
    print(f"Adapter saved to {args.adapter_path}/final.safetensors")

    return summary
