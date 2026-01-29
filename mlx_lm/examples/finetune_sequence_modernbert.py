"""
Fine-tune ModernBERT for sequence classification or regression (Python API).

This example is intentionally minimal and uses the existing `mlx_lm.tuner.trainer.train`
loop with sequence-task utilities from `mlx_lm.tuner.sequence_tasks`.

It demonstrates:
- Load-time head selection:
    - classification: model_config={"task":"sequence_classification","num_labels":K}
    - regression:     model_config={"task":"regression"}
- Batching of {"text": ..., "label": ...} items with padding + attention_mask
- Sequence loss (cross-entropy or MSE/Huber)

Model for validation:
- mlx-community/answerdotai-ModernBERT-base-bf16

Notes:
- `mlx_lm.load` returns a TokenizerWrapper; it is not guaranteed to be callable.
  Use `tokenizer._tokenizer(...)` (the underlying HF tokenizer) for encoding.
- Regression output shape is (B,) by design.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import mlx.optimizers as optim

from mlx_lm import load
from mlx_lm.tuner.trainer import (
    TrainingArgs,
    iterate_sequence_batches,
    sequence_loss,
    train,
)


@dataclass
class ExampleConfig:
    model: str = "mlx-community/answerdotai-ModernBERT-base-bf16"
    task: str = "classification"  # classification | regression
    num_labels: int = 2  # classification only

    # Training
    iters: int = 50
    batch_size: int = 4
    max_seq_length: int = 256
    learning_rate: float = 1e-4

    # Regression loss
    regression_loss: str = "mse"  # mse | huber


def parse_args() -> ExampleConfig:
    p = argparse.ArgumentParser(
        description="Fine-tune ModernBERT for sequence classification/regression (Python API)."
    )
    p.add_argument("--model", type=str, default=ExampleConfig.model)
    p.add_argument(
        "--task",
        type=str,
        default=ExampleConfig.task,
        choices=["classification", "regression"],
    )
    p.add_argument("--num-labels", type=int, default=ExampleConfig.num_labels)
    p.add_argument("--iters", type=int, default=ExampleConfig.iters)
    p.add_argument("--batch-size", type=int, default=ExampleConfig.batch_size)
    p.add_argument("--max-seq-length", type=int, default=ExampleConfig.max_seq_length)
    p.add_argument("--learning-rate", type=float, default=ExampleConfig.learning_rate)
    p.add_argument(
        "--regression-loss",
        type=str,
        default=ExampleConfig.regression_loss,
        choices=["mse", "huber"],
    )
    ns = p.parse_args()

    return ExampleConfig(
        model=ns.model,
        task=ns.task,
        num_labels=ns.num_labels,
        iters=ns.iters,
        batch_size=ns.batch_size,
        max_seq_length=ns.max_seq_length,
        learning_rate=ns.learning_rate,
        regression_loss=ns.regression_loss,
    )


def build_toy_datasets(task: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build tiny toy datasets in {"text": ..., "label": ...} format.

    Replace with your real dataset. Any of these formats are supported by the
    sequence batch iterator:
      - (input_ids, label)
      - (input_ids, attention_mask, label)
      - {"input_ids": ..., "attention_mask": ..., "label": ...}
      - {"text": ..., "label": ...}
    """
    if task == "classification":
        train_items = [
            {"text": "I loved this product.", "label": 1},
            {"text": "This was terrible.", "label": 0},
            {"text": "Amazing quality, will buy again.", "label": 1},
            {"text": "Not what I expected.", "label": 0},
            {"text": "Pretty good overall.", "label": 1},
            {"text": "I do not recommend it.", "label": 0},
        ]
        valid_items = [
            {"text": "I really liked it.", "label": 1},
            {"text": "I hated it.", "label": 0},
        ]
        return train_items, valid_items

    # regression
    train_items = [
        {"text": "This is okay.", "label": 0.5},
        {"text": "This is great!", "label": 0.9},
        {"text": "This is awful.", "label": 0.1},
        {"text": "Mediocre experience.", "label": 0.4},
    ]
    valid_items = [
        {"text": "Excellent.", "label": 1.0},
        {"text": "Bad.", "label": 0.0},
    ]
    return train_items, valid_items


def main() -> None:
    cfg = parse_args()
    train_items, valid_items = build_toy_datasets(cfg.task)

    if cfg.task == "classification":
        model_config = {"task": "sequence_classification", "num_labels": cfg.num_labels}
    else:
        model_config = {"task": "regression"}

    print("Loading model...")
    model, tokenizer, model_cfg = load(
        cfg.model,
        model_config=model_config,
        return_config=True,
    )

    # Read pad_token_id dynamically from the loaded model config.
    pad_token_id = int(model_cfg.get("pad_token_id", 0))

    # Use underlying HF tokenizer (TokenizerWrapper isn't guaranteed callable).
    hf_tokenizer = tokenizer._tokenizer

    opt = optim.Adam(learning_rate=cfg.learning_rate)

    targs = TrainingArgs(
        batch_size=cfg.batch_size,
        iters=cfg.iters,
        max_seq_length=cfg.max_seq_length,
        steps_per_report=max(1, min(10, cfg.iters // 5)),
        steps_per_eval=max(1, min(50, cfg.iters // 2)),
        steps_per_save=max(50, cfg.iters),
    )

    def _iterate(**kw):
        return iterate_sequence_batches(
            dataset=kw["dataset"],
            batch_size=kw["batch_size"],
            max_seq_length=kw["max_seq_length"],
            loop=kw.get("loop", False),
            seed=kw.get("seed", None),
            tokenizer=hf_tokenizer,
            pad_token_id=pad_token_id,
            comm_group=kw.get("comm_group", None),
        )

    def _loss(m, input_ids, attention_mask, labels):
        if cfg.task == "classification":
            return sequence_loss(
                m,
                input_ids,
                attention_mask,
                labels,
                task="classification",
            )
        return sequence_loss(
            m,
            input_ids,
            attention_mask,
            labels,
            task="regression",
            regression_loss=cfg.regression_loss,
        )

    print("Starting training...")
    train(
        model=model,
        optimizer=opt,
        train_dataset=train_items,
        val_dataset=valid_items,
        args=targs,
        loss=_loss,
        iterate_batches=_iterate,
    )

    print("Done.")
    print("Artifacts:")
    print(f"- adapters.safetensors (default from TrainingArgs.adapter_file='{targs.adapter_file}')")


if __name__ == "__main__":
    main()
