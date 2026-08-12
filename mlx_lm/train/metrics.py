# Copyright © 2026 Apple Inc.

import dataclasses
import logging
import time

import mlx.core as mx
import numpy as np


class Losses(dict):
    def __init__(self, terms=(), count=0):
        super().__init__(terms)
        self.count = count

    def total(self):
        return sum(self.values())

    def plus(self, other):
        return Losses({k: self.get(k, 0) + v for k, v in other.items()}, self.count + 1)

    def all_reduce(self, group=None):
        n = self.count * (group.size if group is not None else 1)
        g = group.group if group is not None else None
        return Losses(
            {k: mx.distributed.all_sum(v, group=g) / n for k, v in self.items()}, 1
        )


@dataclasses.dataclass
class Metrics:
    step: int = 0
    tokens: int = 0
    train_loss: list = dataclasses.field(default_factory=list)
    ce_loss: list = dataclasses.field(default_factory=list)
    z_loss: list = dataclasses.field(default_factory=list)
    grad_norm: list = dataclasses.field(default_factory=list)
    learning_rate: list = dataclasses.field(default_factory=list)
    its_per_sec: float = 0.0
    toks_per_sec: float = 0.0
    peak_memory: float = None
    valid_loss: float = None
    valid_ppl: float = None
    tokens_per_step: int = 0
    steps_per_report: int = 1

    def __post_init__(self):
        self.tic = time.perf_counter()

    def append(self, **values):
        for name, value in values.items():
            if value is not None:
                getattr(self, name).append(
                    value.item() if hasattr(value, "item") else value
                )

    def report(self, step):

        elapsed = time.perf_counter() - self.tic
        self.step = step
        self.tokens = step * self.tokens_per_step
        self.its_per_sec = self.steps_per_report / elapsed
        self.toks_per_sec = self.steps_per_report * self.tokens_per_step / elapsed
        self.peak_memory = mx.get_peak_memory() / 1e9
        mx.reset_peak_memory()

        out = self.to_list()
        for field in dataclasses.fields(self):  # the list fields, whichever they are
            if field.default_factory is not dataclasses.MISSING:
                getattr(self, field.name).clear()
        self.tic = time.perf_counter()
        return out

    def to_list(self):
        mean = lambda xs: np.mean(xs).item() if xs else None
        metrics = [
            ("step", self.step),
            ("train_loss", mean(self.train_loss)),
            ("ce_loss", mean(self.ce_loss)),
            ("grad_norm", mean(self.grad_norm)),
            ("its_per_sec", self.its_per_sec),
            ("toks_per_sec", self.toks_per_sec),
            ("tokens", self.tokens),
            ("learning_rate", self.learning_rate[-1] if self.learning_rate else None),
            ("peak_memory", self.peak_memory),
        ]
        if self.z_loss and mean(self.z_loss) > 0:
            metrics.append(("z_loss", mean(self.z_loss)))
        if self.valid_loss is not None:
            metrics.append(("valid_loss", self.valid_loss))
            metrics.append(("valid_ppl", self.valid_ppl))
        return metrics


def log_metrics(metrics):
    list_metrics = metrics if isinstance(metrics, list) else metrics.to_list()

    to_str = lambda v: f"{v:.4f}" if isinstance(v, float) else repr(v)
    logging.info(", ".join(f"{n}: {to_str(v)}" for n, v in list_metrics))

    try:
        import wandb
    except ImportError:
        return

    metrics_dict = dict(list_metrics)
    step = metrics_dict.pop("step")
    if wandb.run is not None:
        wandb.log(metrics_dict, step=step)


def init_wandb(config, args, is_master):
    if not is_master:
        return
    try:
        import wandb
    except ImportError:
        return
    task = config.get("task", "pretrain")
    kwargs = dict(
        project=config.get("project", "mlx-lm"),
        name=args.experiment_name,
        tags=[
            task,
            f"batch_size_{config.batch_size}",
            f"context_size_{config.context_size}",
            f"grad_accum_{config.get('grad_accum_steps', 1)}",
        ],
    )
    if args.experiment_name is None:
        kwargs["mode"] = "disabled"
    wandb.init(**kwargs)
