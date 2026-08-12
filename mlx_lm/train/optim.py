# Copyright © 2026 Apple Inc.

import logging

import mlx.optimizers as optim


def build_schedule(config, num_steps=None):
    warmup_steps = config.get("warmup_steps", 0)
    decay_steps = config.get("decay_steps") or None

    if decay_steps is None:
        if num_steps is None:
            raise ValueError(
                "the schedule needs either config.optimizer.decay_steps or "
                "config.num_steps to size the decay"
            )
        decay_steps = num_steps - warmup_steps
    elif num_steps is not None and warmup_steps + decay_steps != num_steps:
        logging.warning(
            "the schedule spans %d steps (warmup_steps %d + decay_steps %d) but "
            "config.num_steps is %d; the learning rate %s",
            warmup_steps + decay_steps,
            warmup_steps,
            decay_steps,
            num_steps,
            (
                "sits at end_learning_rate for the last %d steps"
                % (num_steps - warmup_steps - decay_steps)
                if warmup_steps + decay_steps < num_steps
                else "never finishes decaying"
            ),
        )

    if decay_steps <= 0:
        raise ValueError(
            f"decay_steps must be positive, got {decay_steps}; num_steps "
            f"({num_steps}) must exceed warmup_steps ({warmup_steps})"
        )

    if warmup_steps > 0:
        warmup = optim.linear_schedule(
            0,
            config.learning_rate,
            warmup_steps,
        )
    if config.schedulers == "cosine_decay":
        decay = optim.cosine_decay(
            config.learning_rate,
            decay_steps,
            config.get("end_learning_rate", 0.0),
        )
    elif config.schedulers == "linear_decay":
        decay = optim.linear_schedule(
            config.learning_rate,
            0,
            decay_steps,
        )
    else:
        raise ValueError(
            f"unknown scheduler {config.schedulers!r}; "
            "expected 'cosine_decay' or 'linear_decay'"
        )
    if warmup_steps > 0:
        lr_schedule = optim.join_schedules([warmup, decay], [warmup_steps])
    else:
        lr_schedule = decay

    return lr_schedule


def build_optimizer(config, num_steps=None):
    lr_schedule = build_schedule(config, num_steps)
    weight_decay = config.get("weight_decay", 0.0)

    if config.optim == "adam" or config.optim == "adamw":
        params = {
            "learning_rate": lr_schedule,
            "eps": config.get("eps", 1e-8),
            "betas": (config.get("beta1", 0.9), config.get("beta2", 0.95)),
        }
        if weight_decay:
            optimizer = optim.AdamW(**params, weight_decay=weight_decay)
        else:
            optimizer = optim.Adam(**params)
    elif config.optim == "sgd":
        optimizer = optim.SGD(learning_rate=lr_schedule, weight_decay=weight_decay)
    else:
        raise ValueError(
            f"unknown optimizer {config.optim!r}; expected 'adam', 'adamw' or 'sgd'"
        )
    return optimizer
