import argparse
import copy
import json
import math
import os
import re
import types
import warnings
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import yaml

from .tuner.callbacks import get_reporting_callbacks
from .tuner.datasets import CacheDataset, PreferenceDataset, load_dataset
from .tuner.trainer import TrainingArgs, TrainingCallback, train_dpo
from .tuner.utils import (
    build_schedule,
    linear_to_lora_layers,
    load_adapters,
    print_trainable_parameters,
)
from .utils import load, save_config

yaml_loader = yaml.SafeLoader
yaml_loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        """^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)

CONFIG_DEFAULTS = {
    "model": "mlx_model",
    "train": False,
    "fine_tune_type": "lora",
    "optimizer": "adam",
    "optimizer_config": {
        "adam": {},
        "adamw": {},
        "muon": {},
        "sgd": {},
        "adafactor": {},
    },
    "data": "data/",
    "seed": 0,
    "num_layers": 16,
    "batch_size": 4,
    "iters": 1000,
    "val_batches": 25,
    "learning_rate": 1e-5,
    "steps_per_report": 10,
    "steps_per_eval": 200,
    "resume_adapter_file": None,
    "adapter_path": "adapters",
    "save_every": 100,
    "test": False,
    "test_batches": 500,
    "max_seq_length": 2048,
    "config": None,
    "grad_checkpoint": False,
    "lr_schedule": None,
    "lora_parameters": {"rank": 8, "dropout": 0.0, "scale": 20.0},
    "beta": 0.1,
    "reference_model": None,
    "reference_model_adapters": None,
    "report_to": None,
    "project_name": None,
}


def build_parser():
    parser = argparse.ArgumentParser(description="DPO finetuning.")
    parser.add_argument(
        "--model",
        type=str,
        help="The path to the local model directory or Hugging Face repo.",
    )

    # Training args
    parser.add_argument(
        "--train",
        action="store_true",
        help="Do training",
        default=None,
    )
    parser.add_argument(
        "--data",
        type=str,
        help=(
            "Directory with {train, valid, test}.jsonl files or the name "
            "of a Hugging Face dataset with preference pairs"
        ),
    )
    parser.add_argument(
        "--fine-tune-type",
        type=str,
        choices=["lora", "dora", "full"],
        help="Type of fine-tuning to perform: lora, dora, or full.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adam", "adamw", "muon", "sgd", "adafactor"],
        default=None,
        help="Optimizer to use for training: adam, adamw, sgd, or adafactor.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        help="Number of layers to fine-tune. Default is 16, use -1 for all.",
    )
    parser.add_argument("--batch-size", type=int, help="Minibatch size.")
    parser.add_argument("--iters", type=int, help="Iterations to train for.")
    parser.add_argument(
        "--val-batches",
        type=int,
        help="Number of validation batches, -1 uses the entire validation set.",
    )
    parser.add_argument("--learning-rate", type=float, help="Adam learning rate.")
    parser.add_argument(
        "--steps-per-report",
        type=int,
        help="Number of training steps between loss reporting.",
    )
    parser.add_argument(
        "--steps-per-eval",
        type=int,
        help="Number of training steps between validations.",
    )
    parser.add_argument(
        "--resume-adapter-file",
        type=str,
        help="Load path to resume training from the given fine-tuned weights.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Save/load path for the fine-tuned weights.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        help="Save the model every N iterations.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Evaluate on the test set after training",
        default=None,
    )
    parser.add_argument(
        "--test-batches",
        type=int,
        help="Number of test set batches, -1 uses the entire test set.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        help="Maximum sequence length.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="A YAML configuration file with the training options",
    )
    parser.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help="Use gradient checkpointing to reduce memory use.",
        default=None,
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO temperature parameter (default: 0.1).",
    )
    parser.add_argument(
        "--reference-model",
        type=str,
        default=None,
        help="Path to reference model (defaults to initial policy model).",
    )
    parser.add_argument(
        "--reference-model-adapters",
        type=str,
        default=None,
        help="Path to adapters for the reference model.",
    )
    parser.add_argument(
        "--report-to",
        type=str,
        default=None,
        help="Services to report logs to ('wandb', 'swanlab', or 'wandb,swanlab').",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default=None,
        help="Project name for logging. Defaults to the name of the root directory.",
    )
    parser.add_argument("--seed", type=int, help="The PRNG seed")
    return parser


def load_preference_dataset(args, tokenizer):
    """
    Load preference datasets for DPO training.
    """
    data_path = Path(args.data)

    def load_subset(path):
        if not path.exists():
            return []
        with open(path, "r") as fid:
            data = [json.loads(l) for l in fid]
        return PreferenceDataset(
            data=data,
            tokenizer=tokenizer,
            max_length=args.max_seq_length,
        )

    if data_path.exists():
        # Load local dataset
        names = ("train", "valid", "test")
        train, valid, test = [load_subset(data_path / f"{n}.jsonl") for n in names]
    else:
        # Load from Hugging Face
        from datasets import load_dataset as hf_load_dataset

        print(f"Loading Hugging Face preference dataset {args.data}.")
        dataset = hf_load_dataset(args.data)

        # Convert to preference dataset format
        train_data = list(dataset["train"]) if "train" in dataset else []
        valid_data = list(dataset.get("validation", dataset.get("valid", [])))
        test_data = list(dataset.get("test", []))

        train = (
            PreferenceDataset(train_data, tokenizer, args.max_seq_length)
            if train_data
            else []
        )
        valid = (
            PreferenceDataset(valid_data, tokenizer, args.max_seq_length)
            if valid_data
            else []
        )
        test = (
            PreferenceDataset(test_data, tokenizer, args.max_seq_length)
            if test_data
            else []
        )

    if args.train and len(train) == 0:
        raise ValueError(
            "Training set not found or empty. Must provide training set for DPO fine-tuning."
        )
    if args.train and len(valid) == 0:
        raise ValueError(
            "Validation set not found or empty. Must provide validation set for DPO fine-tuning."
        )
    if args.test and len(test) == 0:
        raise ValueError(
            "Test set not found or empty. Must provide test set for evaluation."
        )

    return train, valid, test


def train_model(
    args,
    policy_model: nn.Module,
    reference_model: nn.Module,
    train_set,
    valid_set,
    training_callback: TrainingCallback = None,
):
    """
    Train a model using Direct Preference Optimization (DPO).

    This function follows the exact same model setup logic as lora.py
    to ensure compatibility with existing fine-tuning infrastructure.
    """
    mx.random.seed(args.seed)
    policy_model.freeze()
    if args.num_layers > len(policy_model.layers):
        raise ValueError(
            f"Requested to train {args.num_layers} layers "
            f"but the model only has {len(policy_model.layers)} layers."
        )

    # EXACT SAME LOGIC AS lora.py for model setup
    if args.fine_tune_type == "full":
        for l in policy_model.layers[-max(args.num_layers, 0) :]:
            l.unfreeze()
        args.lora_parameters = None
    elif args.fine_tune_type in ["lora", "dora"]:
        # Convert linear layers to lora/dora layers and unfreeze in the process
        linear_to_lora_layers(
            policy_model,
            args.num_layers,
            args.lora_parameters,
            use_dora=(args.fine_tune_type == "dora"),
        )
    else:
        raise ValueError(f"Received unknown fine-tune-type {args.fine_tune_type}")

    # Resume from weights if provided
    if args.resume_adapter_file is not None:
        print(f"Loading fine-tuned weights from {args.resume_adapter_file}")
        policy_model.load_weights(args.resume_adapter_file, strict=False)

    print_trainable_parameters(policy_model)

    adapter_path = Path(args.adapter_path)
    adapter_path.mkdir(parents=True, exist_ok=True)

    adapter_file = adapter_path / "adapters.safetensors"
    save_config(vars(args), adapter_path / "adapter_config.json")

    # Initialize training args - same as lora.py
    training_args = TrainingArgs(
        batch_size=args.batch_size,
        iters=args.iters,
        val_batches=args.val_batches,
        steps_per_report=args.steps_per_report,
        steps_per_eval=args.steps_per_eval,
        steps_per_save=args.save_every,
        adapter_file=adapter_file,
        max_seq_length=args.max_seq_length,
        grad_checkpoint=args.grad_checkpoint,
    )

    # Initialize the selected optimizer - same as lora.py
    lr = build_schedule(args.lr_schedule) if args.lr_schedule else args.learning_rate

    optimizer_name = args.optimizer.lower()
    optimizer_config = args.optimizer_config.get(optimizer_name, {})
    if optimizer_name == "adam":
        opt_class = optim.Adam
    elif optimizer_name == "adamw":
        opt_class = optim.AdamW
    elif optimizer_name == "muon":
        opt_class = optim.Muon
    elif optimizer_name == "sgd":
        opt_class = optim.SGD
    elif optimizer_name == "adafactor":
        opt_class = optim.Adafactor
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    opt = opt_class(learning_rate=lr, **optimizer_config)

    # Train model using DPO
    train_dpo(
        policy_model=policy_model,
        reference_model=reference_model,
        optimizer=opt,
        train_dataset=CacheDataset(train_set),
        val_dataset=CacheDataset(valid_set),
        args=training_args,
        beta=args.beta,
        training_callback=training_callback,
    )


def load_reference_model(args, policy_model=None):
    """
    Load the reference model for DPO training.
    If no reference model path is provided, use shared weights from policy model.
    """
    if args.reference_model:
        print(f"Loading reference model from {args.reference_model}")
        print(f"Reference model adapters: {args.reference_model_adapters}")
        reference_model, _ = load(
            args.reference_model,
            tokenizer_config={"trust_remote_code": True},
            adapter_path=args.reference_model_adapters,
        )
    else:
        if policy_model is not None:
            print("Using shared weights for reference model (memory efficient)")
            import copy

            reference_model = copy.deepcopy(policy_model)
            reference_model.freeze()
        else:
            print("Using policy model as reference model")
            reference_model, _ = load(
                args.model, tokenizer_config={"trust_remote_code": True}
            )

    return reference_model


def evaluate_model(args, policy_model: nn.Module, reference_model: nn.Module, test_set):
    """
    Evaluate the DPO-trained model on test set.
    """
    from .tuner.trainer import dpo_evaluate

    test_results = dpo_evaluate(
        policy_model=policy_model,
        reference_model=reference_model,
        dataset=CacheDataset(test_set),
        batch_size=args.batch_size,
        num_batches=args.test_batches,
        max_seq_length=args.max_seq_length,
        beta=args.beta,
    )

    print(
        f"Test loss {test_results['loss']:.3f}, "
        f"Test accuracy {test_results.get('accuracy', 0):.3f}, "
        f"Test margin {test_results.get('avg_margin', 0):.3f}."
    )


def run(args, training_callback: TrainingCallback = None):
    np.random.seed(args.seed)
    training_callback = get_reporting_callbacks(
        args.report_to,
        project_name=args.project_name,
        log_dir=args.adapter_path,
        config=vars(args),
    )

    print("Loading pretrained model")
    policy_model, tokenizer = load(
        args.model, tokenizer_config={"trust_remote_code": True}
    )

    print("Loading reference model")
    reference_model = load_reference_model(args, policy_model=policy_model)

    print("Loading preference datasets")
    train_set, valid_set, test_set = load_preference_dataset(args, tokenizer)

    if args.test and not args.train:
        # Allow testing without LoRA layers by providing empty path
        if args.adapter_path != "":
            load_adapters(policy_model, args.adapter_path)

    elif args.train:
        print("Training with DPO")
        train_model(
            args, policy_model, reference_model, train_set, valid_set, training_callback
        )
    else:
        raise ValueError("Must provide at least one of --train or --test")

    if args.test:
        print("Testing")
        evaluate_model(args, policy_model, reference_model, test_set)


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    parser = build_parser()
    args = parser.parse_args()
    config = args.config
    args = vars(args)
    if config:
        print("Loading configuration file", config)
        with open(config, "r") as file:
            config = yaml.load(file, yaml_loader)
        # Prefer parameters from command-line arguments
        for k, v in config.items():
            if args.get(k, None) is None:
                args[k] = v

    # Update defaults for unspecified parameters
    for k, v in CONFIG_DEFAULTS.items():
        if args.get(k, None) is None:
            args[k] = v
    run(types.SimpleNamespace(**args))


if __name__ == "__main__":
    print(
        "Calling `python -m mlx_lm.dpo...` directly is deprecated."
        " Use `mlx_lm.dpo...` or `python -m mlx_lm dpo ...` instead."
    )
    main()
