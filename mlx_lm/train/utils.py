# Copyright © 2025 Apple Inc.

import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path

import ml_collections
import mlx.core as mx
from mlx.utils import tree_map

CONFIG_ROOT = Path(__file__).resolve().parent / "configs"


def config_path(name):
    """A path to a config: as given, or a name under the bundled configs."""
    path = Path(name)
    if path.suffix == ".py" or path.is_dir():
        return str(path)
    resolved = CONFIG_ROOT / f"{name}.py"
    if not resolved.exists():
        raise SystemExit(f"no config at {resolved}")
    return str(resolved)


def load_config(config_path):
    if config_path.endswith(".py"):
        import sys

        path = Path(config_path).resolve()
        if str(path.parent) not in sys.path:
            sys.path.append(str(path.parent))
        name = "_config_" + "_".join(path.with_suffix("").parts[-3:])
        spec = importlib.util.spec_from_file_location(name, path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        return config_module.get_config()
    else:
        with open(Path(config_path) / "config.json", "r") as fid:
            return ml_collections.ConfigDict(json.load(fid))


def save_config(dirname, config):
    with open(Path(dirname) / "config.json", "w") as fid:
        fid.write(config.to_json(indent=4))


def load_tokenizer(path):

    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    import transformers

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return transformers.AutoTokenizer.from_pretrained(path)


def build_model(config):

    arch = importlib.import_module(f"mlx_lm.train.{config.model_type}")
    accepted = inspect.signature(arch.ModelArgs).parameters
    model_args = arch.ModelArgs(
        **{k: v for k, v in config.to_dict().items() if k in accepted}
    )
    model = arch.Model(model_args)

    if hasattr(model, "init_weights"):
        model.init_weights()
    return model


def grad_checkpoint(layer, dtype=None):
    """Recompute ``layer``'s forward during the backward pass.

    ``dtype`` casts the layer's parameters inside the checkpointed region, so the
    cast itself is recomputed rather than held.
    """
    fn = type(layer).__call__

    def checkpointed_fn(model, *args, **kwargs):
        def inner_fn(params, *args, **kwargs):
            if dtype is not None:
                params = tree_map(lambda x: x.astype(dtype), params)
            model.update(params)
            return fn(model, *args, **kwargs)

        return mx.checkpoint(inner_fn)(model.trainable_parameters(), *args, **kwargs)

    type(layer).__call__ = checkpointed_fn
