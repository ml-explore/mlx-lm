# Copyright © 2023-2024 Apple Inc.

import copy
import glob
import importlib
import json
import logging
import os
import shutil
from pathlib import Path
from textwrap import dedent
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Tuple,
    Type,
    Union,
)

import mlx.core as mx
import mlx.nn as nn

if os.getenv("MLXLM_USE_MODELSCOPE", "False").lower() == "true":
    try:
        from modelscope import snapshot_download
    except ImportError:
        raise ImportError(
            "Please run `pip install modelscope` to activate the ModelScope."
        )
else:
    from huggingface_hub import snapshot_download

from mlx.utils import tree_flatten, tree_reduce
from transformers import PreTrainedTokenizer

# Local imports
from .tokenizer_utils import TokenizerWrapper, load_tokenizer
from .tuner.utils import dequantize as dequantize_model
from .tuner.utils import load_adapters, nparams

# Constants
MODEL_REMAPPING = {
    "mistral": "llama",  # mistral is compatible with llama
    "phi-msft": "phixtral",
    "falcon_mamba": "mamba",
}

MAX_FILE_SIZE_GB = 5


def _get_classes(config: dict):
    """
    Retrieve the model and model args classes based on the configuration.

    Args:
        config (dict): The model configuration.

    Returns:
        A tuple containing the Model class and the ModelArgs class.
    """
    model_type = config["model_type"]
    model_type = MODEL_REMAPPING.get(model_type, model_type)
    try:
        arch = importlib.import_module(f"mlx_lm.models.{model_type}")
    except ImportError:
        msg = f"Model type {model_type} not supported."
        logging.error(msg)
        raise ValueError(msg)

    return arch.Model, arch.ModelArgs


def compute_bits_per_weight(model):
    model_bytes = tree_reduce(
        lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc, model, 0
    )
    leaf_modules = tree_flatten(
        model.leaf_modules(), is_leaf=lambda m: isinstance(m, nn.Module)
    )
    model_params = sum(nparams(m) for _, m in leaf_modules)
    return model_bytes * 8 / model_params


def get_model_path(path_or_hf_repo: str, revision: Optional[str] = None) -> Path:
    """
    Ensures the model is available locally. If the path does not exist locally,
    it is downloaded from the Hugging Face Hub.

    Args:
        path_or_hf_repo (str): The local path or Hugging Face repository ID of the model.
        revision (str, optional): A revision id which can be a branch name, a tag, or a commit hash.

    Returns:
        Path: The path to the model.
    """
    model_path = Path(path_or_hf_repo)

    if not model_path.exists():
        model_path = Path(
            snapshot_download(
                path_or_hf_repo,
                revision=revision,
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.py",
                    "tokenizer.model",
                    "*.tiktoken",
                    "tiktoken.model",
                    "*.txt",
                    "*.jsonl",
                ],
            )
        )
    return model_path


def load_config(model_path: Path) -> dict:
    try:
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        logging.error(f"Config file not found in {model_path}")
        raise
    return config


def load_model(
    model_path: Path,
    lazy: bool = False,
    strict: bool = True,
    model_config: dict = {},
    get_model_classes: Callable[[dict], Tuple[Type[nn.Module], Type]] = _get_classes,
) -> nn.Module:
    """
    Load and initialize the model from a given path.

    Args:
        model_path (Path): The path to load the model from.
        lazy (bool): If False eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``
        strict (bool): Whether or not to raise an exception if weights don't
            match. Default: ``True``
        model_config (dict, optional): Optional configuration parameters for the
            model. Defaults to an empty dictionary.
        get_model_classes (Callable[[dict], Tuple[Type[nn.Module], Type]], optional):
            A function that returns the model class and model args class given a config.
            Defaults to the ``_get_classes`` function.

    Returns:
        nn.Module: The loaded and initialized model.

    Raises:
        FileNotFoundError: If the weight files (.safetensors) are not found.
        ValueError: If the model class or args class are not found or cannot be instantiated.
    """
    config = load_config(model_path)
    config.update(model_config)

    weight_files = glob.glob(str(model_path / "model*.safetensors"))

    if not weight_files:
        # Try weight for back-compat
        weight_files = glob.glob(str(model_path / "weight*.safetensors"))

    if not weight_files and strict:
        logging.error(f"No safetensors found in {model_path}")
        raise FileNotFoundError(f"No safetensors found in {model_path}")

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    model_class, model_args_class = get_model_classes(config=config)

    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)

    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    if (quantization := config.get("quantization", None)) is not None:

        def class_predicate(p, m):
            # Handle custom per layer quantizations
            if p in config["quantization"]:
                return config["quantization"][p]
            if not hasattr(m, "to_quantized"):
                return False
            # Handle legacy models which may not have everything quantized
            return f"{p}.scales" in weights

        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            class_predicate=class_predicate,
        )

    model.load_weights(list(weights.items()), strict=strict)

    if not lazy:
        mx.eval(model.parameters())

    model.eval()
    return model, config


def load(
    path_or_hf_repo: str,
    tokenizer_config={},
    model_config={},
    adapter_path: Optional[str] = None,
    lazy: bool = False,
) -> Tuple[nn.Module, TokenizerWrapper]:
    """
    Load the model and tokenizer from a given path or a huggingface repository.

    Args:
        path_or_hf_repo (Path): The path or the huggingface repository to load the model from.
        tokenizer_config (dict, optional): Configuration parameters specifically for the tokenizer.
            Defaults to an empty dictionary.
        model_config(dict, optional): Configuration parameters specifically for the model.
            Defaults to an empty dictionary.
        adapter_path (str, optional): Path to the LoRA adapters. If provided, applies LoRA layers
            to the model. Default: ``None``.
        lazy (bool): If ``False`` eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``
    Returns:
        Tuple[nn.Module, TokenizerWrapper]: A tuple containing the loaded model and tokenizer.

    Raises:
        FileNotFoundError: If config file or safetensors are not found.
        ValueError: If model class or args class are not found.
    """
    model_path = get_model_path(path_or_hf_repo)

    model, config = load_model(model_path, lazy)
    if adapter_path is not None:
        model = load_adapters(model, adapter_path)
        model.eval()
    tokenizer = load_tokenizer(
        model_path, tokenizer_config, eos_token_ids=config.get("eos_token_id", None)
    )

    return model, tokenizer


def fetch_from_hub(
    model_path: Path, lazy: bool = False
) -> Tuple[nn.Module, dict, PreTrainedTokenizer]:
    model, config = load_model(model_path, lazy)
    tokenizer = load_tokenizer(
        model_path, eos_token_ids=config.get("eos_token_id", None)
    )
    return model, config, tokenizer


def make_shards(weights: dict, max_file_size_gb: int = MAX_FILE_SIZE_GB) -> list:
    """
    Splits the weights into smaller shards.

    Args:
        weights (dict): Model weights.
        max_file_size_gb (int): Maximum size of each shard in gigabytes.

    Returns:
        list: List of weight shards.
    """
    max_file_size_bytes = max_file_size_gb << 30
    shards = []
    shard, shard_size = {}, 0
    for k, v in weights.items():
        if shard_size + v.nbytes > max_file_size_bytes:
            shards.append(shard)
            shard, shard_size = {}, 0
        shard[k] = v
        shard_size += v.nbytes
    shards.append(shard)
    return shards


def create_model_card(path: Union[str, Path], hf_path: Union[str, Path]):
    """
    Uploads the model to Hugging Face hub.

    Args:
        path (Union[str, Path]): Local path to the model.
        hf_path (Union[str, Path]): Path to the original Hugging Face model.
    """
    from huggingface_hub import ModelCard

    card = ModelCard.load(hf_path)
    card.data.library_name = "mlx"
    card.data.pipeline_tag = "text-generation"
    if card.data.tags is None:
        card.data.tags = ["mlx"]
    elif "mlx" not in card.data.tags:
        card.data.tags += ["mlx"]
    card.data.base_model = str(hf_path)
    card.text = ""
    card.save(os.path.join(path, "README.md"))


def upload_to_hub(path: str, upload_repo: str):
    """
    Uploads the model to Hugging Face hub.

    Args:
        path (str): Local path to the model.
        upload_repo (str): Name of the HF repo to upload to.
    """
    from huggingface_hub import HfApi, ModelCard, logging

    from . import __version__

    logging.set_verbosity_info()
    card_path = Path(path) / "README.md"
    card = ModelCard.load(card_path)
    hf_path = card.data.base_model
    card.text = dedent(
        f"""
        # {upload_repo}

        This model [{upload_repo}](https://huggingface.co/{upload_repo}) was
        converted to MLX format from [{hf_path}](https://huggingface.co/{hf_path})
        using mlx-lm version **{__version__}**.

        ## Use with mlx

        ```bash
        pip install mlx-lm
        ```

        ```python
        from mlx_lm import load, generate

        model, tokenizer = load("{upload_repo}")

        prompt = "hello"

        if tokenizer.chat_template is not None:
            messages = [{{"role": "user", "content": prompt}}]
            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )

        response = generate(model, tokenizer, prompt=prompt, verbose=True)
        ```
        """
    )
    card.save(card_path)

    api = HfApi()
    api.create_repo(repo_id=upload_repo, exist_ok=True)
    api.upload_large_folder(
        folder_path=path,
        repo_id=upload_repo,
        repo_type="model",
    )
    print(f"Upload successful, go to https://huggingface.co/{upload_repo} for details.")


def save_weights(
    save_path: Union[str, Path],
    weights: Dict[str, Any],
    *,
    donate_weights: bool = False,
) -> None:
    """Save model weights into specified directory."""
    if isinstance(save_path, str):
        save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    shards = make_shards(weights)
    shards_count = len(shards)
    shard_file_format = (
        "model-{:05d}-of-{:05d}.safetensors"
        if shards_count > 1
        else "model.safetensors"
    )

    total_size = sum(v.nbytes for v in weights.values())
    index_data = {"metadata": {"total_size": total_size}, "weight_map": {}}

    # Write the weights and make sure no references are kept other than the
    # necessary ones
    if donate_weights:
        weights.clear()
        del weights

    for i in range(len(shards)):
        shard = shards[i]
        shards[i] = None
        shard_name = shard_file_format.format(i + 1, shards_count)
        shard_path = save_path / shard_name

        mx.save_safetensors(str(shard_path), shard, metadata={"format": "mlx"})

        for weight_name in shard.keys():
            index_data["weight_map"][weight_name] = shard_name
        del shard

    index_data["weight_map"] = {
        k: index_data["weight_map"][k] for k in sorted(index_data["weight_map"])
    }

    with open(save_path / "model.safetensors.index.json", "w") as f:
        json.dump(
            index_data,
            f,
            indent=4,
        )


def quantize_model(
    model: nn.Module,
    config: dict,
    q_group_size: int,
    q_bits: int,
    quant_predicate: Optional[
        Callable[[str, nn.Module, dict], Union[bool, dict]]
    ] = None,
) -> Tuple:
    """
    Applies quantization to the model weights.

    Args:
        model (nn.Module): The model to be quantized.
        config (dict): Model configuration.
        q_group_size (int): Group size for quantization.
        q_bits (int): Bits per weight for quantization.
        quant_predicate (Callable): A callable that decides how
            to quantize each layer based on the path.
            Accepts the layer `path`, the `module` and the model `config`.
            Returns either a bool to signify quantize/no quantize or
            a dict of quantization parameters to pass to `to_quantized`.

    Returns:
        Tuple: Tuple containing quantized weights and config.
    """
    if "quantization" in config:
        raise ValueError("Cannot quantize already quantized model")
    quantized_config = copy.deepcopy(config)
    quantized_config["quantization"] = {"group_size": q_group_size, "bits": q_bits}

    # Add any custom quantization parameters to the config as we go
    def _class_predicate(p, m):
        bool_or_params = quant_predicate(p, m, config)
        quantized_config["quantization"][p] = bool_or_params
        return bool_or_params

    nn.quantize(
        model,
        q_group_size,
        q_bits,
        class_predicate=_class_predicate if quant_predicate else None,
    )
    # support hf model tree #957
    quantized_config["quantization_config"] = quantized_config["quantization"]
    quantized_weights = dict(tree_flatten(model.parameters()))

    bpw = compute_bits_per_weight(model)
    print(f"[INFO] Quantized model with {bpw:.3f} bits per weight.")

    return quantized_weights, quantized_config


def save_config(
    config: dict,
    config_path: Union[str, Path],
) -> None:
    """Save the model configuration to the ``config_path``.

    The final configuration will be sorted before saving for better readability.

    Args:
        config (dict): The model configuration.
        config_path (Union[str, Path]): Model configuration file path.
    """
    # Clean unused keys
    config.pop("_name_or_path", None)
    config.pop("vision_config", None)

    # sort the config for better readability
    config = dict(sorted(config.items()))

    # write the updated config to the config_path (if provided)
    with open(config_path, "w") as fid:
        json.dump(config, fid, indent=4)


def save(
    dst_path: Union[str, Path],
    src_path: Union[str, Path],
    weights: Dict[str, mx.array],
    tokenizer: TokenizerWrapper,
    config: Dict[str, Any],
    hf_repo: Optional[str] = None,
    donate_weights: bool = True,
):
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    save_weights(dst_path, weights, donate_weights=True)
    save_config(config, config_path=dst_path / "config.json")
    tokenizer.save_pretrained(dst_path)

    for p in ["*.py", "generation_config.json"]:
        for file in glob.glob(str(src_path / p)):
            shutil.copy(file, dst_path)

    if hf_repo is not None:
        create_model_card(dst_path, hf_repo)


def common_prefix_len(list1, list2):
    """
    Calculates the length of the common prefix of two lists.

    Args:
        list1: The first list of strings.
        list2: The second list of strings.

    Returns:
        The length of the common prefix. Returns 0 if lists are empty
        or do not match at the first element.
    """
    # Determine the maximum possible length of the common prefix
    min_len = min(len(list1), len(list2))

    # Iterate up to the length of the shorter list
    for i in range(min_len):
        if list1[i] != list2[i]:
            # Mismatch found, the common prefix length is the current index
            return i

    # No mismatch found within the bounds of the shorter list,
    # so the common prefix length is the length of the shorter list.
    return min_len
