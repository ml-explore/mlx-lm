# Copyright © 2025 Apple Inc.

import importlib
import sys


def main():
    subcommands = (
        "benchmark",
        "cache_prompt",
        "chat",
        "convert",
        "evaluate",
        "fuse",
        "generate",
        "lora",
        "manage",
        "perplexity",
        "awq",
        "dwq",
        "dynamic_quant",
        "gptq",
        "server",
        "upload",
        "share",
    )
    subpackages = {
        "awq": "quant",
        "dwq": "quant",
        "dynamic_quant": "quant",
        "gptq": "quant",
    }
    if len(sys.argv) < 2:
        raise ValueError(f"CLI requires a subcommand in {subcommands}")
    subcommand = sys.argv.pop(1)
    if subcommand in subcommands:
        if subpackage := subpackages.get(subcommand):
            subcommand = f"{subpackage}.{subcommand}"
        submodule = importlib.import_module(f"mlx_lm.{subcommand}")
        submodule.main()
    elif subcommand == "--version":
        from mlx_lm import __version__

        print(__version__)
    elif subcommand in ("-h", "--help"):
        print(f"The supported subcommands are {subcommands}")
        print()
        print(
            "For help on an individual subcommand, pass --help "
            "to the subcommand. For example: mlx_lm.generate --help"
        )
    else:
        raise ValueError(f"CLI requires a subcommand in {subcommands}")
