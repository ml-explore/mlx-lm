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
        "quant.awq",
        "quant.dwq",
        "quant.dynamic_quant",
        "quant.gptq",
        "server",
        "upload",
    )
    if len(sys.argv) < 2:
        raise ValueError(f"CLI requires a subcommand in {subcommands}")
    subcommand = sys.argv.pop(1).replace("mlx_lm.", "")

    if subcommand not in subcommands and subcommand not in ("--version", "-h", "--help"):
        subcommand = next((s for s in subcommands if s.endswith(f".{subcommand}")), subcommand)

    if subcommand in subcommands:
        submodule = importlib.import_module(f"mlx_lm.{subcommand}")
        submodule.main()
    elif subcommand == "--version":
        from mlx_lm import __version__

        print(__version__)
    elif subcommand in ("-h", "--help"):
        print("The supported subcommands are:\n")
        w = max(map(len, subcommands)) + 2
        for i in range(0, len(subcommands), 3):
            print(" ", *(f"{s:{w}}" for s in subcommands[i : i + 3]))
        print(
            "\nFor help on an individual subcommand, pass --help "
            "to the subcommand. For example: mlx_lm.generate --help"
        )
    else:
        raise ValueError(f"CLI requires a subcommand in {subcommands}")
