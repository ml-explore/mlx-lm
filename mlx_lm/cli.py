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
        "plugin",
        "awq",
        "dwq",
        "dynamic_quant",
        "gptq",
        "server",
        "server_v2",
        "upload",
        "share",
    )
    # The module of a subcommand, if it is not mlx_lm.<subcommand>.
    modules = {
        "awq": "quant.awq",
        "dwq": "quant.dwq",
        "dynamic_quant": "quant.dynamic_quant",
        "gptq": "quant.gptq",
        "plugin": "plugins.cli",
    }
    if len(sys.argv) < 2:
        raise ValueError(f"CLI requires a subcommand in {subcommands}")
    subcommand = sys.argv.pop(1)
    if subcommand in subcommands:
        module = modules.get(subcommand, subcommand)
        submodule = importlib.import_module(f"mlx_lm.{module}")
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
