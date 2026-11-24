# Copyright © 2026 Apple Inc.

import importlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from mlx_lm import models, plugins, utils
from mlx_lm.processors import load_processor

MODEL = (
    "import mlx.nn as nn\n"
    'TOKENIZER_DEFAULTS = {"chat_template_type": "test"}\n'
    "class ModelArgs:\n"
    "    @classmethod\n"
    "    def from_dict(cls, config):\n"
    "        return cls()\n"
    "class Model(nn.Module):\n"
    "    model_args = ModelArgs\n"
    "class NoArgs(nn.Module):\n"
    "    pass\n"
)
PARSER = (
    "class Parser:\n"
    '    tool_call_start = "<call>"\n'
    '    tool_call_end = "</call>"\n'
    "    @staticmethod\n"
    "    def parse_tool_call(text, tools):\n"
    '        return {"name": text}\n'
    "class NotAParser:\n"
    "    pass\n"
)
TEST = {"model_type": "test"}


class TestPlugins(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.TemporaryDirectory()
        sys.path.insert(0, self.root.name)
        model = {
            "test": "test_lm.models.test:Model",
            "llama": "test_lm.models.test:Model",
            "Solo-LM": "test_lm.solo:Model",
            "no_args": "test_lm.solo:NoArgs",
        }
        self.install(
            "test-lm",
            {
                "mlx_lm.models": model,
                "mlx_lm.tool_parsers": {
                    "test": "test_lm.parsers:Parser",
                    "not_a_parser": "test_lm.parsers:NotAParser",
                },
            },
            {
                "test_lm/models/test/__init__.py": MODEL,
                "test_lm/solo.py": MODEL,
                "test_lm/parsers.py": PARSER,
            },
        )

    def tearDown(self):
        sys.path.remove(self.root.name)
        for name in list(sys.modules):
            if name.split(".")[0] in ("test_lm", "other_lm"):
                del sys.modules[name]
        self.root.cleanup()

    def install(
        self,
        name: str,
        entry_points: dict[str, dict[str, str]],
        files: dict[str, str],
    ) -> None:
        """Write an installed project with its entry points and files."""
        root = Path(self.root.name)
        for path, text in files.items():
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            (root / path).write_text(text)
        lines = []
        for group, entries in entry_points.items():
            lines.append(f"[{group}]")
            lines.extend(f"{key} = {value}" for key, value in entries.items())
        self.write_metadata(name, "\n".join(lines) + "\n")

    def write_metadata(self, name: str, entry_points_text: str) -> None:
        """Write the metadata of an installed project, as a wheel does."""
        dist_info = Path(self.root.name) / f"{name.replace('-', '_')}-0.dist-info"
        dist_info.mkdir()
        (dist_info / "METADATA").write_text(f"Name: {name}\nVersion: 0\n")
        (dist_info / "entry_points.txt").write_text(entry_points_text)
        importlib.invalidate_caches()

    def test_new_model(self):
        model, args = utils._get_classes(TEST)
        self.assertEqual(model.__module__, "test_lm.models.test")
        self.assertEqual(args.__module__, "test_lm.models.test")

    def test_builtin_model_wins(self):
        model, _ = utils._get_classes({"model_type": "llama"})
        self.assertEqual(model.__module__, "mlx_lm.models.llama.model")

    def test_broken_package_does_not_break_builtins(self):
        self.write_metadata("broken-lm", "[mlx_lm.models]\nno equals sign\n")
        model, _ = utils._get_classes({"model_type": "llama"})
        self.assertEqual(model.__module__, "mlx_lm.models.llama.model")

    def test_folder_without_init_is_not_a_builtin(self):
        # Left behind by a branch switch.
        leftover = Path(self.root.name) / "leftover"
        (leftover / "test" / "__pycache__").mkdir(parents=True)
        path = [*models.__path__, str(leftover)]
        with mock.patch.object(models, "__path__", path):
            model, _ = utils._get_classes(TEST)
        self.assertEqual(model.__module__, "test_lm.models.test")

    def test_unknown_model(self):
        with self.assertRaises(ValueError):
            utils._get_classes({"model_type": "unknown"})

    def test_names_are_normalized(self):
        for model_type in ("solo_lm", "Solo-LM"):
            model, _ = utils._get_classes({"model_type": model_type})
            self.assertEqual(model.__module__, "test_lm.solo")

    def test_two_entry_points_give_the_same_name(self):
        entry_points = {
            "test": "other_lm.test:Model",
            "dup": "other_lm.a:Model",
            "DUP": "other_lm.b:Model",
        }
        self.install("other-lm", {"mlx_lm.models": entry_points}, {})
        cases = {
            "test": "test from other-lm, test from test-lm",
            "dup": "DUP from other-lm, dup from other-lm",
        }
        for model_type, sources in cases.items():
            with self.assertRaisesRegex(ValueError, sources):
                utils._get_classes({"model_type": model_type})

    def test_tool_parser(self):
        self.assertEqual(plugins.load_tool_parser("test").__name__, "Parser")

    def test_plugin_has_no_processor_or_tokenizer_defaults(self):
        for config in (TEST, {"model_type": "unknown"}):
            self.assertIsNone(load_processor({**config, "vision_config": {"a": 1}}))
            self.assertIsNone(utils._tokenizer_defaults(Path(self.root.name), config))

    def run_command(self, *args: str) -> tuple[int, str]:
        """The exit code and output of ``mlx_lm plugins`` with ``args``."""
        output = io.StringIO()
        code = 0
        argv = ["mlx_lm.plugins", *args]
        with mock.patch.object(sys, "argv", argv), redirect_stdout(output):
            try:
                plugins.main()
            except SystemExit as error:
                code = error.code
        return code, output.getvalue()

    def install_other_lm(self) -> None:
        """Install entries that an alias shadows or that give one name; none load."""
        entry_points = {
            "mistral": "other_lm.a:Model",
            "dup": "other_lm.a:Model",
            "DUP": "other_lm.b:Model",
        }
        self.install("other-lm", {"mlx_lm.models": entry_points}, {})

    def test_list_command(self):
        code, output = self.run_command("list")
        self.assertEqual(code, 0)
        project = (
            "test-lm 0\n"
            "  models: llama, no_args, Solo-LM, test\n"
            "  tool_parsers: not_a_parser, test\n"
        )
        self.assertIn(project, output)

        code, output = self.run_command("list", "--json")
        self.assertEqual(code, 0)
        project = {
            "name": "test-lm",
            "version": "0",
            "models": ["llama", "no_args", "Solo-LM", "test"],
            "tool_parsers": ["not_a_parser", "test"],
        }
        self.assertIn(project, json.loads(output))

    def test_check_command(self):
        self.install_other_lm()
        code, output = self.run_command("check")
        self.assertEqual(code, 1)
        for problem in (
            "] mistral = other_lm.a:Model (other-lm 0): shadowed by built-in",
            "] dup = other_lm.a:Model (other-lm 0): 2 entries give dup",
            "] no_args = test_lm.solo:NoArgs (test-lm 0): no model_args with from_dict",
            "] not_a_parser = test_lm.parsers:NotAParser (test-lm 0): parse_tool_call",
        ):
            self.assertIn(problem, output)
        self.assertEqual(
            self.run_command("check", "test", "solo-lm"), (0, "No problems found.\n")
        )


if __name__ == "__main__":
    unittest.main()
