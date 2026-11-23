# Copyright © 2026 Apple Inc.

import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mlx_lm import models, plugins, utils
from mlx_lm.processors import load_processor

MODEL = (
    "import mlx.nn as nn\n"
    'TOKENIZER_DEFAULTS = {"chat_template_type": "test"}\n'
    "class Model(nn.Module):\n"
    "    pass\n"
    "class ModelArgs:\n"
    "    @classmethod\n"
    "    def from_dict(cls, config):\n"
    "        return cls()\n"
    "class NotAModel:\n"
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
            "extras": "test_lm.solo:Model [cuda]",
            "ghost": "test_lm.missing:Model",
            "orphan": "no_such_package.model:Model",
            "no_attr": "test_lm.solo:Missing",
            "half": "test_lm.solo:Model",
            "split": "test_lm.solo:Model",
            "not_a_model": "test_lm.solo:NotAModel",
        }
        args = {name: "test_lm.solo:ModelArgs" for name in model}
        args.pop("half")
        args.pop("split")
        args["test"] = "test_lm.models.test:ModelArgs"
        self.install(
            "test-lm",
            {
                "mlx_lm.models": model,
                "mlx_lm.model_args": args,
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

    def test_invalid_name(self):
        with self.assertRaisesRegex(ValueError, "not a valid identifier"):
            utils._get_classes({"model_type": "a.b"})

    def test_entry_point_with_extras(self):
        model, _ = utils._get_classes({"model_type": "extras"})
        self.assertEqual(model.__module__, "test_lm.solo")

    def test_entry_point_must_name_an_object(self):
        values = ("other_lm.test", "other_lm.test:", "other-lm.t:Model", ".lm.t:Model")
        entry_points = {f"bad{i}": value for i, value in enumerate(values)}
        self.install("other-lm", {"mlx_lm.models": entry_points}, {})
        for model_type in entry_points:
            with self.assertRaisesRegex(ValueError, "of other-lm must name an object"):
                utils._get_classes({"model_type": model_type})

    def test_entry_point_to_missing_object(self):
        for model_type in ("ghost", "orphan", "no_attr"):
            with self.assertRaisesRegex(ValueError, f"test-lm gives {model_type}"):
                utils._get_classes({"model_type": model_type})

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

    def test_model_and_args_come_from_one_project(self):
        self.install(
            "other-lm",
            {"mlx_lm.model_args": {"split": "other_lm.args:ModelArgs"}},
            {"other_lm/args.py": MODEL},
        )
        with self.assertRaisesRegex(ValueError, "in mlx_lm.model_args"):
            utils._get_classes({"model_type": "half"})
        with self.assertRaisesRegex(
            ValueError, "one project, not test-lm and other-lm"
        ):
            utils._get_classes({"model_type": "split"})

    def test_model_must_be_a_module(self):
        with self.assertRaisesRegex(ValueError, "not an nn.Module"):
            utils._get_classes({"model_type": "not_a_model"})

    def test_tool_parser(self):
        self.assertEqual(plugins.load_tool_parser("test").__name__, "Parser")
        with self.assertRaisesRegex(ValueError, "needs tool_call_start"):
            plugins.load_tool_parser("not_a_parser")

    def test_plugin_has_no_processor_or_tokenizer_defaults(self):
        for config in (TEST, {"model_type": "unknown"}):
            self.assertIsNone(load_processor({**config, "vision_config": {"a": 1}}))
            self.assertIsNone(utils._tokenizer_defaults(Path(self.root.name), config))


if __name__ == "__main__":
    unittest.main()
