# Copyright © 2026 Apple Inc.

import importlib
import sys
import tempfile
import unittest
from pathlib import Path

from mlx_lm import plugins, utils
from mlx_lm.plugins import discovery
from mlx_lm.processors import load_processor

MODEL = "class Model:\n    pass\n\n\nclass ModelArgs:\n    pass\n"
DEFAULTS = 'TOKENIZER_DEFAULTS = {"chat_template_type": "test"}\n'
PROCESSOR = (
    "class Processor:\n"
    "    @classmethod\n"
    "    def from_config(cls, config):\n"
    "        return cls()\n"
)
TEST = {"model_type": "test"}


class TestPlugins(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.TemporaryDirectory()
        sys.path.insert(0, self.root.name)
        self.install(
            "test-lm",
            {
                "mlx_lm.models": {
                    "test": "test_lm.models.test",
                    "llama": "test_lm.models.llama",
                    "solo": "test_lm.solo",
                    "Solo-LM": "test_lm.solo",
                    "extras": "test_lm.solo [cuda]",
                    "ghost": "test_lm.missing",
                    "orphan": "no_such_package.model",
                },
                "mlx_lm.tool_parsers": {"test": "test_lm.tool_parsers.test"},
                "mlx_lm.chat_templates": {"test": "test_lm.chat_templates.test"},
            },
            {
                "test_lm/models/test/__init__.py": MODEL + DEFAULTS,
                "test_lm/models/test/processor.py": PROCESSOR,
                "test_lm/models/llama/__init__.py": MODEL,
                "test_lm/solo.py": MODEL,
                "test_lm/tool_parsers/test.py": "",
                "test_lm/chat_templates/test.py": "",
            },
        )

    def tearDown(self):
        sys.path.remove(self.root.name)
        for name in list(sys.modules):
            if name.split(".")[0] in ("test_lm", "other_lm"):
                del sys.modules[name]
        discovery._entry_points.cache_clear()
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
        discovery._entry_points.cache_clear()

    def test_new_model(self):
        package = utils.model_package(TEST)
        self.assertEqual(package, "test_lm.models.test")
        self.assertNotIn(package, sys.modules)
        model_class, _ = utils._get_classes(TEST)
        self.assertEqual(model_class.__module__, package)

    def test_builtin_model_wins(self):
        package = utils.model_package({"model_type": "llama"})
        self.assertEqual(package, "mlx_lm.models.llama")

    def test_broken_package_does_not_break_builtins(self):
        self.write_metadata("broken-lm", "[mlx_lm.models]\nno equals sign\n")
        package = utils.model_package({"model_type": "llama"})
        self.assertEqual(package, "mlx_lm.models.llama")

    def test_unknown_model(self):
        with self.assertRaises(ValueError):
            utils.model_package({"model_type": "unknown"})

    def test_processor(self):
        processor = load_processor({**TEST, "vision_config": {"patch_size": 16}})
        self.assertEqual(type(processor).__module__, "test_lm.models.test.processor")

    def test_tokenizer_defaults_follow_the_given_config(self):
        with tempfile.TemporaryDirectory() as path:
            (Path(path) / "config.json").write_text('{"model_type": "llama"}')
            defaults = utils._tokenizer_defaults(Path(path), TEST)
        self.assertEqual(defaults, {"chat_template_type": "test"})

    def test_module_without_processor(self):
        config = {"model_type": "solo", "vision_config": {"patch_size": 16}}
        self.assertIsNone(load_processor(config))

    def test_names_are_normalized(self):
        for model_type in ("solo_lm", "Solo-LM"):
            package = utils.model_package({"model_type": model_type})
            self.assertEqual(package, "test_lm.solo")

    def test_entry_point_with_extras(self):
        package = utils.model_package({"model_type": "extras"})
        self.assertEqual(package, "test_lm.solo")

    def test_entry_point_to_missing_module(self):
        for model_type in ("ghost", "orphan"):
            with self.assertRaisesRegex(ValueError, f"test-lm gives {model_type}"):
                utils.model_package({"model_type": model_type})

    def test_invalid_name(self):
        with self.assertRaisesRegex(ValueError, "not a valid identifier"):
            utils.model_package({"model_type": "a.b"})

    def test_unknown_kind(self):
        with self.assertRaisesRegex(ValueError, "is not one of"):
            plugins.module_name("layers", "test")

    def test_tool_parser_and_chat_template(self):
        for kind in ("tool_parsers", "chat_templates"):
            name = plugins.module_name(kind, "test")
            self.assertEqual(name, f"test_lm.{kind}.test")

    def test_two_packages_give_the_same_name(self):
        self.install("other-lm", {"mlx_lm.models": {"test": "other_lm.test"}}, {})
        with self.assertRaises(ValueError):
            utils.model_package(TEST)

    def test_entry_point_must_be_a_module(self):
        entry_points = {"mlx_lm.models": {"other": "other_lm.test:Model"}}
        self.install("other-lm", entry_points, {})
        with self.assertRaises(ValueError):
            utils.model_package({"model_type": "other"})


if __name__ == "__main__":
    unittest.main()
