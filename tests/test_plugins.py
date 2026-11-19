# Copyright © 2026 Apple Inc.

import importlib
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

from mlx_lm import plugins, utils
from mlx_lm.processors import load_processor

MODEL = "class Model:\n    pass\n\n\nclass ModelArgs:\n    pass\n"
PROCESSOR = (
    "class Processor:\n"
    "    @classmethod\n"
    "    def from_config(cls, config):\n"
    "        return cls()\n"
)
ACME = {"model_type": "acme"}


class TestPlugins(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.TemporaryDirectory()
        sys.path.insert(0, self.root.name)
        self.install(
            "acme_lm",
            {
                "mlx_lm.models": {
                    "acme": "acme_lm.models.acme",
                    "llama": "acme_lm.models.llama",
                },
                "mlx_lm.tool_parsers": {"acme": "acme_lm.tool_parsers.acme"},
                "mlx_lm.chat_templates": {"acme": "acme_lm.chat_templates.acme"},
            },
            {
                "acme_lm/models/acme/__init__.py": MODEL,
                "acme_lm/models/acme/processor.py": PROCESSOR,
                "acme_lm/models/llama/__init__.py": MODEL,
            },
        )

    def tearDown(self):
        sys.path.remove(self.root.name)
        for name in list(sys.modules):
            if name.split(".")[0] in ("acme_lm", "other_lm"):
                del sys.modules[name]
        plugins._entry_points.cache_clear()
        self.root.cleanup()

    def install(
        self,
        name: str,
        entry_points: dict[str, dict[str, str]],
        files: dict[str, str],
    ) -> None:
        """Write an installed distribution with its entry points and files."""
        root = Path(self.root.name)
        dist_info = root / f"{name}-0.dist-info"
        dist_info.mkdir()
        (dist_info / "METADATA").write_text(f"Name: {name}\nVersion: 0\n")
        lines = []
        for group, entries in entry_points.items():
            lines.append(f"[{group}]")
            lines.extend(f"{key} = {value}" for key, value in entries.items())
        (dist_info / "entry_points.txt").write_text("\n".join(lines) + "\n")
        for path, text in files.items():
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            (root / path).write_text(text)
        importlib.invalidate_caches()
        plugins._entry_points.cache_clear()

    def test_new_model(self):
        package = utils.model_package(ACME)
        self.assertEqual(package, "acme_lm.models.acme")
        self.assertNotIn(package, sys.modules)
        model_class, _ = utils._get_classes(ACME)
        self.assertEqual(model_class.__module__, package)

    def test_builtin_model_wins(self):
        # assertWarns reads sys.modules, which imports transformers' lazy modules.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            package = utils.model_package({"model_type": "llama"})
        self.assertEqual(package, "mlx_lm.models.llama")
        self.assertEqual(len(caught), 1)

    def test_unknown_model(self):
        with self.assertRaises(ValueError):
            utils.model_package({"model_type": "unknown"})

    def test_processor(self):
        processor = load_processor({**ACME, "vision_config": {"patch_size": 16}})
        self.assertEqual(type(processor).__module__, "acme_lm.models.acme.processor")

    def test_tool_parser_and_chat_template(self):
        for kind in ("tool_parsers", "chat_templates"):
            name = plugins.module_name(kind, "acme")
            self.assertEqual(name, f"acme_lm.{kind}.acme")

    def test_two_packages_give_the_same_name(self):
        self.install("other_lm", {"mlx_lm.models": {"acme": "other_lm.acme"}}, {})
        with self.assertRaises(ValueError):
            utils.model_package(ACME)

    def test_entry_point_must_be_a_module(self):
        entry_points = {"mlx_lm.models": {"other": "other_lm.acme:Model"}}
        self.install("other_lm", entry_points, {})
        with self.assertRaises(ValueError):
            utils.model_package({"model_type": "other"})


if __name__ == "__main__":
    unittest.main()
