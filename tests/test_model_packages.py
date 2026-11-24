# Copyright © 2026 Apple Inc.

"""The shape of the model packages: their files, exports and remapping."""

import importlib
import pkgutil
import unittest
from pathlib import Path

import mlx_lm.models
from mlx_lm.models import MODEL_ARCHITECTURE_REMAPPING, MODEL_REMAPPING, model_package

MODELS = Path(mlx_lm.models.__file__).parent
PACKAGES: list[str] = sorted(
    info.name for info in pkgutil.iter_modules(mlx_lm.models.__path__) if info.ispkg
)
REQUIRED_FILES = {"__init__", "config", "model"}
OPTIONAL_FILES = {"checkpoint", "kernels", "layers", "processor", "vision"}
EXPORTS = {"Model", "ModelArgs"}
OPTIONAL_EXPORTS = {"TOKENIZER_DEFAULTS"}


def module_names(package: str) -> list[str]:
    """The modules of a package: the package itself, then its files."""
    names = [f"mlx_lm.models.{package}"]
    for path in sorted((MODELS / package).glob("*.py")):
        if path.stem != "__init__":
            names.append(f"mlx_lm.models.{package}.{path.stem}")
    return names


class TestModelPackages(unittest.TestCase):
    def test_files(self) -> None:
        for package in PACKAGES:
            with self.subTest(package=package):
                files = {path.stem for path in (MODELS / package).glob("*.py")}
                self.assertEqual(REQUIRED_FILES - files, set())
                self.assertEqual(files - REQUIRED_FILES - OPTIONAL_FILES, set())

    def test_exports(self) -> None:
        for package in PACKAGES:
            with self.subTest(package=package):
                module = importlib.import_module(f"mlx_lm.models.{package}")
                exported = set(module.__all__)
                self.assertEqual(exported - OPTIONAL_EXPORTS, EXPORTS)
                prefix = f"mlx_lm.models.{package}"
                self.assertEqual(module.ModelArgs.__module__, f"{prefix}.config")
                self.assertEqual(module.Model.__module__, f"{prefix}.model")
                self.assertIs(module.Model.model_args, module.ModelArgs)

    def test_all(self) -> None:
        for package in PACKAGES:
            for name in module_names(package):
                with self.subTest(module=name):
                    module = importlib.import_module(name)
                    self.assertEqual(list(module.__all__), sorted(module.__all__))
                    for public in module.__all__:
                        self.assertTrue(hasattr(module, public), public)

    def test_remapping_targets_are_packages(self) -> None:
        targets = list(MODEL_REMAPPING.values())
        targets += list(MODEL_ARCHITECTURE_REMAPPING.values())
        for target in targets:
            with self.subTest(target=target):
                self.assertIn(target, PACKAGES)
                # The lookup can apply the aliases twice, so no target is an alias.
                self.assertNotIn(target, MODEL_REMAPPING)

    def test_model_types_that_are_not_package_names(self) -> None:
        for model_type in (
            "Klear",
            "lfm2-vl",
            "lille-130m",
            "nemotron-nas",
            "phi-msft",
        ):
            with self.subTest(model_type=model_type):
                self.assertIn(model_package({"model_type": model_type}), PACKAGES)
        # Keys in another form match no model type.
        for key in MODEL_REMAPPING:
            self.assertEqual(key, key.lower().replace("-", "_"))


if __name__ == "__main__":
    unittest.main()
