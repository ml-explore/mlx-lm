# Copyright © 2025 Apple Inc.

import unittest
from pathlib import Path

from mlx_lm.train.utils import load_config

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def config_paths(kind):
    return sorted(p for p in (CONFIGS / kind).rglob("*.py") if "__" not in p.name)


class TestConfigs(unittest.TestCase):

    def test_model_configs_carry_no_dataset(self):
        for path in config_paths("models"):
            with self.subTest(config=path.name):
                config = load_config(str(path))
                self.assertIsNone(config.get("dataset"))
                self.assertIsNotNone(config.get("model"))
                self.assertIsNotNone(config.get("optimizer"))

    def test_data_configs_name_a_known_source(self):
        for path in config_paths("data"):
            with self.subTest(config=path.name):
                dataset = load_config(str(path))
                self.assertIn(dataset.source, ("s3", "hf"))
                if dataset.source == "s3":
                    self.assertTrue(dataset.uri.startswith("s3://"))
                else:
                    self.assertTrue(dataset.name)

    def test_every_pairing_composes(self):
        for model in config_paths("models"):
            for data in config_paths("data"):
                with self.subTest(model=model.name, data=data.name):
                    config = load_config(str(model))
                    config.dataset = load_config(str(data))
                    config.to_json()


if __name__ == "__main__":
    unittest.main()
