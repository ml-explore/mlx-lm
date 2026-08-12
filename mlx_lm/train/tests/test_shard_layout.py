# Copyright © 2025 Apple Inc.

import sys
import unittest
from pathlib import Path

from mlx_lm.train.data import s3

SHUFFLE_DIR = Path(__file__).resolve().parent.parent / "dataset_shuffle"


def shuffle_module():
    sys.path.insert(0, str(SHUFFLE_DIR))
    try:
        import shuffle_dataset

        return shuffle_dataset
    finally:
        sys.path.remove(str(SHUFFLE_DIR))


class TestShardLayout(unittest.TestCase):

    def setUp(self):
        self.shuffle = shuffle_module()

    def written_keys(self, output, num_workers, num_output_shards):
        """The keys stage 2 writes, straight from the shuffler's own helpers."""
        return [
            "{}{}/{}{}".format(
                self.shuffle._as_directory(output),
                self.shuffle._shard_name(worker, num_workers),
                self.shuffle._shard_name(shard, num_output_shards),
                self.shuffle.OUTPUT_EXT["jsonl"],
            )
            for worker in range(num_workers)
            for shard in range(num_output_shards)
        ]

    def test_loader_generates_exactly_what_the_shuffler_writes(self):
        for num_workers, num_output_shards in ((4, 8), (64, 32), (1024, 32)):
            with self.subTest(workers=num_workers, shards=num_output_shards):
                uri = "s3://bucket/corpus-shuffled/data/"
                _, expected = s3.split_uri(uri)
                written = self.written_keys(expected, num_workers, num_output_shards)

                _, generated = s3.shard_keys(
                    uri,
                    num_groups=num_workers,
                    shards_per_group=num_output_shards,
                    suffix=self.shuffle.OUTPUT_EXT["jsonl"],
                )
                self.assertEqual(sorted(generated), sorted(written))

    def test_loader_default_suffix_matches_the_shuffler(self):
        _, keys = s3.shard_keys("s3://b/p/", num_groups=1, shards_per_group=1)
        self.assertTrue(keys[0].endswith(self.shuffle.OUTPUT_EXT["jsonl"]))

    def test_shuffler_output_extension_is_a_readable_input(self):
        self.assertIn(self.shuffle.OUTPUT_EXT["jsonl"], self.shuffle.SUFFIXES["jsonl"])


if __name__ == "__main__":
    unittest.main()
