# Copyright © 2025 Apple Inc.

"""Run under a launcher, e.g.

mlx.launch -n 8 -- python mlx_lm/train/tests/distributed_data_tests.py
"""

import gzip
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_lm.train import data
from mlx_lm.train.data import batching, s3

DOCS_PER_SHARD = 24


class Tokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [int(t) for t in text.split()]


def barrier(group):
    mx.eval(mx.distributed.all_sum(mx.array(1.0), group=group))


def fixture(group, num_shards):
    """One shard per file, every document tagged with its own id."""
    root = Path(tempfile.gettempdir()) / f"mlx_lm_train_data_{num_shards}"
    if group.rank() == 0:
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True)
        doc_id = 0
        for shard in range(num_shards):
            with gzip.open(root / f"part-{shard:03d}.jsonl.gz", "wt") as fid:
                for _ in range(DOCS_PER_SHARD):
                    text = " ".join([str(doc_id)] * (3 + doc_id % 7))
                    fid.write(json.dumps({"text": text}) + "\n")
                    doc_id += 1
    barrier(group)
    files = sorted(str(p) for p in root.glob("part-*.jsonl.gz"))
    return files, num_shards * DOCS_PER_SHARD


class TestDistributedData(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.group = mx.distributed.init()
        cls.rank, cls.size = cls.group.rank(), cls.group.size()
        # One shard count that divides the ranks and one that does not.
        cls.shard_counts = (cls.size, cls.size + 1)
        cls.fixtures = {n: fixture(cls.group, n) for n in cls.shard_counts}

    @classmethod
    def tearDownClass(cls):
        barrier(cls.group)
        if cls.group.rank() == 0:
            for n in cls.shard_counts:
                shutil.rmtree(
                    Path(tempfile.gettempdir()) / f"mlx_lm_train_data_{n}",
                    ignore_errors=True,
                )

    def hf_stream(self, files, shuffle_buffer=None):
        return data.load_hf(
            Tokenizer(),
            self.rank,
            self.size,
            dataset="json",
            data_files=files,
            shuffle_buffer=shuffle_buffer,
            seed=0,
        )

    def test_hf_ranks_partition_the_corpus(self):
        """Every document reaches exactly one rank, whatever the shard count."""
        for num_shards in self.shard_counts:
            files, num_docs = self.fixtures[num_shards]
            for shuffle_buffer in (None, 16):
                with self.subTest(num_shards=num_shards, shuffle_buffer=shuffle_buffer):
                    seen = np.zeros(num_docs, np.float32)
                    for doc in self.hf_stream(files, shuffle_buffer):
                        seen[doc["input_ids"][0]] += 1
                    counts = mx.distributed.all_sum(mx.array(seen), group=self.group)
                    mx.eval(counts)
                    self.assertEqual(counts.min().item(), 1.0)
                    self.assertEqual(counts.max().item(), 1.0)

    def test_hf_batches_on_every_rank(self):
        files, _ = self.fixtures[self.size]
        batches = list(
            batching.iterate_batches(
                self.hf_stream(files), context_size=7, batch_size=2, max_batches=3
            )
        )
        self.assertEqual(len(batches), 3)
        for batch in batches:
            self.assertEqual(batch["input_ids"].shape, (2, 8))
            self.assertEqual(batch["input_ids"].dtype, np.int32)
        agreed = mx.distributed.all_sum(
            mx.array(float(len(batches) == 3)), group=self.group
        )
        mx.eval(agreed)
        self.assertEqual(agreed.item(), float(self.size))

    def test_prefetch_survives_the_fork(self):
        """prefetch forks after the distributed backend is already up."""
        files, _ = self.fixtures[self.size]
        reference = list(
            batching.iterate_batches(
                self.hf_stream(files), context_size=7, batch_size=2, max_batches=3
            )
        )
        got = list(
            data.prefetch(
                batching.iterate_batches(
                    self.hf_stream(files), context_size=7, batch_size=2, max_batches=3
                )
            )
        )
        matched = len(got) == len(reference) and all(
            np.array_equal(a["input_ids"], b["input_ids"])
            for a, b in zip(reference, got)
        )
        agreed = mx.distributed.all_sum(mx.array(float(matched)), group=self.group)
        mx.eval(agreed)
        self.assertEqual(agreed.item(), float(self.size))

    def test_s3_ranks_partition_the_generated_keys(self):
        """No S3 access: the keys are generated, so this is pure arithmetic."""
        num_groups, per_group = 8, 32
        _, keys = s3.shard_keys("s3://bucket/corpus/data/", num_groups, per_group)
        index = {key: i for i, key in enumerate(keys)}

        mine = np.zeros(len(keys), np.float32)
        for key in keys[self.rank :: self.size]:
            mine[index[key]] += 1
        counts = mx.distributed.all_sum(mx.array(mine), group=self.group)
        mx.eval(counts)
        self.assertEqual(counts.min().item(), 1.0)
        self.assertEqual(counts.max().item(), 1.0)

    def test_s3_slices_stride_rather_than_block(self):
        _, keys = s3.shard_keys("s3://bucket/corpus/data/", 8, 32)
        groups = {k.split("/")[-2] for k in keys[self.rank :: self.size]}
        agreed = mx.distributed.all_sum(
            mx.array(float(len(groups) > 1)), group=self.group
        )
        mx.eval(agreed)
        self.assertEqual(agreed.item(), float(self.size))


if __name__ == "__main__":
    unittest.main()
