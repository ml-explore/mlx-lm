# Copyright © 2025 Apple Inc.

import gzip
import io
import json
import os
import tempfile
import unittest

import numpy as np
from botocore.exceptions import ClientError

from mlx_lm.train import data
from mlx_lm.train.data import batching, s3


class Tokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [int(t) for t in text.split()]


def documents(num_files=4, per_file=8):
    """One token per position, numbered from 1, so packing is checkable."""
    token = 1
    out = []
    for shard in range(num_files):
        lines = []
        for doc in range(per_file):
            n = 3 + (shard * per_file + doc) % 7
            lines.append(" ".join(str(token + i) for i in range(n)))
            token += n
        out.append(lines)
    return out, token - 1


class TestBatching(unittest.TestCase):
    """iterate_batches, shared by both sources."""

    def stream(self, lines_per_file, file_name="f"):
        return s3.tokenized_data(
            Tokenizer(),
            (
                {"text": line, "file_name": "%s%d" % (file_name, f), "sample_idx": i}
                for f, lines in enumerate(lines_per_file)
                for i, line in enumerate(lines)
            ),
        )

    def batches(self, lines, num_batches=None, **kw):
        return list(
            batching.iterate_batches(
                self.stream(lines),
                context_size=7,
                batch_size=2,
                max_batches=num_batches,
                **kw,
            )
        )

    def test_packing(self):
        lines, num_tokens = documents()
        got = self.batches(lines, 3)
        self.assertEqual(len(got), 3)
        for batch in got:
            self.assertEqual(batch["input_ids"].shape, (2, 8))
            self.assertEqual(batch["input_ids"].dtype, np.int32)

        tokens = np.concatenate([b["input_ids"].ravel() for b in got])
        self.assertIn(Tokenizer.eos_token_id, tokens)
        body = tokens[tokens != Tokenizer.eos_token_id]
        self.assertTrue(np.all(np.diff(body) == 1))

    def test_batches_do_not_share_a_buffer(self):
        lines, _ = documents()
        got = self.batches(lines, 3)
        self.assertFalse(
            np.array_equal(got[0]["input_ids"], got[1]["input_ids"]),
            "each batch must own its buffer; reshape returns a view",
        )

    def test_exhaustion_drops_the_partial_batch(self):
        lines, num_tokens = documents()
        got = self.batches(lines)
        self.assertEqual(len(got), (num_tokens + 4 * 8) // 16)
        for batch in got:
            self.assertEqual(batch["input_ids"].shape, (2, 8))

    def test_state_names_a_position(self):
        lines, _ = documents()
        got = self.batches(lines, 3)
        state = json.loads(json.dumps(got[-1]["_data_state"]))
        self.assertEqual(state["batch_idx"], 3)
        self.assertTrue(state["file_name"].startswith("f"))
        self.assertGreaterEqual(state["sample_idx"], 0)

    def test_resume_from_d_next(self):
        lines, _ = documents()
        reference = self.batches(lines, 6)
        state = json.loads(json.dumps(reference[2]["_data_state"]))

        # Replay from the recorded document, carrying the split tail.
        tail = self.stream(lines)
        for _ in range(state["sample_idx"] + 8 * int(state["file_name"][1:])):
            next(tail)
        resumed = list(
            batching.iterate_batches(
                tail,
                context_size=7,
                batch_size=2,
                max_batches=3,
                resume_d_next=state["d_next"],
            )
        )
        self.assertEqual(len(resumed), 3)
        for expected, got in zip(reference[3:], resumed):
            self.assertTrue(np.array_equal(expected["input_ids"], got["input_ids"]))

    def test_prefetch(self):
        lines, _ = documents()
        reference = self.batches(lines, 4)
        got = list(
            data.prefetch(
                batching.iterate_batches(
                    self.stream(lines), context_size=7, batch_size=2, max_batches=4
                )
            )
        )
        self.assertEqual(len(got), 4)
        for expected, batch in zip(reference, got):
            self.assertTrue(np.array_equal(expected["input_ids"], batch["input_ids"]))


class FakeS3:
    """Just enough of a boto3 s3 client: paginated list plus streaming get."""

    def __init__(self, objects):
        self.objects = objects
        self.gets = []
        self.manifest = None

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "missing"}}, "GetObject"
            )
        if Key.endswith(s3.MANIFEST_NAME):
            return {"Body": io.BytesIO(json.dumps(self.manifest).encode())}
        self.gets.append(Key)
        raw = b"".join(
            json.dumps({"text": t}).encode() + b"\n" for t in self.objects[Key]
        )
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as fid:
            fid.write(raw)
        return {"Body": io.BytesIO(buf.getvalue())}


def fake_objects(num_groups=4, per_group=8, docs=3):
    return {
        "corpus/data/{:05d}-of-{:05d}/{:05d}-of-{:05d}.json.gz".format(
            g, num_groups, s, per_group
        ): ["%d %d %d" % (g, s, d) for d in range(docs)]
        for g in range(num_groups)
        for s in range(per_group)
    }


def fake_objects_varlen(num_groups=4, per_group=8, docs=5):
    out = {}
    token = 1
    for g in range(num_groups):
        for s in range(per_group):
            key = "corpus/data/{:05d}-of-{:05d}/{:05d}-of-{:05d}.json.gz".format(
                g, num_groups, s, per_group
            )
            lines = []
            for d in range(docs):
                n = 3 + (g * per_group + s + d) % 7
                lines.append(" ".join(str(token + i) for i in range(n)))
                token += n
            out[key] = lines
    return out


class TestS3(unittest.TestCase):
    """The S3 source, against a fake client: no network, no local files."""

    URI = "s3://bucket/corpus/data/"

    def setUp(self):
        self.objects = fake_objects()
        self.client = FakeS3(self.objects)

    def load(self, rank, size, **kw):
        return list(
            data.load_s3(
                Tokenizer(),
                rank,
                size,
                uri=self.URI,
                num_groups=4,
                shards_per_group=8,
                client=self.client,
                **kw,
            )
        )

    def test_manifest_supplies_the_layout(self):
        self.objects[s3.split_uri(self.URI)[1] + s3.MANIFEST_NAME] = None
        self.client.manifest = {
            "num_groups": 4,
            "shards_per_group": 8,
            "suffix": ".json.gz",
        }
        docs = list(data.load_s3(Tokenizer(), 0, 1, uri=self.URI, client=self.client))
        self.assertEqual(len(docs), (len(self.objects) - 1) * 3)

    def test_missing_manifest_needs_the_layout_from_the_config(self):
        with self.assertRaises(ValueError) as cm:
            list(data.load_s3(Tokenizer(), 0, 1, uri=self.URI, client=self.client))
        self.assertIn(s3.MANIFEST_NAME, str(cm.exception))

    def test_generated_keys_are_the_objects_that_exist(self):
        _, keys = s3.shard_keys(self.URI, num_groups=4, shards_per_group=8)
        self.assertEqual(sorted(keys), sorted(self.objects))

    def test_ranks_partition_the_files(self):
        size = 8
        seen = [{d["file_name"] for d in self.load(r, size)} for r in range(size)]
        self.assertEqual(len(set().union(*seen)), len(self.objects))
        self.assertEqual(sum(len(s) for s in seen), len(self.objects))

    def test_slices_stride_rather_than_block(self):
        mine = {d["file_name"].split("/")[-2] for d in self.load(0, 8)}
        self.assertGreater(len(mine), 1)

    def test_every_document_arrives_once(self):
        docs = [tuple(d["input_ids"][:-1]) for d in self.load(0, 1)]
        self.assertEqual(len(docs), len(set(docs)))
        self.assertEqual(len(docs), len(self.objects) * 3)

    def test_nothing_is_written_to_disk(self):
        before = set(os.listdir("."))
        self.load(0, 8)
        self.assertEqual(set(os.listdir(".")), before)

    def test_resume_skips_to_file_and_document(self):
        full = self.load(0, 8)
        target = full[5]
        resumed = self.load(
            0,
            8,
            start_file_name=target["file_name"],
            start_sample_idx=target["sample_idx"],
        )
        self.assertEqual(resumed[0]["file_name"], target["file_name"])
        self.assertEqual(resumed[0]["sample_idx"], target["sample_idx"])
        self.assertEqual(resumed[0]["input_ids"], target["input_ids"])

    def test_only_this_ranks_objects_are_fetched(self):
        self.load(3, 8)
        self.assertEqual(len(self.client.gets), len(self.objects) // 8)

    def test_checkpoint_state_resumes_the_exact_token_stream(self):
        self.client = FakeS3(fake_objects_varlen())

        def batches(resume_d_next=None, **kw):
            return list(
                batching.iterate_batches(
                    data.load_s3(
                        Tokenizer(),
                        0,
                        8,
                        uri=self.URI,
                        num_groups=4,
                        shards_per_group=8,
                        client=self.client,
                        **kw,
                    ),
                    context_size=7,
                    batch_size=2,
                    resume_d_next=resume_d_next,
                )
            )

        reference = batches()
        states = [json.loads(json.dumps(b["_data_state"])) for b in reference]
        self.assertTrue(
            any(s["d_next"] for s in states), "fixture must split a document"
        )

        for cut, state in enumerate(states[:-1]):
            with self.subTest(cut=cut, split=bool(state["d_next"])):
                resumed = batches(
                    resume_d_next=state["d_next"],
                    start_file_name=state["file_name"],
                    start_sample_idx=state["sample_idx"],
                )
                self.assertEqual(len(resumed), len(reference) - cut - 1)

                got = np.concatenate([b["input_ids"].ravel() for b in resumed])
                expected = np.concatenate(
                    [b["input_ids"].ravel() for b in reference[cut + 1 :]]
                )
                self.assertTrue(np.array_equal(expected, got))

                stitched = np.concatenate(
                    [b["input_ids"].ravel() for b in reference[: cut + 1]] + [got]
                )
                body = stitched[stitched != Tokenizer.eos_token_id]
                self.assertEqual(len(set(body.tolist())), body.size)


if __name__ == "__main__":
    unittest.main()
