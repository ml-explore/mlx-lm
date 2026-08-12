import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import shuffle_dataset
import storage

SEED = "test-seed"

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - parquet tests are skipped instead
    pa = pq = None

try:
    import zstandard
except ImportError:  # pragma: no cover - zstd tests are skipped instead
    zstandard = None


def make_records(n):
    return [{"id": i, "text": f"record number {i}"} for i in range(n)]


def write_jsonl_inputs(dirname, records, num_files, kind="gz", trailing_newline=True):
    """Split ``records`` across ``num_files`` JSON Lines files. Returns their paths."""
    os.makedirs(dirname, exist_ok=True)
    ext = {"gz": ".jsonl.gz", "zst": ".jsonl.zst", "none": ".jsonl"}[kind]
    paths = []
    for i in range(num_files):
        lines = [json.dumps(r).encode("utf-8") + b"\n" for r in records[i::num_files]]
        if not trailing_newline and lines:
            lines[-1] = lines[-1].rstrip(b"\n")
        path = os.path.join(dirname, f"part-{i:03d}{ext}")
        if kind == "zst":
            with open(path, "wb") as raw:
                with zstandard.ZstdCompressor().stream_writer(raw) as fout:
                    fout.write(b"".join(lines))
        elif kind == "gz":
            with gzip.open(path, "wb") as fout:
                fout.writelines(lines)
        else:
            with open(path, "wb") as fout:
                fout.writelines(lines)
        paths.append(path)
    return paths


def write_parquet_inputs(dirname, records, num_files, extra_column_in_last=False):
    os.makedirs(dirname, exist_ok=True)
    paths = []
    for i in range(num_files):
        chunk = records[i::num_files]
        data = {
            "id": [r["id"] for r in chunk],
            "text": [r["text"] for r in chunk],
        }
        if extra_column_in_last and i == num_files - 1:
            data["extra"] = [f"x{r['id']}" for r in chunk]
        path = os.path.join(dirname, f"part-{i:03d}.parquet")
        pq.write_table(pa.table(data), path)
        paths.append(path)
    return paths


def run_both_stages(
    input_path,
    output,
    num_workers,
    num_output_shards,
    fmt="auto",
    seed=SEED,
    add_source=True,
):
    for worker_id in range(num_workers):
        shuffle_dataset.stage_1(
            input_path, output, num_workers, worker_id, fmt, seed, add_source
        )
    for worker_id in range(num_workers):
        shuffle_dataset.stage_2(
            output, num_workers, worker_id, num_output_shards, fmt, seed
        )


def final_shards(output, ext):
    """The final shards, in shard-name order, excluding stage 1's buckets."""
    return sorted(
        f for f in storage.list_files(output) if "_tmp/" not in f and f.endswith(ext)
    )


def read_jsonl_shard(path):
    with gzip.open(path, "rb") as fin:
        return [json.loads(line) for line in fin]


def read_all(output, fmt):
    """Every record in the final shards, in shard order."""
    if fmt == "jsonl":
        out = []
        for path in final_shards(output, ".jsonl.gz"):
            out.extend(read_jsonl_shard(path))
        return out
    out = []
    for path in final_shards(output, ".parquet"):
        out.extend(pq.read_table(path).to_pylist())
    return out


class ShuffleTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def path(self, *parts):
        return os.path.join(self.root, *parts)

    def assert_is_permutation(self, records, expected, fmt):
        """Every input record appears in the output exactly once."""
        got = sorted(r["id"] for r in records)
        self.assertEqual(got, sorted(r["id"] for r in expected))
        self.assertEqual(len(got), len(set(got)), f"{fmt}: duplicated records")

    def assert_spread_across_shards(self, output, ext, num_inputs):
        """Records from one input file must end up in several output shards."""
        shards = final_shards(output, ext)
        by_source = {}
        for path in shards:
            records = (
                read_jsonl_shard(path)
                if ext == ".jsonl.gz"
                else pq.read_table(path).to_pylist()
            )
            for r in records:
                source = r["source_file"].rsplit(":", 1)[0]
                by_source.setdefault(source, set()).add(path)
        self.assertEqual(len(by_source), num_inputs)
        for source, paths in by_source.items():
            self.assertGreater(
                len(paths), 1, f"{source} was not spread across output shards"
            )


class TestJsonl(ShuffleTestCase):
    def test_roundtrip_is_a_shuffled_permutation(self):
        records = make_records(240)
        write_jsonl_inputs(self.path("in"), records, num_files=7)
        output = self.path("out") + "/"

        run_both_stages(
            self.path("in") + "/", output, num_workers=3, num_output_shards=2
        )

        shards = final_shards(output, ".jsonl.gz")
        self.assertEqual(
            [os.path.relpath(s, self.path("out")) for s in shards],
            [
                "00000-of-00003/00000-of-00002.jsonl.gz",
                "00000-of-00003/00001-of-00002.jsonl.gz",
                "00001-of-00003/00000-of-00002.jsonl.gz",
                "00001-of-00003/00001-of-00002.jsonl.gz",
                "00002-of-00003/00000-of-00002.jsonl.gz",
                "00002-of-00003/00001-of-00002.jsonl.gz",
            ],
        )

        got = read_all(output, "jsonl")
        self.assert_is_permutation(got, records, "jsonl")
        self.assertNotEqual(
            [r["id"] for r in got], [r["id"] for r in records], "output is not shuffled"
        )
        self.assert_spread_across_shards(output, ".jsonl.gz", num_inputs=7)

    def test_source_column_identifies_every_record(self):
        records = make_records(60)
        write_jsonl_inputs(self.path("in"), records, num_files=3)
        output = self.path("out") + "/"

        run_both_stages(
            self.path("in") + "/", output, num_workers=2, num_output_shards=1
        )

        got = read_all(output, "jsonl")
        sources = [r["source_file"] for r in got]
        self.assertEqual(len(sources), len(set(sources)), "source_file is not unique")
        for r in got:
            name, _, line_no = r["source_file"].rpartition(":")
            self.assertTrue(name.endswith(".jsonl.gz"), name)
            self.assertLessEqual(len(name.split("/")), 3, name)
            self.assertGreaterEqual(int(line_no), 0)

    def test_no_source_column(self):
        records = make_records(40)
        write_jsonl_inputs(self.path("in"), records, num_files=2)
        output = self.path("out") + "/"

        run_both_stages(
            self.path("in") + "/",
            output,
            num_workers=2,
            num_output_shards=1,
            add_source=False,
        )

        got = read_all(output, "jsonl")
        self.assert_is_permutation(got, records, "jsonl")
        self.assertNotIn("source_file", got[0])

    def test_input_without_trailing_newline_does_not_glue_records(self):
        records = make_records(40)
        write_jsonl_inputs(
            self.path("in"), records, num_files=4, trailing_newline=False
        )
        output = self.path("out") + "/"

        # add_source=False passes the raw bytes straight through, which is the
        # case where a missing newline would corrupt the shard.
        run_both_stages(
            self.path("in") + "/",
            output,
            num_workers=2,
            num_output_shards=1,
            add_source=False,
        )

        self.assert_is_permutation(read_all(output, "jsonl"), records, "jsonl")

    @unittest.skipIf(zstandard is None, "zstandard not installed")
    def test_zstd_input(self):
        records = make_records(80)
        write_jsonl_inputs(self.path("in"), records, num_files=4, kind="zst")
        output = self.path("out") + "/"

        run_both_stages(
            self.path("in") + "/", output, num_workers=2, num_output_shards=2
        )

        self.assert_is_permutation(read_all(output, "jsonl"), records, "jsonl")

    def test_uncompressed_input(self):
        records = make_records(50)
        write_jsonl_inputs(self.path("in"), records, num_files=5, kind="none")
        output = self.path("out") + "/"

        run_both_stages(
            self.path("in") + "/", output, num_workers=2, num_output_shards=1
        )

        self.assert_is_permutation(read_all(output, "jsonl"), records, "jsonl")

    def test_non_data_files_under_the_input_prefix_are_ignored(self):
        records = make_records(40)
        write_jsonl_inputs(self.path("in"), records, num_files=2)
        with open(self.path("in", "README.md"), "w") as fout:
            fout.write("not data\n")
        with open(self.path("in", "manifest.txt"), "w") as fout:
            fout.write("part-000.jsonl.gz\n")
        output = self.path("out") + "/"

        run_both_stages(
            self.path("in") + "/", output, num_workers=2, num_output_shards=1
        )

        self.assert_is_permutation(read_all(output, "jsonl"), records, "jsonl")

    def test_input_may_be_a_file_list(self):
        records = make_records(40)
        paths = write_jsonl_inputs(self.path("in"), records, num_files=4)
        # A text file of names relative to its own directory, listing a subset.
        listed = paths[:2]
        with open(self.path("in", "files.txt"), "w") as fout:
            fout.writelines(os.path.basename(p) + "\n" for p in listed)
        output = self.path("out") + "/"

        run_both_stages(
            self.path("in", "files.txt"), output, num_workers=2, num_output_shards=1
        )

        expected = [r for i in range(4) if i < 2 for r in records[i::4]]
        self.assert_is_permutation(read_all(output, "jsonl"), expected, "jsonl")

    def test_file_list_with_no_directory_part(self):
        """A bare "files.txt" lists paths relative to the working directory."""
        records = make_records(40)
        write_jsonl_inputs(self.path("in"), records, num_files=2)
        with open(self.path("in", "files.txt"), "w") as fout:
            fout.write("part-000.jsonl.gz\npart-001.jsonl.gz\n")
        cwd = os.getcwd()
        os.chdir(self.path("in"))
        self.addCleanup(os.chdir, cwd)

        run_both_stages("files.txt", "out/", num_workers=2, num_output_shards=1)

        self.assert_is_permutation(read_all("out/", "jsonl"), records, "jsonl")

    def test_output_prefix_without_a_trailing_slash(self):
        """ "--output x" must behave exactly like "--output x/", not write "x00000-...".

        Getting this wrong silently produces a mangled directory name rather
        than failing, so it is worth pinning.
        """
        records = make_records(40)
        write_jsonl_inputs(self.path("in"), records, num_files=4)

        run_both_stages(
            self.path("in") + "/",
            self.path("out", "data"),  # deliberately no trailing slash
            num_workers=2,
            num_output_shards=1,
        )

        shards = final_shards(self.path("out") + "/", ".jsonl.gz")
        self.assertEqual(
            [os.path.relpath(s, self.path("out")) for s in shards],
            [
                "data/00000-of-00002/00000-of-00001.jsonl.gz",
                "data/00001-of-00002/00000-of-00001.jsonl.gz",
            ],
        )
        self.assert_is_permutation(
            read_all(self.path("out", "data") + "/", "jsonl"), records, "jsonl"
        )

    def test_same_seed_gives_identical_bytes(self):
        records = make_records(120)
        write_jsonl_inputs(self.path("in"), records, num_files=4)
        outputs = []
        for run in ("a", "b"):
            output = self.path("out-" + run) + "/"
            run_both_stages(
                self.path("in") + "/", output, num_workers=2, num_output_shards=2
            )
            shards = []
            for path in final_shards(output, ".jsonl.gz"):
                with open(path, "rb") as fin:
                    shards.append(fin.read())
            outputs.append(shards)
        self.assertEqual(outputs[0], outputs[1])

    def test_different_seed_gives_a_different_order(self):
        records = make_records(120)
        write_jsonl_inputs(self.path("in"), records, num_files=4)
        orders = []
        for seed in ("seed-one", "seed-two"):
            output = self.path("out-" + seed) + "/"
            run_both_stages(
                self.path("in") + "/",
                output,
                num_workers=2,
                num_output_shards=2,
                seed=seed,
            )
            orders.append([r["id"] for r in read_all(output, "jsonl")])
        self.assertNotEqual(orders[0], orders[1])
        self.assertEqual(sorted(orders[0]), sorted(orders[1]))


@unittest.skipIf(pq is None, "pyarrow not installed")
class TestParquet(ShuffleTestCase):
    def test_roundtrip_is_a_shuffled_permutation(self):
        records = make_records(240)
        write_parquet_inputs(self.path("in"), records, num_files=7)
        output = self.path("out") + "/"

        run_both_stages(
            self.path("in") + "/", output, num_workers=3, num_output_shards=2
        )

        self.assertEqual(len(final_shards(output, ".parquet")), 6)
        got = read_all(output, "parquet")
        self.assert_is_permutation(got, records, "parquet")
        self.assertNotEqual([r["id"] for r in got], [r["id"] for r in records])
        self.assert_spread_across_shards(output, ".parquet", num_inputs=7)

    def test_columns_survive(self):
        records = make_records(60)
        write_parquet_inputs(self.path("in"), records, num_files=3)
        output = self.path("out") + "/"

        run_both_stages(
            self.path("in") + "/", output, num_workers=2, num_output_shards=1
        )

        for path in final_shards(output, ".parquet"):
            schema = pq.read_schema(path)
            self.assertEqual(set(schema.names), {"id", "text", "source_file"})
        for r in read_all(output, "parquet"):
            self.assertEqual(r["text"], f"record number {r['id']}")

    def test_schemas_are_unioned_across_input_files(self):
        records = make_records(90)
        write_parquet_inputs(
            self.path("in"), records, num_files=3, extra_column_in_last=True
        )
        output = self.path("out") + "/"

        run_both_stages(
            self.path("in") + "/", output, num_workers=1, num_output_shards=2
        )

        got = read_all(output, "parquet")
        self.assert_is_permutation(got, records, "parquet")
        # Rows from the file that has "extra" keep it; the others get null.
        with_extra = [r for r in got if r.get("extra") is not None]
        self.assertEqual(len(with_extra), len(records[2::3]))
        for r in with_extra:
            self.assertEqual(r["extra"], f"x{r['id']}")

    def test_same_seed_gives_the_same_order(self):
        records = make_records(120)
        write_parquet_inputs(self.path("in"), records, num_files=4)
        orders = []
        for run in ("a", "b"):
            output = self.path("out-" + run) + "/"
            run_both_stages(
                self.path("in") + "/", output, num_workers=2, num_output_shards=2
            )
            orders.append([r["id"] for r in read_all(output, "parquet")])
        self.assertEqual(orders[0], orders[1])

    def test_more_shards_than_records(self):
        """Empty shards are legal: take() needs typed indices to produce one."""
        records = make_records(5)
        write_parquet_inputs(self.path("in"), records, num_files=1)
        output = self.path("out") + "/"

        run_both_stages(
            self.path("in") + "/", output, num_workers=1, num_output_shards=10
        )

        shards = final_shards(output, ".parquet")
        self.assertEqual(len(shards), 10)
        self.assert_is_permutation(read_all(output, "parquet"), records, "parquet")
        # The empty shards must still be readable and carry the schema.
        empty = [p for p in shards if pq.read_table(p).num_rows == 0]
        self.assertEqual(len(empty), 5)
        for path in empty:
            self.assertEqual(
                set(pq.read_schema(path).names), {"id", "text", "source_file"}
            )

    def test_fewer_records_than_workers_squared(self):
        """Stage 1 scatters into num_workers buckets, so buckets go empty first."""
        records = make_records(20)
        write_parquet_inputs(self.path("in"), records, num_files=8)
        output = self.path("out") + "/"

        run_both_stages(
            self.path("in") + "/", output, num_workers=8, num_output_shards=4
        )

        self.assert_is_permutation(read_all(output, "parquet"), records, "parquet")

    def test_input_file_with_zero_rows(self):
        records = make_records(30)
        write_parquet_inputs(self.path("in"), records, num_files=3)
        pq.write_table(
            pa.table(
                {"id": pa.array([], pa.int64()), "text": pa.array([], pa.string())}
            ),
            self.path("in", "empty.parquet"),
        )
        output = self.path("out") + "/"

        run_both_stages(
            self.path("in") + "/", output, num_workers=4, num_output_shards=1
        )

        self.assert_is_permutation(read_all(output, "parquet"), records, "parquet")


class TestEmptyShardsJsonl(ShuffleTestCase):
    def test_more_shards_than_records(self):
        """JSON Lines must behave the same as Parquet on the same input."""
        records = make_records(5)
        write_jsonl_inputs(self.path("in"), records, num_files=1)
        output = self.path("out") + "/"

        run_both_stages(
            self.path("in") + "/", output, num_workers=1, num_output_shards=10
        )

        self.assertEqual(len(final_shards(output, ".jsonl.gz")), 10)
        self.assert_is_permutation(read_all(output, "jsonl"), records, "jsonl")


class TestHuggingFaceUris(ShuffleTestCase):
    """URI handling only -- these make no network calls."""

    def test_split(self):
        self.assertEqual(
            storage._hf_split("hf://datasets/allenai/dolma/data/a.jsonl.gz"),
            ("allenai/dolma", None, "data/a.jsonl.gz"),
        )
        # A revision, and a prefix rather than a full key.
        self.assertEqual(
            storage._hf_split("hf://datasets/allenai/dolma@v1.7/data/"),
            ("allenai/dolma", "v1.7", "data/"),
        )
        # The repo root: no key at all.
        self.assertEqual(
            storage._hf_split("hf://datasets/allenai/dolma"),
            ("allenai/dolma", None, ""),
        )

    def test_round_trips_through_uri(self):
        for uri in (
            "hf://datasets/allenai/dolma/data/a.jsonl.gz",
            "hf://datasets/allenai/dolma@abc123/data/b.parquet",
        ):
            repo, revision, key = storage._hf_split(uri)
            self.assertEqual(storage._hf_uri(repo, revision, key), uri)

    def test_is_remote_and_is_hf(self):
        uri = "hf://datasets/allenai/dolma/data/a.jsonl.gz"
        self.assertTrue(storage.is_remote(uri))
        self.assertTrue(storage.is_hf(uri))
        self.assertFalse(storage.is_hf("s3://bucket/key"))
        self.assertFalse(storage.is_hf("/local/path"))

    def test_malformed_uris_are_rejected(self):
        for uri in (
            "hf://allenai/dolma/data/",  # missing "datasets/"
            "hf://datasets/dolma",  # a bare name, no owner
            "hf://datasets/",
        ):
            with self.assertRaises(SystemExit, msg=uri):
                storage._hf_split(uri)

    def test_output_to_the_hub_is_refused(self):
        with self.assertRaises(SystemExit) as caught:
            storage.upload("/tmp/whatever", "hf://datasets/me/mine/data/x.parquet")
        self.assertIn("source only", str(caught.exception))

    def test_file_list_may_hold_absolute_uris(self):
        """A curated list can name Hub files, or mix schemes."""
        listing = self.path("files.txt")
        with open(listing, "w") as fout:
            fout.write(
                "hf://datasets/allenai/dolma/data/a.jsonl.gz\n"
                "s3://bucket/b.jsonl.gz\n"
                "relative.jsonl.gz\n"
            )
        self.assertEqual(
            storage.read_file_list(listing),
            sorted(
                [
                    "hf://datasets/allenai/dolma/data/a.jsonl.gz",
                    "s3://bucket/b.jsonl.gz",
                    self.path("relative.jsonl.gz"),
                ]
            ),
        )


class TestHuggingFaceOffline(ShuffleTestCase):
    """The full hf:// path end to end, with the Hub client stubbed out.

    Exercises list_files, download and both shuffle stages over hf:// URIs
    without touching the network, so the wiring is covered even where the Hub is
    unreachable. TestHuggingFaceNetwork does the same against the real thing.
    """

    def stub_hub(self, repo, root):
        """Serve ``repo`` out of the local directory ``root``."""

        class FakeApi:
            def list_repo_files(self, name, repo_type=None, revision=None):
                assert name == repo and repo_type == "dataset", (name, repo_type)
                return sorted(
                    os.path.relpath(os.path.join(dirpath, f), root)
                    for dirpath, _, files in os.walk(root)
                    for f in files
                )

            def file_exists(self, name, key, repo_type=None, revision=None):
                return os.path.exists(os.path.join(root, key))

        def fake_download(name, key, repo_type=None, revision=None, local_dir=None):
            assert name == repo and repo_type == "dataset", (name, repo_type)
            dst = os.path.join(local_dir, os.path.basename(key))
            shutil.copyfile(os.path.join(root, key), dst)
            return dst

        real_api, real_download = storage._hf_api, storage._hf_download

        def patched_download(src, dst):
            from unittest import mock

            with mock.patch("huggingface_hub.hf_hub_download", fake_download):
                real_download(src, dst)

        storage._hf_api = lambda: FakeApi()
        storage._hf_download = patched_download
        self.addCleanup(setattr, storage, "_hf_api", real_api)
        self.addCleanup(setattr, storage, "_hf_download", real_download)

    def test_shuffle_jsonl_from_a_hub_prefix(self):
        records = make_records(120)
        write_jsonl_inputs(self.path("repo", "data"), records, num_files=4)
        self.stub_hub("allenai/dolma", self.path("repo"))
        output = self.path("out") + "/"

        run_both_stages(
            "hf://datasets/allenai/dolma/data/",
            output,
            num_workers=2,
            num_output_shards=2,
        )

        got = read_all(output, "jsonl")
        self.assert_is_permutation(got, records, "jsonl")
        # Provenance names the Hub file, not the temp path it was staged in.
        # Last three components of the URI happens to read repo/dir/file.
        for r in got:
            self.assertTrue(
                r["source_file"].startswith("dolma/data/part-"), r["source_file"]
            )

    def test_shuffle_parquet_from_a_hub_prefix_with_revision(self):
        records = make_records(90)
        write_parquet_inputs(self.path("repo", "data"), records, num_files=3)
        self.stub_hub("allenai/Dolci-Think-SFT-7B", self.path("repo"))
        output = self.path("out") + "/"

        run_both_stages(
            "hf://datasets/allenai/Dolci-Think-SFT-7B@main/data/",
            output,
            num_workers=3,
            num_output_shards=2,
        )

        self.assert_is_permutation(read_all(output, "parquet"), records, "parquet")

    def test_a_hub_uri_needs_no_trailing_slash(self):
        records = make_records(40)
        write_jsonl_inputs(self.path("repo", "data"), records, num_files=2)
        self.stub_hub("allenai/dolma", self.path("repo"))

        # Unlike a local or s3 prefix, a bare hf:// URI is still a prefix and
        # must not be mistaken for a text file listing.
        self.assertEqual(len(storage.read_file_list("hf://datasets/allenai/dolma")), 2)

    def test_non_data_files_in_the_repo_are_ignored(self):
        records = make_records(40)
        write_jsonl_inputs(self.path("repo", "data"), records, num_files=2)
        with open(self.path("repo", "README.md"), "w") as fout:
            fout.write("dataset card\n")
        self.stub_hub("allenai/dolma", self.path("repo"))
        output = self.path("out") + "/"

        run_both_stages(
            "hf://datasets/allenai/dolma", output, num_workers=2, num_output_shards=1
        )

        self.assert_is_permutation(read_all(output, "jsonl"), records, "jsonl")


@unittest.skipUnless(
    os.environ.get("TEST_HF_NETWORK"), "set TEST_HF_NETWORK=1 to hit the Hub"
)
class TestHuggingFaceNetwork(ShuffleTestCase):
    """End to end against real Dolma repos. Downloads a few files."""

    def test_shuffle_dolci_parquet_from_the_hub(self):
        # The Parquet SFT set, the same shape the internal pipeline shuffled.
        listing = self.path("files.txt")
        repo = "hf://datasets/allenai/Dolci-Think-SFT-7B/data/"
        with open(listing, "w") as fout:
            fout.writelines(f + "\n" for f in storage.list_files(repo)[:2])
        output = self.path("out") + "/"

        run_both_stages(listing, output, num_workers=2, num_output_shards=2)

        rows = read_all(output, "parquet")
        self.assertGreater(len(rows), 0)
        self.assertEqual(len(rows), len({r["source_file"] for r in rows}))
        self.assertEqual(len(final_shards(output, ".parquet")), 4)

    def test_shuffle_dolmino_jsonl_from_the_hub(self):
        listing = self.path("files.txt")
        repo = "hf://datasets/allenai/dolmino-mix-1124/data/flan/"
        with open(listing, "w") as fout:
            fout.writelines(f + "\n" for f in storage.list_files(repo)[:2])
        output = self.path("out") + "/"

        run_both_stages(listing, output, num_workers=2, num_output_shards=1)

        rows = read_all(output, "jsonl")
        self.assertGreater(len(rows), 0)
        self.assertEqual(len(rows), len({r["source_file"] for r in rows}))


class TestErrors(ShuffleTestCase):
    def test_more_workers_than_input_files(self):
        write_jsonl_inputs(self.path("in"), make_records(10), num_files=2)
        with self.assertRaises(SystemExit) as caught:
            shuffle_dataset.stage_1(
                self.path("in") + "/", self.path("out") + "/", 4, 3, "auto", SEED, True
            )
        self.assertIn("--num-workers", str(caught.exception))

    def test_unrecognised_format(self):
        os.makedirs(self.path("in"))
        with open(self.path("in", "data.txt"), "w") as fout:
            fout.write("hello\n")
        with self.assertRaises(SystemExit) as caught:
            shuffle_dataset.stage_1(
                self.path("in") + "/", self.path("out") + "/", 1, 0, "auto", SEED, True
            )
        self.assertIn("--format", str(caught.exception))

    def test_mixed_formats_need_an_explicit_choice(self):
        write_jsonl_inputs(self.path("in"), make_records(10), num_files=1)
        with open(self.path("in", "extra.parquet"), "wb") as fout:
            fout.write(b"not really parquet")
        with self.assertRaises(SystemExit) as caught:
            shuffle_dataset.stage_1(
                self.path("in") + "/", self.path("out") + "/", 1, 0, "auto", SEED, True
            )
        self.assertIn("--format", str(caught.exception))

    def test_stage_2_before_stage_1(self):
        with self.assertRaises(SystemExit) as caught:
            shuffle_dataset.stage_2(self.path("out") + "/", 2, 0, 1, "auto", SEED)
        self.assertIn("stage-1", str(caught.exception))

    def test_a_failed_write_leaves_nothing_behind(self):
        records = make_records(20)
        write_jsonl_inputs(self.path("in"), records, num_files=2)
        output = self.path("out") + "/"
        with self.assertRaises(RuntimeError):
            with storage.publish(output + "doomed.jsonl.gz"):
                raise RuntimeError("worker died mid-write")
        self.assertFalse(os.path.exists(output + "doomed.jsonl.gz"))


class TestCommandLine(ShuffleTestCase):
    def test_cli_end_to_end_from_another_directory(self):
        records = make_records(60)
        write_jsonl_inputs(self.path("in"), records, num_files=3)
        output = self.path("out") + "/"
        script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "shuffle_dataset.py"
        )
        common = [
            sys.executable,
            script,
            "--output",
            output,
            "--num-workers",
            "2",
            "--num-output-shards",
            "2",
            "--seed",
            SEED,
        ]
        for stage in ("1", "2"):
            for worker_id in ("0", "1"):
                # cwd is deliberately not the script's directory.
                result = subprocess.run(
                    common
                    + [
                        "--stage",
                        stage,
                        "--worker-id",
                        worker_id,
                        "--input",
                        self.path("in") + "/",
                    ],
                    cwd=self.root,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

        self.assert_is_permutation(read_all(output, "jsonl"), records, "jsonl")

    def test_worker_id_out_of_range_is_rejected(self):
        script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "shuffle_dataset.py"
        )
        result = subprocess.run(
            [
                sys.executable,
                script,
                "--stage",
                "1",
                "--input",
                self.path("in") + "/",
                "--output",
                self.path("out") + "/",
                "--num-workers",
                "4",
                "--worker-id",
                "4",
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--worker-id", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
