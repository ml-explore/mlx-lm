"""Tests for the mirror tool.

Local directories and a stubbed Hub -- no network.

    python test_mirror_dataset.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mirror_dataset
import storage
from test_shuffle_dataset import make_records, write_jsonl_inputs


def files_under(root):
    return sorted(os.path.relpath(p, root) for p in storage.list_files(root))


def mirror_all(input_path, output, num_workers=1, **kw):
    for worker_id in range(num_workers):
        mirror_dataset.mirror(input_path, output, num_workers, worker_id, **kw)


class MirrorTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def path(self, *parts):
        return os.path.join(self.root, *parts)

    def stub_hub(self, repo, source_root):
        class FakeApi:
            def list_repo_files(self, name, repo_type=None, revision=None):
                return sorted(
                    os.path.relpath(os.path.join(d, f), source_root)
                    for d, _, fs in os.walk(source_root)
                    for f in fs
                )

            def file_exists(self, name, key, repo_type=None, revision=None):
                return os.path.exists(os.path.join(source_root, key))

        def fake_download(name, key, repo_type=None, revision=None, local_dir=None):
            dst = os.path.join(local_dir, os.path.basename(key))
            shutil.copyfile(os.path.join(source_root, key), dst)
            return dst

        real_api, real_download = storage._hf_api, storage._hf_download

        def patched(src, dst):
            from unittest import mock

            with mock.patch("huggingface_hub.hf_hub_download", fake_download):
                real_download(src, dst)

        storage._hf_api = lambda: FakeApi()
        storage._hf_download = patched
        self.addCleanup(setattr, storage, "_hf_api", real_api)
        self.addCleanup(setattr, storage, "_hf_download", real_download)


class TestMirror(MirrorTestCase):
    def test_copies_every_file_and_keeps_the_layout(self):
        write_jsonl_inputs(self.path("src", "data", "train"), make_records(40), 3)
        write_jsonl_inputs(self.path("src", "data", "eval"), make_records(10), 1)
        out = self.path("dst") + "/"

        mirror_all(self.path("src") + "/", out)

        self.assertEqual(
            files_under(out),
            [
                "data/eval/part-000.jsonl.gz",
                "data/train/part-000.jsonl.gz",
                "data/train/part-001.jsonl.gz",
                "data/train/part-002.jsonl.gz",
            ],
        )

    def test_contents_are_byte_identical(self):
        paths = write_jsonl_inputs(self.path("src"), make_records(40), 3)
        out = self.path("dst") + "/"

        mirror_all(self.path("src") + "/", out)

        for src in paths:
            with open(src, "rb") as a, open(
                os.path.join(self.path("dst"), os.path.basename(src)), "rb"
            ) as b:
                self.assertEqual(a.read(), b.read())

    def test_workers_partition_the_files(self):
        write_jsonl_inputs(self.path("src"), make_records(70), 7)
        out = self.path("dst") + "/"

        mirror_all(self.path("src") + "/", out, num_workers=3)

        self.assertEqual(len(files_under(out)), 7)

    def test_output_without_trailing_slash(self):
        write_jsonl_inputs(self.path("src"), make_records(20), 2)

        mirror_all(self.path("src") + "/", self.path("dst"))

        self.assertEqual(len(files_under(self.path("dst"))), 2)

    def test_resumes_by_skipping_what_is_already_there(self):
        write_jsonl_inputs(self.path("src"), make_records(40), 4)
        out = self.path("dst") + "/"
        mirror_all(self.path("src") + "/", out)

        # Delete two, then rerun: only the missing ones come back.
        present = files_under(out)
        for name in present[:2]:
            os.remove(os.path.join(self.path("dst"), name))
        marker = os.path.join(self.path("dst"), present[2])
        before = os.stat(marker).st_mtime_ns

        mirror_all(self.path("src") + "/", out)

        self.assertEqual(files_under(out), present)
        self.assertEqual(
            os.stat(marker).st_mtime_ns, before, "untouched file rewritten"
        )

    def test_overwrite_recopies(self):
        write_jsonl_inputs(self.path("src"), make_records(20), 2)
        out = self.path("dst") + "/"
        mirror_all(self.path("src") + "/", out)
        victim = os.path.join(self.path("dst"), files_under(out)[0])
        with open(victim, "wb") as fout:
            fout.write(b"corrupted")

        mirror_all(self.path("src") + "/", out, overwrite=True)

        with open(victim, "rb") as fin:
            self.assertNotEqual(fin.read(), b"corrupted")

    def test_dry_run_writes_nothing(self):
        write_jsonl_inputs(self.path("src"), make_records(20), 2)
        out = self.path("dst") + "/"

        mirror_all(self.path("src") + "/", out, dry_run=True)

        self.assertFalse(os.path.exists(self.path("dst")))

    def test_empty_input_is_an_error(self):
        os.makedirs(self.path("src"))
        with self.assertRaises(SystemExit) as caught:
            mirror_all(self.path("src") + "/", self.path("dst") + "/")
        self.assertIn("no files found", str(caught.exception))

    def test_check_accepts_a_usable_pair(self):
        write_jsonl_inputs(self.path("src"), make_records(20), 2)

        files = mirror_dataset.check(self.path("src") + "/", self.path("dst") + "/")

        self.assertEqual(len(files), 2)
        self.assertFalse(os.path.exists(self.path("dst")), "check must not write")

    def test_check_rejects_an_empty_input(self):
        os.makedirs(self.path("src"))
        with self.assertRaises(SystemExit) as caught:
            mirror_dataset.check(self.path("src") + "/", self.path("dst") + "/")
        self.assertIn("no files found", str(caught.exception))

    def test_from_a_file_list(self):
        paths = write_jsonl_inputs(self.path("src", "data"), make_records(40), 4)
        listing = self.path("src", "files.txt")
        with open(listing, "w") as fout:
            fout.writelines("data/" + os.path.basename(p) + "\n" for p in paths[:2])
        out = self.path("dst") + "/"

        mirror_all(listing, out)

        self.assertEqual(len(files_under(out)), 2)


class TestMirrorFromHub(MirrorTestCase):
    def test_mirrors_a_hub_repo(self):
        write_jsonl_inputs(self.path("repo", "data"), make_records(60), 3)
        self.stub_hub("allenai/dolma", self.path("repo"))
        out = self.path("dst") + "/"

        mirror_all("hf://datasets/allenai/dolma/data/", out, num_workers=2)

        self.assertEqual(
            files_under(out),
            ["part-000.jsonl.gz", "part-001.jsonl.gz", "part-002.jsonl.gz"],
        )

    def test_bare_repo_uri_keeps_the_repo_layout(self):
        write_jsonl_inputs(self.path("repo", "data"), make_records(40), 2)
        with open(self.path("repo", "README.md"), "w") as fout:
            fout.write("card\n")
        self.stub_hub("allenai/dolma", self.path("repo"))
        out = self.path("dst") + "/"

        mirror_all("hf://datasets/allenai/dolma", out)

        # Everything in the repo is mirrored, including the card: unlike the
        # shuffle, this is a byte-for-byte copy and does not filter by extension.
        self.assertEqual(
            files_under(out),
            ["README.md", "data/part-000.jsonl.gz", "data/part-001.jsonl.gz"],
        )

    def test_mirrored_copy_then_shuffles(self):
        """The point of the tool: mirror once, then shuffle from the mirror."""
        import shuffle_dataset
        from test_shuffle_dataset import read_all

        records = make_records(120)
        write_jsonl_inputs(self.path("repo", "data"), records, 4)
        self.stub_hub("allenai/dolma", self.path("repo"))
        mirrored = self.path("mirror") + "/"

        mirror_all("hf://datasets/allenai/dolma/data/", mirrored, num_workers=2)

        out = self.path("out") + "/"
        for w in range(2):
            shuffle_dataset.stage_1(mirrored, out, 2, w, "auto", "s", True)
        for w in range(2):
            shuffle_dataset.stage_2(out, 2, w, 2, "auto", "s")

        got = read_all(out, "jsonl")
        self.assertEqual(sorted(r["id"] for r in got), list(range(120)))


class TestMirrorCli(MirrorTestCase):
    def script(self):
        return os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "mirror_dataset.py"
        )

    def test_cli_end_to_end(self):
        write_jsonl_inputs(self.path("src"), make_records(30), 3)
        result = subprocess.run(
            [
                sys.executable,
                self.script(),
                "--input",
                self.path("src") + "/",
                "--output",
                self.path("dst") + "/",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(files_under(self.path("dst"))), 3)

    def test_cli_check_exits_zero_and_writes_nothing(self):
        write_jsonl_inputs(self.path("src"), make_records(10), 2)
        result = subprocess.run(
            [
                sys.executable,
                self.script(),
                "--check",
                "--input",
                self.path("src") + "/",
                "--output",
                self.path("dst") + "/",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ready: 2 files", result.stderr)
        self.assertFalse(os.path.exists(self.path("dst")))

    def test_cli_check_fails_on_empty_input(self):
        os.makedirs(self.path("src"))
        result = subprocess.run(
            [
                sys.executable,
                self.script(),
                "--check",
                "--input",
                self.path("src") + "/",
                "--output",
                self.path("dst") + "/",
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no files found", result.stderr)

    def test_cli_rejects_hub_output(self):
        # Refused by the storage layer, which is the single place that knows the
        # Hub is read-only, so it holds however the tool is driven.
        write_jsonl_inputs(self.path("src"), make_records(10), 1)
        result = subprocess.run(
            [
                sys.executable,
                self.script(),
                "--input",
                self.path("src") + "/",
                "--output",
                "hf://datasets/me/mine/",
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("source only", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
