# Copyright © 2024 Apple Inc.
import pathlib
import re
import unittest

import mlx.core as mx

from mlx_lm.models.mla import latent_length, max_absorbed_queries

MODELS = [
    "deepseek_v3",
    "deepseek_v32",
    "kimi_k3",
    "kimi_linear",
    "longcat_flash",
    "glm4_moe_lite",
    "bailing_moe_v3",
]


class TestMaxAbsorbedQueries(unittest.TestCase):
    def test_asymptotic_limit(self):
        # cache_len omitted -> the T -> inf answer
        self.assertEqual(max_absorbed_queries(512, 128, 128), 170)
        self.assertEqual(max_absorbed_queries(512, 192, 256), 398)

    def test_cold_cache_rejects_the_absorbed_path(self):
        # At T == L materializing is cheaper for every current model, so the
        # threshold must fall below L rather than admit it.
        for r, n, v in ((512, 128, 128), (512, 192, 256)):
            for L in (2, 32, 64, 169, 398):
                self.assertLess(
                    max_absorbed_queries(r, n, v, cache_len=L),
                    L,
                    f"cold cache admitted L={L} for dims {(r, n, v)}",
                )

    def test_warm_cache_approaches_the_asymptote(self):
        r, n, v = 512, 128, 128
        limit = max_absorbed_queries(r, n, v)
        self.assertEqual(max_absorbed_queries(r, n, v, cache_len=32768), limit - 1)
        self.assertLess(max_absorbed_queries(r, n, v, cache_len=1024), limit)
        seq = [max_absorbed_queries(r, n, v, cache_len=t) for t in (256, 1024, 8192)]
        self.assertEqual(seq, sorted(seq))

    def test_degenerate_dims_keep_the_decode_path(self):
        self.assertEqual(max_absorbed_queries(64, 128, 128), 1)
        self.assertGreaterEqual(max_absorbed_queries(1, 1, 1), 1)


class TestLatentLength(unittest.TestCase):
    def test_plain_array(self):
        self.assertEqual(latent_length(mx.zeros((1, 1, 37, 8))), 37)

    def test_quantized_tuple(self):
        q = (mx.zeros((1, 1, 37, 2)), mx.zeros((1, 1, 37, 1)), mx.zeros((1, 1, 37, 1)))
        self.assertEqual(latent_length(q), 37)


class TestGatePairing(unittest.TestCase):
    """Both gates must be driven by one boolean, so they cannot disagree."""

    def test_one_decision_two_uses(self):
        root = pathlib.Path(__file__).resolve().parents[1] / "mlx_lm" / "models"
        decision = re.compile(
            r"absorbed = (?:L|length) == 1 or (?:L|length) <= max_absorbed_queries\("
        )
        for m in MODELS:
            src = (root / f"{m}.py").read_text()
            with self.subTest(model=m):
                self.assertEqual(
                    len(decision.findall(src)),
                    1,
                    f"{m}: expected exactly one gate decision",
                )
                self.assertEqual(
                    src.count("if absorbed:"),
                    2,
                    f"{m}: expected both gates to use that decision",
                )


class TestIndexerGateUnchanged(unittest.TestCase):
    """The sparse top-k gather stays at L == 1.

    It selects with ``topk_indices[:, :, 0, :]``, the first query's top-k, so
    widening it would apply one query's selection to every query.
    """

    def test_first_query_gather_is_still_gated_on_one_query(self):
        root = pathlib.Path(__file__).resolve().parents[1] / "mlx_lm" / "models"
        src = (root / "deepseek_v32.py").read_text()
        self.assertRegex(
            src,
            re.compile(r"if L == 1:\s*\n\s*idx = topk_indices"),
            "deepseek_v32: the first-query top-k gather is no longer gated on L == 1",
        )


if __name__ == "__main__":
    unittest.main()
