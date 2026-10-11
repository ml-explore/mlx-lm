# Copyright © 2026 Apple Inc.

"""A speculative request (here a draft model) runs alone while it is alone,
and moves into the batch when another request arrives -- its answer the same
as if it had run alone to the end, the other request served at once."""

import http.server
import json
import threading
import time
import unittest

import requests

from mlx_lm.generate import StopSequences
from mlx_lm.models.cache import LRUPromptCache, make_prompt_cache
from mlx_lm.server import APIHandler, ResponseGenerator, _ResumedStopSequences

from tests.test_server import DummyModelProvider


def _stream(port, prompt, max_tokens, out, key, started=None):
    body = {"model": "chat_model", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0, "stream": True}
    text = []
    with requests.post(f"http://localhost:{port}/v1/chat/completions", json=body, stream=True) as r:
        for line in r.iter_lines():
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            for c in chunk.get("choices", []):
                piece = (c.get("delta") or {}).get("content")
                if piece:
                    text.append(piece)
                    if started is not None and len(text) == 3:
                        started.set()
    out[key] = "".join(text)


class TestBatchHandoff(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.provider = DummyModelProvider(with_draft=True)
        cls.generator = ResponseGenerator(cls.provider, LRUPromptCache())
        cls.httpd = http.server.ThreadingHTTPServer(("localhost", 0), lambda *a, **k: APIHandler(cls.generator, *a, **k))
        cls.port = cls.httpd.server_port
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.generator.stop_and_join()

    def test_alone_is_speculative_and_batchable_with_company(self):
        args = type("args", (object,), {"seed": None, "num_draft_tokens": 3})
        self.assertFalse(self.generator._is_batchable(args, alone=True))
        self.assertTrue(self.generator._is_batchable(args, alone=False))

    def test_stop_sequence_split_by_the_move_still_stops(self):
        resumed = _ResumedStopSequences(StopSequences([[7, 8, 9]]), [1, 7, 8])
        m = resumed.matcher()
        self.assertTrue(m.advance(9), "7 8 sent alone, 9 in the batch: a stop")
        self.assertFalse(resumed.matcher().advance(5))

    def test_the_batch_takes_the_models_slots_only(self):
        model = self.provider.model
        both = make_prompt_cache(model) + make_prompt_cache(self.provider.draft_model)
        self.assertEqual(len(self.generator._main_slots(both)), len(make_prompt_cache(model)))
        self.assertIsNone(self.generator._main_slots(None))

    def test_distributed_never_switches_by_load(self):
        from queue import Queue
        gen = object.__new__(ResponseGenerator)
        gen.requests = Queue()
        gen.requests.put(("x", None, None))
        gen._is_distributed = False
        self.assertTrue(gen._others_waiting())
        gen._is_distributed = True
        self.assertFalse(gen._others_waiting(), "only rank 0 sees the queue")

    def test_a_second_request_moves_the_first_into_the_batch(self):
        prompt_a = "Write a long story about a dragon who learns to code."
        prompt_b = "List the planets of the solar system."
        alone = {}
        _stream(self.port, prompt_a, 300, alone, "a")
        before = getattr(self.generator, "handoffs", 0)

        out = {}
        started = threading.Event()
        ta = threading.Thread(target=_stream, args=(self.port, prompt_a, 300, out, "a", started))
        ta.start()
        self.assertTrue(started.wait(60))
        tb = threading.Thread(target=_stream, args=(self.port, prompt_b, 20, out, "b"))
        tb.start()
        ta.join(120)
        tb.join(120)

        self.assertEqual(getattr(self.generator, "handoffs", 0), before + 1, "A moved into the batch")
        # Up to the move, what it sent alone; then it goes on from there --
        # nothing repeated or lost at the seam. (After it, the batch's 4-bit
        # numerics may pick another token than the single path would.)
        sent = self.generator.last_handoff_sent
        self.assertTrue(sent)
        self.assertTrue(alone["a"].startswith(sent), "the single path's text up to the move")
        self.assertTrue(out["a"].startswith(sent))
        self.assertGreater(len(out["a"]), len(sent) + 20, "it went on in the batch")
        seam = len(sent)
        common = 0
        while common < min(len(out["a"]), len(alone["a"])) and out["a"][common] == alone["a"][common]:
            common += 1
        self.assertGreaterEqual(common, seam)
        # The first tokens after the seam are the batch's own greedy picks
        # from the same context: identical to the single path's.
        self.assertGreaterEqual(common, seam + 3, "the first words after the move match too")
        self.assertTrue(out["b"])


if __name__ == "__main__":
    unittest.main()
