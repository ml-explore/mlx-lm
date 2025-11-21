# ABOUTME: Ensures ModelRunner retires contexts when generator goes idle with no responses.
# ABOUTME: Prevents scheduler hangs when BatchGenerator finishes without emitting tokens.

import sys
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mlx_lm.server_batched.engine import ModelRunner
from mlx_lm.server_batched.state import SequenceContext, SequenceState


class DummyDetokenizer:
    def __init__(self):
        self.last_segment = ""

    def add_token(self, token_id):
        self.last_segment = ""

    def finalize(self):
        self.last_segment = ""


class DummyTokenizer:
    eos_token_id = 1
    eos_token_ids = [1]

    def __init__(self):
        self.detokenizer = DummyDetokenizer()

    def encode(self, prompt):
        return [0]


class DummyModel:
    def __call__(self, *args, **kwargs):
        return mx.zeros((1, 1, 2))


class DummyGenerator:
    def __init__(self):
        self.active_batch = type("_AB", (), {"uids": []})()

    def next(self):
        return []


def test_idle_generator_retire_finishes_contexts():
    runner = ModelRunner(
        DummyModel(), DummyTokenizer(), max_num_seqs=2, prefill_chunk=1
    )

    # Replace generator with idle stub.
    runner.generator = DummyGenerator()

    # Build a context and register it as active.
    ctx = runner.build_context(
        request_id="r1",
        prompt=[1, 2],
        max_new_tokens=4,
        sampler_settings={"temp": 0.0, "top_p": 1.0, "min_p": 0.0, "top_k": 0},
        stopping_settings={"eos_token_id": 1},
    )
    # Pretend the generator registered the uid
    runner.generator.active_batch.uids.append(1)
    runner.uid_to_context[1] = ctx

    stats = runner.decode([ctx])

    assert ctx.state.finished
    assert runner.uid_to_context == {}
    assert stats["decode_iterations"] == 0
    assert stats["decode_tokens"] == 0


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
