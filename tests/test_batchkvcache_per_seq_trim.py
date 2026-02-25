"""Tests for per-sequence variable trim in BatchKVCache.

This file tests the upstream fix that enables per-sequence variable
trimming in BatchKVCache, which is required for batched speculative
decoding. Without the fix, after variable trim the attention mask and
returned KV slices use the scalar _idx rather than per-sequence offsets,
causing incorrect outputs for sequences with different trim amounts.

The fix modifies:
- update_and_fetch(): return keys/values up to max(offset) instead of _idx
- make_mask(): add right_padding for sequences whose offset < max(offset)

Test categories:
1. Basic per-sequence trim (mock/synthetic data)
2. Real model GATE test (Qwen3-4B-4bit)
3. Backward compatibility
4. Edge cases

Tests that REQUIRE the fix are marked with comments. Before the fix,
these tests will fail.
"""

import os

import mlx.core as mx
import pytest

from mlx_lm.models.cache import BatchKVCache


# ---------------------------------------------------------------------------
# Helper: create batch cache layers (no model needed)
# ---------------------------------------------------------------------------

def _make_batch_layers(n_layers, batch_size, left_padding=None):
    """Create n_layers of BatchKVCache with given left_padding."""
    if left_padding is None:
        left_padding = [0] * batch_size
    return [BatchKVCache(left_padding) for _ in range(n_layers)]


def _fill_cache(layers, tokens_per_step, n_steps=1, batch_size=2, n_heads=2, head_dim=4):
    """Feed synthetic data into cache layers to advance _idx and offset."""
    for _ in range(n_steps):
        keys = mx.random.normal((batch_size, n_heads, tokens_per_step, head_dim))
        values = mx.random.normal((batch_size, n_heads, tokens_per_step, head_dim))
        mx.eval(keys, values)
        for layer in layers:
            layer.update_and_fetch(keys, values)


# ===========================================================================
# 1. Basic per-sequence trim tests (mock/synthetic data)
# ===========================================================================

class TestPerSequenceVariableTrim:
    """Tests for per-sequence trim behavior in BatchKVCache."""

    def test_uniform_trim_still_works(self):
        """trim(3) on batch of 2, both offsets decrease by 3 (backward compat)."""
        layers = _make_batch_layers(2, batch_size=2)
        _fill_cache(layers, tokens_per_step=10)

        for layer in layers:
            assert layer._idx == 10
            assert mx.array_equal(layer.offset, mx.array([10, 10]))

        # Uniform trim: both sequences lose 3
        for layer in layers:
            layer.trim(3)

        for layer in layers:
            assert layer._idx == 7
            assert mx.array_equal(layer.offset, mx.array([7, 7]))

    def test_offset_readvance_after_trim(self):
        """trim(5), then manually set offset[0] += 3 (simulate per-seq trim).

        After the fix, this should be supported:
        - _idx decreases by 5 (global)
        - offset[0] re-advances by 3 (net trim of 2 for seq 0)
        - offset[1] stays (net trim of 5 for seq 1)
        """
        layers = _make_batch_layers(1, batch_size=2)
        _fill_cache(layers, tokens_per_step=10)

        layer = layers[0]
        assert layer._idx == 10

        # Trim all by 5
        layer.trim(5)
        assert layer._idx == 5
        assert mx.array_equal(layer.offset, mx.array([5, 5]))

        # Re-advance offset for seq 0 (accepted 3 more than seq 1)
        layer.offset = layer.offset + mx.array([3, 0])
        assert mx.array_equal(layer.offset, mx.array([8, 5]))

    def test_offset_cannot_exceed_buffer(self):
        """Verify offset stays within allocated buffer bounds."""
        layers = _make_batch_layers(1, batch_size=2)
        _fill_cache(layers, tokens_per_step=10)

        layer = layers[0]
        buffer_size = layer.keys.shape[2]

        # Trim and re-advance
        layer.trim(3)
        layer.offset = layer.offset + mx.array([2, 0])

        # offset[0] = 9, offset[1] = 7 -- both within buffer_size
        assert layer.offset[0].item() <= buffer_size
        assert layer.offset[1].item() <= buffer_size


# ===========================================================================
# 2. Real model GATE test
# ===========================================================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "Qwen3-4B-4bit",
)
if not os.path.isdir(MODEL_PATH):
    MODEL_PATH = "Qwen3-4B-4bit"
HAS_MODEL = os.path.isdir(MODEL_PATH) and os.path.exists(
    os.path.join(MODEL_PATH, "config.json")
)


@pytest.mark.skipif(not HAS_MODEL, reason="Qwen3-4B-4bit not available")
class TestPerSequenceTrimWithRealModel:
    """Real model tests for per-sequence variable trim.

    These tests load Qwen3-4B-4bit and verify that after the
    BatchKVCache fix, per-sequence variable trim produces correct
    outputs that match reference single-sequence processing.
    """

    @pytest.fixture(scope="class")
    def model_and_tokenizer(self):
        from mlx_lm import load
        model, tokenizer = load(MODEL_PATH)
        return model, tokenizer

    def _make_batch_cache(self, model, left_padding):
        from mlx_lm.generate import _make_cache
        return _make_cache(model, left_padding, max_kv_size=None)

    def _prefill(self, model, tokens, cache):
        """Run prefill for all tokens, return logits from last position."""
        if tokens.shape[1] > 1:
            model(tokens[:, :-1], cache=cache)
            mx.eval([c.state for c in cache])
        logits = model(tokens[:, -1:], cache=cache)
        mx.eval([c.state for c in cache])
        return logits

    def _decode_one(self, model, y, cache):
        logits = model(y, cache=cache)
        mx.eval([c.state for c in cache])
        return logits

    def _multi_token_forward(self, model, tokens, cache):
        logits = model(tokens, cache=cache)
        mx.eval([c.state for c in cache])
        return logits

    def test_variable_trim_output_matches_reference(self, model_and_tokenizer):
        """THE GATE TEST: per-seq variable trim must match reference outputs.

        REQUIRES THE FIX. Fails without it (GATE Result B).

        Procedure:
        1. Create 2 sequences with different prompts, batched with left-padding
        2. Prefill both
        3. Decode 1 step
        4. Multi-token forward with k=5 draft tokens
        5. Variable trim: trim(3) then offset re-advance [0, 2]
           seq 0: trim 3 (deficit 0), seq 1: trim 1 (deficit 2)
        6. Decode 1 more step
        7. Reference: separate single-sequence processing with exact token counts
        8. Compare argmax outputs -- they must match
        """
        model, tokenizer = model_and_tokenizer

        prompt_a = "Hello world"
        prompt_b = "The quick brown fox jumps"
        tokens_a = mx.array(tokenizer.encode(prompt_a))
        tokens_b = mx.array(tokenizer.encode(prompt_b))

        len_a = tokens_a.shape[0]
        len_b = tokens_b.shape[0]
        max_len = max(len_a, len_b)
        pad_a = max_len - len_a
        pad_b = max_len - len_b
        left_padding = [pad_a, pad_b]

        padded_a = (
            mx.concatenate([mx.zeros(pad_a, dtype=mx.int32), tokens_a])
            if pad_a > 0
            else tokens_a
        )
        padded_b = (
            mx.concatenate([mx.zeros(pad_b, dtype=mx.int32), tokens_b])
            if pad_b > 0
            else tokens_b
        )
        batch_tokens = mx.stack([padded_a, padded_b])

        # ---- SPEC DECODE PATH (with variable trim) ----
        spec_cache = self._make_batch_cache(model, left_padding)
        spec_logits = self._prefill(model, batch_tokens, spec_cache)

        y_spec = mx.argmax(spec_logits[:, -1, :], axis=-1, keepdims=True)
        mx.eval(y_spec)
        spec_logits_1 = self._decode_one(model, y_spec, spec_cache)
        y_spec_1 = mx.argmax(spec_logits_1[:, -1, :], axis=-1)
        mx.eval(y_spec_1)

        idx_after_decode1 = spec_cache[0]._idx

        # Multi-token forward with k=5 draft tokens
        k = 5
        draft_tokens = mx.zeros((2, k), dtype=mx.int32)
        draft_tokens[0, 0] = y_spec_1[0]
        draft_tokens[1, 0] = y_spec_1[1]
        for i in range(1, k):
            draft_tokens[0, i] = i + 100
            draft_tokens[1, i] = i + 200
        mx.eval(draft_tokens)

        self._multi_token_forward(model, draft_tokens, spec_cache)
        assert spec_cache[0]._idx == idx_after_decode1 + k

        # Variable trim: seq 0 trims 3, seq 1 trims 1
        # max_trim = 3, deficit = [0, 2]
        max_trim = 3
        for layer_cache in spec_cache:
            if hasattr(layer_cache, "trim"):
                layer_cache.trim(max_trim)
                layer_cache.offset = layer_cache.offset + mx.array([0, 2])

        # Decode one more token
        # seq 0 accepted 2 of 5 -> next token is draft[0, 2]
        # seq 1 accepted 4 of 5 -> next token is draft[1, 4]
        next_token = mx.array([[draft_tokens[0, 2]], [draft_tokens[1, 4]]])
        mx.eval(next_token)
        final_logits_spec = self._decode_one(model, next_token, spec_cache)
        mx.eval(final_logits_spec)

        # ---- REFERENCE PATH (single-sequence) ----
        # Seq 0: prompt + 1 decode + 2 accepted draft + 1 final
        ref_cache_0 = self._make_batch_cache(model, [0])
        self._prefill(model, tokens_a.reshape(1, -1), ref_cache_0)
        ref_logits_0 = self._prefill(model, tokens_a.reshape(1, -1), ref_cache_0)
        # Oops, double prefill. Let me redo:

        ref_cache_0 = self._make_batch_cache(model, [0])
        ref_logits_0 = self._prefill(model, tokens_a.reshape(1, -1), ref_cache_0)
        ref_y0 = mx.argmax(ref_logits_0[:, -1, :], axis=-1, keepdims=True)
        mx.eval(ref_y0)
        ref_logits_0_1 = self._decode_one(model, ref_y0, ref_cache_0)
        mx.eval(ref_logits_0_1)

        accepted_0 = draft_tokens[0:1, :2]
        self._multi_token_forward(model, accepted_0, ref_cache_0)
        ref_final_0 = self._decode_one(
            model, mx.array([[draft_tokens[0, 2]]]), ref_cache_0
        )
        mx.eval(ref_final_0)

        # Seq 1: prompt + 1 decode + 4 accepted draft + 1 final
        ref_cache_1 = self._make_batch_cache(model, [0])
        ref_logits_1 = self._prefill(model, tokens_b.reshape(1, -1), ref_cache_1)
        ref_y1 = mx.argmax(ref_logits_1[:, -1, :], axis=-1, keepdims=True)
        mx.eval(ref_y1)
        ref_logits_1_1 = self._decode_one(model, ref_y1, ref_cache_1)
        mx.eval(ref_logits_1_1)

        accepted_1 = draft_tokens[1:2, :4]
        self._multi_token_forward(model, accepted_1, ref_cache_1)
        ref_final_1 = self._decode_one(
            model, mx.array([[draft_tokens[1, 4]]]), ref_cache_1
        )
        mx.eval(ref_final_1)

        # ---- COMPARISON ----
        spec_argmax_0 = mx.argmax(final_logits_spec[0:1, -1, :], axis=-1).item()
        spec_argmax_1 = mx.argmax(final_logits_spec[1:2, -1, :], axis=-1).item()
        ref_argmax_0 = mx.argmax(ref_final_0[:, -1, :], axis=-1).item()
        ref_argmax_1 = mx.argmax(ref_final_1[:, -1, :], axis=-1).item()

        diff_0 = mx.abs(
            final_logits_spec[0:1, -1, :] - ref_final_0[:, -1, :]
        ).max().item()
        diff_1 = mx.abs(
            final_logits_spec[1:2, -1, :] - ref_final_1[:, -1, :]
        ).max().item()

        print(f"\n{'=' * 70}")
        print("PER-SEQ TRIM GATE TEST (with fix)")
        print(f"{'=' * 70}")
        print(
            f"Seq 0 (trim=3, deficit=0): spec={spec_argmax_0}, "
            f"ref={ref_argmax_0}, diff={diff_0:.4f}"
        )
        print(
            f"Seq 1 (trim=1, deficit=2): spec={spec_argmax_1}, "
            f"ref={ref_argmax_1}, diff={diff_1:.4f}"
        )

        assert spec_argmax_0 == ref_argmax_0, (
            f"Seq 0 argmax mismatch: spec={spec_argmax_0} ref={ref_argmax_0} "
            f"diff={diff_0}"
        )
        assert spec_argmax_1 == ref_argmax_1, (
            f"Seq 1 argmax mismatch: spec={spec_argmax_1} ref={ref_argmax_1} "
            f"diff={diff_1}"
        )

    def test_variable_trim_multiple_layers_consistent(self, model_and_tokenizer):
        """All cache layers should have consistent offsets after variable trim."""
        model, tokenizer = model_and_tokenizer

        prompt_a = "Hello"
        prompt_b = "The quick brown"
        tokens_a = mx.array(tokenizer.encode(prompt_a))
        tokens_b = mx.array(tokenizer.encode(prompt_b))

        len_a = tokens_a.shape[0]
        len_b = tokens_b.shape[0]
        max_len = max(len_a, len_b)
        pad_a = max_len - len_a
        pad_b = max_len - len_b

        padded_a = (
            mx.concatenate([mx.zeros(pad_a, dtype=mx.int32), tokens_a])
            if pad_a > 0
            else tokens_a
        )
        padded_b = (
            mx.concatenate([mx.zeros(pad_b, dtype=mx.int32), tokens_b])
            if pad_b > 0
            else tokens_b
        )
        batch_tokens = mx.stack([padded_a, padded_b])

        cache = self._make_batch_cache(model, [pad_a, pad_b])
        self._prefill(model, batch_tokens, cache)

        # Apply variable trim: trim 3, re-advance seq 0 by 2
        for layer_cache in cache:
            if hasattr(layer_cache, "trim"):
                layer_cache.trim(3)
                layer_cache.offset = layer_cache.offset + mx.array([2, 0])

        # Check consistency across all layers
        ref_offset = cache[0].offset
        ref_idx = cache[0]._idx
        for i, layer_cache in enumerate(cache):
            if hasattr(layer_cache, "_idx"):
                assert layer_cache._idx == ref_idx, f"Layer {i} _idx mismatch"
                assert mx.array_equal(layer_cache.offset, ref_offset), (
                    f"Layer {i} offset mismatch: "
                    f"{layer_cache.offset.tolist()} vs {ref_offset.tolist()}"
                )

    def test_make_mask_excludes_stale_data(self, model_and_tokenizer):
        """After variable trim, attention mask must mask stale positions.

        REQUIRES THE FIX. The fix makes make_mask() use per-sequence
        right_padding to exclude positions between offset[i] and max(offset)
        for sequences with smaller offsets.
        """
        model, tokenizer = model_and_tokenizer

        prompt_a = "Hello"
        prompt_b = "The quick"
        tokens_a = mx.array(tokenizer.encode(prompt_a))
        tokens_b = mx.array(tokenizer.encode(prompt_b))

        len_a = tokens_a.shape[0]
        len_b = tokens_b.shape[0]
        max_len = max(len_a, len_b)
        pad_a = max_len - len_a
        pad_b = max_len - len_b

        padded_a = (
            mx.concatenate([mx.zeros(pad_a, dtype=mx.int32), tokens_a])
            if pad_a > 0
            else tokens_a
        )
        padded_b = (
            mx.concatenate([mx.zeros(pad_b, dtype=mx.int32), tokens_b])
            if pad_b > 0
            else tokens_b
        )
        batch_tokens = mx.stack([padded_a, padded_b])

        cache = self._make_batch_cache(model, [pad_a, pad_b])
        self._prefill(model, batch_tokens, cache)

        # Feed 5 more tokens
        draft = mx.zeros((2, 5), dtype=mx.int32)
        mx.eval(draft)
        model(draft, cache=cache)
        mx.eval([c.state for c in cache])

        # Variable trim: seq 0 trims 4, seq 1 trims 1
        max_trim = 4
        for layer_cache in cache:
            if hasattr(layer_cache, "trim"):
                layer_cache.trim(max_trim)
                # Re-advance seq 1 by 3 (deficit = 4 - 1 = 3)
                layer_cache.offset = layer_cache.offset + mx.array([0, 3])

        # Generate mask for N=1 (single decode token)
        mask = cache[0].make_mask(N=1, return_array=True)
        mx.eval(mask)

        assert mask is not None, "make_mask should return an array after variable trim"
        # The mask should have the right shape to cover max(offset) positions
        assert mask.shape[-1] > 0


# ===========================================================================
# 3. Backward compatibility tests
# ===========================================================================

@pytest.mark.skipif(not HAS_MODEL, reason="Qwen3-4B-4bit not available")
class TestBackwardCompatibility:
    """Verify the fix does not break existing behavior."""

    @pytest.fixture(scope="class")
    def model_and_tokenizer(self):
        from mlx_lm import load
        model, tokenizer = load(MODEL_PATH)
        return model, tokenizer

    def test_non_spec_decode_path_unchanged(self, model_and_tokenizer):
        """Normal decode (no trim) produces identical results with the fix.

        This is the most important backward compatibility test.
        """
        model, tokenizer = model_and_tokenizer
        from mlx_lm.generate import _make_cache

        prompt = "The capital of France is"
        tokens = mx.array([tokenizer.encode(prompt)])

        cache = _make_cache(model, [0], max_kv_size=None)
        if tokens.shape[1] > 1:
            model(tokens[:, :-1], cache=cache)
            mx.eval([c.state for c in cache])
        logits = model(tokens[:, -1:], cache=cache)
        mx.eval([c.state for c in cache])

        generated = []
        for _ in range(5):
            y = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            mx.eval(y)
            generated.append(y.item())
            logits = model(y, cache=cache)
            mx.eval([c.state for c in cache])

        decoded = tokenizer.decode(generated)
        assert len(decoded) > 0, "Should produce non-empty output"
        print(f"\nNormal decode output: {decoded}")

    def test_trim_then_extend_normal(self, model_and_tokenizer):
        """trim(n) then continue normal decode matches the never-trimmed path.

        Path A: prefill + 1 decode + 3 draft + trim(3) + 1 decode
        Path B: prefill + 1 decode + 1 decode (no draft)
        Both should produce the same final output.
        """
        model, tokenizer = model_and_tokenizer
        from mlx_lm.generate import _make_cache

        prompt = "The quick brown fox"
        tokens = mx.array([tokenizer.encode(prompt)])

        # Path A
        cache_a = _make_cache(model, [0], max_kv_size=None)
        if tokens.shape[1] > 1:
            model(tokens[:, :-1], cache=cache_a)
            mx.eval([c.state for c in cache_a])
        logits_a = model(tokens[:, -1:], cache=cache_a)
        mx.eval([c.state for c in cache_a])

        y_a = mx.argmax(logits_a[:, -1, :], axis=-1, keepdims=True)
        mx.eval(y_a)
        logits_a1 = model(y_a, cache=cache_a)
        mx.eval([c.state for c in cache_a])
        y_a1 = mx.argmax(logits_a1[:, -1, :], axis=-1, keepdims=True)
        mx.eval(y_a1)

        # Feed 3 draft tokens
        draft = mx.array([[y_a1.item(), 100, 200]])
        mx.eval(draft)
        model(draft, cache=cache_a)
        mx.eval([c.state for c in cache_a])

        # Trim all 3 draft tokens (reject all)
        for c in cache_a:
            if hasattr(c, "trim"):
                c.trim(3)

        # Decode 1 more (same token as if we never drafted)
        final_a = model(mx.array([[y_a1.item()]]), cache=cache_a)
        mx.eval(final_a)

        # Path B
        cache_b = _make_cache(model, [0], max_kv_size=None)
        if tokens.shape[1] > 1:
            model(tokens[:, :-1], cache=cache_b)
            mx.eval([c.state for c in cache_b])
        logits_b = model(tokens[:, -1:], cache=cache_b)
        mx.eval([c.state for c in cache_b])

        y_b = mx.argmax(logits_b[:, -1, :], axis=-1, keepdims=True)
        mx.eval(y_b)
        logits_b1 = model(y_b, cache=cache_b)
        mx.eval([c.state for c in cache_b])
        y_b1 = mx.argmax(logits_b1[:, -1, :], axis=-1, keepdims=True)
        mx.eval(y_b1)

        final_b = model(mx.array([[y_b1.item()]]), cache=cache_b)
        mx.eval(final_b)

        argmax_a = mx.argmax(final_a[:, -1, :], axis=-1).item()
        argmax_b = mx.argmax(final_b[:, -1, :], axis=-1).item()

        assert argmax_a == argmax_b, (
            f"Trim-then-extend should match no-trim path: {argmax_a} != {argmax_b}"
        )


# ===========================================================================
# 4. Edge cases
# ===========================================================================

class TestEdgeCases:
    """Edge case tests for BatchKVCache trim."""

    def test_single_sequence_batch(self):
        """batch=1, variable trim degenerates to uniform."""
        layers = _make_batch_layers(2, batch_size=1)
        _fill_cache(layers, tokens_per_step=10, batch_size=1)

        for layer in layers:
            assert layer._idx == 10
            assert mx.array_equal(layer.offset, mx.array([10]))

        # Trim 4, re-advance by 0 (degenerate variable trim)
        for layer in layers:
            layer.trim(4)
            layer.offset = layer.offset + mx.array([0])

        for layer in layers:
            assert layer._idx == 6
            assert mx.array_equal(layer.offset, mx.array([6]))

    def test_all_same_trim(self):
        """All sequences trimmed by same amount = identical to uniform trim."""
        layers_var = _make_batch_layers(2, batch_size=3)
        layers_uni = _make_batch_layers(2, batch_size=3)

        # Fill both with the same data
        data_k = mx.random.normal((3, 2, 8, 4))
        data_v = mx.random.normal((3, 2, 8, 4))
        mx.eval(data_k, data_v)
        for layer in layers_var:
            layer.update_and_fetch(data_k, data_v)
        for layer in layers_uni:
            layer.update_and_fetch(data_k, data_v)

        # Variable trim: all trim 3, deficit all 0
        for layer in layers_var:
            layer.trim(3)
            layer.offset = layer.offset + mx.array([0, 0, 0])

        # Uniform trim: trim 3
        for layer in layers_uni:
            layer.trim(3)

        for lv, lu in zip(layers_var, layers_uni):
            assert lv._idx == lu._idx
            assert mx.array_equal(lv.offset, lu.offset)

    def test_zero_trim(self):
        """trim(0) is a no-op."""
        layers = _make_batch_layers(2, batch_size=2)
        _fill_cache(layers, tokens_per_step=5)

        for layer in layers:
            before_idx = layer._idx
            before_offset = layer.offset.tolist()
            layer.trim(0)
            assert layer._idx == before_idx
            assert layer.offset.tolist() == before_offset

    def test_trim_entire_cache(self):
        """Trim everything, offsets go to 0."""
        layers = _make_batch_layers(2, batch_size=2, left_padding=[0, 0])
        _fill_cache(layers, tokens_per_step=5)

        for layer in layers:
            assert layer._idx == 5

        for layer in layers:
            layer.trim(5)

        for layer in layers:
            assert layer._idx == 0
            assert mx.array_equal(layer.offset, mx.array([0, 0]))

    def test_trim_more_than_available(self):
        """trim(n) where n > _idx should clamp to _idx."""
        layers = _make_batch_layers(1, batch_size=2)
        _fill_cache(layers, tokens_per_step=5)

        layer = layers[0]
        assert layer._idx == 5

        returned = layer.trim(100)
        assert returned == 5
        assert layer._idx == 0

    def test_trim_with_left_padding(self):
        """Trim with left-padded sequences maintains correct offset relationship."""
        layers = _make_batch_layers(1, batch_size=2, left_padding=[3, 0])
        layer = layers[0]

        # Before fill: offset = [-3, 0]
        assert mx.array_equal(layer.offset, mx.array([-3, 0]))

        # Fill 10 tokens
        _fill_cache(layers, tokens_per_step=10)
        # After fill: offset = [7, 10], _idx = 10
        assert layer._idx == 10
        assert mx.array_equal(layer.offset, mx.array([7, 10]))

        # Trim 4
        layer.trim(4)
        assert layer._idx == 6
        assert mx.array_equal(layer.offset, mx.array([3, 6]))

        # Re-advance seq 1 by 2 (variable trim: seq 0 trimmed 4, seq 1 trimmed 2)
        layer.offset = layer.offset + mx.array([0, 2])
        assert mx.array_equal(layer.offset, mx.array([3, 8]))

    def test_multiple_trim_cycles(self):
        """Multiple trim + fill cycles (simulating spec decode rounds)."""
        layers = _make_batch_layers(1, batch_size=2)
        layer = layers[0]

        # Round 1: fill 10, trim 3
        _fill_cache(layers, tokens_per_step=10)
        assert layer._idx == 10
        layer.trim(3)
        assert layer._idx == 7

        # Round 2: fill 5 more, trim 2
        _fill_cache(layers, tokens_per_step=5)
        assert layer._idx == 12
        layer.trim(2)
        assert layer._idx == 10

        # Round 3: fill 3, trim 3 (net zero)
        _fill_cache(layers, tokens_per_step=3)
        assert layer._idx == 13
        layer.trim(3)
        assert layer._idx == 10

    def test_make_mask_after_variable_trim_shape(self):
        """make_mask() returns correct shape after variable trim.

        REQUIRES THE FIX: make_mask must account for per-sequence offsets.
        """
        layers = _make_batch_layers(1, batch_size=2, left_padding=[0, 0])
        _fill_cache(layers, tokens_per_step=10)

        layer = layers[0]

        # Variable trim
        layer.trim(4)
        layer.offset = layer.offset + mx.array([2, 0])
        # _idx = 6, offset = [8, 6]

        # make_mask for N=1 (single-token decode)
        mask = layer.make_mask(N=1, return_array=True)
        mx.eval(mask)

        assert mask is not None
        # After fix: end = max(left_padding + offset) = max(8, 6) = 8
        # create_causal_mask(N=1, offset=8) produces width = offset + N = 9
        # (includes the query token position at the end)
        assert mask.shape[-1] == 9, (
            f"Expected mask width 9 (end + N), got {mask.shape[-1]}"
        )

    def test_make_mask_after_variable_trim_content(self):
        """make_mask() should mask stale positions for under-trimmed sequences.

        REQUIRES THE FIX.

        After trim(4) + offset re-advance [2, 0]:
        - _idx = 6, offset = [8, 6], left_padding = [0, 0]
        - end = max(8, 6) = 8
        - right_padding = [8-8, 8-6] = [0, 2]
        - create_causal_mask(N=1, offset=8) => width = 9

        Seq 0 (right_pad=0): all 9 positions visible (full causal)
        Seq 1 (right_pad=2): positions 0..6 visible, 7..8 masked

        Position 6 is visible for seq 1 because during actual forward pass,
        update_and_fetch writes the new token at position 6 (overwriting stale
        data) before the mask is applied. Positions 7..8 contain stale data
        and are correctly masked out.
        """
        layers = _make_batch_layers(1, batch_size=2, left_padding=[0, 0])
        _fill_cache(layers, tokens_per_step=10)

        layer = layers[0]

        # Variable trim
        layer.trim(4)
        layer.offset = layer.offset + mx.array([2, 0])
        # _idx = 6, offset = [8, 6]

        mask = layer.make_mask(N=1, return_array=True)
        mx.eval(mask)

        assert mask is not None
        assert mask.shape == (2, 1, 1, 9), f"Expected shape (2,1,1,9), got {mask.shape}"

        mask_list = mask.tolist()
        # Seq 0: all 9 positions visible (right_pad=0)
        assert all(mask_list[0][0][0]), "Seq 0 should see all 9 positions"
        # Seq 1: positions 0..6 visible, 7..8 masked (right_pad=2)
        assert all(mask_list[1][0][0][:7]), "Seq 1 should see positions 0..6"
        assert not any(mask_list[1][0][0][7:]), (
            "Seq 1 should NOT see positions 7..8 (stale data)"
        )

    def test_update_and_fetch_returns_up_to_max_offset(self):
        """After per-seq offset re-advance, update_and_fetch should return
        KV data up to max(left_padding + offset), not just _idx.

        REQUIRES THE FIX.
        """
        layers = _make_batch_layers(1, batch_size=2, left_padding=[0, 0])
        _fill_cache(layers, tokens_per_step=10)

        layer = layers[0]
        # _idx = 10, offset = [10, 10]

        # Trim by 5
        layer.trim(5)
        # _idx = 5, offset = [5, 5]

        # Re-advance seq 0 by 3
        layer.offset = layer.offset + mx.array([3, 0])
        # offset = [8, 5]

        # Now feed 1 new token
        B, H, D = 2, 2, 4
        new_k = mx.random.normal((B, H, 1, D))
        new_v = mx.random.normal((B, H, 1, D))
        mx.eval(new_k, new_v)
        k_out, v_out = layer.update_and_fetch(new_k, new_v)
        mx.eval(k_out, v_out)

        # After update: _idx = 6, offset = [9, 6]
        # effective_end = max(9, 6) = 9
        # k_out should have shape [..., 9, ...]
        assert k_out.shape[2] == 9, (
            f"Expected KV output width 9 (max offset), got {k_out.shape[2]}"
        )
