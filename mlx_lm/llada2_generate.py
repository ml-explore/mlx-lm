"""
LLaDA2 MoE (Large Language Diffusion with mAsking) generation implementation.

LLaDA2 uses a block-wise iterative refinement (diffusion) process for text
generation, which differs from traditional autoregressive generation.

Reference: https://huggingface.co/inclusionAI/LLaDA2.0-mini
"""

import mlx.core as mx

from .models.cache import make_prompt_cache


def get_num_transfer_tokens(block_length: int, steps: int) -> mx.array:
    """
    Compute the number of tokens to unmask at each step.

    Distributes tokens evenly across steps, with remainder going to early steps.

    Args:
        block_length: Number of tokens in a block
        steps: Number of denoising steps

    Returns:
        Array of shape (steps,) with number of tokens to unmask per step
    """
    if steps == 0:
        return mx.array([], dtype=mx.int32)

    base = block_length // steps
    remainder = block_length % steps

    num_transfer_tokens = mx.full((steps,), base, dtype=mx.int32)

    if remainder > 0:
        num_transfer_tokens = mx.concatenate([
            num_transfer_tokens[:remainder] + 1,
            num_transfer_tokens[remainder:]
        ])

    return num_transfer_tokens


def create_block_diagonal_mask(num_blocks: int, block_length: int) -> mx.array:
    """
    Create block-diagonal causal attention mask.

    Tokens within a block can attend to all previous blocks but not future blocks.

    Args:
        num_blocks: Number of blocks
        block_length: Tokens per block

    Returns:
        Attention mask of shape (1, 1, total_len, total_len)
    """
    total_length = num_blocks * block_length

    row_idx = mx.arange(total_length)
    col_idx = mx.arange(total_length)

    row_blocks = row_idx[:, None] // block_length
    col_blocks = col_idx[None, :] // block_length

    attend_mask = row_blocks >= col_blocks

    mask = mx.where(
        attend_mask,
        mx.array(0.0, dtype=mx.bfloat16),
        mx.array(float("-inf"), dtype=mx.bfloat16)
    )

    return mask[None, None, :, :]


def top_k_logits(logits: mx.array, k: int) -> mx.array:
    """Apply top-k filtering to logits."""
    if k is None or k <= 0:
        return logits
    top_k_values = mx.topk(logits, k=k, axis=-1)
    min_value = top_k_values[:, :, -1:]
    return mx.where(logits < min_value, mx.full(logits.shape, float("-inf")), logits)


def top_p_logits(logits: mx.array, p: float) -> mx.array:
    """Apply top-p (nucleus) filtering to logits."""
    if p is None or p >= 1.0:
        return logits

    sorted_indices = mx.argsort(-logits, axis=-1)
    sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)

    sorted_probs = mx.softmax(sorted_logits, axis=-1)
    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

    sorted_mask = cumulative_probs > p
    sorted_mask = mx.concatenate([
        mx.zeros((*sorted_mask.shape[:-1], 1), dtype=mx.bool_),
        sorted_mask[..., :-1]
    ], axis=-1)

    mask = mx.zeros_like(logits, dtype=mx.bool_)
    mask = mx.put_along_axis(mask, sorted_indices, sorted_mask, axis=-1)

    return mx.where(mask, mx.full(logits.shape, float("-inf")), logits)


def sample_tokens(
    logits: mx.array,
    temperature: float = 0.0,
    top_k: int = None,
    top_p: float = None,
) -> tuple:
    """
    Sample tokens with temperature, top-k, and top-p filtering.

    Args:
        logits: Model output logits (batch, seq_len, vocab_size)
        temperature: Sampling temperature (0 = greedy)
        top_k: Top-k filtering threshold
        top_p: Top-p (nucleus) filtering threshold

    Returns:
        Tuple of (sampled_tokens, token_probabilities)
    """
    if temperature > 0 and temperature != 1.0:
        logits = logits / temperature

    if top_k is not None and top_k > 0:
        logits = top_k_logits(logits, top_k)

    if top_p is not None and top_p < 1.0:
        logits = top_p_logits(logits, top_p)

    probs = mx.softmax(logits, axis=-1)

    if temperature == 0:
        tokens = mx.argmax(logits, axis=-1)
    else:
        orig_shape = probs.shape[:-1]
        flat_probs = probs.reshape(-1, probs.shape[-1])
        flat_tokens = mx.random.categorical(mx.log(flat_probs + 1e-10))
        tokens = flat_tokens.reshape(orig_shape)

    token_probs = mx.take_along_axis(
        probs, mx.expand_dims(tokens, axis=-1), axis=-1
    ).squeeze(-1)

    return tokens, token_probs


def _generate_no_cache(
    model,
    input_ids: mx.array,
    max_new_tokens: int = 256,
    block_length: int = 32,
    steps: int = 32,
    temperature: float = 0.0,
    top_p: float = None,
    top_k: int = None,
    threshold: float = 0.95,
    mask_id: int = 156895,
    eos_id: int = 156892,
    eos_early_stop: bool = True,
) -> mx.array:
    """
    Generate tokens without KV-cache (original implementation).

    Forwards the full prefix at every denoising step.
    """
    prompt_length = input_ids.shape[1]

    total_gen_length = prompt_length + max_new_tokens
    num_blocks = (total_gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length

    # Initialize sequence with mask tokens, copy prompt
    x = mx.full((1, total_length), mask_id, dtype=mx.int32)
    x = mx.concatenate([input_ids, x[:, prompt_length:]], axis=1)

    # Create block-diagonal attention mask
    full_mask = create_block_diagonal_mask(num_blocks, block_length)

    # Token transfer schedule
    denoising_steps = min(steps, block_length)
    num_transfer_schedule = get_num_transfer_tokens(block_length, denoising_steps)

    # Process blocks starting after prompt
    prefill_blocks = prompt_length // block_length

    for num_block in range(prefill_blocks, num_blocks):
        current_window_end = (num_block + 1) * block_length

        cur_x = x[:, :current_window_end]
        cur_mask = full_mask[:, :, :current_window_end, :current_window_end]

        for step in range(denoising_steps):
            active_block = cur_x[:, -block_length:]
            active_mask = active_block == mask_id
            num_masks = mx.sum(active_mask).item()

            if num_masks == 0:
                break

            # Forward pass with block-diagonal mask
            logits = model(cur_x, mask=cur_mask)

            # Get logits for current block only
            active_logits = logits[:, -block_length:, :]

            # Sample tokens
            sampled_tokens, token_probs = sample_tokens(
                active_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            # Determine which tokens to transfer
            num_to_transfer = int(num_transfer_schedule[step].item())

            # Mask out non-mask positions for confidence calculation
            confidence = mx.where(active_mask, token_probs, mx.array(-float("inf")))

            # Find high confidence tokens above threshold
            high_conf_mask = confidence > threshold
            num_high_conf = mx.sum(high_conf_mask).item()

            if num_high_conf >= num_to_transfer:
                transfer_mask = high_conf_mask
            else:
                flat_conf = confidence.reshape(-1)
                k = min(num_to_transfer, int(num_masks))
                if k > 0:
                    top_indices = mx.argpartition(-flat_conf, kth=k-1)[:k]
                    transfer_mask = mx.zeros(flat_conf.shape, dtype=mx.bool_)
                    transfer_mask = transfer_mask.at[top_indices].add(True)
                    transfer_mask = transfer_mask.reshape(active_mask.shape)
                else:
                    transfer_mask = mx.zeros(active_mask.shape, dtype=mx.bool_)

            # Update tokens
            new_block = mx.where(transfer_mask, sampled_tokens, active_block)
            cur_x = mx.concatenate([cur_x[:, :-block_length], new_block], axis=1)

            # Check for EOS
            if eos_early_stop:
                if mx.any(new_block == eos_id).item():
                    eos_mask = cur_x[0] == eos_id
                    indices = mx.arange(cur_x.shape[1])
                    eos_indices = mx.where(eos_mask, indices, mx.array(cur_x.shape[1]))
                    eos_pos = int(mx.min(eos_indices).item())
                    if eos_pos < cur_x.shape[1]:
                        prefix = cur_x[0, prompt_length:eos_pos]
                        if not mx.any(prefix == mask_id).item():
                            return cur_x[:, :eos_pos + 1]

            mx.eval(cur_x)

        # Update full sequence
        x = mx.concatenate([cur_x, x[:, current_window_end:]], axis=1)

    # Remove trailing mask tokens
    non_mask = x[0] != mask_id
    if mx.any(non_mask).item():
        indices = mx.arange(x.shape[1])
        non_mask_indices = mx.where(non_mask, indices, mx.array(-1))
        last_non_mask = int(mx.max(non_mask_indices).item())
        x = x[:, :last_non_mask + 1]

    return x


def _generate_cached(
    model,
    input_ids: mx.array,
    max_new_tokens: int = 256,
    block_length: int = 32,
    steps: int = 32,
    temperature: float = 0.0,
    top_p: float = None,
    top_k: int = None,
    threshold: float = 0.95,
    mask_id: int = 156895,
    eos_id: int = 156892,
    eos_early_stop: bool = True,
) -> mx.array:
    """
    Generate tokens with KV-cache.

    Caches K/V for the static prefix (prompt + completed blocks) and only
    forwards the current block's tokens through the model at each denoising step.
    """
    prompt_length = input_ids.shape[1]

    total_gen_length = prompt_length + max_new_tokens
    num_blocks = (total_gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length

    # Initialize sequence with mask tokens, copy prompt
    x = mx.full((1, total_length), mask_id, dtype=mx.int32)
    x = mx.concatenate([input_ids, x[:, prompt_length:]], axis=1)

    # Create KV-cache (one per layer)
    cache = make_prompt_cache(model)

    # Token transfer schedule
    denoising_steps = min(steps, block_length)
    num_transfer_schedule = get_num_transfer_tokens(block_length, denoising_steps)

    # Number of complete prompt blocks
    prefill_blocks = prompt_length // block_length

    # Prefill prompt blocks to populate cache
    if prefill_blocks > 0:
        prefill_length = prefill_blocks * block_length
        prefill_tokens = x[:, :prefill_length]
        prefill_mask = create_block_diagonal_mask(prefill_blocks, block_length)
        logits = model(prefill_tokens, cache=cache, mask=prefill_mask)
        mx.eval(logits)

    for num_block in range(prefill_blocks, num_blocks):
        block_start = num_block * block_length
        block_end = (num_block + 1) * block_length

        # Record prefix offset for trimming back after each step
        prefix_offset = cache[0].offset

        # Full-attention mask: block tokens attend to all cached + block tokens
        block_mask = mx.zeros(
            (1, 1, block_length, prefix_offset + block_length),
            dtype=mx.bfloat16,
        )

        block_tokens = x[:, block_start:block_end]

        for step in range(denoising_steps):
            active_mask = block_tokens == mask_id
            num_masks = mx.sum(active_mask).item()

            if num_masks == 0:
                break

            # Trim cache back to prefix (remove stale block K/V)
            trim_amount = cache[0].offset - prefix_offset
            if trim_amount > 0:
                for c in cache:
                    c.trim(trim_amount)

            # Forward pass with only the block tokens
            logits = model(block_tokens, cache=cache, mask=block_mask)

            # Sample tokens (logits are already block-sized)
            sampled_tokens, token_probs = sample_tokens(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            # Determine which tokens to transfer
            num_to_transfer = int(num_transfer_schedule[step].item())

            # Mask out non-mask positions for confidence calculation
            confidence = mx.where(active_mask, token_probs, mx.array(-float("inf")))

            # Find high confidence tokens above threshold
            high_conf_mask = confidence > threshold
            num_high_conf = mx.sum(high_conf_mask).item()

            if num_high_conf >= num_to_transfer:
                transfer_mask = high_conf_mask
            else:
                flat_conf = confidence.reshape(-1)
                k = min(num_to_transfer, int(num_masks))
                if k > 0:
                    top_indices = mx.argpartition(-flat_conf, kth=k-1)[:k]
                    transfer_mask = mx.zeros(flat_conf.shape, dtype=mx.bool_)
                    transfer_mask = transfer_mask.at[top_indices].add(True)
                    transfer_mask = transfer_mask.reshape(active_mask.shape)
                else:
                    transfer_mask = mx.zeros(active_mask.shape, dtype=mx.bool_)

            # Update block tokens
            block_tokens = mx.where(transfer_mask, sampled_tokens, block_tokens)

            # Check for EOS
            if eos_early_stop:
                if mx.any(block_tokens == eos_id).item():
                    x_so_far = mx.concatenate(
                        [x[:, :block_start], block_tokens], axis=1
                    )
                    eos_mask = x_so_far[0] == eos_id
                    indices = mx.arange(x_so_far.shape[1])
                    eos_indices = mx.where(
                        eos_mask, indices, mx.array(x_so_far.shape[1])
                    )
                    eos_pos = int(mx.min(eos_indices).item())
                    if eos_pos < x_so_far.shape[1]:
                        prefix = x_so_far[0, prompt_length:eos_pos]
                        if not mx.any(prefix == mask_id).item():
                            return x_so_far[:, :eos_pos + 1]

            mx.eval(block_tokens)

        # Update full sequence with finalized block
        x = mx.concatenate(
            [x[:, :block_start], block_tokens, x[:, block_end:]], axis=1
        )

        # Commit finalized block to cache (K/V from last denoising step are
        # stale because tokens changed after that forward pass)
        trim_amount = cache[0].offset - prefix_offset
        if trim_amount > 0:
            for c in cache:
                c.trim(trim_amount)
        commit_logits = model(block_tokens, cache=cache, mask=block_mask)
        mx.eval(commit_logits)

    # Remove trailing mask tokens
    non_mask = x[0] != mask_id
    if mx.any(non_mask).item():
        indices = mx.arange(x.shape[1])
        non_mask_indices = mx.where(non_mask, indices, mx.array(-1))
        last_non_mask = int(mx.max(non_mask_indices).item())
        x = x[:, :last_non_mask + 1]

    return x


def generate(
    model,
    input_ids: mx.array,
    max_new_tokens: int = 256,
    block_length: int = 32,
    steps: int = 32,
    temperature: float = 0.0,
    top_p: float = None,
    top_k: int = None,
    threshold: float = 0.95,
    mask_id: int = 156895,
    eos_id: int = 156892,
    eos_early_stop: bool = True,
    use_cache: bool = True,
) -> mx.array:
    """
    Generate tokens using LLaDA2's block-wise iterative refinement (diffusion) strategy.

    Args:
        model: LLaDA2 MoE model instance
        input_ids: Input token IDs (1, prompt_length)
        max_new_tokens: Maximum number of new tokens to generate
        block_length: Size of each generation block
        steps: Number of refinement steps per block
        temperature: Sampling temperature (0 = greedy)
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling threshold
        threshold: Confidence threshold for accepting tokens
        mask_id: Token ID used for masked positions
        eos_id: End-of-sequence token ID
        eos_early_stop: Whether to stop early on EOS token
        use_cache: Whether to use KV-cache for the static prefix

    Returns:
        Generated token IDs including prompt
    """
    fn = _generate_cached if use_cache else _generate_no_cache
    return fn(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        block_length=block_length,
        steps=steps,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        threshold=threshold,
        mask_id=mask_id,
        eos_id=eos_id,
        eos_early_stop=eos_early_stop,
    )


def _stream_generate_no_cache(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    block_length: int = 32,
    steps: int = 32,
    temperature: float = 0.0,
    top_p: float = None,
    top_k: int = None,
    threshold: float = 0.95,
    mask_id: int = 156895,
    eos_id: int = 156892,
):
    """
    Stream-generate text without KV-cache (original implementation).

    Forwards the full prefix at every denoising step.
    """
    # Format prompt with chat template if available
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    else:
        formatted = prompt

    input_ids = mx.array([tokenizer.encode(formatted)])
    prompt_length = input_ids.shape[1]

    total_gen_length = prompt_length + max_new_tokens
    num_blocks = (total_gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length

    # Initialize sequence
    x = mx.full((1, total_length), mask_id, dtype=mx.int32)
    x = mx.concatenate([input_ids, x[:, prompt_length:]], axis=1)

    full_mask = create_block_diagonal_mask(num_blocks, block_length)
    denoising_steps = min(steps, block_length)
    num_transfer_schedule = get_num_transfer_tokens(block_length, denoising_steps)

    prefill_blocks = prompt_length // block_length

    for num_block in range(prefill_blocks, num_blocks):
        current_window_end = (num_block + 1) * block_length

        cur_x = x[:, :current_window_end]
        cur_mask = full_mask[:, :, :current_window_end, :current_window_end]

        for step in range(denoising_steps):
            active_block = cur_x[:, -block_length:]
            active_mask = active_block == mask_id
            num_masks = mx.sum(active_mask).item()

            if num_masks == 0:
                break

            logits = model(cur_x, mask=cur_mask)
            active_logits = logits[:, -block_length:, :]

            sampled_tokens, token_probs = sample_tokens(
                active_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            num_to_transfer = int(num_transfer_schedule[step].item())
            confidence = mx.where(active_mask, token_probs, mx.array(-float("inf")))

            high_conf_mask = confidence > threshold
            num_high_conf = mx.sum(high_conf_mask).item()

            if num_high_conf >= num_to_transfer:
                transfer_mask = high_conf_mask
            else:
                flat_conf = confidence.reshape(-1)
                k = min(num_to_transfer, int(num_masks))
                if k > 0:
                    top_indices = mx.argpartition(-flat_conf, kth=k-1)[:k]
                    transfer_mask = mx.zeros(flat_conf.shape, dtype=mx.bool_)
                    transfer_mask = transfer_mask.at[top_indices].add(True)
                    transfer_mask = transfer_mask.reshape(active_mask.shape)
                else:
                    transfer_mask = mx.zeros(active_mask.shape, dtype=mx.bool_)

            new_block = mx.where(transfer_mask, sampled_tokens, active_block)
            cur_x = mx.concatenate([cur_x[:, :-block_length], new_block], axis=1)

            # Check for EOS
            if mx.any(new_block == eos_id).item():
                eos_mask = cur_x[0] == eos_id
                indices = mx.arange(cur_x.shape[1])
                eos_indices = mx.where(eos_mask, indices, mx.array(cur_x.shape[1]))
                eos_pos = int(mx.min(eos_indices).item())
                if eos_pos < cur_x.shape[1]:
                    prefix = cur_x[0, prompt_length:eos_pos]
                    if not mx.any(prefix == mask_id).item():
                        # Yield final text and return
                        generated = cur_x[0, prompt_length:eos_pos + 1].tolist()
                        yield tokenizer.decode(generated, skip_special_tokens=True)
                        return

            mx.eval(cur_x)

        x = mx.concatenate([cur_x, x[:, current_window_end:]], axis=1)

        # Yield text generated so far
        generated = x[0, prompt_length:current_window_end].tolist()
        # Filter out mask tokens for display
        generated = [t for t in generated if t != mask_id]
        yield tokenizer.decode(generated, skip_special_tokens=True)


def _stream_generate_cached(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    block_length: int = 32,
    steps: int = 32,
    temperature: float = 0.0,
    top_p: float = None,
    top_k: int = None,
    threshold: float = 0.95,
    mask_id: int = 156895,
    eos_id: int = 156892,
):
    """
    Stream-generate text with KV-cache.

    Caches K/V for the static prefix and only forwards the current block's
    tokens through the model at each denoising step.
    """
    # Format prompt with chat template if available
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    else:
        formatted = prompt

    input_ids = mx.array([tokenizer.encode(formatted)])
    prompt_length = input_ids.shape[1]

    total_gen_length = prompt_length + max_new_tokens
    num_blocks = (total_gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length

    # Initialize sequence
    x = mx.full((1, total_length), mask_id, dtype=mx.int32)
    x = mx.concatenate([input_ids, x[:, prompt_length:]], axis=1)

    # Create KV-cache (one per layer)
    cache = make_prompt_cache(model)

    denoising_steps = min(steps, block_length)
    num_transfer_schedule = get_num_transfer_tokens(block_length, denoising_steps)

    prefill_blocks = prompt_length // block_length

    # Prefill prompt blocks to populate cache
    if prefill_blocks > 0:
        prefill_length = prefill_blocks * block_length
        prefill_tokens = x[:, :prefill_length]
        prefill_mask = create_block_diagonal_mask(prefill_blocks, block_length)
        logits = model(prefill_tokens, cache=cache, mask=prefill_mask)
        mx.eval(logits)

    for num_block in range(prefill_blocks, num_blocks):
        block_start = num_block * block_length
        block_end = (num_block + 1) * block_length

        prefix_offset = cache[0].offset

        block_mask = mx.zeros(
            (1, 1, block_length, prefix_offset + block_length),
            dtype=mx.bfloat16,
        )

        block_tokens = x[:, block_start:block_end]

        for step in range(denoising_steps):
            active_mask = block_tokens == mask_id
            num_masks = mx.sum(active_mask).item()

            if num_masks == 0:
                break

            trim_amount = cache[0].offset - prefix_offset
            if trim_amount > 0:
                for c in cache:
                    c.trim(trim_amount)

            logits = model(block_tokens, cache=cache, mask=block_mask)

            sampled_tokens, token_probs = sample_tokens(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            num_to_transfer = int(num_transfer_schedule[step].item())
            confidence = mx.where(active_mask, token_probs, mx.array(-float("inf")))

            high_conf_mask = confidence > threshold
            num_high_conf = mx.sum(high_conf_mask).item()

            if num_high_conf >= num_to_transfer:
                transfer_mask = high_conf_mask
            else:
                flat_conf = confidence.reshape(-1)
                k = min(num_to_transfer, int(num_masks))
                if k > 0:
                    top_indices = mx.argpartition(-flat_conf, kth=k-1)[:k]
                    transfer_mask = mx.zeros(flat_conf.shape, dtype=mx.bool_)
                    transfer_mask = transfer_mask.at[top_indices].add(True)
                    transfer_mask = transfer_mask.reshape(active_mask.shape)
                else:
                    transfer_mask = mx.zeros(active_mask.shape, dtype=mx.bool_)

            block_tokens = mx.where(transfer_mask, sampled_tokens, block_tokens)

            # Check for EOS
            if mx.any(block_tokens == eos_id).item():
                x_so_far = mx.concatenate(
                    [x[:, :block_start], block_tokens], axis=1
                )
                eos_mask = x_so_far[0] == eos_id
                indices = mx.arange(x_so_far.shape[1])
                eos_indices = mx.where(
                    eos_mask, indices, mx.array(x_so_far.shape[1])
                )
                eos_pos = int(mx.min(eos_indices).item())
                if eos_pos < x_so_far.shape[1]:
                    prefix = x_so_far[0, prompt_length:eos_pos]
                    if not mx.any(prefix == mask_id).item():
                        generated = x_so_far[0, prompt_length:eos_pos + 1].tolist()
                        yield tokenizer.decode(generated, skip_special_tokens=True)
                        return

            mx.eval(block_tokens)

        # Update full sequence with finalized block
        x = mx.concatenate(
            [x[:, :block_start], block_tokens, x[:, block_end:]], axis=1
        )

        # Commit finalized block to cache
        trim_amount = cache[0].offset - prefix_offset
        if trim_amount > 0:
            for c in cache:
                c.trim(trim_amount)
        commit_logits = model(block_tokens, cache=cache, mask=block_mask)
        mx.eval(commit_logits)

        # Yield text generated so far
        generated = x[0, prompt_length:block_end].tolist()
        generated = [t for t in generated if t != mask_id]
        yield tokenizer.decode(generated, skip_special_tokens=True)


def stream_generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    block_length: int = 32,
    steps: int = 32,
    temperature: float = 0.0,
    top_p: float = None,
    top_k: int = None,
    threshold: float = 0.95,
    mask_id: int = 156895,
    eos_id: int = 156892,
    use_cache: bool = True,
):
    """
    Generate text with streaming output per block.

    Yields generated text after each block is completed.

    Args:
        model: LLaDA2 MoE model instance
        tokenizer: Tokenizer instance
        prompt: Input text prompt
        max_new_tokens: Maximum tokens to generate
        block_length: Block size for generation
        steps: Refinement steps per block
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling threshold
        threshold: Confidence threshold
        mask_id: Mask token ID
        eos_id: EOS token ID
        use_cache: Whether to use KV-cache for the static prefix

    Yields:
        Generated text after each block
    """
    fn = _stream_generate_cached if use_cache else _stream_generate_no_cache
    yield from fn(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        block_length=block_length,
        steps=steps,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        threshold=threshold,
        mask_id=mask_id,
        eos_id=eos_id,
    )
