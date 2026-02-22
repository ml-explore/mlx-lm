#!/usr/bin/env python3
"""
Test suite for hybrid model caching in mlx_lm server.

This tests the prompt chaining functionality for hybrid models (like Qwen3.5)
that use a mix of KVCache and ArraysCache.
"""

import json
import subprocess
import sys
import time
import urllib.request
import urllib.error

# Test configuration
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 1117
MODEL_PATH = "/Users/sombrax/.lmstudio/models/sombra/QWEN-3.5-MXFP4"
CHAT_TEMPLATE_PATH = "/Users/sombrax/VibeCoding/mlx_server/templates/qwen35.jinja"
SERVER_STARTUP_TIMEOUT = 120  # seconds
REQUEST_TIMEOUT = 180  # seconds


def start_server():
    """Start the mlx_lm server."""
    with open(CHAT_TEMPLATE_PATH) as f:
        chat_template = f.read()

    cmd = [
        "python3", "-m", "mlx_lm.server",
        "--host", SERVER_HOST,
        "--port", str(SERVER_PORT),
        "--model", MODEL_PATH,
        "--trust-remote-code",
        "--log-level", "INFO",
        "--chat-template", chat_template,
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Wait for server to be ready
    start_time = time.time()
    while time.time() - start_time < SERVER_STARTUP_TIMEOUT:
        try:
            req = urllib.request.urlopen(
                f"http://{SERVER_HOST}:{SERVER_PORT}/health",
                timeout=5
            )
            if req.status == 200:
                return proc
        except (urllib.error.URLError, ConnectionRefusedError):
            pass
        time.sleep(2)

    proc.terminate()
    raise RuntimeError("Server failed to start within timeout")


def stop_server(proc):
    """Stop the mlx_lm server."""
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def make_request(messages, max_tokens=50, enable_thinking=False):
    """Make a chat completion request."""
    data = {
        "model": MODEL_PATH,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": enable_thinking}
    }

    req = urllib.request.Request(
        f"http://{SERVER_HOST}:{SERVER_PORT}/v1/chat/completions",
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        response = urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT)
        return json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode()}


def test_basic_generation():
    """Test basic generation works."""
    print("\n=== Test: Basic Generation ===")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello"}
    ]
    response = make_request(messages, enable_thinking=False)

    if "error" in response:
        print(f"FAILED: {response['error']}")
        return False

    cached = response.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
    content = response.get("choices", [{}])[0].get("message", {}).get("content", "")

    print(f"Prompt tokens: {response.get('usage', {}).get('prompt_tokens', 'N/A')}")
    print(f"Cached tokens: {cached}")
    print(f"Generated: {content[:100]}")

    assert cached == 0, "First request should have 0 cached tokens"
    assert len(content) > 0, "Should generate content"

    print("PASSED")
    return True


def test_prompt_chaining():
    """Test that prompt chaining uses cache."""
    print("\n=== Test: Prompt Chaining ===")

    # First request
    messages1 = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    response1 = make_request(messages1, enable_thinking=False)

    if "error" in response1:
        print(f"FAILED (request 1): {response1['error']}")
        return False

    content1 = response1.get("choices", [{}])[0].get("message", {}).get("content", "")
    print(f"First response: {content1[:100]}")

    # Second request (extending the conversation)
    messages2 = messages1 + [
        {"role": "assistant", "content": content1.split('.')[0] + "."},  # Use part of the actual response
        {"role": "user", "content": "What about Germany?"}
    ]
    response2 = make_request(messages2, enable_thinking=False)

    if "error" in response2:
        print(f"FAILED (request 2): {response2['error']}")
        return False

    cached2 = response2.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
    content2 = response2.get("choices", [{}])[0].get("message", {}).get("content", "")

    print(f"Second request prompt tokens: {response2.get('usage', {}).get('prompt_tokens', 'N/A')}")
    print(f"Second request cached tokens: {cached2}")
    print(f"Second response: {content2[:100]}")

    assert cached2 > 0, f"Second request should use cache, but cached_tokens={cached2}"

    print("PASSED")
    return True


def test_extended_prompt_chaining():
    """Test extended prompt chaining with multiple turns."""
    print("\n=== Test: Extended Prompt Chaining ===")

    # Use unique system prompt to avoid interference from other tests
    import time
    unique_id = str(int(time.time() * 1000))

    # Build up a conversation
    messages = [
        {"role": "system", "content": f"You are a helpful assistant. [{unique_id}]"},
    ]

    cached_tokens_history = []

    # First turn
    messages.append({"role": "user", "content": "My name is Alice."})
    response = make_request(messages, enable_thinking=False)
    if "error" in response:
        print(f"FAILED (turn 1): {response['error']}")
        return False
    content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
    messages.append({"role": "assistant", "content": content.split('.')[0] + "."})
    cached = response.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
    cached_tokens_history.append(cached)
    print(f"Turn 1: cached={cached}, response={content[:50]}...")

    # Second turn
    messages.append({"role": "user", "content": "What is my name?"})
    response = make_request(messages, enable_thinking=False)
    if "error" in response:
        print(f"FAILED (turn 2): {response['error']}")
        return False
    content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
    messages.append({"role": "assistant", "content": content.split('.')[0] + "."})
    cached = response.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
    cached_tokens_history.append(cached)
    print(f"Turn 2: cached={cached}, response={content[:50]}...")

    # Third turn
    messages.append({"role": "user", "content": "Can you count to 5?"})
    response = make_request(messages, enable_thinking=False)
    if "error" in response:
        print(f"FAILED (turn 3): {response['error']}")
        return False
    cached = response.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
    cached_tokens_history.append(cached)
    print(f"Turn 3: cached={cached}")

    # Verify cache is being used
    assert cached_tokens_history[0] == 0, "First request should have 0 cached tokens"
    assert cached_tokens_history[1] > 0, "Second request should use cache"
    assert cached_tokens_history[2] > cached_tokens_history[1], "Third request should use more cache"

    print("PASSED")
    return True


def test_cache_persistence():
    """Test that cache persists when extending the same conversation."""
    print("\n=== Test: Cache Persistence (Same Conversation Extension) ===")

    # Use unique system prompt to avoid interference from other tests
    import time
    unique_id = str(int(time.time() * 1000))

    # First request
    messages1 = [
        {"role": "system", "content": f"You are a helpful assistant. [{unique_id}]"},
        {"role": "user", "content": "My favorite color is blue."}
    ]
    response1 = make_request(messages1, enable_thinking=False)
    if "error" in response1:
        print(f"FAILED (request 1): {response1['error']}")
        return False
    content1 = response1.get("choices", [{}])[0].get("message", {}).get("content", "")

    # Second request - extend with actual generated response (should use cache from first)
    messages2 = messages1 + [
        {"role": "assistant", "content": content1.strip()},
        {"role": "user", "content": "What is my favorite color?"}
    ]
    response2 = make_request(messages2, enable_thinking=False)
    if "error" in response2:
        print(f"FAILED (request 2): {response2['error']}")
        return False
    cached2 = response2.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
    content2 = response2.get("choices", [{}])[0].get("message", {}).get("content", "")

    # Third request - continue extending with actual generated response (should use cache from second)
    messages3 = messages2 + [
        {"role": "assistant", "content": content2.strip()},
        {"role": "user", "content": "Do you remember my favorite color?"}
    ]
    response3 = make_request(messages3, enable_thinking=False)
    if "error" in response3:
        print(f"FAILED (request 3): {response3['error']}")
        return False
    cached3 = response3.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)

    print(f"Request 1 cached: 0 (first request)")
    print(f"Request 2 cached: {cached2}")
    print(f"Request 3 cached: {cached3}")

    # Each subsequent request should have more cached tokens than the previous
    assert cached2 > 0, f"Request 2 should use cache, but cached_tokens={cached2}"
    assert cached3 > cached2, f"Request 3 should have more cached tokens than request 2, but cached3={cached3} vs cached2={cached2}"

    print("PASSED")
    return True


def test_cache_invalidation():
    """Test that different conversations don't share cache."""
    print("\n=== Test: Cache Invalidation ===")

    # Use unique system prompts to avoid interference
    import time
    unique_id = str(int(time.time() * 1000))

    # Conversation A
    messages_a = [
        {"role": "system", "content": f"You are a helpful assistant. [{unique_id}A]"},
        {"role": "user", "content": "Remember the number 42."}
    ]
    response_a1 = make_request(messages_a, enable_thinking=False)
    if "error" in response_a1:
        print(f"FAILED (conv A): {response_a1['error']}")
        return False

    # Conversation B (different system prompt)
    messages_b = [
        {"role": "system", "content": f"You are a pirate assistant. Arr! [{unique_id}B]"},
        {"role": "user", "content": "Remember the number 42."}
    ]
    response_b = make_request(messages_b, enable_thinking=False)
    if "error" in response_b:
        print(f"FAILED (conv B): {response_b['error']}")
        return False

    cached_b = response_b.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)

    # Conversation B should not use cache from conversation A (different system prompt)
    assert cached_b == 0, f"Different conversations should not share cache, but cached_tokens={cached_b}"

    print("PASSED")
    return True


def test_conversation_branching():
    """Test that extending a conversation properly moves the cache.

    Note: With the 'move' caching strategy, branching from intermediate points
    (like the system prompt) is not supported. Each conversation path has its
    own cache that gets moved as the conversation extends.
    """
    print("\n=== Test: Conversation Extension (Cache Move) ===")

    # Use unique system prompt to avoid interference
    import time
    unique_id = str(int(time.time() * 1000))
    system_prompt = f"You are a math bot. Answer briefly. [{unique_id}]"

    # Request 1: Establish conversation
    messages1 = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "1+1=?"}
    ]
    response1 = make_request(messages1, enable_thinking=False)
    if "error" in response1:
        print(f"FAILED (request 1): {response1['error']}")
        return False

    content1 = response1.get("choices", [{}])[0].get("message", {}).get("content", "")
    cached1 = response1.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
    print(f"Request 1 (first): {cached1} cached")

    # Request 2: Extend conversation (should use cache from request 1)
    messages2 = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "1+1=?"},
        {"role": "assistant", "content": content1.strip()},
        {"role": "user", "content": "2+2=?"}
    ]
    response2 = make_request(messages2, enable_thinking=False)
    if "error" in response2:
        print(f"FAILED (request 2): {response2['error']}")
        return False

    content2 = response2.get("choices", [{}])[0].get("message", {}).get("content", "")
    cached2 = response2.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
    print(f"Request 2 (extend): {cached2} cached")

    # Request 3: Extend further (should use cache from request 2)
    messages3 = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "1+1=?"},
        {"role": "assistant", "content": content1.strip()},
        {"role": "user", "content": "2+2=?"},
        {"role": "assistant", "content": content2.strip()},
        {"role": "user", "content": "3+3=?"}
    ]
    response3 = make_request(messages3, enable_thinking=False)
    if "error" in response3:
        print(f"FAILED (request 3): {response3['error']}")
        return False

    cached3 = response3.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
    print(f"Request 3 (extend further): {cached3} cached")

    # Verify caching behavior: each extension should have more cached tokens
    assert cached1 == 0, "First request should have 0 cached tokens"
    assert cached2 > cached1, f"Request 2 should have more cache than request 1"
    assert cached3 > cached2, f"Request 3 should have more cache than request 2"

    print("PASSED")
    return True


def test_cache_survives_multiple_branches():
    """Test that multiple conversation extensions work correctly.

    Note: With the 'move' caching strategy, we maintain one cache per
    conversation path, not per system prompt.
    """
    print("\n=== Test: Multiple Extensions ===")

    # Use unique system prompt to avoid interference
    import time
    unique_id = str(int(time.time() * 1000))
    system_prompt = f"You are a helpful assistant. Be very brief. [{unique_id}]"

    # Establish conversation
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "First message"}
    ]
    response = make_request(messages, enable_thinking=False)
    if "error" in response:
        print(f"FAILED (initial): {response['error']}")
        return False

    last_content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
    cached_tokens_list = [0]

    # Multiple extensions should each use the previous cache
    for i in range(5):
        messages.append({"role": "assistant", "content": last_content.strip()})
        messages.append({"role": "user", "content": f"Message {i+2}"})
        response = make_request(messages, enable_thinking=False)
        if "error" in response:
            print(f"FAILED (extension {i}): {response['error']}")
            return False

        cached = response.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
        cached_tokens_list.append(cached)
        last_content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        print(f"Extension {i+1}: {cached} cached")

    # Each extension should have more cached tokens than the previous
    for i in range(1, len(cached_tokens_list)):
        assert cached_tokens_list[i] > cached_tokens_list[i-1], \
            f"Extension {i} should have more cache than extension {i-1}"

    print("PASSED")
    return True


def test_exact_match_reuse():
    """Test that identical requests reuse exact cache match."""
    print("\n=== Test: Exact Match Reuse ===")

    # Use unique system prompt to avoid interference
    import time
    unique_id = str(int(time.time() * 1000))

    messages = [
        {"role": "system", "content": f"You are a helpful assistant. [{unique_id}]"},
        {"role": "user", "content": "What is 2+2?"}
    ]

    # First request
    response1 = make_request(messages, enable_thinking=False)
    if "error" in response1:
        print(f"FAILED (request 1): {response1['error']}")
        return False
    cached1 = response1.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
    print(f"First request cached: {cached1}")

    # Second identical request (should get exact match)
    response2 = make_request(messages, enable_thinking=False)
    if "error" in response2:
        print(f"FAILED (request 2): {response2['error']}")
        return False
    cached2 = response2.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
    prompt_tokens2 = response2.get("usage", {}).get("prompt_tokens", 0)
    print(f"Second request cached: {cached2}, prompt_tokens: {prompt_tokens2}")

    # Second request should have all tokens cached (exact match)
    assert cached1 == 0, "First request should have 0 cached tokens"
    assert cached2 == prompt_tokens2, f"Second request should have all tokens cached (exact match), got {cached2}/{prompt_tokens2}"

    print("PASSED")
    return True


def test_thinking_mode_caching():
    """Test that thinking mode works with caching."""
    print("\n=== Test: Thinking Mode Caching ===")

    # Use unique system prompt
    import time
    unique_id = str(int(time.time() * 1000))

    messages1 = [
        {"role": "system", "content": f"You are a helpful assistant. [{unique_id}]"},
        {"role": "user", "content": "Think about what 5+5 equals."}
    ]

    # First request with thinking enabled
    response1 = make_request(messages1, max_tokens=100, enable_thinking=True)
    if "error" in response1:
        print(f"FAILED (request 1): {response1['error']}")
        return False
    cached1 = response1.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
    content1 = response1.get("choices", [{}])[0].get("message", {}).get("content", "")
    reasoning1 = response1.get("choices", [{}])[0].get("message", {}).get("reasoning", "")
    print(f"First request cached: {cached1}")
    print(f"Has reasoning: {len(reasoning1) > 0}")

    # Second request extending the conversation
    messages2 = messages1 + [
        {"role": "assistant", "content": content1[:50] if content1 else "10"},
        {"role": "user", "content": "Now what about 6+6?"}
    ]
    response2 = make_request(messages2, max_tokens=100, enable_thinking=True)
    if "error" in response2:
        print(f"FAILED (request 2): {response2['error']}")
        return False
    cached2 = response2.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
    print(f"Second request cached: {cached2}")

    assert cached1 == 0, "First request should have 0 cached tokens"
    assert cached2 > 0, f"Second request should use cache, but cached_tokens={cached2}"

    print("PASSED")
    return True


def test_interleaved_conversations():
    """Test multiple interleaved conversations don't interfere."""
    print("\n=== Test: Interleaved Conversations ===")

    import time
    unique_id = str(int(time.time() * 1000))

    # Conversation A
    messages_a = [
        {"role": "system", "content": f"You are a math bot. [{unique_id}A]"},
        {"role": "user", "content": "Remember the number 100."}
    ]
    response_a1 = make_request(messages_a, enable_thinking=False)
    if "error" in response_a1:
        print(f"FAILED (conv A1): {response_a1['error']}")
        return False
    content_a1 = response_a1.get("choices", [{}])[0].get("message", {}).get("content", "")
    print(f"Conv A1: Got response")

    # Conversation B (different)
    messages_b = [
        {"role": "system", "content": f"You are a math bot. [{unique_id}B]"},
        {"role": "user", "content": "Remember the number 200."}
    ]
    response_b1 = make_request(messages_b, enable_thinking=False)
    if "error" in response_b1:
        print(f"FAILED (conv B1): {response_b1['error']}")
        return False
    content_b1 = response_b1.get("choices", [{}])[0].get("message", {}).get("content", "")
    print(f"Conv B1: Got response")

    # Extend conversation A
    messages_a2 = messages_a + [
        {"role": "assistant", "content": content_a1[:50]},
        {"role": "user", "content": "What number did I tell you?"}
    ]
    response_a2 = make_request(messages_a2, enable_thinking=False)
    if "error" in response_a2:
        print(f"FAILED (conv A2): {response_a2['error']}")
        return False
    cached_a2 = response_a2.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
    print(f"Conv A2: cached={cached_a2}")

    # Extend conversation B
    messages_b2 = messages_b + [
        {"role": "assistant", "content": content_b1[:50]},
        {"role": "user", "content": "What number did I tell you?"}
    ]
    response_b2 = make_request(messages_b2, enable_thinking=False)
    if "error" in response_b2:
        print(f"FAILED (conv B2): {response_b2['error']}")
        return False
    cached_b2 = response_b2.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
    print(f"Conv B2: cached={cached_b2}")

    # Both conversations should use their own cache
    assert cached_a2 > 0, f"Conv A2 should use cache, but cached_tokens={cached_a2}"
    assert cached_b2 > 0, f"Conv B2 should use cache, but cached_tokens={cached_b2}"

    print("PASSED")
    return True


def test_long_conversation_chain():
    """Test caching with a long conversation chain."""
    print("\n=== Test: Long Conversation Chain ===")

    import time
    unique_id = str(int(time.time() * 1000))

    messages = [
        {"role": "system", "content": f"You are a helpful assistant. Be very brief. [{unique_id}]"}
    ]

    cached_tokens_history = []

    # Build up a conversation with 10 turns
    for i in range(10):
        messages.append({"role": "user", "content": f"Turn {i+1}: Say 'ok'."})
        response = make_request(messages, max_tokens=5, enable_thinking=False)
        if "error" in response:
            print(f"FAILED (turn {i+1}): {response['error']}")
            return False

        cached = response.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
        cached_tokens_history.append(cached)

        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        messages.append({"role": "assistant", "content": content[:20]})

        if i < 3 or i >= 8:  # Print first few and last few
            print(f"Turn {i+1}: cached={cached}")

    # Verify cache grows with each turn
    assert cached_tokens_history[0] == 0, "First request should have 0 cached tokens"

    # Check that cache generally increases (with some tolerance for slight variations)
    for i in range(1, len(cached_tokens_history)):
        # Allow small variations due to tokenization differences
        assert cached_tokens_history[i] >= cached_tokens_history[i-1] - 5, \
            f"Turn {i+1} cache ({cached_tokens_history[i]}) should not be much less than turn {i} ({cached_tokens_history[i-1]})"

    print(f"Final cached tokens: {cached_tokens_history[-1]}")
    print("PASSED")
    return True


def test_cache_with_different_params():
    """Test that different generation params still use same cache for prompt."""
    print("\n=== Test: Cache with Different Generation Params ===")

    import time
    unique_id = str(int(time.time() * 1000))

    messages = [
        {"role": "system", "content": f"You are a helpful assistant. [{unique_id}]"},
        {"role": "user", "content": "Count to 3."}
    ]

    # First request with temp 0
    data1 = {
        "model": MODEL_PATH,
        "messages": messages,
        "max_tokens": 10,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False}
    }
    req1 = urllib.request.Request(
        f"http://{SERVER_HOST}:{SERVER_PORT}/v1/chat/completions",
        data=json.dumps(data1).encode(),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    response1 = json.loads(urllib.request.urlopen(req1, timeout=REQUEST_TIMEOUT).read().decode())
    if "error" in response1:
        print(f"FAILED (request 1): {response1['error']}")
        return False
    cached1 = response1.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
    print(f"Request 1 (temp=0.0): cached={cached1}")

    # Second request with temp 1.0 (different generation param)
    data2 = {
        "model": MODEL_PATH,
        "messages": messages,
        "max_tokens": 10,
        "temperature": 1.0,
        "chat_template_kwargs": {"enable_thinking": False}
    }
    req2 = urllib.request.Request(
        f"http://{SERVER_HOST}:{SERVER_PORT}/v1/chat/completions",
        data=json.dumps(data2).encode(),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    response2 = json.loads(urllib.request.urlopen(req2, timeout=REQUEST_TIMEOUT).read().decode())
    if "error" in response2:
        print(f"FAILED (request 2): {response2['error']}")
        return False
    cached2 = response2.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
    prompt_tokens2 = response2.get("usage", {}).get("prompt_tokens", 0)
    print(f"Request 2 (temp=1.0): cached={cached2}, prompt_tokens={prompt_tokens2}")

    # Second request should get exact match (all tokens cached) despite different temperature
    assert cached1 == 0, "First request should have 0 cached tokens"
    assert cached2 == prompt_tokens2, f"Second request should have all tokens cached (exact match), got {cached2}/{prompt_tokens2}"

    print("PASSED")
    return True


def test_partial_cache_extension():
    """Test that partial cache matches extend correctly."""
    print("\n=== Test: Partial Cache Extension ===")

    import time
    unique_id = str(int(time.time() * 1000))

    # First, establish a base conversation
    messages1 = [
        {"role": "system", "content": f"You are a helpful assistant. [{unique_id}]"},
        {"role": "user", "content": "My name is Test."}
    ]
    response1 = make_request(messages1, enable_thinking=False)
    if "error" in response1:
        print(f"FAILED (request 1): {response1['error']}")
        return False
    content1 = response1.get("choices", [{}])[0].get("message", {}).get("content", "")
    print(f"Request 1: Base conversation established")

    # Extend the conversation
    messages2 = messages1 + [
        {"role": "assistant", "content": content1[:30]},
        {"role": "user", "content": "What is my name?"}
    ]
    response2 = make_request(messages2, enable_thinking=False)
    if "error" in response2:
        print(f"FAILED (request 2): {response2['error']}")
        return False
    cached2 = response2.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
    content2 = response2.get("choices", [{}])[0].get("message", {}).get("content", "")
    print(f"Request 2: cached={cached2}")

    # Extend further
    messages3 = messages2 + [
        {"role": "assistant", "content": content2[:30]},
        {"role": "user", "content": "Can you spell it?"}
    ]
    response3 = make_request(messages3, enable_thinking=False)
    if "error" in response3:
        print(f"FAILED (request 3): {response3['error']}")
        return False
    cached3 = response3.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
    print(f"Request 3: cached={cached3}")

    # Verify cache increases with each extension
    assert cached2 > 0, f"Request 2 should use cache, got {cached2}"
    assert cached3 > cached2, f"Request 3 ({cached3}) should have more cache than request 2 ({cached2})"

    print("PASSED")
    return True


def run_tests():
    """Run all tests."""
    print("=" * 60)
    print("Hybrid Model Caching Tests")
    print("=" * 60)

    # Start server
    print("\nStarting server...")
    server_proc = start_server()
    print("Server started successfully!")

    try:
        results = {}

        # Run tests
        results["basic_generation"] = test_basic_generation()
        results["prompt_chaining"] = test_prompt_chaining()
        results["extended_prompt_chaining"] = test_extended_prompt_chaining()
        results["cache_persistence"] = test_cache_persistence()
        results["cache_invalidation"] = test_cache_invalidation()
        results["conversation_branching"] = test_conversation_branching()
        results["cache_survives_multiple_branches"] = test_cache_survives_multiple_branches()
        results["exact_match_reuse"] = test_exact_match_reuse()
        results["thinking_mode_caching"] = test_thinking_mode_caching()
        results["interleaved_conversations"] = test_interleaved_conversations()
        results["long_conversation_chain"] = test_long_conversation_chain()
        results["cache_with_different_params"] = test_cache_with_different_params()
        results["partial_cache_extension"] = test_partial_cache_extension()

        # Summary
        print("\n" + "=" * 60)
        print("Test Results Summary")
        print("=" * 60)

        all_passed = True
        for test_name, passed in results.items():
            status = "PASSED" if passed else "FAILED"
            print(f"  {test_name}: {status}")
            if not passed:
                all_passed = False

        print()
        if all_passed:
            print("All tests PASSED!")
            return 0
        else:
            print("Some tests FAILED!")
            return 1

    finally:
        print("\nStopping server...")
        stop_server(server_proc)


if __name__ == "__main__":
    sys.exit(run_tests())
