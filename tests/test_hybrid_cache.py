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
    """Test that cache persists across multiple requests (not deleted after use)."""
    print("\n=== Test: Cache Persistence ===")

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

    # Second request (should use cache from first)
    messages2 = messages1 + [
        {"role": "assistant", "content": content1.split('.')[0] + "."},
        {"role": "user", "content": "What is my favorite color?"}
    ]
    response2 = make_request(messages2, enable_thinking=False)
    if "error" in response2:
        print(f"FAILED (request 2): {response2['error']}")
        return False
    cached2 = response2.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)

    # Third request (should STILL use cache from first - this is the key test)
    messages3 = messages1 + [
        {"role": "assistant", "content": content1.split('.')[0] + "."},
        {"role": "user", "content": "Do you remember my favorite color?"}
    ]
    response3 = make_request(messages3, enable_thinking=False)
    if "error" in response3:
        print(f"FAILED (request 3): {response3['error']}")
        return False
    cached3 = response3.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)

    print(f"Request 2 cached: {cached2}")
    print(f"Request 3 cached: {cached3}")

    # Both request 2 and 3 should use cache (cache should not be deleted after use)
    assert cached2 > 0, f"Request 2 should use cache, but cached_tokens={cached2}"
    assert cached3 > 0, f"Request 3 should use cache (cache persistence), but cached_tokens={cached3}"
    assert cached3 == cached2, f"Request 3 should use same cache as request 2 (same prefix)"

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
    """Test that branching conversations share system prompt cache."""
    print("\n=== Test: Conversation Branching ===")

    # Use unique system prompt to avoid interference
    import time
    unique_id = str(int(time.time() * 1000))
    system_prompt = f"You are a math bot. Answer briefly. [{unique_id}]"

    # Request 1: Establish first conversation branch
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

    # Request 2: Different user message, same system (should use system cache)
    messages2 = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "2+2=?"}
    ]
    response2 = make_request(messages2, enable_thinking=False)
    if "error" in response2:
        print(f"FAILED (request 2): {response2['error']}")
        return False

    content2 = response2.get("choices", [{}])[0].get("message", {}).get("content", "")
    cached2 = response2.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
    print(f"Request 2 (branch from system): {cached2} cached")

    # Request 3: Extend first conversation branch
    messages3 = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "1+1=?"},
        {"role": "assistant", "content": content1.strip()},
        {"role": "user", "content": "3+3=?"}
    ]
    response3 = make_request(messages3, enable_thinking=False)
    if "error" in response3:
        print(f"FAILED (request 3): {response3['error']}")
        return False

    cached3 = response3.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
    print(f"Request 3 (extend branch 1): {cached3} cached")

    # Request 4: Extend second conversation branch
    messages4 = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "2+2=?"},
        {"role": "assistant", "content": content2.strip()},
        {"role": "user", "content": "4+4=?"}
    ]
    response4 = make_request(messages4, enable_thinking=False)
    if "error" in response4:
        print(f"FAILED (request 4): {response4['error']}")
        return False

    cached4 = response4.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
    print(f"Request 4 (extend branch 2): {cached4} cached")

    # Verify caching behavior
    assert cached1 == 0, "First request should have 0 cached tokens"
    assert cached2 > 0, f"Request 2 should use system cache, but cached_tokens={cached2}"
    assert cached3 > 0, f"Request 3 should use cache, but cached_tokens={cached3}"
    assert cached4 > 0, f"Request 4 should use cache, but cached_tokens={cached4}"

    print("PASSED")
    return True


def test_cache_survives_multiple_branches():
    """Test that the system prompt cache survives multiple different branches."""
    print("\n=== Test: Cache Survives Multiple Branches ===")

    # Use unique system prompt to avoid interference
    import time
    unique_id = str(int(time.time() * 1000))
    system_prompt = f"You are a helpful assistant. Be very brief. [{unique_id}]"

    # Establish system cache
    messages1 = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "First message"}
    ]
    response1 = make_request(messages1, enable_thinking=False)
    if "error" in response1:
        print(f"FAILED (request 1): {response1['error']}")
        return False

    cached1 = response1.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
    print(f"Request 1: {cached1} cached")

    # Multiple different branches should all use system cache
    cached_tokens_list = []
    for i in range(5):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Different message {i}"}
        ]
        response = make_request(messages, enable_thinking=False)
        if "error" in response:
            print(f"FAILED (branch {i}): {response['error']}")
            return False

        cached = response.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
        cached_tokens_list.append(cached)
        print(f"Branch {i}: {cached} cached")

    # All branches should use system cache
    for i, cached in enumerate(cached_tokens_list):
        assert cached > 0, f"Branch {i} should use system cache, but cached_tokens={cached}"

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
