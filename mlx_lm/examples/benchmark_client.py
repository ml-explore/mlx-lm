import argparse
import time
import threading
import random
from queue import Queue
import sys

try:
    from openai import OpenAI
except ImportError:
    print("Error: 'openai' module not found. Please install it using 'pip install openai'")
    sys.exit(1)

def send_request(base_url, api_key, model, prompt, max_tokens, stream):
    client = OpenAI(base_url=base_url, api_key=api_key)
    
    start_time = time.time()
    first_token_time = None
    token_times = []
    token_count = 0
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            stream=stream
        )
        
        if stream:
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    current_time = time.time()
                    if first_token_time is None:
                        first_token_time = current_time
                    token_times.append(current_time)
                    token_count += 1
        else:
            # Non-streaming
            _ = response.choices[0].message.content
            # Try to get usage if available
            if response.usage:
                token_count = response.usage.completion_tokens
            else:
                token_count = -1 # Unknown without tokenizer
            
    except Exception as e:
        print(f"Request failed: {e}")
        return 0, 0, 0, 0, 0, 0

    end_time = time.time()
    duration = end_time - start_time
    ttft = (first_token_time - start_time) if first_token_time else duration 
    
    # Calculate smoothness stats (Inter-Token Latency)
    latencies = []
    if len(token_times) > 1:
        latencies = [token_times[i] - token_times[i-1] for i in range(1, len(token_times))]
    
    avg_itl = 0
    std_itl = 0
    max_itl = 0
    
    if latencies:
        avg_itl = sum(latencies) / len(latencies)
        max_itl = max(latencies)
        if len(latencies) > 1:
            variance = sum((x - avg_itl) ** 2 for x in latencies) / (len(latencies) - 1)
            std_itl = variance ** 0.5
            
    return token_count, duration, ttft, avg_itl, std_itl, max_itl

def worker(base_url, api_key, model, prompts, max_tokens_range, stream, num_requests, results_queue, min_delay, max_delay):
    for _ in range(num_requests):
        # Simulate random user think time/delay
        delay = random.uniform(min_delay, max_delay)
        time.sleep(delay)

        # Pick a random prompt and max_tokens
        prompt = random.choice(prompts)
        max_tokens = random.randint(max_tokens_range[0], max_tokens_range[1])

        result = send_request(base_url, api_key, model, prompt, max_tokens, stream)
        results_queue.put(result)

def main():
    parser = argparse.ArgumentParser(description="Benchmark MLX Dynamic Batch Server")
    parser.add_argument("--base-url", type=str, default="http://localhost:8080/v1", help="Base URL for the API")
    parser.add_argument("--api-key", type=str, default="EMPTY", help="API Key")
    parser.add_argument("--model", type=str, default="default_model", help="Model name")
    parser.add_argument("--concurrency", type=int, default=4, help="Number of concurrent threads")
    parser.add_argument("--requests", type=int, default=20, help="Total number of requests to send")
    parser.add_argument("--stream", action="store_true", help="Use streaming (required for accurate client-side token counting)")
    parser.add_argument("--min-delay", type=float, default=0.1, help="Min delay between requests (seconds)")
    parser.add_argument("--max-delay", type=float, default=2.0, help="Max delay between requests (seconds)")
    parser.add_argument("--min-tokens", type=int, default=10, help="Min max_tokens for generation")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max max_tokens for generation")
    args = parser.parse_args()

    # Sanitize base_url to avoid double path issues with OpenAI client
    if args.base_url.endswith("/chat/completions"):
        args.base_url = args.base_url[:-17]
    if args.base_url.endswith("/"):
        args.base_url = args.base_url[:-1]

    # A diverse set of prompts with varying lengths
    prompts = [
        "Hi",
        "What is the capital of France?",
        "Write a haiku about winter.",
        "Explain quantum entanglement to a 5-year-old.",
        "Write a Python function to calculate the Fibonacci sequence recursively.",
        "Summarize the plot of Romeo and Juliet in one paragraph.",
        "Write a short story about a robot who discovers emotions. The story should be at least 500 words long.",
        "Translate 'Hello, how are you?' into Spanish, French, German, and Japanese.",
        "List 10 fun facts about octopuses.",
        "Write a detailed essay about the impact of the industrial revolution on modern society.",
        "what is the meaning of life, the universe, and everything?",
        "how to make a perfect cup of coffee?",
        "is there a god?",
        "what is the airspeed velocity of an unladen swallow?" ,
        "why is the sky blue?",
        "who won the world series in 2020?",       
        "Describe the process of photosynthesis in detail.",
        "Generate a list of 20 unique startup ideas in the field of renewable energy.",
        "Create a meal plan for a week that is vegan and high in protein.",
        "Explain the theory of relativity in simple terms.",
        "Design a workout routine for building muscle mass over 3 months.",
        "What are the health benefits of meditation?",
        "How does blockchain technology work?"       
        
    ]

    print(f"Starting benchmark against {args.base_url}")
    print(f"Concurrency: {args.concurrency}, Total Requests: {args.requests}")
    print(f"Delay Range: {args.min_delay}s - {args.max_delay}s")
    print(f"Max Tokens Range: {args.min_tokens} - {args.max_tokens}")
    
    queue = Queue()
    threads = []
    requests_per_thread = args.requests // args.concurrency
    remainder = args.requests % args.concurrency
    
    start_time = time.time()
    
    for i in range(args.concurrency):
        count = requests_per_thread + (1 if i < remainder else 0)
        t = threading.Thread(target=worker, args=(args.base_url, args.api_key, args.model, prompts, (args.min_tokens, args.max_tokens), args.stream, count, queue, args.min_delay, args.max_delay))
        t.start()
        threads.append(t)
        
    for t in threads:
        t.join()
        
    total_time = time.time() - start_time
    
    total_tokens = 0
    total_duration_sum = 0
    total_ttft_sum = 0
    
    # Smoothness stats aggregators
    total_avg_itl_sum = 0
    total_std_itl_sum = 0
    global_max_itl = 0
    valid_itl_count = 0
    
    count = 0
    
    while not queue.empty():
        t, d, ttft, avg_itl, std_itl, max_itl = queue.get()
        if t >= 0:
            total_tokens += t
        total_duration_sum += d
        total_ttft_sum += ttft
        
        if t > 1: # Only count ITL stats if we had at least 2 tokens
            total_avg_itl_sum += avg_itl
            total_std_itl_sum += std_itl
            global_max_itl = max(global_max_itl, max_itl)
            valid_itl_count += 1
            
        count += 1
        
    print("\nBenchmark Results:")
    print(f"Total Wall Time: {total_time:.2f}s")
    print(f"Total Requests Completed: {count}")
    print(f"Average Latency: {total_duration_sum/count:.2f}s")
    print(f"Average TTFT (Time To First Token): {total_ttft_sum/count:.2f}s")
    print(f"Request Throughput: {count/total_time:.2f} req/s")
    
    if valid_itl_count > 0:
        print(f"Average Inter-Token Latency (ITL): {total_avg_itl_sum/valid_itl_count:.4f}s")
        print(f"Average ITL Jitter (Std Dev): {total_std_itl_sum/valid_itl_count:.4f}s")
        print(f"Max ITL Observed: {global_max_itl:.4f}s")
    
    if args.stream:
        print(f"Total Tokens Received: {total_tokens}")
        print(f"Token Throughput (Client-side): {total_tokens/total_time:.2f} tokens/s")
    else:
        print("Note: Token throughput not available in non-streaming mode without client-side tokenizer.")
        print("Check server logs for server-side TPS.")

if __name__ == "__main__":
    main()
