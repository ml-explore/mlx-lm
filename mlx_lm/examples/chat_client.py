import argparse
import sys
import time

try:
    from openai import OpenAI
except ImportError:
    print("Error: 'openai' module not found. Please install it using 'pip install openai'")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Simple CLI Chat Client for MLX Dynamic Batch Server")
    parser.add_argument("--base-url", type=str, default="http://localhost:8080/v1", help="Base URL for the API")
    parser.add_argument("--api-key", type=str, default="EMPTY", help="API Key")
    parser.add_argument("--model", type=str, default="default_model", help="Model name")
    parser.add_argument("--prompt", type=str, required=True, help="User prompt")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    
    args = parser.parse_args()

    # Sanitize base_url to avoid double path issues with OpenAI client
    if args.base_url.endswith("/chat/completions"):
        args.base_url = args.base_url[:-17]
    if args.base_url.endswith("/"):
        args.base_url = args.base_url[:-1]

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    print(f"Sending request to {args.base_url}...")
    print(f"Prompt: {args.prompt}")
    print("-" * 40)

    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": args.prompt}],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            stream=True
        )

        first_token_time = None
        token_count = 0
        
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                if first_token_time is None:
                    first_token_time = time.time()
                
                print(content, end="", flush=True)
                token_count += 1
        
        end_time = time.time()
        print("\n" + "-" * 40)
        
        total_time = end_time - start_time
        ttft = (first_token_time - start_time) if first_token_time else total_time
        tps = token_count / (total_time - ttft) if (total_time - ttft) > 0 else 0

        print(f"\nStats:")
        print(f"Total Time: {total_time:.2f}s")
        print(f"TTFT: {ttft:.2f}s")
        print(f"Tokens: {token_count}")
        print(f"Generation TPS: {tps:.2f} tokens/s")

    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
