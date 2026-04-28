import argparse
import time

from transformers import AutoTokenizer
from mlx_lm import load, generate

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--max-tokens", type=int, default=180)
args = parser.parse_args()

messages = [
    {
        "role": "user",
        "content": (
            "Spiegami in italiano, in circa 120 parole, che cos'è un modello "
            "Mixture of Experts e perché può essere efficiente."
        ),
    }
]

print("=" * 80)
print("MODEL:", args.model)
print("=" * 80)

tokenizer_hf = AutoTokenizer.from_pretrained(
    args.model,
    trust_remote_code=True,
)

prompt = tokenizer_hf.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False,
    enable_thinking=False,
)

t0 = time.time()
model, tokenizer = load(
    args.model,
    tokenizer_config={"trust_remote_code": True},
)
print(f"Load time: {time.time() - t0:.2f} s")

t0 = time.time()
response = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=args.max_tokens,
    verbose=True,
)
print(f"Wall time: {time.time() - t0:.2f} s")

print("\nRESPONSE:")
print(response)
