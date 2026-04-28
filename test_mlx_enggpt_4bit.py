from transformers import AutoTokenizer
from mlx_lm import load, generate

model_path = "../EngGPT2/models/EngGPT2-16B-A3B-MLX-4bit"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
)

model, tokenizer_mlx = load(
    model_path,
    tokenizer_config={"trust_remote_code": True},
)

messages = [
    {
        "role": "user",
        "content": "Spiegami in italiano, in due frasi, che cos'è un modello Mixture of Experts.",
    }
]

prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False,
    enable_thinking=False,
)

response = generate(
    model,
    tokenizer_mlx,
    prompt=prompt,
    max_tokens=150,
    verbose=True,
)

print(response)
