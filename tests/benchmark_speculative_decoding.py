import mlx_lm

main_model, tokenizer = mlx_lm.load("mlx-community/Qwen2.5-Coder-32B-Instruct-3bit")
# draft_model, _ = mlx_lm.load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
# draft_model, _ = mlx_lm.load("mlx-community/Qwen2.5-Coder-3B-4bit")
draft_model, _ = mlx_lm.load("mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit")


prompt = "Write a Python program to sum up the Fibonacci series"
messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

mlx_lm.generate(
    model=main_model,
    tokenizer=tokenizer,
    prompt=prompt,
    verbose=True,
    max_tokens=512,
)

for num_draft_tokens in range(1, 10):
    mlx_lm.generate(
        model=main_model,
        tokenizer=tokenizer,
        prompt=prompt,
        verbose=True,
        draft_model=draft_model,
        num_draft_tokens=num_draft_tokens,
        max_tokens=512,
    )
