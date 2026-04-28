import time
from transformers import AutoTokenizer
from mlx_lm import load, generate

MODELS = [
    "../EngGPT2/models/EngGPT2-16B-A3B-MLX",
    "../EngGPT2/models/EngGPT2-16B-A3B-MLX-4bit",
    "../EngGPT2/models/EngGPT2-16B-A3B-MLX-8bit",
]

messages = [
    {
        "role": "user",
        "content": (
            "Spiegami in italiano, in circa 120 parole, che cos'è un modello "
            "Mixture of Experts e perché può essere efficiente."
        ),
    }
]

for model_path in MODELS:
    print("\n" + "=" * 80)
    print("MODEL:", model_path)
    print("=" * 80)

    tokenizer_hf = AutoTokenizer.from_pretrained(
        model_path,
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
        model_path,
        tokenizer_config={"trust_remote_code": True},
    )
    t_load = time.time() - t0

    print(f"Load time: {t_load:.2f} s")

    t0 = time.time()
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=180,
        verbose=True,
    )
    t_gen = time.time() - t0

    print(f"\nWall generation time: {t_gen:.2f} s")
    print("\nRESPONSE:")
    print(response)
