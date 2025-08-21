from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("AntonV/mamba2-130m-hf")
model = AutoModelForCausalLM.from_pretrained("AntonV/mamba2-130m-hf")

input_ids = tokenizer("Hey,", return_tensors="pt")["input_ids"]
out = model.generate(input_ids, max_new_tokens=10, temperature=0.8)
print(tokenizer.batch_decode(out))


"""

python -m mlx_lm.generate --model AntonV/mamba2-130m-hf --prompt "Hey," --temp 0.8
python -m mlx_lm.generate --model AntonV/mamba2-1.3b-hf  --prompt "Einstein was a" --temp 0.8
python -m mlx_lm.generate --model mlx-community/Mamba-Codestral-7B-v0.1-4bit --prompt "<s>[SYSTEM_PROMPT]You are a helpful assistant.[/SYSTEM_PROMPT][INST]Write a function in python[/INST]"
python -m mlx_lm.convert --hf-path
"""